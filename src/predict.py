from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.io import load_metadata, load_sessions
from src.model import TemporalCNNBaseline
from src.preprocess import apply_denorm, apply_norm, compute_test_session_stats
from src.utils import resolve_device


class InferenceWindowDataset(Dataset):
    def __init__(
        self,
        sbp_norm: np.ndarray,
        obs_mask_full: np.ndarray,
        kinematics: np.ndarray,
        trial_ids: np.ndarray,
        window_size: int,
    ) -> None:
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        self.sbp_norm = sbp_norm
        self.obs_mask_full = obs_mask_full
        self.kinematics = kinematics
        self.trial_ids = trial_ids
        self.window_size = window_size
        self.half = window_size // 2
        self.centers = self._valid_centers()

    def _valid_centers(self) -> np.ndarray:
        trial_ids = self.trial_ids
        n_bins = len(trial_ids)
        if n_bins == 0:
            return np.empty((0,), dtype=np.int64)

        boundaries = np.where(np.diff(trial_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [n_bins]])

        centers = []
        for s, e in zip(starts, ends):
            tid = trial_ids[s]
            if tid < 0:
                continue
            lo = s + self.half
            hi = e - self.half
            if lo < hi:
                centers.append(np.arange(lo, hi, dtype=np.int64))

        if not centers:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(centers)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        center = int(self.centers[idx])
        start = center - self.half
        end = center + self.half + 1
        x_sbp = self.sbp_norm[start:end]
        obs_mask = self.obs_mask_full[start:end]
        x_kin = self.kinematics[start:end]
        return {
            "center": center,
            "x_sbp": torch.from_numpy(x_sbp.astype(np.float32)),
            "obs_mask": torch.from_numpy(obs_mask.astype(np.float32)),
            "x_kin": torch.from_numpy(x_kin.astype(np.float32)),
        }


def _load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoint stores numpy arrays (e.g., train_global_mean/std), so we need full unpickling.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions that do not expose weights_only.
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = ckpt["model_kwargs"]
    model = TemporalCNNBaseline(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def _predict_session_matrix(
    model: torch.nn.Module,
    sbp_masked: np.ndarray,
    kin: np.ndarray,
    trial_ids: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    sbp_norm = apply_norm(sbp_masked, mean, std)
    # Match train-time masking semantics: masked input channels are zero in normalized space.
    sbp_norm = sbp_norm.copy()
    sbp_norm[mask] = 0.0
    obs_mask_full = (~mask).astype(np.float32)

    # Baseline fallback at bins where full window inference is unavailable.
    pred_norm = np.broadcast_to(np.zeros((1, 96), dtype=np.float32), sbp_masked.shape).copy()

    ds = InferenceWindowDataset(
        sbp_norm=sbp_norm,
        obs_mask_full=obs_mask_full,
        kinematics=kin,
        trial_ids=trial_ids,
        window_size=window_size,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for batch in loader:
        x_sbp = batch["x_sbp"].to(device)
        x_kin = batch["x_kin"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        centers = batch["center"].cpu().numpy().astype(np.int64)
        y_hat = model(x_sbp, x_kin, obs_mask).cpu().numpy().astype(np.float32)
        pred_norm[centers] = y_hat

    pred = apply_denorm(pred_norm, mean, std)

    # Keep observed entries unchanged; only masked entries are reconstructed.
    out = sbp_masked.copy()
    out[mask] = pred[mask]
    return out


def build_submission(
    predictions_by_session: Dict[str, np.ndarray],
    sample_submission_path: Path | None,
    output_path: Path,
) -> None:
    if sample_submission_path is not None and sample_submission_path.exists():
        sample = pd.read_csv(sample_submission_path)
        required = {"sample_id", "session_id", "time_bin", "channel"}
        if not required.issubset(sample.columns):
            raise ValueError(
                f"Sample submission missing required columns {sorted(required)}. Found {list(sample.columns)}"
            )

        preds = []
        for row in sample.itertuples(index=False):
            session_pred = predictions_by_session[row.session_id]
            preds.append(float(session_pred[int(row.time_bin), int(row.channel)]))

        sample["predicted_sbp"] = preds
        sample.to_csv(output_path, index=False)
        return

    # TODO: Adapt this fallback if your competition expects a different submission schema.
    rows: List[Dict[str, object]] = []
    sample_id = 0
    for session_id, pred in sorted(predictions_by_session.items()):
        n_bins, n_channels = pred.shape
        for t in range(n_bins):
            for c in range(n_channels):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "session_id": session_id,
                        "time_bin": t,
                        "channel": c,
                        "predicted_sbp": float(pred[t, c]),
                    }
                )
                sample_id += 1
    pd.DataFrame(rows).to_csv(output_path, index=False)


def predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, ckpt = _load_model(args.checkpoint_path, device=device)

    metadata = load_metadata(Path(args.data_dir) / args.metadata_file)
    if "split" not in metadata.columns:
        raise ValueError("metadata.csv must include a 'split' column to identify test sessions.")

    test_ids = (
        metadata.loc[metadata["split"] == "test", "session_id"].astype(str).sort_values().tolist()
    )
    if not test_ids:
        raise ValueError("No test sessions found in metadata.")

    test_sessions = load_sessions(
        data_dir=args.data_dir,
        session_ids=test_ids,
        split="test",
        infer_mask_from_zero=args.infer_mask_from_zero,
    )

    fallback_mean = ckpt.get("train_global_mean")
    fallback_std = ckpt.get("train_global_std")
    if fallback_mean is None or fallback_std is None:
        raise ValueError("Checkpoint missing train_global_mean/std; retrain or patch checkpoint.")

    predictions_by_session: Dict[str, np.ndarray] = {}

    for session in test_sessions:
        assert session.mask is not None
        mean, std, observed_counts = compute_test_session_stats(
            sbp_masked=session.sbp,
            mask=session.mask,
            fallback_mean=fallback_mean,
            fallback_std=fallback_std,
        )

        pred_matrix = _predict_session_matrix(
            model=model,
            sbp_masked=session.sbp,
            kin=session.kinematics,
            trial_ids=session.trial_ids,
            mask=session.mask,
            mean=mean,
            std=std,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        predictions_by_session[session.session_id] = pred_matrix
        print(
            f"Predicted session {session.session_id} | "
            f"masked_entries={int(session.mask.sum())} | "
            f"channels_with_low_obs={(observed_counts < 16).sum()}"
        )

    sample_path = Path(args.data_dir) / args.sample_submission_file
    output_path = Path(args.output_file)
    build_submission(predictions_by_session, sample_submission_path=sample_path, output_path=output_path)
    print(f"Wrote submission: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test predictions and Kaggle submission CSV.")
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--window-size", type=int, default=201)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-submission-file", type=str, default="sample_submission.csv")
    parser.add_argument("--output-file", type=str, default="submission.csv")
    parser.add_argument("--infer-mask-from-zero", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    return args


if __name__ == "__main__":
    predict(parse_args())
