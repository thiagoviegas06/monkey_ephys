from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        skip_negative_trials: bool = False,
    ) -> None:
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        self.sbp_norm = sbp_norm
        self.obs_mask_full = obs_mask_full
        self.kinematics = kinematics
        self.trial_ids = trial_ids
        self.window_size = window_size
        self.half = window_size // 2
        self.skip_negative_trials = skip_negative_trials
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
            if self.skip_negative_trials and tid < 0:
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


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_dict_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _print_checkpoint_info(label: str, checkpoint_path: str | Path, state_dict: Dict[str, torch.Tensor]) -> None:
    path = Path(checkpoint_path)
    file_hash = _sha256_file(path)
    mtime = path.stat().st_mtime
    model_hash = _state_dict_fingerprint(state_dict)
    print(
        f"[{label}] checkpoint={path} | sha256={file_hash} | mtime={mtime:.0f} | "
        f"state_dict_fp={model_hash}"
    )


def _load_model(
    checkpoint_path: str,
    device: torch.device,
    debug: bool = True,
    label: str = "MODEL",
) -> Tuple[torch.nn.Module, Dict]:
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoint stores numpy arrays (e.g., train_global_mean/std), so we need full unpickling.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions that do not expose weights_only.
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = ckpt["model_kwargs"]
    if debug:
        _print_checkpoint_info(label=label, checkpoint_path=checkpoint_path, state_dict=ckpt["model_state_dict"])
    model = TemporalCNNBaseline(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _max_trial_len(trial_ids: np.ndarray, skip_negative_trials: bool) -> int:
    boundaries = np.where(np.diff(trial_ids) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(trial_ids)]])
    lengths = []
    for s, e in zip(starts, ends):
        tid = int(trial_ids[s])
        if skip_negative_trials and tid < 0:
            continue
        lengths.append(int(e - s))
    return max(lengths) if lengths else 0

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
    skip_negative_trials: bool = False,
    debug: bool = True,
    label: str = "MODEL",
    session_id: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    sbp_norm = apply_norm(sbp_masked, mean, std)
    # Match train-time masking semantics: masked input channels are zero in normalized space.
    sbp_norm = sbp_norm.copy()
    sbp_norm[mask] = 0.0
    obs_mask_full = (~mask).astype(np.float32)

    # Accumulate overlapping window predictions and average.
    pred_sum = np.zeros_like(sbp_norm, dtype=np.float32)
    pred_count = np.zeros_like(sbp_norm, dtype=np.float32)

    max_len = _max_trial_len(trial_ids, skip_negative_trials)
    ws = int(window_size)

    if max_len < ws:
        ws = max_len if (max_len % 2 == 1) else max_len - 1
        ws = max(ws, 31)  # minimum odd window
        if debug:
            print(f"[{label}] session={session_id} reducing window_size {window_size} -> {ws} (max_trial_len={max_len})")

    window_size = ws

    ds = InferenceWindowDataset(
        sbp_norm=sbp_norm,
        obs_mask_full=obs_mask_full,
        kinematics=kin,
        trial_ids=trial_ids,
        window_size=window_size,
        skip_negative_trials=skip_negative_trials,
    )
    if debug:
        print(f"[{label}] session={session_id} windows={len(ds)} skip_negative_trials={skip_negative_trials}")
    if len(ds) == 0:
        boundaries = np.where(np.diff(trial_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(trial_ids)]])
        lengths = []
        for s, e in zip(starts, ends):
            tid = int(trial_ids[s])
            if skip_negative_trials and tid < 0:
                continue
            lengths.append(int(e - s))
        max_len = max(lengths) if lengths else 0
        raise RuntimeError(
            f"[{label}] session={session_id} has zero inference windows. "
            f"window_size={window_size}, max_trial_len={max_len}, "
            f"skip_negative_trials={skip_negative_trials}. "
            "Use a smaller odd --window-size, disable --skip-negative-trials, or inspect trial_ids."
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    first_batch_y_hat = None
    for batch_idx, batch in enumerate(loader):
        x_sbp = batch["x_sbp"].to(device)
        x_kin = batch["x_kin"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        centers = batch["center"].cpu().numpy().astype(np.int64)
        y_hat = model(x_sbp, x_kin, obs_mask).cpu().numpy().astype(np.float32)  # (B,96,T)
        if batch_idx == 0:
            first_batch_y_hat = y_hat.copy()
            if debug:
                print(
                    f"[{label}] session={session_id} first_batch_y_hat "
                    f"mean={y_hat.mean():.6f} std={y_hat.std():.6f} "
                    f"min={y_hat.min():.6f} max={y_hat.max():.6f}"
                )

        half = window_size // 2
        for i, c in enumerate(centers):
            start = int(c - half)
            end = int(c + half + 1)
            pred_bt = y_hat[i].T  # (T,96)
            pred_sum[start:end] += pred_bt
            pred_count[start:end] += 1.0

    valid = pred_count > 0
    pred_norm = np.zeros_like(pred_sum, dtype=np.float32)
    pred_norm[valid] = pred_sum[valid] / pred_count[valid]
    # Fallback where no full-window prediction is available.
    pred_norm[~valid] = sbp_norm[~valid]

    pred = apply_denorm(pred_norm, mean, std)

    # Keep observed entries unchanged; only masked entries are reconstructed.
    out = sbp_masked.copy()
    out[mask] = pred[mask]
    valid_fraction = float(valid.mean())
    masked_valid_fraction = float((valid & mask).mean())

    if debug:
        print(
            f"[{label}] session={session_id} valid_fraction={valid_fraction:.6f} "
            f"masked_valid_fraction={masked_valid_fraction:.6f}"
        )
        masked_values = out[mask]
        if masked_values.size > 0:
            print(
                f"[{label}] session={session_id} out[mask] "
                f"mean={masked_values.mean():.6f} std={masked_values.std():.6f} "
                f"min={masked_values.min():.6f} max={masked_values.max():.6f}"
            )
        else:
            print(f"[{label}] session={session_id} out[mask] has no masked entries.")

    if masked_valid_fraction == 0.0:
        print(
            f"[{label}] WARNING: session={session_id} masked_valid_fraction is 0.0. "
            "Masked bins received no model predictions; fallback dominates."
        )

    return out, {
        "first_batch_y_hat": first_batch_y_hat,
        "valid_fraction": valid_fraction,
        "masked_valid_fraction": masked_valid_fraction,
    }


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
    ab_mode = bool(args.checkpoint_a and args.checkpoint_b)
    if (args.checkpoint_a is None) ^ (args.checkpoint_b is None):
        raise ValueError("Provide both --checkpoint-a and --checkpoint-b for A/B mode, or neither.")

    if ab_mode:
        model_a, ckpt_a = _load_model(args.checkpoint_a, device=device, debug=args.debug, label="A")
        model_b, ckpt_b = _load_model(args.checkpoint_b, device=device, debug=args.debug, label="B")
        fallback_mean_a = ckpt_a.get("train_global_mean")
        fallback_std_a = ckpt_a.get("train_global_std")
        fallback_mean_b = ckpt_b.get("train_global_mean")
        fallback_std_b = ckpt_b.get("train_global_std")
        if fallback_mean_a is None or fallback_std_a is None:
            raise ValueError("Checkpoint A missing train_global_mean/std; retrain or patch checkpoint.")
        if fallback_mean_b is None or fallback_std_b is None:
            raise ValueError("Checkpoint B missing train_global_mean/std; retrain or patch checkpoint.")
        if args.window_size is None:
            ws_a = ckpt_a.get("window_size")
            ws_b = ckpt_b.get("window_size")
            if ws_a is None or ws_b is None:
                raise ValueError(
                    "Missing window_size in one/both checkpoints. Please pass --window-size explicitly."
                )
            if int(ws_a) != int(ws_b):
                raise ValueError(
                    f"Checkpoint window_size mismatch: A={ws_a}, B={ws_b}. "
                    "Pass --window-size explicitly for A/B comparison."
                )
            infer_window_size = int(ws_a)
        else:
            infer_window_size = int(args.window_size)
    else:
        model, ckpt = _load_model(args.checkpoint_path, device=device, debug=args.debug, label="MODEL")
        fallback_mean = ckpt.get("train_global_mean")
        fallback_std = ckpt.get("train_global_std")
        if fallback_mean is None or fallback_std is None:
            raise ValueError("Checkpoint missing train_global_mean/std; retrain or patch checkpoint.")
        if args.window_size is None:
            infer_window_size = int(ckpt.get("window_size", 201))
        else:
            infer_window_size = int(args.window_size)

    if infer_window_size % 2 == 0:
        raise ValueError(f"Inference window_size must be odd, got {infer_window_size}")
    if args.debug:
        print(f"[predict] using window_size={infer_window_size}")

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

    if ab_mode:
        predictions_by_session_a: Dict[str, np.ndarray] = {}
        predictions_by_session_b: Dict[str, np.ndarray] = {}
    else:
        predictions_by_session: Dict[str, np.ndarray] = {}

    compare_ids = set(test_ids[: max(0, args.compare_sessions)])

    for session in test_sessions:
        assert session.mask is not None
        if args.debug:
            uniq = np.unique(session.trial_ids)
            uniq_head = uniq[:10].tolist()
            print(
                f"[session={session.session_id}] trial_ids min={int(uniq.min())} max={int(uniq.max())} "
                f"n_unique={len(uniq)} first={uniq_head}"
            )

        if ab_mode:
            mean_a, std_a, observed_counts_a = compute_test_session_stats(
                sbp_masked=session.sbp,
                mask=session.mask,
                fallback_mean=fallback_mean_a,
                fallback_std=fallback_std_a,
            )
            mean_b, std_b, observed_counts_b = compute_test_session_stats(
                sbp_masked=session.sbp,
                mask=session.mask,
                fallback_mean=fallback_mean_b,
                fallback_std=fallback_std_b,
            )

            pred_matrix_a, debug_a = _predict_session_matrix(
                model=model_a,
                sbp_masked=session.sbp,
                kin=session.kinematics,
                trial_ids=session.trial_ids,
                mask=session.mask,
                mean=mean_a,
                std=std_a,
                window_size=infer_window_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                skip_negative_trials=args.skip_negative_trials,
                debug=args.debug,
                label="A",
                session_id=session.session_id,
            )
            pred_matrix_b, debug_b = _predict_session_matrix(
                model=model_b,
                sbp_masked=session.sbp,
                kin=session.kinematics,
                trial_ids=session.trial_ids,
                mask=session.mask,
                mean=mean_b,
                std=std_b,
                window_size=infer_window_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                skip_negative_trials=args.skip_negative_trials,
                debug=args.debug,
                label="B",
                session_id=session.session_id,
            )

            predictions_by_session_a[session.session_id] = pred_matrix_a
            predictions_by_session_b[session.session_id] = pred_matrix_b
            print(
                f"Predicted session {session.session_id} | "
                f"masked_entries={int(session.mask.sum())} | "
                f"low_obs_channels_A={(observed_counts_a < 16).sum()} | "
                f"low_obs_channels_B={(observed_counts_b < 16).sum()}"
            )

            if session.session_id in compare_ids:
                first_a = debug_a["first_batch_y_hat"]
                first_b = debug_b["first_batch_y_hat"]
                if first_a is not None and first_b is not None:
                    yhat_diff = float(np.max(np.abs(first_a - first_b)))
                    print(f"[A/B] session={session.session_id} first_batch_y_hat max_abs_diff={yhat_diff:.9f}")
                    if yhat_diff == 0.0:
                        print(
                            f"[A/B] WARNING: session={session.session_id} first-batch model outputs are identical."
                        )

                out_diff = float(np.max(np.abs(pred_matrix_a - pred_matrix_b)))
                print(f"[A/B] session={session.session_id} reconstructed_out max_abs_diff={out_diff:.9f}")
                if out_diff == 0.0:
                    print(
                        f"[A/B] WARNING: session={session.session_id} final outputs are identical. "
                        "Models may be identical or inference may be bypassing outputs."
                    )
        else:
            mean, std, observed_counts = compute_test_session_stats(
                sbp_masked=session.sbp,
                mask=session.mask,
                fallback_mean=fallback_mean,
                fallback_std=fallback_std,
            )

            pred_matrix, _debug_info = _predict_session_matrix(
                model=model,
                sbp_masked=session.sbp,
                kin=session.kinematics,
                trial_ids=session.trial_ids,
                mask=session.mask,
                mean=mean,
                std=std,
                window_size=infer_window_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                skip_negative_trials=args.skip_negative_trials,
                debug=args.debug,
                label="MODEL",
                session_id=session.session_id,
            )

            predictions_by_session[session.session_id] = pred_matrix
            print(
                f"Predicted session {session.session_id} | "
                f"masked_entries={int(session.mask.sum())} | "
                f"channels_with_low_obs={(observed_counts < 16).sum()}"
            )

    sample_path = Path(args.data_dir) / args.sample_submission_file
    if ab_mode:
        output_a = Path(args.output_a)
        output_b = Path(args.output_b)
        build_submission(predictions_by_session_a, sample_submission_path=sample_path, output_path=output_a)
        build_submission(predictions_by_session_b, sample_submission_path=sample_path, output_path=output_b)
        print(f"Wrote submission A: {output_a}")
        print(f"Wrote submission B: {output_b}")
    else:
        output_path = Path(args.output_file)
        build_submission(predictions_by_session, sample_submission_path=sample_path, output_path=output_path)
        print(f"Wrote submission: {output_path}")


def _str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test predictions and Kaggle submission CSV.")
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--checkpoint-a", type=str, default=None)
    parser.add_argument("--checkpoint-b", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-submission-file", type=str, default="sample_submission.csv")
    parser.add_argument("--output-file", type=str, default="submission.csv")
    parser.add_argument("--output-a", type=str, default="submission_a.csv")
    parser.add_argument("--output-b", type=str, default="submission_b.csv")
    parser.add_argument("--compare-sessions", type=int, default=2)
    parser.add_argument("--infer-mask-from-zero", action="store_true")
    parser.add_argument("--skip-negative-trials", action="store_true")
    parser.add_argument("--debug", type=_str2bool, default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.window_size is not None and args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    return args


if __name__ == "__main__":
    predict(parse_args())
