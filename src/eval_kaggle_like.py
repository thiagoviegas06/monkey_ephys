from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.io import load_metadata, load_sessions
from src.predict import _load_model, _predict_session_matrix
from src.preprocess import compute_test_session_stats
from src.utils import resolve_device, seed_everything, split_train_val_sessions


def _trial_edges(trial_ids: np.ndarray, edge_width: int) -> np.ndarray:
    edge = np.zeros((len(trial_ids),), dtype=bool)
    if len(trial_ids) == 0:
        return edge

    boundaries = np.where(np.diff(trial_ids) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(trial_ids)]])

    for start, end in zip(starts, ends):
        tid = int(trial_ids[start])
        if tid < 0:
            continue
        lo = min(end, start + edge_width)
        hi = max(start, end - edge_width)
        edge[start:lo] = True
        edge[hi:end] = True
    return edge


def _sample_kaggle_like_mask(
    trial_ids: np.ndarray,
    n_channels: int,
    target_fraction: float,
    edge_width: int,
    edge_only: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if target_fraction < 0.0 or target_fraction > 1.0:
        raise ValueError("target_fraction must be in [0, 1]")

    valid_bins = np.flatnonzero(trial_ids >= 0)
    if len(valid_bins) == 0:
        return np.zeros((len(trial_ids), n_channels), dtype=bool)

    if edge_only:
        edge_bins = np.flatnonzero(_trial_edges(trial_ids, edge_width=edge_width))
        candidate_bins = edge_bins
    else:
        candidate_bins = valid_bins

    if len(candidate_bins) == 0:
        candidate_bins = valid_bins

    candidate_total = len(candidate_bins) * n_channels
    n_target = int(round(target_fraction * len(valid_bins) * n_channels))
    n_target = min(candidate_total, n_target)
    if target_fraction > 0 and n_target == 0:
        n_target = 1

    mask = np.zeros((len(trial_ids), n_channels), dtype=bool)
    if n_target == 0:
        return mask

    selected = rng.choice(candidate_total, size=n_target, replace=False)
    bin_idx = selected // n_channels
    chan_idx = selected % n_channels
    rows = candidate_bins[bin_idx]
    mask[rows, chan_idx] = True
    return mask


@torch.no_grad()
def run_eval_kaggle_like(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    model, ckpt = _load_model(args.checkpoint_path, device=device, debug=args.debug, label="MODEL")
    fallback_mean = ckpt.get("train_global_mean")
    fallback_std = ckpt.get("train_global_std")
    if fallback_mean is None or fallback_std is None:
        raise ValueError("Checkpoint missing train_global_mean/std; retrain or patch checkpoint.")

    window_size = int(args.window_size) if args.window_size is not None else int(ckpt.get("window_size", 201))
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    metadata = load_metadata(Path(args.data_dir) / args.metadata_file)
    _, val_ids = split_train_val_sessions(metadata, val_fraction=args.val_fraction, seed=args.seed)
    if args.max_sessions is not None:
        val_ids = val_ids[: args.max_sessions]
    if not val_ids:
        raise ValueError("No validation sessions available after filtering.")

    val_sessions = load_sessions(data_dir=args.data_dir, session_ids=val_ids, split="train")

    total_num = 0.0
    total_den = 0.0
    total_masked = 0
    session_nmse: List[float] = []
    session_mse: List[float] = []
    session_mask_frac: List[float] = []

    for idx, session in enumerate(val_sessions):
        rng = np.random.default_rng(args.seed + idx)
        mask = _sample_kaggle_like_mask(
            trial_ids=session.trial_ids,
            n_channels=session.sbp.shape[1],
            target_fraction=args.target_fraction,
            edge_width=args.edge_width,
            edge_only=args.edge_only,
            rng=rng,
        )
        sbp_masked = session.sbp.copy()
        sbp_masked[mask] = 0.0

        mean, std, _ = compute_test_session_stats(
            sbp_masked=sbp_masked,
            mask=mask,
            fallback_mean=fallback_mean,
            fallback_std=fallback_std,
        )

        pred_matrix, _ = _predict_session_matrix(
            model=model,
            sbp_masked=sbp_masked,
            kin=session.kinematics,
            trial_ids=session.trial_ids,
            mask=mask,
            mean=mean,
            std=std,
            window_size=window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            skip_negative_trials=False,
            debug=args.debug,
            label="MODEL",
            session_id=session.session_id,
        )

        y_true = session.sbp[mask]
        y_pred = pred_matrix[mask]
        if y_true.size == 0:
            continue

        num = float(np.sum((y_pred - y_true) ** 2))
        den = float(np.sum(y_true ** 2))
        mse = num / max(1, int(y_true.size))
        nmse = num / max(den, 1e-8)

        total_num += num
        total_den += den
        total_masked += int(y_true.size)
        session_nmse.append(nmse)
        session_mse.append(mse)
        session_mask_frac.append(float(mask.mean()))

        print(
            f"session={session.session_id} masked={int(y_true.size)} "
            f"mask_frac={mask.mean():.6f} nmse={nmse:.6f} mse={mse:.6f}"
        )

    if total_masked == 0:
        raise RuntimeError("No masked entries were evaluated. Increase --target-fraction.")

    global_nmse = total_num / max(total_den, 1e-8)
    global_mse = total_num / total_masked
    mean_session_nmse = float(np.mean(session_nmse)) if session_nmse else float("nan")
    mean_session_mse = float(np.mean(session_mse)) if session_mse else float("nan")
    mean_mask_frac = float(np.mean(session_mask_frac)) if session_mask_frac else float("nan")

    print("\n=== Kaggle-like Validation Summary ===")
    print(f"sessions={len(session_nmse)} total_masked={total_masked}")
    print(f"target_fraction={args.target_fraction:.6f} realized_mean_mask_fraction={mean_mask_frac:.6f}")
    print(f"global_nmse={global_nmse:.6f}")
    print(f"global_mse={global_mse:.6f}")
    print(f"mean_session_nmse={mean_session_nmse:.6f}")
    print(f"mean_session_mse={mean_session_mse:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint using Kaggle-like sparse, time-varying masking and full inference path."
    )
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--target-fraction", type=float, default=0.0086)
    parser.add_argument("--edge-width", type=int, default=100)
    parser.add_argument("--edge-only", dest="edge_only", action="store_true")
    parser.add_argument("--no-edge-only", dest="edge_only", action="store_false")
    parser.set_defaults(edge_only=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.window_size is not None and args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    if args.edge_width < 0:
        raise ValueError("--edge-width must be non-negative")
    if args.max_sessions is not None and args.max_sessions <= 0:
        raise ValueError("--max-sessions must be positive")
    return args


if __name__ == "__main__":
    run_eval_kaggle_like(parse_args())
