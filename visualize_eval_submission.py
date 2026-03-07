#!/usr/bin/env python3
"""Visualize eval submission predictions against test masked SBP."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_session_arrays(data_path: str, session_id: str):
    base = Path(data_path) / "test"
    sbp_masked_path = base / f"{session_id}_sbp_masked.npy"
    mask_path = base / f"{session_id}_mask.npy"

    if not sbp_masked_path.exists():
        raise FileNotFoundError(f"Missing file: {sbp_masked_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing file: {mask_path}")

    sbp_masked = np.load(sbp_masked_path).astype(np.float32)
    mask = np.load(mask_path).astype(bool)

    if sbp_masked.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for session {session_id}: "
            f"sbp_masked={sbp_masked.shape}, mask={mask.shape}"
        )

    return sbp_masked, mask


def build_predicted_array(submission_df: pd.DataFrame, session_id: str, sbp_masked: np.ndarray):
    required_cols = {"session_id", "time_bin", "channel", "predicted_sbp"}
    missing = required_cols - set(submission_df.columns)
    if missing:
        raise ValueError(f"Submission missing columns: {sorted(missing)}")

    sess = submission_df[submission_df["session_id"] == session_id].copy()
    if sess.empty:
        raise ValueError(f"No rows for session_id={session_id} in submission CSV")

    time_idx = sess["time_bin"].to_numpy(dtype=np.int64)
    ch_idx = sess["channel"].to_numpy(dtype=np.int64)
    values = sess["predicted_sbp"].to_numpy(dtype=np.float32)

    pred = sbp_masked.copy()
    pred[time_idx, ch_idx] = values

    return pred, len(sess)


def plot_session(
    session_id: str,
    sbp_masked: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    n_pred_rows: int,
    save_path: str | None,
):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc

    masked_positions = int(mask.sum())
    coverage_pct = 100.0 * n_pred_rows / max(masked_positions, 1)

    pred_delta = np.where(mask, pred - sbp_masked, 0.0)

    vmax = float(np.nanpercentile(np.abs(pred_delta[mask]), 99)) if masked_positions > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(sbp_masked.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[0, 0].set_title("Input: sbp_masked")
    axes[0, 0].set_xlabel("Time bin")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(pred.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[0, 1].set_title("Reconstructed from submission CSV")
    axes[0, 1].set_xlabel("Time bin")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(mask.T.astype(np.float32), aspect="auto", origin="lower", interpolation="nearest", cmap="gray_r")
    axes[1, 0].set_title("Mask (white = masked)")
    axes[1, 0].set_xlabel("Time bin")
    axes[1, 0].set_ylabel("Channel")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(
        pred_delta.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1, 1].set_title("Predicted change (masked positions only)")
    axes[1, 1].set_xlabel("Time bin")
    axes[1, 1].set_ylabel("Channel")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Session {session_id} | masked positions={masked_positions:,} | "
        f"rows in submission={n_pred_rows:,} | coverage={coverage_pct:.2f}%",
        fontsize=12,
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved figure to: {out}")
    else:
        plt.show()


def _normalize_argv(argv):
    normalized = list(argv)
    i = 0
    while i < len(normalized) - 2:
        if normalized[i] == "--session" and normalized[i + 1] == "id":
            normalized[i] = "--session-id"
            del normalized[i + 1]
            continue
        i += 1
    return normalized


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)

    parser = argparse.ArgumentParser(
        description="Visualize model predictions from eval submission CSV for one session."
    )
    parser.add_argument(
        "--submission-csv",
        type=str,
        default="submission_eval.csv",
        help="Path to submission CSV from eval.py",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="kaggle_data",
        help="Data root containing test/*.npy",
    )
    parser.add_argument(
        "--session-id",
        "--session",
        type=str,
        default=None,
        help="Session ID to visualize (e.g. S008). If omitted, uses first session in CSV.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save figure instead of displaying.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    submission_df = pd.read_csv(args.submission_csv)
    if args.session_id is None:
        args.session_id = str(submission_df["session_id"].iloc[0])

    sbp_masked, mask = load_session_arrays(args.data_path, args.session_id)
    pred, n_pred_rows = build_predicted_array(submission_df, args.session_id, sbp_masked)

    plot_session(
        session_id=args.session_id,
        sbp_masked=sbp_masked,
        pred=pred,
        mask=mask,
        n_pred_rows=n_pred_rows,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
