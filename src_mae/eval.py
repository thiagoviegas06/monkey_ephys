import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor, TCNReconstructor
from preprocessing import sample_span_start


def mask_segments(mask_2d: np.ndarray):
    """
    mask_2d: (N, C) bool, True where masked.
    Returns list of (start, end) time segments where any channel is masked.
    end is exclusive.
    """
    t = np.where(mask_2d.any(axis=1))[0]
    if len(t) == 0:
        return []

    cuts = np.where(np.diff(t) > 1)[0]
    starts = np.r_[t[0], t[cuts + 1]]
    ends = np.r_[t[cuts] + 1, t[-1] + 1]
    return list(zip(starts, ends))


def build_randomized_windows_from_mask(mask_2d: np.ndarray, window_size: int, rng):
    """
    One window per masked region.
    Each window has fixed size `window_size`, fully contains that region,
    and randomizes where the region lands inside the window.
    """
    N = mask_2d.shape[0]
    segs = mask_segments(mask_2d)
    if not segs:
        return []

    if N < window_size:
        raise ValueError(
            f"Session length N={N} is smaller than window_size={window_size}; "
            "cannot create fixed-size windows."
        )

    windows = []
    for seg_start, seg_end in segs:
        L = seg_end - seg_start
        if L > window_size:
            raise ValueError(
                f"Masked segment length ({L}) exceeds window_size ({window_size}); "
                "cannot cover this region with a single window."
            )

        lo = max(0, seg_end - window_size)
        hi = min(seg_start, N - window_size)

        if lo > hi:
            w0 = max(0, min(seg_start, N - window_size))
        else:
            t0 = sample_span_start(rng, W=window_size, L=L)
            w0 = int(np.clip(seg_start - t0, lo, hi))

        windows.append((w0, w0 + window_size))

    return windows


def preprocess_test(data_path, window_size, metadata_csv, seed=42, expected_regions=10):
    masked_files = os.path.join(data_path, "test/*_sbp_masked.npy")
    session_data = {}

    for file in sorted(glob(masked_files)):
        session_id = Path(file).stem.split("_")[0]
        rng = np.random.default_rng(seed + (hash(session_id) & 0xFFFFFFFF))

        masked_sbp = np.load(file)
        mask = np.load(file.replace("sbp_masked", "mask"))
        kin = np.load(file.replace("sbp_masked", "kinematics"))

        segs = mask_segments(mask)
        windows = build_randomized_windows_from_mask(mask, window_size, rng)

        if len(segs) != expected_regions:
            print(
                f"WARNING: Session {session_id} has {len(segs)} masked regions "
                f"(expected {expected_regions})."
            )

        if len(windows) != len(segs):
            raise RuntimeError(
                f"Session {session_id}: expected {len(segs)} windows (one per region), "
                f"got {len(windows)}."
            )

        for w0, w1 in windows:
            print(
                f"Session {session_id} | window ({w0},{w1}) covers masked segment "
                f"with {mask[w0:w1].sum()} masked positions."
            )

        session_data[session_id] = {
            "masked_sbp": masked_sbp,
            "mask": mask,
            "kinematics": kin,
            "windows": windows,
            "n_regions": len(segs),
        }

    return session_data


def build_model(model_name: str, base_channels: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "unet":
        return SBP_Reconstruction_UNet(base_channels=base_channels)
    if model_name == "simple_cnn":
        return SimpleCNN(hidden_channels=128, num_layers=6)
    if model_name == "resnet":
        return ResNetReconstructor(hidden_channels=128, num_blocks=8)
    raise ValueError(f"Unknown model_name '{model_name}'")


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = sorted(glob(os.path.join(checkpoint_dir, "model_epoch_*.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in '{checkpoint_dir}'. "
            "Provide --model-path or train/save a checkpoint first."
        )
    return candidates[-1]


def load_model(model_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint format not recognized. Expected a dict with 'model_state_dict'."
        )

    config = ckpt.get("config", {})
    model_name = config.get("model_name", "unet")
    base_channels = int(config.get("base_channels", 64))

    model = build_model(model_name=model_name, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_sessions(model: nn.Module, session_data: dict, device: torch.device):
    predictions = {}

    for session_id, info in session_data.items():
        masked_sbp = info["masked_sbp"]
        mask = info["mask"]
        windows = info["windows"]

        pred_full = masked_sbp.copy()
        covered = np.zeros_like(mask, dtype=np.bool_)

        for w0, w1 in windows:
            x_window = torch.from_numpy(masked_sbp[w0:w1]).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            m_window = torch.from_numpy(mask[w0:w1]).unsqueeze(0).to(
                device=device, dtype=torch.bool
            )

            pred_window = model(x_window, m_window).squeeze(0).cpu().numpy()

            m_np = mask[w0:w1]
            block = pred_full[w0:w1]
            block[m_np] = pred_window[m_np]
            pred_full[w0:w1] = block
            covered[w0:w1] |= m_np

        n_missing = int((mask & ~covered).sum())
        if n_missing > 0:
            raise RuntimeError(
                f"Session {session_id} has {n_missing} masked positions not covered by evaluation windows."
            )

        predictions[session_id] = pred_full

    return predictions


def build_submission(sample_submission_path: str, predictions: dict, output_csv: str):
    sub = pd.read_csv(sample_submission_path)

    for session_id, pred_full in predictions.items():
        idx = sub["session_id"] == session_id
        if not idx.any():
            continue

        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)
        channels = sub.loc[idx, "channel"].to_numpy(dtype=np.int64)
        sub.loc[idx, "predicted_sbp"] = pred_full[time_bins, channels].astype(np.float32)

    if sub["predicted_sbp"].isna().any():
        n_nan = int(sub["predicted_sbp"].isna().sum())
        raise RuntimeError(f"Submission has {n_nan} NaN predictions.")

    sub.to_csv(output_csv, index=False)
    print(f"Saved submission: {output_csv}")
    return sub


def run_eval(model_path, data_path, output_csv, window_size, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = os.path.join(data_path, "metadata.csv")
    sample_submission_csv = os.path.join(data_path, "sample_submission.csv")

    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    print("Building randomized one-window-per-region eval windows...")
    session_data = preprocess_test(
        data_path=data_path,
        window_size=window_size,
        metadata_csv=metadata_csv,
        seed=seed,
        expected_regions=10,
    )
    total_windows = sum(len(v["windows"]) for v in session_data.values())
    print(f"Total windows created: {total_windows}")

    print("Running inference...")
    predictions = predict_sessions(model, session_data, device)

    print("Constructing submission CSV...")
    build_submission(sample_submission_csv, predictions, output_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick evaluation and Kaggle submission export.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for auto-picking latest checkpoint",
    )
    parser.add_argument("--data-path", type=str, default="kaggle_data", help="Data root path")
    parser.add_argument("--window-size", type=int, default=200, help="Evaluation window size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for window randomization")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission_eval.csv",
        help="Output submission path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path if args.model_path else find_latest_checkpoint(args.checkpoint_dir)
    run_eval(
        model_path=model_path,
        data_path=args.data_path,
        output_csv=args.output_csv,
        window_size=args.window_size,
        seed=args.seed,
    )
