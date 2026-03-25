import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preprocessing import sample_span_start
from config import Config
from model import SBP_TCN_Transformer

config = Config()

# ============================================================================
# Model Builder
# ============================================================================
def build_model(config):
    """Build SBP_TCN_Transformer based on config."""
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    print("Built Hybrid TCN + Cross-Channel Transformer")
    return model.to(config.device)


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
    """
    N = mask_2d.shape[0]
    segs = mask_segments(mask_2d)
    if not segs:
        return []

    if N < window_size:
        raise ValueError(f"Session length N={N} < window_size={window_size}")

    windows = []
    for seg_start, seg_end in segs:
        L = seg_end - seg_start
        if L > window_size:
            raise ValueError(f"Masked segment length ({L}) > window_size ({window_size})")

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
    global config
    masked_files = os.path.join(data_path, "test/*_sbp_masked.npy")
    session_data = {}

    for file in sorted(glob(masked_files)):
        session_id = Path(file).stem.split("_")[0]
        rng = np.random.default_rng(seed + (hash(session_id) & 0xFFFFFFFF))

        masked_sbp = np.load(file)
        mask = np.load(file.replace("sbp_masked", "mask"))

        segs = mask_segments(mask)
        windows = build_randomized_windows_from_mask(mask, window_size, rng)

        if len(segs) != expected_regions:
            print(f"WARNING: Session {session_id} has {len(segs)} masked regions.")

        session_data[session_id] = {
            "masked_sbp": masked_sbp,
            "mask": mask,
            "windows": windows,
        }

    return session_data


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
    candidates.sort(key=natural_keys)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}'")
    return candidates[-1]


def load_model(model_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
    else:
        model = build_model(config).to(device)
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
        macro_timestamp = info.get("macro_timestamp", 0.0)

        pred_full = masked_sbp.copy()
        covered = np.zeros_like(mask, dtype=np.bool_)

        for w0, w1 in windows:
            x_window = torch.from_numpy(masked_sbp[w0:w1]).unsqueeze(0).to(device, dtype=torch.float32)
            m_window = torch.from_numpy(mask[w0:w1]).unsqueeze(0).to(device, dtype=torch.float32)
            macro_ts_tensor = torch.tensor([[macro_timestamp]], device=device, dtype=torch.float32)

            pred_window = model(x_window, m_window, macro_ts_tensor).squeeze(0).cpu().numpy()

            m_np = mask[w0:w1]
            block = pred_full[w0:w1]
            block[m_np] = pred_window[m_np]
            pred_full[w0:w1] = block
            covered[w0:w1] |= m_np

        n_missing = int((mask & ~covered).sum())
        if n_missing > 0:
            raise RuntimeError(f"Session {session_id}: {n_missing} positions not covered.")

        predictions[session_id] = pred_full
    return predictions


def build_submission(sample_submission_path: str, predictions: dict, output_csv: str):
    sub = pd.read_csv(sample_submission_path)
    for session_id, pred_full in predictions.items():
        idx = sub["session_id"] == session_id
        if not idx.any(): continue

        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)
        channels = sub.loc[idx, "channel"].to_numpy(dtype=np.int64)
        sub.loc[idx, "predicted_sbp"] = pred_full[time_bins, channels].astype(np.float32)

    sub.to_csv(output_csv, index=False)
    print(f"Saved submission: {output_csv}")


def run_eval(model_path, data_path, output_csv, window_size, seed, args):
    global config
    config.window_size = window_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    session_data = preprocess_test(
        data_path=data_path,
        window_size=window_size,
        metadata_csv=os.path.join(data_path, "metadata.csv"),
        seed=seed
    )

    print("Running inference...")
    predictions = predict_sessions(model, session_data, device)

    print("Constructing submission CSV...")
    build_submission(os.path.join(data_path, "sample_submission.csv"), predictions, output_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation and submission export.")
    parser.add_argument("--window-size", type=int, default=200, help="Evaluation window size")
    parser.add_argument("--data-path", type=str, default="kaggle_data", help="Data root path")
    parser.add_argument("--seed", type=int, default=42, help="Seed for window randomization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_dir = f"checkpoints_{args.window_size}"
    model_path = os.path.join(checkpoint_dir, f"best_model_{config.model_name}.pt")
    if not os.path.exists(model_path):
        model_path = find_latest_checkpoint(checkpoint_dir)
    
    output_csv = f"submission_eval_{args.window_size}.csv"
    run_eval(model_path, args.data_path, output_csv, args.window_size, args.seed, args)
