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
        num_layers=config.num_layers,
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


def load_training_session_stats(data_path):
    """Load sbp_mean/std from 5 nearest training sessions for each day."""
    import pickle

    train_pickle_dir = os.path.join(data_path, "masked_windows_200")
    if not os.path.exists(train_pickle_dir):
        print(f"Warning: Training pickle directory not found at {train_pickle_dir}")
        return {}

    train_sessions = {}  # day -> [(session_id, sbp_mean, sbp_std), ...]

    for pkl_file in glob(os.path.join(train_pickle_dir, "*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                sample = pickle.load(f)

            session_id = sample["session_id"]
            day = sample.get("day", 0.0)

            # Only load once per session (first window is representative)
            if session_id not in train_sessions:
                sbp_mean = sample.get("sbp_mean")
                sbp_std = sample.get("sbp_std")

                if sbp_mean is not None and sbp_std is not None:
                    if day not in train_sessions:
                        train_sessions[day] = []
                    train_sessions[day].append({
                        "session_id": session_id,
                        "sbp_mean": sbp_mean,
                        "sbp_std": sbp_std,
                    })
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    return train_sessions


def get_nearest_session_stats(test_day, train_sessions, k=5):
    """Get sbp stats from k nearest training sessions by day distance, weighted by inverse distance."""
    if not train_sessions:
        return None, None

    # Find distances to all training days
    days = sorted(train_sessions.keys())
    distances = [(abs(d - test_day), d) for d in days]
    distances.sort()

    # Collect stats from k nearest days with inverse distance weighting
    all_stats = []
    for dist, day in distances[:k]:
        for session_info in train_sessions[day]:
            # Inverse distance weighting: closer sessions get higher weight
            weight = 1.0 / (dist + 1e-8)  # Add epsilon to avoid division by zero
            all_stats.append({
                "sbp_mean": session_info["sbp_mean"],
                "sbp_std": session_info["sbp_std"],
                "weight": weight,
                "day_dist": dist,
            })

    if not all_stats:
        return None, None

    # Weighted average of stats
    total_weight = np.sum([s["weight"] for s in all_stats])
    sbp_mean_avg = np.sum([s["sbp_mean"] * s["weight"] for s in all_stats]) / total_weight
    sbp_std_avg = np.sum([s["sbp_std"] * s["weight"] for s in all_stats]) / total_weight

    return np.float32(sbp_mean_avg), np.float32(sbp_std_avg)


def preprocess_test(data_path, window_size, seed=42, expected_regions=10):
    global config
    masked_files = os.path.join(data_path, "test/*_sbp_masked.npy")
    session_data = {}
    session_sbp_stats = {}

    # Load training session stats for nearby day matching
    print("Loading training session statistics...")
    train_sessions = load_training_session_stats(data_path)

    for file in sorted(glob(masked_files)):
        session_id = Path(file).stem.split("_")[0]
        rng = np.random.default_rng(seed + (hash(session_id) & 0xFFFFFFFF))

        masked_sbp = np.load(file)
        mask = np.load(file.replace("sbp_masked", "mask"))

        # Extract day from session ID (e.g., "008" -> day 8)
        try:
            test_day = float(session_id)
        except:
            test_day = 0.0

        # Get stats from 5 nearest training sessions
        sbp_mean, sbp_std = get_nearest_session_stats(test_day, train_sessions, k=5)
        print(f"  Session {session_id} (day {test_day}): Using stats from 5 nearest training sessions")

        session_sbp_stats[session_id] = {
            "mean": sbp_mean,
            "std": sbp_std,
        }

        segs = mask_segments(mask)
        windows = build_randomized_windows_from_mask(mask, window_size, rng)

        if len(segs) != expected_regions:
            print(f"WARNING: Session {session_id} has {len(segs)} masked regions.")

        session_data[session_id] = {
            "masked_sbp": masked_sbp,
            "mask": mask,
            "windows": windows,
        }

    return session_data, session_sbp_stats


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
def predict_sessions(model: nn.Module, session_data: dict, device: torch.device, session_sbp_stats: dict = None, denormalize: bool = False):
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

        # Denormalize if requested
        if denormalize and session_sbp_stats and session_id in session_sbp_stats:
            sbp_mean = session_sbp_stats[session_id]["mean"]
            sbp_std = session_sbp_stats[session_id]["std"]
            pred_full = pred_full * sbp_std + sbp_mean

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


def run_eval(model_path, data_path, output_csv, window_size, seed, denormalize):
    global config
    config.window_size = window_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    session_data, session_sbp_stats = preprocess_test(
        data_path=data_path,
        window_size=window_size,
        seed=seed
    )

    print("Running inference...")
    if denormalize:
        print("  → Denormalizing predictions using per-session statistics")
    predictions = predict_sessions(model, session_data, device, session_sbp_stats, denormalize=denormalize)

    print("Constructing submission CSV...")
    build_submission(os.path.join(data_path, "sample_submission.csv"), predictions, output_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation and submission export.")
    parser.add_argument("--window-size", type=int, default=200, help="Evaluation window size")
    parser.add_argument("--data-path", type=str, default="kaggle_data", help="Data root path")
    parser.add_argument("--seed", type=int, default=42, help="Seed for window randomization")
    parser.add_argument("--denormalize", action="store_true", help="Denormalize predictions using per-session statistics")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint (optional, auto-detects best model if not provided)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Auto-detect best model or latest checkpoint
        checkpoint_dir = f"checkpoints_{args.window_size}"
        model_path = os.path.join(checkpoint_dir, f"best_model_{config.model_name}.pt")
        if not os.path.exists(model_path):
            model_path = find_latest_checkpoint(checkpoint_dir)

    output_csv = f"submission_eval_{args.window_size}.csv"
    run_eval(model_path, args.data_path, output_csv, args.window_size, args.seed, args.denormalize)
