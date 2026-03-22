#!/usr/bin/env python3
"""Visualize masked windows from preprocess_non_overlapping."""

import argparse
from pathlib import Path
import numpy as np
from src_mae.old_preprocessing import sessionData, non_overlapping_windows

def apply_elementwise_random_mask(sbp, rng, mask_fraction=0.4):
    """
    Randomly masks `mask_fraction` of ALL individual data points 
    in the window (not full channels or specific time spans).
    """
    W, C = sbp.shape
    x = sbp.copy()
    mask = np.zeros((W, C), dtype=np.bool_)

    total_elements = W * C
    num_masked = int(np.round(total_elements * mask_fraction))

    # Pick random 1D indices representing the locations to mask
    masked_indices_1d = rng.choice(total_elements, size=num_masked, replace=False)
    
    # Convert back to 2D indices
    t_idx, c_idx = np.unravel_index(masked_indices_1d, (W, C))

    x[t_idx, c_idx] = 0.0
    mask[t_idx, c_idx] = True

    return x, mask

def compute_session_channel_variance(sbp):
    session_variance = np.var(sbp, axis=0)
    return session_variance

def visualize_masked_window(sample, save_path=None):
    """
    Visualize a masked window from preprocess_non_overlapping.
    
    Args:
        sample: dict with x_sbp, y_sbp, mask, kin, session_id, w0, span
        save_path: optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc
    
    x_sbp = sample["x_sbp"]
    y_sbp = sample["y_sbp"]
    kin = sample["kin"]
    mask = sample["mask"]  # (W, 96) 2D mask
    session_id = sample["session_id"]
    w0 = sample["w0"]
    W, C = y_sbp.shape
    
    # Which channels are masked anywhere in the window
    mask_vec = mask.any(axis=0)  # (96,) boolean - True if channel masked in any bin
    masked_channels = np.flatnonzero(mask_vec)
    
    # Create display copy where masked values are NaN (will show as white)
    x_sbp_display = x_sbp.copy()
    x_sbp_display[x_sbp == 0] = np.nan
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    
    # Ground truth SBP
    im0 = axes[0, 0].imshow(y_sbp.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 0].set_title("Ground-truth SBP window")
    axes[0, 0].set_xlabel("Window bin")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Masked input SBP - NaN values appear white
    im1 = axes[0, 1].imshow(x_sbp_display.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 1].set_title("Masked input SBP window (white = masked)")
    axes[0, 1].set_xlabel("Window bin")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Time profile
    mean_all = y_sbp.mean(axis=1)
    axes[1, 0].plot(mean_all, label="mean SBP (all channels)", linewidth=2)
    if masked_channels.size > 0:
        mean_masked_true = y_sbp[:, masked_channels].mean(axis=1)
        mean_masked_in = x_sbp[:, masked_channels].mean(axis=1)
        axes[1, 0].plot(mean_masked_true, label="mean masked channels (true)", linewidth=1.7)
        axes[1, 0].plot(mean_masked_in, label="mean masked channels (input)", linewidth=1.7)
    axes[1, 0].set_title("SBP time profile in window")
    axes[1, 0].set_xlabel("Window bin")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(loc="upper right")
    
    # Kinematics
    axes[1, 1].plot(kin)
    axes[1, 1].set_title("Kinematics (4 channels)")
    axes[1, 1].set_xlabel("Window bin")
    axes[1, 1].set_ylabel("Value")
    
    mask_count_positions = int(mask.sum())  # Total masked (bin, channel) positions
    mask_pct = (mask_count_positions / (W * C)) * 100
    
    fig.suptitle(
        (
            f"session={session_id}  w0={w0} | "
            f"Total window size={W}x{C} | Masked points={mask_count_positions} ({mask_pct:.1f}%)"
        ),
        fontsize=12,
    )
    
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Saved plot to: {path}")
    else:
        plt.show()



def generate_one_masked_window(data_path, window_size=200, seed=0, session_id=None):
    """
    Generate a single masked window for visualization from preprocess_non_overlapping.
    
    Returns:
        dict with keys: x_sbp, y_sbp, mask, kin, session_id, w0, span
    """
    sessions, _ = sessionData(f"{data_path}/metadata.csv").generate_session_obj()
    
    if session_id is None:
        # Find first train session
        for s in sessions:
            if s.train:
                session_id = s.session_id
                session = s
                break
    else:
        # Find specific session
        session = None
        for s in sessions:
            if s.session_id == session_id:
                session = s
                break
        if session is None:
            raise ValueError(f"Session {session_id} not found")
    
    if session.isTest():
        raise ValueError(f"Session {session_id} is a test session, need train session")
    
    sbp, kin, starts_bins, end_bins = session.load_data(data_path)
    if sbp is None:
        raise RuntimeError(f"Failed to load data for session {session.session_id}")
    
    N = sbp.shape[0]
    if N < window_size:
        raise ValueError(f"Session {session.session_id} has {N} bins, smaller than window_size={window_size}")
    
    rng = np.random.default_rng(seed + (hash(session.session_id) & 0xFFFFFFFF))
    
    # Get first non-overlapping window
    w0s = non_overlapping_windows(N, window_size)
    if not w0s:
        raise RuntimeError("No windows generated")
    
    w0 = w0s[0]
    y = sbp[w0:w0 + window_size].copy()
    kin_w = kin[w0:w0 + window_size].copy()
    
    x, M = apply_elementwise_random_mask(y, rng, mask_fraction=0.4)

    print(f"Generated masked window for session {session.session_id}, w0={w0}, masked positions={int(M.sum())}")
    
    return {
        "x_sbp": x.astype(np.float32),
        "y_sbp": y.astype(np.float32),
        "mask": M,
        "kin": kin_w.astype(np.float32),
        "session_id": session.session_id,
        "w0": int(w0),
        "span": (0, window_size),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize a masked window from non-overlapping preprocessing")
    parser.add_argument("--data-path", "--data_path", type=str, default="kaggle_data",
                       help="Path to data directory")
    parser.add_argument("--session-id", "--session_id", type=str, default=None,
                       help="Train session ID (e.g. S008). If omitted, uses first train session.")
    parser.add_argument("--window-size", "--window_size", type=int, default=200,
                       help="Window size")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--save-path", "--save_path", type=str, default=None,
                       help="Save figure to this path instead of displaying")
    
    args = parser.parse_args()
    
    print(f"Generating masked window from {args.data_path}...")
    sample = generate_one_masked_window(
        data_path=args.data_path,
        window_size=args.window_size,
        seed=args.seed,
        session_id=args.session_id,
    )
    
    print(f"Generated sample from session {sample['session_id']}")
    print(f"  Masked span: [{sample['span'][0]}, {sample['span'][1]})")
    print(f"  Masked channels: {int(sample['mask'].sum())}")
    print(f"\nVisualizing...")
    
    visualize_masked_window(sample, save_path=args.save_path)


if __name__ == "__main__":
    main()
