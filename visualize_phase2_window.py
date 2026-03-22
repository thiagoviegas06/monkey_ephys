"""Visualize windows from Phase 2 dataset (SBP and Kinematics)."""

import argparse
import os
from pathlib import Path
import numpy as np
from src_kin.preprocessing import SessionDataPhase2

def visualize_phase2_window(sample, save_path=None):
    """
    Visualize a Phase 2 window.
    
    Args:
        sample: dict with sbp, kin, session_id, w0, active_mask
        save_path: optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc
    
    sbp = sample["sbp"]
    kin = sample["kin"]
    session_id = sample["session_id"]
    w0 = sample["w0"]
    active_mask = sample["active_mask"]  # (96,) boolean
    
    W, C = sbp.shape
    
    # Create display copy where masked (inactive) values are NaN (will show as white)
    sbp_display = sbp.copy()
    # Find channels that are zeroed out in this window
    window_zeros = (sbp == 0).all(axis=0)
    # Actually, we should use the active_mask from the whole session
    # but for visualization, showing where it's zero is fine.
    
    # Let's mark inactive channels with NaN for imshow
    sbp_display[:, ~active_mask] = np.nan
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    
    # SBP Heatmap
    im0 = axes[0, 0].imshow(sbp.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 0].set_title(f"SBP window (session {session_id}, w0={w0})")
    axes[0, 0].set_xlabel("Window bin")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # SBP with Masked Channels Highlighted
    # We'll use a different cmap or just show where data is missing
    im1 = axes[0, 1].imshow(sbp_display.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 1].set_title("SBP with inactive channels (white = inactive)")
    axes[0, 1].set_xlabel("Window bin")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Time profile
    mean_active = sbp[:, active_mask].mean(axis=1) if active_mask.any() else np.zeros(W)
    axes[1, 0].plot(mean_active, label="mean SBP (active channels)", linewidth=2)
    axes[1, 0].set_title("SBP time profile (active channels)")
    axes[1, 0].set_xlabel("Window bin")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(loc="upper right")
    
    # Kinematics
    if kin is not None:
        labels = ["index_pos", "mrp_pos", "index_vel", "mrp_vel"]
        for i in range(kin.shape[1]):
            axes[1, 1].plot(kin[:, i], label=labels[i])
        axes[1, 1].set_title("Kinematics (Positions and Velocities)")
        axes[1, 1].set_xlabel("Window bin")
        axes[1, 1].set_ylabel("Normalized Value")
        axes[1, 1].legend(loc="upper right")
    else:
        axes[1, 1].text(0.5, 0.5, "Kinematics not available (Test set)", ha='center', va='center')
        axes[1, 1].set_title("Kinematics (N/A)")

    active_count = int(active_mask.sum())
    inactive_count = C - active_count
    
    fig.suptitle(
        (
            f"Phase 2 Session: {session_id} | w0={w0} | "
            f"Active Channels={active_count}/96 ({active_count/C*100:.1f}%)"
        ),
        fontsize=14,
    )
    
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Saved plot to: {path}")
    else:
        plt.show()

def generate_one_phase2_window(data_path, window_size=200, session_id=None, is_train=True):
    """
    Generate a single window for visualization from Phase 2.
    """
    session_manager = SessionDataPhase2(data_path, is_train=is_train)
    sessions = session_manager.generate_session_obj()
    
    if session_id is None:
        session = sessions[0]
    else:
        session = next((s for s in sessions if s.session_id == session_id), None)
        if session is None:
            raise ValueError(f"Session {session_id} not found in {'train' if is_train else 'test'} set")
            
    sbp, kin = session.load_data()
    if sbp is None:
        raise RuntimeError(f"Failed to load SBP data for session {session.session_id}")
    
    N = sbp.shape[0]
    if N < window_size:
        raise ValueError(f"Session {session.session_id} has {N} bins, smaller than window_size={window_size}")
    
    # Identify active channels for the whole session
    active_mask = ~(sbp == 0).all(axis=0)
    
    # Pick a random window from the middle of the session
    w0 = (N - window_size) // 2
    
    sbp_w = sbp[w0:w0 + window_size].copy()
    kin_w = kin[w0:w0 + window_size].copy() if kin is not None else None
    
    return {
        "sbp": sbp_w,
        "kin": kin_w,
        "session_id": session.session_id,
        "w0": w0,
        "active_mask": active_mask
    }

def main():
    parser = argparse.ArgumentParser(description="Visualize a window from Phase 2 dataset")
    parser.add_argument("--data-path", "--data_path", type=str, default="kaggle_data_phase2",
                       help="Path to data directory")
    parser.add_argument("--session-id", "--session_id", type=str, default=None,
                       help="Session ID (e.g. D001). If omitted, uses first session.")
    parser.add_argument("--window-size", "--window_size", type=int, default=200,
                       help="Window size")
    parser.add_argument("--test", action="store_true", help="Look in test set instead of train set")
    parser.add_argument("--save-path", "--save_path", type=str, default=None,
                       help="Save figure to this path instead of displaying")
    
    args = parser.parse_args()
    
    print(f"Generating Phase 2 window from {args.data_path}...")
    sample = generate_one_phase2_window(
        data_path=args.data_path,
        window_size=args.window_size,
        session_id=args.session_id,
        is_train=not args.test
    )
    
    print(f"Visualizing session {sample['session_id']}, w0={sample['w0']}...")
    visualize_phase2_window(sample, save_path=args.save_path)

if __name__ == "__main__":
    main()
