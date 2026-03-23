import os
import numpy as np
from tqdm import tqdm
import sys
import torch

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.preprocessing import SessionDataPhase2

def compute_kin_stats(data_path):
    """
    DEPRECATED: This computes GLOBAL statistics (across all sessions).

    Per-project requirements, this BREAKS cross-session generalization due to drift.
    Use compute_per_session_kin_stats() instead.

    Kept for backward compatibility only.
    """
    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()

    all_kin = []
    for session in tqdm(sessions, desc="Loading kinematics for stats"):
        _, kin = session.load_data()
        if kin is not None:
            all_kin.append(kin)

    if not all_kin:
        print("No kinematics found.")
        return None, None

    all_kin = np.concatenate(all_kin, axis=0)
    mean = np.mean(all_kin, axis=0)
    std = np.std(all_kin, axis=0) + 1e-8 # Prevent division by zero

    # Return as tensors for easy use in training loop
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def compute_per_session_kin_stats(data_path):
    """
    Compute kinematics statistics PER SESSION (z-score normalization statistics).

    This is CRITICAL per project requirements:
    "Drift handling matters: the key ingredient is per-session z-score normalization.
     Without it, cross-session models fail."

    Returns: dict {session_id: {'mean': (4,), 'std': (4,)}}
    """
    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()

    session_stats = {}

    for session in tqdm(sessions, desc="Computing per-session kinematics stats"):
        _, kin = session.load_data()
        if kin is not None and kin.shape[0] > 0:
            mean = np.mean(kin, axis=0)  # (4,)
            std = np.std(kin, axis=0) + 1e-8  # (4,) prevent division by zero

            session_stats[session.session_id] = {
                'mean': torch.tensor(mean, dtype=torch.float32),
                'std': torch.tensor(std, dtype=torch.float32)
            }

    print(f"✓ Computed per-session kinematics stats for {len(session_stats)} sessions")
    return session_stats

if __name__ == "__main__":
    mean, std = compute_kin_stats("kaggle_data_phase2")
    if mean is not None:
        print("\nKinematics Stats (4 channels: index_pos, mrp_pos, index_vel, mrp_vel):")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")
