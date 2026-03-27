import os
import numpy as np
from tqdm import tqdm
import sys
import torch

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.preprocessing import SessionDataPhase2

def compute_kin_stats(data_path, phase1_data_path=None):
    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()
    
    if phase1_data_path and os.path.exists(phase1_data_path):
        session_manager_p1 = SessionDataPhase2(phase1_data_path, is_train=True)
        sessions_p1 = session_manager_p1.generate_session_obj(source_name="phase1")
        sessions.extend(sessions_p1)
        
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

def compute_sbp_stats(data_path, phase1_data_path=None):
    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()
    
    if phase1_data_path and os.path.exists(phase1_data_path):
        session_manager_p1 = SessionDataPhase2(phase1_data_path, is_train=True)
        sessions_p1 = session_manager_p1.generate_session_obj(source_name="phase1")
        sessions.extend(sessions_p1)
        
    all_sbp_means = []
    all_sbp_vars = []
    all_counts = []
    
    for session in tqdm(sessions, desc="Computing SBP stats"):
        sbp, _ = session.load_data()
        if sbp is not None:
            # In Phase 2, certain channels are fully 0 for the session.
            # We only want to compute stats over active channels.
            mask = (sbp == 0.0)
            visible_mask = (~mask)
            
            # Global mean and var for THIS session (across all active channels)
            num_visible = visible_mask.sum()
            if num_visible == 0: continue
            
            sess_mean = sbp.sum() / num_visible
            sess_var = ((sbp - sess_mean)**2 * visible_mask).sum() / num_visible
            
            all_sbp_means.append(sess_mean)
            all_sbp_vars.append(sess_var)
            all_counts.append(num_visible)
            
    if not all_sbp_means:
        print("No SBP data found.")
        return None, None
        
    # Aggregate across sessions (weighted average)
    all_counts = np.array(all_counts)
    all_sbp_means = np.array(all_sbp_means)
    all_sbp_vars = np.array(all_sbp_vars)
    
    total_count = all_counts.sum()
    global_mean = (all_sbp_means * all_counts).sum() / total_count
    
    # Global variance (law of total variance simplified for shared mean assumption or proper aggregation)
    # Correct way to aggregate variance:
    global_var = (all_counts * (all_sbp_vars + (all_sbp_means - global_mean)**2)).sum() / total_count
    global_std = np.sqrt(global_var) + 1e-8
    
    return torch.tensor(global_mean, dtype=torch.float32), torch.tensor(global_std, dtype=torch.float32)

if __name__ == "__main__":
    mean, std = compute_kin_stats("kaggle_data_phase2")
    if mean is not None:
        print("\nKinematics Stats (4 channels: index_pos, mrp_pos, index_vel, mrp_vel):")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")
        
    sbp_mean, sbp_std = compute_sbp_stats("kaggle_data_phase2")
    if sbp_mean is not None:
        print("\nSBP Stats (Global across active channels):")
        print(f"Mean: {sbp_mean:.4f}")
        print(f"Std:  {sbp_std:.4f}")

