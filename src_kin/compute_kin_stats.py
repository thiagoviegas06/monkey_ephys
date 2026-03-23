import os
import numpy as np
from tqdm import tqdm
import sys
import torch

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.preprocessing import SessionDataPhase2

def compute_kin_stats(data_path):
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

if __name__ == "__main__":
    mean, std = compute_kin_stats("kaggle_data_phase2")
    if mean is not None:
        print("\nKinematics Stats (4 channels: index_pos, mrp_pos, index_vel, mrp_vel):")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")
