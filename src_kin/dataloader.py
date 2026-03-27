import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .preprocessing import SessionDataPhase2

class KinematicsDataset(Dataset):
    def __init__(self, sessions_data, window_size=200, step_size=50, is_train=True):
        self.sessions_data = sessions_data
        self.window_size = window_size
        self.is_train = is_train
        self.windows = []
        
        # Create sliding windows
        for i, session in enumerate(sessions_data):
            N = session["N"]
            if N < self.window_size:
                continue
            
            # Using step_size for training, non-overlapping for validation/testing
            step = step_size if is_train else self.window_size
            for w0 in range(0, N - self.window_size + 1, step):
                self.windows.append((i, w0))
                
        print(f"Prepared {len(self.windows)} windows (is_train={is_train})")
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        sess_idx, w0 = self.windows[idx]
        session = self.sessions_data[sess_idx]
        
        sbp_w = torch.from_numpy(session["sbp"][w0:w0 + self.window_size]).clone()
        
        # Determine source and apply masking if Phase 1
        is_phase1 = session.get("source_name", "phase2") == "phase1"
        
        if is_phase1 and self.is_train:
            # Randomly mask 30% of channels for Phase 1 data (replicate Phase 2)
            C = sbp_w.shape[1]
            num_to_mask = int(C * 0.3)
            # Mask the same channels across the entire window (like Phase 2)
            masked_channels = torch.randperm(C)[:num_to_mask]
            sbp_w[:, masked_channels] = 0.0
            
        # Calculate mask: True where channels are fully 0.
        mask = (sbp_w == 0.0)
        
        # Extract numerical session ID for Perceiver IO
        try:
            num_id = float(''.join(filter(str.isdigit, session["session_id"])))
        except ValueError:
            num_id = 0.0
            
        item = {
            "sbp_masked": sbp_w.float(),
            "mask": mask.float(),
            "session_id": session["session_id"],
            "session_num": torch.tensor([num_id], dtype=torch.float32),
            "macro_timestamp": torch.tensor(w0, dtype=torch.float32) # required by MAE
        }
        
        if "kin" in session and session["kin"] is not None:
            kin_w = torch.from_numpy(session["kin"][w0:w0 + self.window_size])
            item["kin"] = kin_w.float()
            
        return item

def get_dataloaders(config, val_split=0.2, shuffle=True, num_workers=4, pin_memory=False):
    print("Loading full sessions into RAM...")
    
    all_sessions_data = []
    
    # --- Load Phase 2 Sessions ---
    session_manager_p2 = SessionDataPhase2(config.data_path, is_train=True)
    sessions_p2 = session_manager_p2.generate_session_obj(source_name="phase2")
    
    for session in tqdm(sessions_p2, desc="Processing Phase 2 sessions"):
        sbp, kin = session.load_data()
        if sbp is None or sbp.shape[0] < config.window_size:
            continue
            
        session_dict = {
            "sbp": sbp,
            "kin": kin,
            "N": sbp.shape[0],
            "session_id": session.session_id,
            "source_name": "phase2"
        }
        all_sessions_data.append(session_dict)

    # --- Load Phase 1 Sessions ---
    if hasattr(config, 'phase1_data_path') and os.path.exists(config.phase1_data_path):
        session_manager_p1 = SessionDataPhase2(config.phase1_data_path, is_train=True)
        sessions_p1 = session_manager_p1.generate_session_obj(source_name="phase1")
        
        for session in tqdm(sessions_p1, desc="Processing Phase 1 sessions"):
            sbp, kin = session.load_data()
            if sbp is None or sbp.shape[0] < config.window_size:
                continue
                
            session_dict = {
                "sbp": sbp,
                "kin": kin,
                "N": sbp.shape[0],
                "session_id": session.session_id,
                "source_name": "phase1"
            }
            all_sessions_data.append(session_dict)
        
    print(f"Loaded {len(all_sessions_data)} full sessions into RAM.")
    
    random.seed(config.seed)
    random.shuffle(all_sessions_data)
    val_size = max(1, int(len(all_sessions_data) * val_split))
    
    val_sessions = all_sessions_data[:val_size]
    train_sessions = all_sessions_data[val_size:]
    
    train_dataset = KinematicsDataset(train_sessions, window_size=config.window_size, step_size=50, is_train=True)
    val_dataset = KinematicsDataset(val_sessions, window_size=config.window_size, step_size=config.window_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, train_dataset, val_dataset
