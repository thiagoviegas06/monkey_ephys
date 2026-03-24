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
        import numpy as np

        sess_idx, w0 = self.windows[idx]
        session = self.sessions_data[sess_idx]

        sbp_w = torch.from_numpy(session["sbp"][w0:w0 + self.window_size]).float()

        # Apply 30% channel masking consistent per session (matches test distribution)
        mask = torch.zeros_like(sbp_w, dtype=torch.bool)
        if self.is_train:
            # Use session_id to deterministically select which channels to mask
            # Same channels masked throughout the entire session (like real test data)
            session_id = session["session_id"]
            rng = np.random.RandomState(hash(session_id) & 0xFFFFFFFF)

            num_channels = sbp_w.shape[1]  # 96
            num_to_mask = max(1, int(num_channels * 0.3))
            channels_to_mask = rng.choice(num_channels, size=num_to_mask, replace=False)

            sbp_w[:, channels_to_mask] = 0.0
            mask[:, channels_to_mask] = True
        else:
            # Validation: no masking
            mask = (sbp_w == 0.0)

        item = {
            "sbp_masked": sbp_w,
            "mask": mask.float(),
            "session_id": session["session_id"],
            "macro_timestamp": torch.tensor(w0, dtype=torch.float32)
        }

        if "kin" in session and session["kin"] is not None:
            kin_w = torch.from_numpy(session["kin"][w0:w0 + self.window_size])
            item["kin"] = kin_w.float()

        return item

def get_dataloaders(config, val_split=0.2, shuffle=True, num_workers=4, pin_memory=False):
    print("Loading full sessions into RAM...")
    session_manager = SessionDataPhase2(config.data_path, is_train=True)
    sessions = session_manager.generate_session_obj()
    
    all_sessions_data = []
    
    for session in tqdm(sessions, desc="Processing Phase 2 sessions"):
        sbp, kin = session.load_data()
        if sbp is None or sbp.shape[0] < config.window_size:
            continue
            
        session_dict = {
            "sbp": sbp,
            "kin": kin,
            "N": sbp.shape[0],
            "session_id": session.session_id,
        }
        all_sessions_data.append(session_dict)
        
    print(f"Loaded {len(all_sessions_data)} full sessions into RAM.")
    
    random.seed(config.seed)
    random.shuffle(all_sessions_data)
    val_size = max(1, int(len(all_sessions_data) * val_split))
    
    val_sessions = all_sessions_data[:val_size]
    train_sessions = all_sessions_data[val_size:]
    
    train_dataset = KinematicsDataset(train_sessions, window_size=config.window_size, step_size=config.window_step_size, is_train=True)
    val_dataset = KinematicsDataset(val_sessions, window_size=config.window_size, step_size=config.window_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, train_dataset, val_dataset
