import os
import torch
import numpy as np
import pickle
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import random
from tqdm import tqdm

from preprocessing import sessionData, sample_multi_span_lengths_and_starts, apply_multi_span_mask_to_window, compute_session_channel_variance

# ============================================================================
# Dataset Class
# ============================================================================
class SBPDataset(Dataset):
    """
    PyTorch Dataset that loads full sessions into RAM and serves non-overlapping 
    windows dynamically with fast inline masking and behavioral priors.
    """
    def __init__(self, sessions_data, is_train=True, window_size=200, config=None):
        self.sessions_data = sessions_data
        self.is_train = is_train
        self.window_size = window_size
        self.windows = []
        self.config = config

        # Precompute all non-overlapping windows
        for i, session in enumerate(sessions_data):
            N = session["N"]
            for w0 in range(0, N - self.window_size + 1, self.window_size):
                self.windows.append((i, w0))
                
        print(f"Prepared {len(self.windows)} non-overlapping windows (is_train={is_train})")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        sess_idx, w0 = self.windows[idx]
        session = self.sessions_data[sess_idx]
        
        y_sbp = torch.from_numpy(session["sbp"][w0:w0 + self.window_size])
        kin_w = torch.from_numpy(session["kin"][w0:w0 + self.window_size])

        if self.is_train:
            x_sbp = y_sbp.clone()
            mask = torch.zeros_like(y_sbp, dtype=torch.bool)
            C = y_sbp.shape[1]
            
            # 4. Fast inline dynamic masking
            num_spans = torch.randint(2, 4, (1,)).item()
            # Approx triangular distribution
            span_lengths = torch.randint(45, 85, (num_spans,)) 
            
            total_len = span_lengths.sum().item() + (num_spans - 1) * 10
            if total_len < self.window_size:
                available_starts = self.window_size - total_len
                offsets = torch.rand(num_spans)
                offsets = (offsets / (offsets.sum() + 1e-6) * available_starts).int()
                
                curr_t = 0
                for i in range(num_spans):
                    curr_t += offsets[i].item()
                    t0 = curr_t
                    t1 = t0 + span_lengths[i].item()
                    
                    num_channels = torch.randint(20, 40, (1,)).item()
                    channels = torch.randperm(C)[:num_channels]
                    
                    x_sbp[t0:t1, channels] = 0.0
                    mask[t0:t1, channels] = True
                    
                    curr_t = t1 + 10

        else:
            # Deterministic for validation
            rng = np.random.default_rng(idx + 42)
            y_np = y_sbp.numpy().copy()
            
            num_spans = 2
            spans = sample_multi_span_lengths_and_starts(rng, self.window_size, num_spans=num_spans, min_gap=10)
            x_np, m_np = apply_multi_span_mask_to_window(y_np, spans, num_spans=num_spans, rng=rng)
            
            x_sbp = torch.from_numpy(x_np)
            mask = torch.from_numpy(m_np)

        return {
            "x_sbp": x_sbp.float(),
            "y_sbp": y_sbp.float(),
            "mask": mask.float(),
            "kin": kin_w.float(),
            "channel_var": torch.from_numpy(session["channel_var"]).float(),
            "session_id": session["session_id"],
            "macro_timestamp": w0,
        }



def get_dataloaders(config, batch_size=32, val_split=0.2, shuffle=True, num_workers=8, pin_memory=False):
    """
    Creates Training and Validation DataLoaders directly from RAM.
    Splits sessions 80/20 to prevent data leakage.
    Uses Kinematic-Neural Signatures for robust channel alignment.
    """
    
    print("Loading full sessions into RAM for dynamic augmentation...")
    sessions, _ = sessionData(f"{config.data_path}/metadata.csv").generate_session_obj()
    
    all_sessions_data = []
    base_sig = None
    
    for session in tqdm(sessions, desc="Processing sessions"):
        if session.isTest():
            continue
            
        sbp, kin, _, _ = session.load_data(config.data_path)
        if sbp is None or sbp.shape[0] < config.window_size:
            continue
            
        session_dict = {
            "sbp": sbp,
            "kin": kin,
            "N": sbp.shape[0],
            "channel_var": compute_session_channel_variance(sbp),
            "session_id": session.session_id,
        }
        all_sessions_data.append(session_dict)
        
    print(f"Loaded {len(all_sessions_data)} full sessions into RAM.")
    
    # Shuffle and split the SESSIONS, not the individual windows
    random.seed(config.seed)
    random.shuffle(all_sessions_data)
    val_size = max(1, int(len(all_sessions_data) * val_split))
    
    val_sessions = all_sessions_data[:val_size]
    train_sessions = all_sessions_data[val_size:]
    
    train_dataset = SBPDataset(train_sessions, is_train=True, window_size=config.window_size, config=config)
    val_dataset = SBPDataset(val_sessions, is_train=False, window_size=config.window_size, config=config)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, train_dataset, val_dataset
