import os
import torch
import numpy as np
import pickle
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import random
from tqdm import tqdm

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
        
        # Load global signature for behavioral prior
        self.base_sig = None
        if config is not None:
            sig_path = os.path.join(config.data_path, "base_kinematic_signature.npy")
            if os.path.exists(sig_path):
                self.base_sig = torch.from_numpy(np.load(sig_path)).float()
        
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
        
        # Compute behavioral prior (intended signature based on movement)
        if self.base_sig is not None:
            # kin_w: (W, 4), base_sig: (4, 96) -> prior: (W, 96)
            sbp_prior = torch.matmul(kin_w.float(), self.base_sig)
        else:
            sbp_prior = torch.zeros_like(y_sbp)

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
            channel_shift = session["channel_shift"]
        else:
            # Deterministic for validation
            rng = np.random.default_rng(idx + 42)
            y_np = y_sbp.numpy().copy()
            
            num_spans = 2
            spans = sample_multi_span_lengths_and_starts(rng, self.window_size, num_spans=num_spans, min_gap=10)
            x_np, m_np = apply_multi_span_mask_to_window(y_np, spans, num_spans=num_spans, rng=rng)
            
            x_sbp = torch.from_numpy(x_np)
            mask = torch.from_numpy(m_np)
            channel_shift = session["channel_shift"]

        return {
            "x_sbp": x_sbp.float(),
            "y_sbp": y_sbp.float(),
            "sbp_prior": sbp_prior.float(),
            "mask": mask.float(),
            "kin": kin_w.float(),
            "channel_var": torch.from_numpy(session["channel_var"]).float(),
            "session_id": session["session_id"],
            "macro_timestamp": w0,
            "channel_shift": torch.tensor(channel_shift).float(),
        }

from preprocessing import sample_multi_span_lengths_and_starts, apply_multi_span_mask_to_window, compute_session_channel_variance

class SBPDatasetDynamic(Dataset):
    def __init__(self, sessions_data, is_train=True, samples_per_epoch=128000, window_size=200):
        self.sessions_data = sessions_data
        self.is_train = is_train
        self.window_size = window_size
        self.samples_per_epoch = samples_per_epoch
        
    def __len__(self):
        # We define an arbitrary epoch length since data is sampled randomly
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        if self.is_train:
            # ==========================================================
            # FAST PATH: Training (Ultra-fast PyTorch Native Ops)
            # ==========================================================
            # 1. Randomly pick a session (no slow RNG object instantiation)
            sess_idx = torch.randint(0, len(self.sessions_data), (1,)).item()
            session = self.sessions_data[sess_idx]
            
            # 2. Randomly pick a start index
            max_start = session["N"] - self.window_size
            w0 = torch.randint(0, max_start + 1, (1,)).item()
            
            # 3. Zero-Copy Tensor Creation: View the array, clone once for x_sbp
            y_sbp = torch.from_numpy(session["sbp"][w0:w0 + self.window_size])
            kin_w = torch.from_numpy(session["kin"][w0:w0 + self.window_size])
            
            x_sbp = y_sbp.clone()
            mask = torch.zeros_like(y_sbp, dtype=torch.bool)
            C = y_sbp.shape[1]
            
            # 4. Fast inline dynamic masking (bypasses old_preprocessing.py)
            num_spans = torch.randint(2, 4, (1,)).item()
            # Approx triangular distribution (45 to 85 length for 400 window, 20, 50 for 200)
            span_lengths = torch.randint(45, 85, (num_spans,)) 
            
            total_len = span_lengths.sum().item() + (num_spans - 1) * 10
            if total_len < self.window_size:
                available_starts = self.window_size - total_len
                # Distribute the remaining gap space randomly
                offsets = torch.rand(num_spans)
                offsets = (offsets / (offsets.sum() + 1e-6) * available_starts).int()
                
                curr_t = 0
                for i in range(num_spans):
                    curr_t += offsets[i].item()
                    t0 = curr_t
                    t1 = t0 + span_lengths[i].item()
                    
                    # Fast channel selection using randperm
                    num_channels = torch.randint(20, 40, (1,)).item()
                    channels = torch.randperm(C)[:num_channels]
                    
                    x_sbp[t0:t1, channels] = 0.0
                    mask[t0:t1, channels] = True
                    
                    curr_t = t1 + 10 # Enforce minimum gap
        else:
            # ==========================================================
            # DETERMINISTIC PATH: Validation 
            # ==========================================================
            # We keep the old logic here to ensure validation metrics are perfectly reproducible across epochs and runs.
            rng = np.random.default_rng(idx + 42) 
            session = rng.choice(self.sessions_data)
            
            max_start = session["N"] - self.window_size
            w0 = rng.integers(0, max_start + 1)
            
            y_np = session["sbp"][w0:w0 + self.window_size].copy()
            kin_np = session["kin"][w0:w0 + self.window_size].copy()
            
            num_spans = 2
            spans = sample_multi_span_lengths_and_starts(rng, self.window_size, num_spans=num_spans, min_gap=10)
            x_np, m_np = apply_multi_span_mask_to_window(y_np, spans, num_spans=num_spans, rng=rng)
            
            x_sbp = torch.from_numpy(x_np)
            y_sbp = torch.from_numpy(y_np)
            mask = torch.from_numpy(m_np)
            kin_w = torch.from_numpy(kin_np)

        return {
            "x_sbp": x_sbp.float(),
            "y_sbp": y_sbp.float(),
            "mask": mask.float(),
            "kin": kin_w.float(),
            "channel_var": torch.from_numpy(session["channel_var"]).float(),
            "session_id": session["session_id"],
            "macro_timestamp": w0,
            "channel_shift": torch.tensor(0.0).float(),
        }



def get_dataloaders(config, batch_size=32, val_split=0.2, shuffle=True, num_workers=4, pin_memory=False):
    """
    Creates Training and Validation DataLoaders directly from RAM.
    Splits sessions 80/20 to prevent data leakage.
    Uses Kinematic-Neural Signatures for robust channel alignment.
    """
    from preprocessing import sessionData, compute_session_channel_variance, compute_kinematic_signature
    
    print("Loading full sessions into RAM for dynamic augmentation...")
    sessions, _ = sessionData(f"{config.data_path}/metadata.csv").generate_session_obj()
    
    all_sessions_data = []
    lag_bins = config.lag_bins  # Biological kinematic shift
    base_sig = None
    sig_path = os.path.join(config.data_path, "base_kinematic_signature.npy")
    
    # Pre-load/calculate global kinematic signature template
    if os.path.exists(sig_path):
        base_sig = np.load(sig_path)
        print(f"Loaded base signature from {sig_path}")
    
    for session in tqdm(sessions, desc="Processing sessions"):
        if session.isTest():
            continue
            
        sbp, kin, _, _ = session.load_data(config.data_path)
        if sbp is None or sbp.shape[0] < config.window_size:
            continue
            
        # Apply biological kinematic shift globally
        kin_aligned = np.zeros_like(kin)
        if lag_bins > 0:
            kin_aligned[lag_bins:] = kin[:-lag_bins]
        else:
            kin_aligned = kin

        # Extract current session's kinematic-neural signature
        sig = compute_kinematic_signature(sbp, kin_aligned)
        
        if base_sig is None:
            # First session defines the coordinate system
            base_sig = sig.copy()
            np.save(sig_path, base_sig)
            optimal_shift = 0
            print(f"Set base signature from session {session.session_id}")
        else:
            # Find the shift that maximizes correlation with the base signature
            best_corr = -1.0
            optimal_shift = 0
            for k in range(96):
                rolled_sig = np.roll(sig, shift=k, axis=1)
                # Correlation of the 4x96 maps
                corr = np.corrcoef(rolled_sig.flatten(), base_sig.flatten())[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    optimal_shift = k
            
            # Align the session
            sbp = np.roll(sbp, shift=optimal_shift, axis=1)
            
        session_dict = {
            "sbp": sbp,
            "kin": kin_aligned,
            "N": sbp.shape[0],
            "channel_var": compute_session_channel_variance(sbp),
            "session_id": session.session_id,
            "channel_shift": optimal_shift
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

def get_dataloaders_dynamic(config, window_size=200, batch_size=32, val_split=0.2, num_workers=4, pin_memory=False):
    """
    Creates Training and Validation DataLoaders directly from RAM.
    Splits sessions 80/20 to prevent data leakage and assigns distinct Datasets.
    """
    from preprocessing import sessionData
    print("Loading full sessions into RAM for dynamic augmentation...")
    sessions, _ = sessionData(f"{config.data_path}/metadata.csv").generate_session_obj()
    
    from config import Config
    config = Config()
    all_sessions_data = []
    lag_bins = config.lag_bins  # Biological kinematic shift
    
    for session in sessions:
        if session.isTest():
            continue
            
        sbp, kin, _, _ = session.load_data(config.data_path)
        if sbp is None or sbp.shape[0] < window_size:
            continue
            
        # Apply biological kinematic shift globally
        kin_aligned = np.zeros_like(kin)
        if lag_bins > 0:
            kin_aligned[lag_bins:] = kin[:-lag_bins]
        else:
            kin_aligned = kin
       

        session_dict = {
            "sbp": sbp,
            "kin": kin_aligned,
            "N": sbp.shape[0],
            "channel_var": compute_session_channel_variance(sbp),
            "session_id": session.session_id
        }
        all_sessions_data.append(session_dict)
        
    print(f"Loaded {len(all_sessions_data)} full sessions into RAM.")
    
    # Shuffle and split the SESSIONS, not the individual windows
    random.seed(config.seed)
    random.shuffle(all_sessions_data)
    val_size = max(1, int(len(all_sessions_data) * val_split))
    
    val_sessions = all_sessions_data[:val_size]
    train_sessions = all_sessions_data[val_size:]
    
    # Initialize separate datasets to control `is_train` flags
    train_dataset = SBPDatasetDynamic(
        train_sessions, is_train=True, window_size=window_size, samples_per_epoch=8192
    )
    val_dataset = SBPDatasetDynamic(
        val_sessions, is_train=False, window_size=window_size, samples_per_epoch=2048
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    # Validation strictly does not shuffle to maintain perfect deterministic alignment
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, train_dataset, val_dataset
