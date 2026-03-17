import os
import torch
import numpy as np
import pickle
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import random

# ============================================================================
# Dataset Class
# ============================================================================
class SBPDataset(Dataset):
    """
    PyTorch Dataset that loads pre-generated windows from preprocessing.
    
    Each .pkl file contains:
        - x_sbp: (W, 96) masked SBP (zeros where masked)
        - y_sbp: (W, 96) ground truth SBP
        - mask: (W, 96) boolean, True where masked
        - kin: (W, 4) kinematics (not used in current model)
        - session_id: str
        - w0: int, window start position
        - span: (t0, t1) masked time span
        - day: float
        - day_from_nearest: float
    """
    
    def __init__(self, windows_dir):
        """
        Args:
            windows_dir: Directory containing preprocessed .pkl files
        """
        self.windows_dir = windows_dir
        
        # Find all .pkl files
        pkl_pattern = os.path.join(windows_dir, "*.pkl")
        # self.sample_files = sorted(glob(pkl_pattern))
        self.sample_files = glob(pkl_pattern) 
        random.shuffle(self.sample_files)
        
        if len(self.sample_files) == 0:
            raise ValueError(
                f"No .pkl files found in {windows_dir}. "
                f"Run with Config.preprocess=True first!"
            )
        
        print(f"Found {len(self.sample_files)} preprocessed windows")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        """
        Load one preprocessed window.
        
        Returns:
            dict with keys:
                - x_sbp: (W, C) tensor, masked input
                - y_sbp: (W, C) tensor, ground truth
                - mask: (W, C) boolean tensor, True=masked
                - session_id: str
        """
        # Load pickle file
        with open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)

        # Convert to tensors (data already in correct format from preprocessing)
        return {
            "x_sbp": torch.from_numpy(sample["x_sbp"]).float(),  # (W, 96) float32
            "y_sbp": torch.from_numpy(sample["y_sbp"]).float(),  # (W, 96) float32
            "mask": torch.from_numpy(sample["mask"]).float(),    # (W, 96) bool
            "kin": torch.from_numpy(sample["kin"]).float(),      # (W, 4) float32
            "channel_var": torch.from_numpy(sample["channel_var"]).float(), # (96,) float32
            "session_id": sample["session_id"],
            "macro_timestamp": sample["w0"],  # Using window start position as macro timestamp
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
        }



def get_dataloaders(windows_dir, batch_size=32, val_split=0.2, shuffle=True, num_workers=4,  pin_memory=False):
    """
    Creates Training and Validation DataLoaders directly from the preprocessed directory.
    Uses PyTorch's random_split to handle the 80/20 division.
    """
    dataset = SBPDataset(windows_dir)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Use a fixed generator seed so the train/val split is reproducible across runs
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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
