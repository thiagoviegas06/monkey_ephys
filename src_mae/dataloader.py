import os
import torch
import numpy as np
import pickle
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

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
        self.sample_files = sorted(glob(pkl_pattern))
        
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
            "session_id": sample["session_id"],
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
