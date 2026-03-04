"""
Training script for SBP masked reconstruction.

Uses preprocess_non_overlapping() to generate windows, then trains on them.
Clear 2-step process: preprocess -> train
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor
from losses import masked_nmse_loss, masked_mse_loss
from preprocessing import preprocess_non_overlapping

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Training configuration - modify these parameters"""
    
    # Data
    data_path = "kaggle_data"
    window_size = 200
    seed = 42
    
    # Preprocessing
    preprocess = False  # Set True to run preprocessing (only needed once)
    windows_dir = "kaggle_data/masked_windows"  # Where preprocessed windows are saved
    
    # Model
    model_name = "unet"  # Options: "unet", "simple_cnn", "resnet"
    base_channels = 64   # For UNet/ResNet
    
    # Training
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 10
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = "checkpoints"
    save_every = 1  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches


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
            "x_sbp": torch.from_numpy(sample["x_sbp"]),  # (W, 96) float32
            "y_sbp": torch.from_numpy(sample["y_sbp"]),  # (W, 96) float32
            "mask": torch.from_numpy(sample["mask"]),    # (W, 96) bool
            "session_id": sample["session_id"],
        }


# ============================================================================
# Model Builder
# ============================================================================
def build_model(config):
    """Build model based on config."""
    if config.model_name == "unet":
        model = SBP_Reconstruction_UNet(base_channels=config.base_channels)
        print(f"Built U-Net with base_channels={config.base_channels}")
    elif config.model_name == "simple_cnn":
        model = SimpleCNN(hidden_channels=128, num_layers=6)
        print("Built SimpleCNN")
    elif config.model_name == "resnet":
        model = ResNetReconstructor(hidden_channels=128, num_blocks=8)
        print("Built ResNet Reconstructor")
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model.to(config.device)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        optimizer: Optimizer
        config: Config object
        epoch: Current epoch number
    
    Returns:
        Average loss over the epoch
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        x_sbp = batch["x_sbp"].to(config.device)  # (B, W, C)
        y_sbp = batch["y_sbp"].to(config.device)  # (B, W, C)
        mask = batch["mask"].to(config.device)     # (B, W, C)
        
        batch_size = x_sbp.size(0)
        
        # ===== Forward pass =====
        pred = model(x_sbp, mask)  # (B, W, C)
        
        # ===== Compute loss =====
        # Loss is computed ONLY on masked positions
        loss = masked_nmse_loss(pred, y_sbp, mask)
        
        # ===== Backward pass =====
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # ===== Logging =====
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / total_samples:.4f}'
        })
        
        # Detailed logging every N batches
        if (batch_idx + 1) % config.log_every == 0:
            # Compute additional metrics for debugging
            with torch.no_grad():
                mse = masked_mse_loss(pred, y_sbp, mask)
                
                # Check how many positions are masked
                n_masked = mask.sum().item()
                n_total = mask.numel()
                pct_masked = 100.0 * n_masked / n_total
                
                # Check prediction statistics on masked positions
                masked_pred = pred[mask]
                masked_true = y_sbp[mask]
                
                pred_mean = masked_pred.mean().item()
                pred_std = masked_pred.std().item()
                true_mean = masked_true.mean().item()
                true_std = masked_true.std().item()
            
            print(f"\n[Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}]")
            print(f"  NMSE Loss: {loss.item():.6f}")
            print(f"  MSE Loss:  {mse.item():.6f}")
            print(f"  Masked: {n_masked}/{n_total} ({pct_masked:.2f}%)")
            print(f"  Pred (masked): mean={pred_mean:.4f}, std={pred_std:.4f}")
            print(f"  True (masked): mean={true_mean:.4f}, std={true_std:.4f}")
    
    avg_loss = total_loss / total_samples
    return avg_loss


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    """Main training function."""
    
    # Configuration
    config = Config()
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Window size: {config.window_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Windows dir: {config.windows_dir}")
    print("=" * 70)
    
    # ===== Step 1: Preprocessing (if needed) =====
    if config.preprocess:
        print("\n" + "=" * 70)
        print("STEP 1: Preprocessing - Generating Windows")
        print("=" * 70)
        print(f"Calling preprocess_non_overlapping()...")
        print(f"  data_path: {config.data_path}")
        print(f"  window_size: {config.window_size}")
        print(f"  seed: {config.seed}")
        print()
        
        preprocess_non_overlapping(
            data_path=config.data_path,
            window_size=config.window_size,
            seed=config.seed
        )
        
        print(f"\n✓ Preprocessing complete! Windows saved to {config.windows_dir}")
    else:
        print(f"\nSkipping preprocessing (config.preprocess=False)")
        print(f"Using existing windows from: {config.windows_dir}")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # ===== Step 2: Build Dataset & DataLoader =====
    print("\n" + "=" * 70)
    print("STEP 2: Loading Preprocessed Data")
    print("=" * 70)
    
    train_dataset = SBPDataset(windows_dir=config.windows_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if config.device == "cuda" else False
    )
    
    print(f"\nDataLoader created:")
    print(f"  Total windows: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Samples per epoch: {len(train_dataset)}")
    
    # ===== Step 3: Build Model =====
    print("\n" + "=" * 70)
    print("STEP 3: Building Model")
    print("=" * 70)
    
    model = build_model(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    
    # ===== Step 4: Setup Optimizer =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # ===== Step 5: Training Loop =====
    print("\n" + "=" * 70)
    print("STEP 4: Training")
    print("=" * 70)
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 70}")
        
        # Train for one epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
        
        print(f"\n[Epoch {epoch}] Average NMSE Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"model_epoch_{epoch}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.__dict__,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
