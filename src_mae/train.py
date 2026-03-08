"""
Training script for SBP masked reconstruction.

Uses preprocess_non_overlapping() to generate windows, then trains on them.
Clear 2-step process: preprocess -> train
"""
import os
import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor, SBPImputer, SBP_TCN_Transformer
from dataloader import SBPDataset, get_dataloaders
from losses import masked_nmse_loss, kaggle_aligned_nmse_loss
from preprocessing import preprocess_non_overlapping
from config import Config


# Create configuration instance as global variable for easy access in functions
config = Config()

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
    
    elif config.model_name == "transformer":
        model = SBPImputer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            sbp_channels=config.sbp_channels,
            kin_channels=config.kin_channels
        )
        print(f"Built Transformer Reconstructor with d_model={config.d_model}, nhead={config.nhead}, num_layers={config.num_layers}, dropout={config.dropout}")

    elif config.model_name == "tcn_transformer":
        model = SBP_TCN_Transformer(
            sbp_channels=config.sbp_channels,
            kin_channels=config.kin_channels,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            tcn_levels=config.tcn_levels,  # Pass the new config variable
            dropout=config.dropout
        )
        print("Built Hybrid TCN + Cross-Channel Transformer")
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model.to(config.device)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch, scheduler=None):
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
        kin = batch["kin"].to(config.device)      # (B, W, 4)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
        
        # The model's concat step needs floats, but the inverse logic in the loss (~mask) needs bools
        mask_float = batch["mask"].to(config.device).float()
        mask = batch["mask"].to(config.device)     # (B, W, C)
        channel_var = batch["channel_var"].to(config.device)  # (B, C) per-channel variance from session
        session_ids = batch["session_id"]  # list of session IDs for proper grouping
        
        batch_size = x_sbp.size(0)
        optimizer.zero_grad()
        
        # ===== Forward pass =====
        pred = model(x_sbp, kin, mask_float, macro_timestamp)  # (B, W, C)
        
        # ===== Compute loss =====
        # Loss is computed ONLY on masked positions
        # Uses Kaggle-aligned NMSE that groups by (session, channel)
        loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
        
        # ===== Backward pass =====
        loss.backward()

        # Gradient clipping for model stability
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
    
    avg_loss = total_loss / total_samples
    return avg_loss


def validate_one_epoch(model, dataloader, config, epoch):
    """
    Validation for one epoch - computes loss without backprop.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        config: Config object
        epoch: Current epoch number
    
    Returns:
        Average validation loss over the epoch
    """
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            x_sbp = batch["x_sbp"].to(config.device)  # (B, W, C)
            y_sbp = batch["y_sbp"].to(config.device)  # (B, W, C)
            kin = batch["kin"].to(config.device)      # (B, W, 4)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
            
            # The model's concat step needs floats, but the inverse logic in the loss (~mask) needs bools
            mask_float = batch["mask"].to(config.device).float()
            mask = batch["mask"].to(config.device)     # (B, W, C)
            channel_var = batch["channel_var"].to(config.device)  # (B, C) per-channel variance from session
            session_ids = batch["session_id"]  # list of session IDs for proper grouping
            
            batch_size = x_sbp.size(0)
            
            # ===== Forward pass (no grad tracking) =====
            pred = model(x_sbp, kin, mask_float, macro_timestamp)  # (B, W, C)
            
            # ===== Compute loss =====
            loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
            
            # ===== Logging =====
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{total_loss / total_samples:.4f}'
            })
    
    avg_loss = total_loss / total_samples
    return avg_loss



# ============================================================================
# Main Training Script
# ============================================================================
def main():
    """Main training function."""
    global config
    
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
    

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        windows_dir=config.windows_dir,
        batch_size=config.batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if config.device == "cuda" else False
    )

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
    print(f"  Train Batches per epoch: {len(train_loader)}")
    print(f"  Train Samples per epoch: {len(train_dataset)}")
    print(f"  Validation Batches per epoch: {len(val_loader)}")
    print(f"  Validation Samples per epoch: {len(val_dataset)}")
    
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    
    # ===== Step 5: Training Loop =====
    print("\n" + "=" * 70)
    print("STEP 4: Training")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 70}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch, scheduler=scheduler)
        print(f"\n[Epoch {epoch}] Train NMSE Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate_one_epoch(model, val_loader, config, epoch)
        print(f"[Epoch {epoch}] Val NMSE Loss: {val_loss:.6f}")

        scheduler.step()
        print(f"[Epoch {epoch}] LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Track best validation loss
        if val_loss < (best_val_loss - config.early_stopping_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            best_model_path = os.path.join(config.checkpoint_dir, f"best_model{config.model_name}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
            }, best_model_path)
            print(f"✓ Best model saved (val_loss: {val_loss:.6f}): {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"[Epoch {epoch}] No val improvement "
                f"({epochs_without_improvement}/{config.early_stopping_patience})"
            )
        
        # Save regular checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"model_epoch_{epoch}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
            }, checkpoint_path)
            print(f"--> Checkpoint saved: {checkpoint_path}")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best epoch: {best_epoch}, best val: {best_val_loss:.6f})"
            )
            break
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best model: epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
