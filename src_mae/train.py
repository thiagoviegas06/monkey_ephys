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

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor, SBPImputer
from dataloader import SBPDataset, get_dataloaders
from losses import masked_nmse_loss, masked_mse_loss
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
        kin = batch["kin"].to(config.device)      # (B, W, 4)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
        
        # The model's concat step needs floats, but the inverse logic in the loss (~mask) needs bools
        mask_float = batch["mask"].to(config.device).float()
        mask_bool = batch["mask"].to(config.device).bool()
        
        batch_size = x_sbp.size(0)
        optimizer.zero_grad()
        
        # ===== Forward pass =====
        pred = model(x_sbp, kin, mask_float, macro_timestamp)  # (B, W, C)
        
        # ===== Compute loss =====
        # Loss is computed ONLY on masked positions
        loss = masked_nmse_loss(pred, y_sbp, mask_bool)
        
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
        
        # Detailed logging every N batches
        if (batch_idx + 1) % config.log_every == 0:
            # Compute additional metrics for debugging
            with torch.no_grad():
                mse = masked_mse_loss(pred, y_sbp, mask_bool)
                
                # Check how many positions are masked
                n_masked = mask_bool.sum().item()
                n_total = mask_bool.numel()
                pct_masked = 100.0 * n_masked / n_total
                
                # Check prediction statistics on masked positions
                masked_pred = pred[mask_bool]
                masked_true = y_sbp[mask_bool]
                
                pred_mean = masked_pred.mean().item() if n_masked > 0 else 0.0
                pred_std = masked_pred.std().item() if n_masked > 1 else 0.0
                true_mean = masked_true.mean().item() if n_masked > 0 else 0.0
                true_std = masked_true.std().item() if n_masked > 1 else 0.0
            
            # print(f"\n[Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}]")
            # print(f"  NMSE Loss: {loss.item():.6f}")
            # print(f"  MSE Loss:  {mse.item():.6f}")
            # print(f"  Masked: {n_masked}/{n_total} ({pct_masked:.2f}%)")
            # print(f"  Pred (masked): mean={pred_mean:.4f}, std={pred_std:.4f}")
            # print(f"  True (masked): mean={true_mean:.4f}, std={true_std:.4f}")
    
    avg_loss = total_loss / total_samples
    return avg_loss


def validate_one_epoch(model, dataloader, device, epoch, num_epochs, log_every=50):
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{num_epochs}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x_sbp = batch["x_sbp"].to(device)
            y_sbp = batch["y_sbp"].to(device)
            kin = batch["kin"].to(device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            mask_float = batch["mask"].to(device).float()
            mask_bool = batch["mask"].to(device).bool()
            
            batch_size = x_sbp.size(0)
            
            # ===== Forward pass =====
            pred = model(x_sbp, kin, mask_float, macro_timestamp)
            
            # ===== Compute loss =====
            loss = masked_nmse_loss(pred, y_sbp, mask_bool)
            
            # ===== Logging =====
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / total_samples:.4f}'
            })
            
            # Detailed logging every N batches or at the very end of the validation epoch
            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(dataloader):
                mse = masked_mse_loss(pred, y_sbp, mask_bool)
                
                n_masked = mask_bool.sum().item()
                n_total = mask_bool.numel()
                pct_masked = 100.0 * n_masked / n_total
                
                masked_pred = pred[mask_bool]
                masked_true = y_sbp[mask_bool]
                
                pred_mean = masked_pred.mean().item() if n_masked > 0 else 0.0
                pred_std = masked_pred.std().item() if n_masked > 1 else 0.0
                true_mean = masked_true.mean().item() if n_masked > 0 else 0.0
                true_std = masked_true.std().item() if n_masked > 1 else 0.0
                
                # print(f"\n[Val Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}]")
                # print(f"  NMSE Loss: {loss.item():.6f}")
                # print(f"  MSE Loss:  {mse.item():.6f}")
                # print(f"  Masked: {n_masked}/{n_total} ({pct_masked:.2f}%)")
                # print(f"  Pred (masked): mean={pred_mean:.4f}, std={pred_std:.4f}")
                # print(f"  True (masked): mean={true_mean:.4f}, std={true_std:.4f}")
                
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
    
    # ===== Step 5: Training Loop =====
    print("\n" + "=" * 70)
    print("STEP 4: Training")
    print("=" * 70)
    
    best_val_nmse = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 70}")
        
        # Train for one epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
        
        print(f"\n[Epoch {epoch}] Average NMSE Loss: {avg_loss:.6f}")
        
        # Validation Phase
        print("\nValidating on validation set...")
        epoch_val_nmse = validate_one_epoch(model, val_loader, config.device, epoch, config.num_epochs)
        
        print(f"Epoch [{epoch}/{config.num_epochs}] | Train NMSE: {avg_loss:.4f} | Val NMSE: {epoch_val_nmse:.4f}")

        # Save the best model weights
        if epoch_val_nmse < best_val_nmse:
            best_val_nmse = epoch_val_nmse
            best_val_path = os.path.join(config.checkpoint_dir, f"best_{config.model_name}.pth")
            torch.save(model.state_dict(), best_val_path)
            print("  -> Saved new best model.")


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
            print(f"  --> Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
