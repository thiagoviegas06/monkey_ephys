"""
Training script for kinematics prediction from SBP.

Two-stage training:
  Stage 1: Freeze SBP encoder, train kinematics head
  Stage 2: Unfreeze SBP encoder with lower LR for fine-tuning

Uses weighted MSE loss:
  - Position channels (index_pos, mrp_pos): weight=0.8
  - Velocity channels (index_vel, mrp_vel): weight=0.2
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob

from model import SBP_TCN_Transformer, SBPtoKinematicsModel
from dataloader import SBPDataset, get_dataloaders
from config import Config

# Create configuration instance
config = Config()

# ============================================================================
# Loss Functions
# ============================================================================
def weighted_kin_loss(pred, target, position_weight=0.8, velocity_weight=0.2):
    """
    Weighted MSE loss for kinematics.

    Channels:
    - [0, 1]: Position channels (index_pos, mrp_pos) → higher weight
    - [2, 3]: Velocity channels (index_vel, mrp_vel) → lower weight

    Args:
        pred: (B, W, 4) predicted kinematics
        target: (B, W, 4) target kinematics
        position_weight: weight for position channels (default 0.8)
        velocity_weight: weight for velocity channels (default 0.2)

    Returns:
        scalar loss
    """
    # Compute per-channel MSE
    mse = (pred - target) ** 2  # (B, W, 4)

    # Apply weights per channel
    # Channels 0, 1: position → position_weight
    # Channels 2, 3: velocity → velocity_weight
    weights = torch.tensor(
        [position_weight, position_weight, velocity_weight, velocity_weight],
        device=pred.device,
        dtype=pred.dtype
    )  # (4,)

    # Expand weights to match mse shape: (B, W, 4)
    weighted_mse = mse * weights.view(1, 1, 4)

    # Return mean across all dimensions
    return weighted_mse.mean()


# ============================================================================
# Model Builder
# ============================================================================
def build_sbp_model(config, checkpoint_path):
    """Load pre-trained SBP model."""
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        kin_channels=config.kin_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"✓ Loaded SBP model from {checkpoint_path}")
    return model


def build_kin_model(sbp_model, config):
    """Build kinematics model on top of frozen SBP model."""
    model = SBPtoKinematicsModel(
        sbp_model=sbp_model,
        freeze_sbp=True,
        hidden_dim=config.kin_hidden_dim,
        dropout=config.dropout
    )
    print(f"✓ Built kinematics model with frozen SBP encoder")
    return model


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch, scheduler=None):
    """Train for one epoch with weighted loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        x_sbp = batch["x_sbp"].to(config.device)      # (B, W, 96)
        kin_true = batch["kin"].to(config.device)     # (B, W, 4)
        mask = batch["mask"].to(config.device).float() # (B, W, 96)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)

        batch_size = x_sbp.size(0)
        optimizer.zero_grad()

        # ===== Forward pass =====
        kin_pred = model(x_sbp, mask, macro_timestamp)  # (B, W, 4)

        # ===== Compute weighted loss =====
        # Position channels (0, 1): weight=0.8
        # Velocity channels (2, 3): weight=0.2
        loss = weighted_kin_loss(kin_pred, kin_true, position_weight=0.8, velocity_weight=0.2)

        # ===== Backward pass =====
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ===== Logging =====
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss / total_samples:.6f}'
        })

    avg_loss = total_loss / total_samples
    return avg_loss


def validate_one_epoch(model, dataloader, config, epoch):
    """Validation for one epoch with weighted loss."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            x_sbp = batch["x_sbp"].to(config.device)      # (B, W, 96)
            kin_true = batch["kin"].to(config.device)     # (B, W, 4)
            mask = batch["mask"].to(config.device).float() # (B, W, 96)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)

            batch_size = x_sbp.size(0)

            # ===== Forward pass =====
            kin_pred = model(x_sbp, mask, macro_timestamp)  # (B, W, 4)

            # ===== Compute weighted loss =====
            loss = weighted_kin_loss(kin_pred, kin_true, position_weight=0.8, velocity_weight=0.2)

            # ===== Logging =====
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({
                'val_loss': f'{loss.item():.6f}',
                'avg_val_loss': f'{total_loss / total_samples:.6f}'
            })

    avg_loss = total_loss / total_samples
    return avg_loss


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    global config

    print("\n" + "=" * 70)
    print("KINEMATICS PREDICTION FROM SBP (Transfer Learning)")
    print("=" * 70)

    # ===== Setup =====
    device = config.device
    print(f"\nDevice: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")

    # ===== Create checkpoint directory =====
    kin_checkpoint_dir = f"checkpoints_{config.window_size}_kinematics"
    os.makedirs(kin_checkpoint_dir, exist_ok=True)
    print(f"Checkpoint dir: {kin_checkpoint_dir}")

    # ===== Load pre-trained SBP model =====
    print("\n" + "=" * 70)
    print("Loading Pre-trained SBP Model")
    print("=" * 70)
    sbp_checkpoint = f"checkpoints_{config.window_size}/best_model_tcn_transformer.pt"
    if not os.path.exists(sbp_checkpoint):
        raise FileNotFoundError(f"SBP checkpoint not found: {sbp_checkpoint}")
    sbp_model = build_sbp_model(config, sbp_checkpoint).to(device)

    # ===== Build kinematics model =====
    print("\n" + "=" * 70)
    print("Building Kinematics Model")
    print("=" * 70)
    kin_model = build_kin_model(sbp_model, config).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in kin_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in kin_model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} (SBP frozen)")

    # ===== Load data =====
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        windows_dir=config.windows_dir,
        batch_size=config.batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ===== Two-Stage Training Setup =====
    print("\n" + "=" * 70)
    print("STAGE 1: Training Kinematics Head (SBP Encoder Frozen)")
    print("=" * 70)

    stage1_epochs = min(20, config.num_epochs // 3)  # First ~20 epochs or 1/3 of total
    stage2_epochs = config.num_epochs - stage1_epochs

    # Stage 1: Frozen encoder
    optimizer_stage1 = torch.optim.AdamW(
        [p for p in kin_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1, T_max=stage1_epochs, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    stage2_started = False

    # ===== Training loop =====
    print("\n" + "=" * 70)
    print("Training Kinematics Model")
    print("=" * 70)

    for epoch in range(1, config.num_epochs + 1):
        # ===== Stage 2: Transition =====
        if epoch == stage1_epochs + 1 and not stage2_started:
            print("\n" + "=" * 70)
            print("STAGE 2: Fine-tuning SBP Encoder (Lower Learning Rate)")
            print("=" * 70)
            stage2_started = True

            # Unfreeze SBP encoder for fine-tuning
            for param in kin_model.sbp_model.parameters():
                param.requires_grad = True

            # Create new optimizer with different LRs
            optimizer_stage2 = torch.optim.AdamW([
                {'params': kin_model.sbp_model.parameters(), 'lr': 1e-5},  # Low LR for encoder
                {'params': kin_model.kin_head.parameters(), 'lr': config.learning_rate}  # Normal LR for head
            ], weight_decay=config.weight_decay)

            scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_stage2, T_max=stage2_epochs, eta_min=1e-6
            )
            optimizer = optimizer_stage2
            scheduler = scheduler_stage2
            patience_counter = 0  # Reset patience for stage 2
            best_val_loss = float('inf')  # Reset best loss
            print(f"✓ Unfroze SBP encoder with LR=1e-5")
        else:
            optimizer = optimizer_stage1 if epoch <= stage1_epochs else optimizer
            scheduler = scheduler_stage1 if epoch <= stage1_epochs else scheduler

        # Train
        train_loss = train_one_epoch(kin_model, train_loader, optimizer, config, epoch)

        # Validate
        val_loss = validate_one_epoch(kin_model, val_loader, config, epoch)

        # Learning rate step
        scheduler.step()

        # Get current stage
        stage = "1 (Head)" if epoch <= stage1_epochs else "2 (Fine-tune)"
        print(f"\nEpoch {epoch} (Stage {stage}) | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = os.path.join(kin_checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': kin_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            best_path = os.path.join(kin_checkpoint_dir, "best_model_kinematics.pt")
            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': kin_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"✓ Best model saved: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"Best model: {best_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
