#!/usr/bin/env python3
"""
Train kinematics decoder using pre-trained MAE encoder representation.

Uses transformer-based decoder architecture (alternative to LSTM in train.py).
Trains attention-based decoder on downstream kinematics prediction task.

Usage:
    python src_kin/train_transformer.py \
      --mae_checkpoint checkpoints_200/best_model_tcn_transformer.pt \
      --data_dir kaggle_data/custom_unmasked_windows_p2 \
      --output_dir checkpoints_kin_decoder \
      --num_epochs 30 \
      --batch_size 32
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_mae.model import SBP_TCN_Transformer
from src_kin.model import KinematicDecoderTransformer
from src_mae.config import Config as MAEConfig
from src_kin.config import Config as KinConfig
from src_kin.dataloader import get_dataloaders
from src_kin.losses import pearson_correlation_loss, acceleration_penalty, get_r2_components, calculate_global_r2

import torch.nn as nn


def build_mae_model(checkpoint_path, device):
    """Load pre-trained MAE model."""
    config = MAEConfig()

    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Freeze MAE
    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ Loaded MAE checkpoint from {checkpoint_path}")
    return model, config


def build_kin_decoder(d_model=64, window_size=300, device='cpu'):
    """Build kinematics decoder."""
    decoder = KinematicDecoderTransformer(
        d_model=d_model,
        window_size=window_size,
        num_channels=96,
        num_temporal_layers=2,
        num_heads=8,
        output_dim=4,
        dropout=0.1
    )
    return decoder.to(device)


def train_one_epoch(mae_model, kin_decoder, dataloader, optimizer, device):
    """Train kinematics decoder for one epoch."""
    kin_decoder.train()

    total_loss = 0.0
    total_samples = 0

    # For global R² calculation
    total_ss_res = 0.0
    total_sum_y = 0.0
    total_sum_y_sq = 0.0
    total_count = 0.0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        x_sbp = batch["sbp_masked"].to(device)  # (B, W, 96)
        kin_target = batch["kin"].to(device)  # (B, W, 4)
        mask = batch["mask"].to(device).float()  # (B, W, 96)
        macro_time = batch["macro_timestamp"].unsqueeze(-1).to(device)  # (B, 1)

        batch_size = x_sbp.size(0)
        optimizer.zero_grad()

        # Extract encoder representation from MAE (frozen)
        with torch.no_grad():
            encoder_repr, _, _ = mae_model.extract_encoder_repr(x_sbp, mask, macro_time)

        # Predict kinematics
        kin_pred = kin_decoder(encoder_repr)  # (B, W, 4)

        # Loss: MSE + Pearson correlation + acceleration penalty
        recon_loss = nn.MSELoss()(kin_pred, kin_target)
        corr_loss = pearson_correlation_loss(kin_pred, kin_target)
        accel_loss = acceleration_penalty(kin_pred)

        loss = recon_loss + 0.1 * corr_loss + 0.01 * accel_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(kin_decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Accumulate R² components
        ss_res, sum_y, sum_y_sq, count = get_r2_components(kin_pred.detach(), kin_target)
        total_ss_res += ss_res
        total_sum_y += sum_y
        total_sum_y_sq += sum_y_sq
        total_count += count

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(total_samples, 1)
    r2 = calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count)

    return avg_loss, r2


@torch.no_grad()
def validate(mae_model, kin_decoder, dataloader, device):
    """Validate kinematics decoder."""
    kin_decoder.eval()

    total_loss = 0.0
    total_samples = 0

    # For global R² calculation
    total_ss_res = 0.0
    total_sum_y = 0.0
    total_sum_y_sq = 0.0
    total_count = 0.0

    pbar = tqdm(dataloader, desc="Validation")

    for batch in pbar:
        x_sbp = batch["sbp_masked"].to(device)
        kin_target = batch["kin"].to(device)
        mask = batch["mask"].to(device).float()
        macro_time = batch["macro_timestamp"].unsqueeze(-1).to(device)  # (B, 1)

        batch_size = x_sbp.size(0)

        # Extract encoder representation
        encoder_repr, _, _ = mae_model.extract_encoder_repr(x_sbp, mask, macro_time)

        # Predict kinematics
        kin_pred = kin_decoder(encoder_repr)

        # Loss
        recon_loss = nn.MSELoss()(kin_pred, kin_target)
        corr_loss = pearson_correlation_loss(kin_pred, kin_target)
        accel_loss = acceleration_penalty(kin_pred)
        loss = recon_loss + 0.1 * corr_loss + 0.01 * accel_loss

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Accumulate R² components
        ss_res, sum_y, sum_y_sq, count = get_r2_components(kin_pred, kin_target)
        total_ss_res += ss_res
        total_sum_y += sum_y
        total_sum_y_sq += sum_y_sq
        total_count += count

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(total_samples, 1)
    r2 = calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count)

    return avg_loss, r2


def main():
    parser = argparse.ArgumentParser(description="Train kinematics decoder on MAE encoder representation")
    parser.add_argument('--mae_checkpoint', type=str, default='checkpoints_200/best_model_tcn_transformer.pt',
                        help='Path to MAE checkpoint')
    parser.add_argument('--data_dir', type=str, default='phase2_v2_kaggle_data',
                        help='Directory with phase 2 data (must contain train/ subdir with *_sbp.npy files)')
    parser.add_argument('--output_dir', type=str, default='checkpoints_kin_decoder',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early stopping patience')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("KINEMATICS DECODER TRAINING (Transformer)")
    print("="*70)

    # Load MAE (frozen)
    mae_model, mae_config = build_mae_model(args.mae_checkpoint, device)

    # Load kin config and override from CLI args
    kin_config = KinConfig()
    kin_config.batch_size = args.batch_size
    kin_config.data_path = args.data_dir

    # Build decoder using kin window size (must match the data)
    kin_decoder = build_kin_decoder(
        d_model=mae_config.d_model,
        window_size=kin_config.window_size,
        device=device
    )
    print(f"✓ Built kinematics decoder (window_size={kin_config.window_size})")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, _, _ = get_dataloaders(
        kin_config,
        val_split=0.2,
        num_workers=4
    )
    print(f"✓ Data loaded")

    # Optimizer
    optimizer = torch.optim.AdamW(kin_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss, train_r2 = train_one_epoch(mae_model, kin_decoder, train_loader, optimizer, device)
        val_loss, val_r2 = validate(mae_model, kin_decoder, val_loader, device)

        print(f"  Train Loss: {train_loss:.6f} | Train R²: {train_r2:.4f}")
        print(f"  Val Loss:   {val_loss:.6f} | Val R²:   {val_r2:.4f}")

        scheduler.step(val_loss)

        # Best model checkpoint
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(args.output_dir, 'best_kin_decoder.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': kin_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2
            }, checkpoint_path)
            print(f"✓ Best model saved: {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered (patience={args.early_stopping_patience})")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Best validation R²: {best_r2:.4f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Output: {os.path.join(args.output_dir, 'best_kin_decoder.pt')}")
    print("="*70)


if __name__ == '__main__':
    main()
