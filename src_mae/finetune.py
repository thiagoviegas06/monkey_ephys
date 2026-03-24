"""
Finetuning script: Load best checkpoint (0.71 NMSE) and finetune on 30% masked channels per session.
Uses SBP_TCN_Transformer model with macro_timestamp inputs.
"""
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SBP_TCN_Transformer
from losses import kaggle_aligned_nmse_loss
from config import Config
from dataloader import get_dataloaders

# Configuration
config = Config()

def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both state_dict and full checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    return model

def finetune_one_epoch(model, dataloader, optimizer, config, epoch, scheduler=None):
    """Finetune for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        x_sbp = batch["x_sbp"].to(config.device)              # (B, W, C)
        y_sbp = batch["y_sbp"].to(config.device)              # (B, W, C)
        mask = batch["mask"].to(config.device)                # (B, W, C)
        mask_float = mask.float()
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
        channel_var = batch["channel_var"].to(config.device)  # (B, C)
        session_ids = batch["session_id"]

        # Forward pass
        pred = model(x_sbp, mask_float, macro_timestamp)  # (B, W, C)

        # Loss: only on masked positions
        loss = kaggle_aligned_nmse_loss(
            pred, y_sbp, mask,
            channel_variance=channel_var,
            session_ids=session_ids
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x_sbp.size(0)
        total_samples += x_sbp.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss

def validate(model, dataloader, config):
    """Validation loop."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            x_sbp = batch["x_sbp"].to(config.device)
            y_sbp = batch["y_sbp"].to(config.device)
            mask = batch["mask"].to(config.device)
            mask_float = mask.float()
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            channel_var = batch["channel_var"].to(config.device)
            session_ids = batch["session_id"]

            pred = model(x_sbp, mask_float, macro_timestamp)
            loss = kaggle_aligned_nmse_loss(
                pred, y_sbp, mask,
                channel_variance=channel_var,
                session_ids=session_ids
            )

            total_loss += loss.item() * x_sbp.size(0)
            total_samples += x_sbp.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Finetune SBP model on 30% masked channels")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of finetuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='checkpoints_finetuned', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("FINETUNING: Loading checkpoint and preparing data")
    print("="*70)

    # Load dataloaders
    data_dir = config.windows_dir
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        raise FileNotFoundError(f"Preprocessed data not found at: {data_dir}")

    train_loader, val_loader = get_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        train_fraction=0.9
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Build and load model
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    model = load_checkpoint(args.checkpoint, model, config.device)

    # Optimizer: lower learning rate for finetuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Finetuning loop
    best_val_loss = float('inf')
    best_epoch = 0

    print("\n" + "="*70)
    print("STEP: Finetuning")
    print("="*70)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss = finetune_one_epoch(model, train_loader, optimizer, config, epoch, scheduler)
        val_loss = validate(model, val_loader, config)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir, 'best_model_finetuned.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"✓ Best model saved: {checkpoint_path}")

        scheduler.step(val_loss)

    print("\n" + "="*70)
    print(f"FINETUNING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Output: {os.path.join(args.output_dir, 'best_model_finetuned.pt')}")
    print("="*70)

if __name__ == '__main__':
    main()
