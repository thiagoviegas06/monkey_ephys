#!/usr/bin/env python3
"""
Finetuning script for SBP masked reconstruction.
Loads pre-generated masked windows from preprocessing function.
"""

import os
import torch
import pickle
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SBP_TCN_Transformer
from losses import kaggle_aligned_nmse_loss
from config import Config

# ============================================================================
# Preprocessed Data Dataset Class
# ============================================================================
class PreprocessedMaskedDataset(Dataset):
    """
    Loads pre-generated masked windows from pickle files.
    Each file contains: x_sbp (masked), y_sbp (full), kin, mask, channel_var, session_id, etc.
    """
    def __init__(self, data_dir, is_train=True, train_split=0.9, seed=42):
        self.data_dir = data_dir
        self.is_train = is_train
        self.seed = seed

        # Load all pkl file paths
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

        if not all_files:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")

        # Sort for reproducibility
        all_files.sort()

        # Split train/val
        random.seed(seed)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * train_split)

        if is_train:
            self.file_list = all_files[:split_idx]
        else:
            self.file_list = all_files[split_idx:]

        print(f"Loaded {len(self.file_list)} {'train' if is_train else 'val'} samples from {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        # Extract data from pickle
        x_sbp = torch.from_numpy(sample["x_sbp"]).float()      # (W, C) - masked SBP
        y_sbp = torch.from_numpy(sample["y_sbp"]).float()      # (W, C) - full SBP
        kin = torch.from_numpy(sample["kin"]).float()          # (W, 4) - kinematics
        mask = torch.from_numpy(sample["mask"]).bool()         # (W, C) - True where masked
        channel_var = torch.from_numpy(sample["channel_var"]).float()  # (C,)
        session_id = sample["session_id"]

        # Create macro_timestamp (approximate from day info if available)
        day = sample.get("day", 0.0)
        macro_timestamp = torch.tensor([day], dtype=torch.float32)

        return {
            "x_sbp": x_sbp,
            "y_sbp": y_sbp,
            "mask": mask,
            "kin": kin,
            "channel_var": channel_var,
            "macro_timestamp": macro_timestamp,
            "session_id": session_id,
        }

# ============================================================================
# Dataloader Helper
# ============================================================================
def get_dataloaders_from_preprocessed(data_dir, batch_size=32, num_workers=4, train_split=0.9, seed=42):
    """Load preprocessed data from directory."""
    train_dataset = PreprocessedMaskedDataset(data_dir, is_train=True, train_split=train_split, seed=seed)
    val_dataset = PreprocessedMaskedDataset(data_dir, is_train=False, train_split=train_split, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch}")

    for batch in pbar:
        x_sbp = batch["x_sbp"].to(config.device)
        y_sbp = batch["y_sbp"].to(config.device)
        macro_timestamp = batch["macro_timestamp"].to(config.device).float()
        mask_float = batch["mask"].to(config.device).float()
        mask = batch["mask"].to(config.device)
        channel_var = batch["channel_var"].to(config.device)
        session_ids = batch["session_id"]

        batch_size = x_sbp.size(0)
        optimizer.zero_grad()

        pred = model(x_sbp, mask_float, macro_timestamp)
        loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_samples

def validate_one_epoch(model, dataloader, config, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")

    with torch.no_grad():
        for batch in pbar:
            x_sbp = batch["x_sbp"].to(config.device)
            y_sbp = batch["y_sbp"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].to(config.device).float()
            mask_float = batch["mask"].to(config.device).float()
            mask = batch["mask"].to(config.device)
            channel_var = batch["channel_var"].to(config.device)
            session_ids = batch["session_id"]

            batch_size = x_sbp.size(0)
            pred = model(x_sbp, mask_float, macro_timestamp)
            loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    return total_loss / total_samples

# ============================================================================
# Main Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Finetune SBP model on 30% masked channels")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best checkpoint')
    parser.add_argument('--data_dir', type=str, default='kaggle_data/masked_window_p2',
                        help='Directory with preprocessed masked windows (from preprocess_channel_level_masking)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of finetuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='checkpoints_finetuned', help='Output directory')

    args = parser.parse_args()
    config = Config()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("FINETUNING: Loading preprocessed data")
    print("="*70)

    # Check that data exists
    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        raise FileNotFoundError(
            f"No preprocessed data found at: {args.data_dir}\n"
            f"Run preprocessing first:\n"
            f"  python src_mae/preprocessing.py --preprocess_type channel_level_masking --out_dir {args.data_dir}"
        )

    # Load dataloaders
    train_loader, val_loader = get_dataloaders_from_preprocessed(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Build and load model
    print("\nBuilding model...")
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    print(f"✓ Loaded checkpoint from: {args.checkpoint}")

    # Optimizer
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

        train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
        val_loss = validate_one_epoch(model, val_loader, config, epoch)

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
