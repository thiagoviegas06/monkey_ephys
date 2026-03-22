#!/usr/bin/env python3
"""
Finetuning script for SBP masked reconstruction.
Masks entire channels instead of time-spans to improve robustness to channel dropout.
"""

import os
import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SBP_TCN_Transformer
from losses import kaggle_aligned_nmse_loss
from config import Config
from preprocessing import sessionData, compute_session_channel_variance

# ============================================================================
# New Dataset Class for Finetuning
# ============================================================================
class SBPChannelMaskDataset(Dataset):
    """
    Dataset that masks ENTIRE channels for the whole window.
    This mimics the channel dropout seen in Phase 2.
    """
    def __init__(self, sessions_data, is_train=True, window_size=200, config=None):
        self.sessions_data = sessions_data
        self.is_train = is_train
        self.window_size = window_size
        self.windows = []
        self.config = config

        # Precompute all non-overlapping windows
        for i, session in enumerate(sessions_data):
            N = session["N"]
            for w0 in range(0, N - self.window_size + 1, self.window_size):
                self.windows.append((i, w0))
                
        print(f"Prepared {len(self.windows)} windows for channel-mask finetuning (is_train={is_train})")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        if self.is_train:
            sess_idx, w0 = self.windows[idx]
            session = self.sessions_data[sess_idx]
            
            y_sbp = torch.from_numpy(session["sbp"][w0:w0 + self.window_size])
            kin_w = torch.from_numpy(session["kin"][w0:w0 + self.window_size])
            C = y_sbp.shape[1]

            x_sbp = y_sbp.clone()
            mask = torch.zeros_like(y_sbp, dtype=torch.bool)
            
            # Mask entire channels: 20 to 40 channels zeroed out for the WHOLE window
            num_channels = torch.randint(20, 40, (1,)).item()
            channels = torch.randperm(C)[:num_channels]
            
            x_sbp[:, channels] = 0.0
            mask[:, channels] = True
            
        else:
            # Deterministic for validation
            rng = np.random.default_rng(idx + 42) 
            sess_idx, w0 = self.windows[idx % len(self.windows)]
            session = self.sessions_data[sess_idx]
            
            y_np = session["sbp"][w0:w0 + self.window_size].copy()
            kin_np = session["kin"][w0:w0 + self.window_size].copy()
            C = y_np.shape[1]
            
            x_np = y_np.copy()
            mask_np = np.zeros_like(y_np, dtype=bool)
            
            # Use deterministic channel selection for validation
            num_channels = rng.integers(20, 40)
            channels = rng.choice(C, size=num_channels, replace=False)
            
            x_np[:, channels] = 0.0
            mask_np[:, channels] = True
            
            x_sbp = torch.from_numpy(x_np)
            y_sbp = torch.from_numpy(y_np)
            mask = torch.from_numpy(mask_np)
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

# ============================================================================
# Dataloader Helper
# ============================================================================
def get_finetune_dataloaders(config, val_split=0.2, shuffle=True, num_workers=8):
    print("Loading full sessions into RAM for channel-mask finetuning...")
    sessions, _ = sessionData(f"{config.data_path}/metadata.csv").generate_session_obj()
    
    all_sessions_data = []
    for session in tqdm(sessions, desc="Processing sessions"):
        if session.isTest(): continue
        sbp, kin, _, _ = session.load_data(config.data_path)
        if sbp is None or sbp.shape[0] < config.window_size: continue
            
        session_dict = {
            "sbp": sbp,
            "kin": kin,
            "N": sbp.shape[0],
            "channel_var": compute_session_channel_variance(sbp),
            "session_id": session.session_id,
        }
        all_sessions_data.append(session_dict)
        
    random.seed(config.seed)
    random.shuffle(all_sessions_data)
    val_size = max(1, int(len(all_sessions_data) * val_split))
    val_sessions = all_sessions_data[:val_size]
    train_sessions = all_sessions_data[val_size:]
    
    train_dataset = SBPChannelMaskDataset(train_sessions, is_train=True, window_size=config.window_size, config=config)
    val_dataset = SBPChannelMaskDataset(val_sessions, is_train=False, window_size=config.window_size, config=config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True if config.device == "cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if config.device == "cuda" else False)
    
    return train_loader, val_loader, train_dataset, val_dataset

# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch}/{config.num_epochs}")
    
    for batch in pbar:
        x_sbp = batch["x_sbp"].to(config.device)
        y_sbp = batch["y_sbp"].to(config.device)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
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
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss / total_samples:.4f}'})
    
    return total_loss / total_samples

def validate_one_epoch(model, dataloader, config, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")

    with torch.no_grad():
        for batch in pbar:
            x_sbp = batch["x_sbp"].to(config.device)
            y_sbp = batch["y_sbp"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            mask_float = batch["mask"].to(config.device).float()
            mask = batch["mask"].to(config.device)
            channel_var = batch["channel_var"].to(config.device)
            session_ids = batch["session_id"]
            
            batch_size = x_sbp.size(0)
            pred = model(x_sbp, mask_float, macro_timestamp)
            loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'avg_val_loss': f'{total_loss / total_samples:.4f}'})
    
    return total_loss / total_samples

# ============================================================================
# Main Loop
# ============================================================================
def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Finetune SBP reconstruction with channel masking.")
    parser.add_argument("--window-size", type=int, default=config.window_size, help="Window size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of finetuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for finetuning")
    args = parser.parse_args()

    config.window_size = args.window_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.checkpoint_dir = f"checkpoints_{config.window_size}"

    print("=" * 70)
    print(f"Starting Channel-Mask Finetuning")
    print(f"Window size: {config.window_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(config.checkpoint_dir, f"best_model_tcn_transformer.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Train the model first using train.py.")
    
    print(f"Loading best model from {model_path}")
    checkpoint = torch.load(model_path, map_location=config.device)
    
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Prepare dataloaders
    train_loader, val_loader, train_dataset, val_dataset = get_finetune_dataloaders(
        config, num_workers=8
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
        print(f"\tTrain NMSE: {train_loss:.6f}")
        val_loss = validate_one_epoch(model, val_loader, config, epoch)
        print(f"\tVal NMSE: {val_loss:.6f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            finetune_model_path = os.path.join(config.checkpoint_dir, f"ft_best_model_tcn_transformer_finetuned.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__,
            }, finetune_model_path)
            print(f"✓ Best finetuned model saved: {finetune_model_path}")

    print(f"\nFinetuning complete. Best val_loss: {best_val_loss:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
