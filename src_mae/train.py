"""
Training script for SBP masked reconstruction using TCN + Transformer.

Uses preprocess_non_overlapping() to generate windows, then trains on them.
Clear 2-step process: preprocess -> train
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

# Create configuration instance as global variable for easy access in functions
config = Config()

if config.preprocess_type == "non_overlapping":
    from dataloader import get_dataloaders

elif config.preprocess_type == "overlapping_dynamic":
    from dataloader import get_dataloaders_dynamic
else:
    raise ValueError(f"Unknown preprocessing option: {config.preprocess_type}")

# ============================================================================
# Model Builder
# ============================================================================
def build_model(config):
    """Build SBP_TCN_Transformer based on config."""
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    print(f"Built Interleaved Axial TCN + Transformer (d_model={config.d_model}, layers={config.num_axial_layers})")
    return model.to(config.device)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, config, epoch, scheduler=None):
    """
    Train for one epoch.
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
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
        
        mask_float = batch["mask"].to(config.device).float()
        mask = batch["mask"].to(config.device)     # (B, W, C)
        channel_mean = batch["channel_mean"].to(config.device)
        channel_var = batch["channel_var"].to(config.device)  # (B, C) per-channel variance
        session_ids = batch["session_id"]
        
        batch_size = x_sbp.size(0)
        optimizer.zero_grad()
        
        # ===== Forward pass =====
        pred = model(x_sbp, mask_float, macro_timestamp, channel_mean, channel_var)  # (B, W, C)
        
        # ===== Compute loss =====
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
    Validation for one epoch.
    """
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x_sbp = batch["x_sbp"].to(config.device)  # (B, W, C)
            y_sbp = batch["y_sbp"].to(config.device)  # (B, W, C)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()  # (B, 1)
            
            mask_float = batch["mask"].to(config.device).float()
            mask = batch["mask"].to(config.device)     # (B, W, C)
            channel_mean = batch["channel_mean"].to(config.device)
            channel_var = batch["channel_var"].to(config.device)
            session_ids = batch["session_id"]
            
            batch_size = x_sbp.size(0)
            
            # ===== Forward pass =====
            pred = model(x_sbp, mask_float, macro_timestamp, channel_mean, channel_var)  # (B, W, C)
            
            # ===== Compute loss =====
            loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
            
            if loss.item() > 20.0:
                print(f"\n[WARNING] High loss ({loss.item():.2f}) in batch. Sessions: {list(set(session_ids))}")
                # Optional: find which session in the batch is the worst
                for sid in set(session_ids):
                    s_idx = [i for i, s in enumerate(session_ids) if s == sid]
                    s_loss = kaggle_aligned_nmse_loss(pred[s_idx], y_sbp[s_idx], mask[s_idx], channel_var[s_idx], [sid]*len(s_idx))
                    if s_loss.item() > 10.0:
                         print(f"  - Session {sid} loss: {s_loss.item():.2f}")
            
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


def train_val_loader():
    global config

    if config.preprocess_type == "non_overlapping":
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
            config=config,
            batch_size=config.batch_size,
            val_split=0.2,
            shuffle=True,
            num_workers=12,  # Parallel data loading
            pin_memory=True if config.device == "cuda" else False
        )

    elif config.preprocess_type == "overlapping_dynamic":
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders_dynamic(
            config, 
            batch_size=config.batch_size,
            val_split=0.2,
            num_workers=12,  # Parallel data loading
        )

    else:
        raise ValueError(f"Unknown preprocess_type: {config.preprocess_type}")
    
    return train_loader, val_loader, train_dataset, val_dataset


def main():
    """Main training function."""
    global config
    
    args = parse_args()
    config.window_size = args.window_size
    config.windows_dir = f"kaggle_data/masked_windows_{config.window_size}"
    config.checkpoint_dir = f"checkpoints_{config.window_size}"

    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Window size: {config.window_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("\nSTEP 2: Loading Data")
    print("\n" + "=" * 70)
    train_loader, val_loader, train_dataset, val_dataset = train_val_loader()
    print(f"\nDataLoader created:")
    print(f"  Train Batches per epoch: {len(train_loader)}")
    print(f"  Train Samples per epoch: {len(train_dataset)}")
    print(f"  Validation Batches per epoch: {len(val_loader)}")
    print(f"  Validation Samples per epoch: {len(val_dataset)}")


    print("\n" + "=" * 70)
    print("\nSTEP 3: Building Model")
    print("\n" + "=" * 70)
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    print("\n" + "=" * 70)
    print("\nSTEP 4: Training")
    print("\n" + "=" * 70)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch, scheduler=scheduler)
        print(f"\tTrain NMSE Loss: {train_loss:.6f}")
        val_loss = validate_one_epoch(model, val_loader, config, epoch)
        print(f"\tVal NMSE Loss: {val_loss:.6f}")

        scheduler.step()
        
        if val_loss < (best_val_loss - config.early_stopping_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_path = os.path.join(config.checkpoint_dir, f"best_model_{config.model_name}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__,
            }, best_model_path)
            print(f"✓ Best model saved: {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % config.save_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

    print("\n" + "=" * 70)
    print(f"\nTraining Complete! Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Best model: epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    print("\n" + "=" * 70)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SBP reconstruction.")
    parser.add_argument("--window-size", type=int, default=200, help="Window size")
    return parser.parse_args()

if __name__ == "__main__":
    main()
