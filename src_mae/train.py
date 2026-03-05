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

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor
from losses import masked_nmse_loss, masked_mse_loss, kaggle_aligned_nmse_loss
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
                        # NOTE: When running preprocess=True for first time, use updated pipeline
                        # that saves channel_var from full session for Kaggle metric alignment
    windows_dir = "kaggle_data/masked_windows"  # Where preprocessed windows are saved
    
    # Model
    model_name = "unet"  # Options: "unet", "simple_cnn", "resnet"
    base_channels = 64   # For UNet/ResNet
    
    # Training
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 25
    val_ratio = 0.2
    split_mode = "test_proximity_session"  # options: "random_window", "random_session", "temporal_session", "test_proximity_session"
    proximity_temperature_days = 80.0
    proximity_randomness = 0.25
    early_stopping_patience = 6
    early_stopping_min_delta = 1e-4
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = "checkpoints_sbp_reconstruction"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches


def _session_id_from_sample_path(sample_path: str) -> str:
    return os.path.basename(sample_path).split("_")[0]


def _load_session_days(metadata_csv_path: str):
    day_by_session = {}
    with open(metadata_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row["session_id"]
            day_by_session[session_id] = float(row["day"])
    return day_by_session


def _load_metadata_day_split(metadata_csv_path: str):
    rows = []
    with open(metadata_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "session_id": row["session_id"],
                    "day": float(row["day"]),
                    "split": row["split"],
                }
            )
    return rows


def _nearest_test_day_distance(day, test_days):
    if len(test_days) == 0:
        return float("inf")
    return float(min(abs(day - td) for td in test_days))


def build_split_indices(
    sample_files,
    metadata_csv_path,
    val_ratio=0.2,
    split_mode="temporal_session",
    seed=42,
    proximity_temperature_days=80.0,
    proximity_randomness=0.25,
):
    session_to_indices = {}
    for idx, sample_file in enumerate(sample_files):
        sid = _session_id_from_sample_path(sample_file)
        session_to_indices.setdefault(sid, []).append(idx)

    all_sessions = sorted(session_to_indices.keys())
    if len(all_sessions) < 2:
        raise ValueError("Need at least 2 sessions to create train/validation split")

    n_val_sessions = max(1, int(round(len(all_sessions) * val_ratio)))
    n_val_sessions = min(n_val_sessions, len(all_sessions) - 1)

    if split_mode == "random_window":
        num_samples = len(sample_files)
        indices = np.arange(num_samples)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_size = int(round(num_samples * val_ratio))
        val_indices = np.sort(indices[:val_size]).tolist()
        train_indices = np.sort(indices[val_size:]).tolist()
        train_sessions = sorted({_session_id_from_sample_path(sample_files[i]) for i in train_indices})
        val_sessions = sorted({_session_id_from_sample_path(sample_files[i]) for i in val_indices})
        return train_indices, val_indices, train_sessions, val_sessions

    sessions = list(all_sessions)
    if split_mode == "temporal_session":
        day_by_session = _load_session_days(metadata_csv_path)
        sessions.sort(key=lambda sid: day_by_session.get(sid, float("inf")))
        val_sessions = set(sessions[-n_val_sessions:])
    elif split_mode == "random_session":
        rng = np.random.default_rng(seed)
        rng.shuffle(sessions)
        val_sessions = set(sessions[:n_val_sessions])
    elif split_mode == "test_proximity_session":
        rng = np.random.default_rng(seed)
        metadata_rows = _load_metadata_day_split(metadata_csv_path)
        day_by_session = {r["session_id"]: r["day"] for r in metadata_rows}
        test_days = [r["day"] for r in metadata_rows if r["split"] == "test"]

        if len(test_days) == 0:
            rng.shuffle(sessions)
            val_sessions = set(sessions[:n_val_sessions])
        else:
            dists = np.array(
                [_nearest_test_day_distance(day_by_session.get(sid, float("inf")), test_days) for sid in sessions],
                dtype=np.float64,
            )
            temp = max(1e-6, float(proximity_temperature_days))
            proximity_scores = np.exp(-dists / temp)
            if float(proximity_scores.sum()) == 0.0:
                proximity_scores = np.ones_like(proximity_scores)
            proximity_probs = proximity_scores / proximity_scores.sum()

            rand_mix = float(np.clip(proximity_randomness, 0.0, 1.0))
            uniform_probs = np.full_like(proximity_probs, 1.0 / len(proximity_probs))
            sample_probs = (1.0 - rand_mix) * proximity_probs + rand_mix * uniform_probs
            sample_probs = sample_probs / sample_probs.sum()

            chosen = rng.choice(np.arange(len(sessions)), size=n_val_sessions, replace=False, p=sample_probs)
            val_sessions = {sessions[i] for i in chosen}
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    train_sessions = [sid for sid in sessions if sid not in val_sessions]
    val_sessions = [sid for sid in sessions if sid in val_sessions]

    train_indices = []
    val_indices = []
    for sid in train_sessions:
        train_indices.extend(session_to_indices[sid])
    for sid in val_sessions:
        val_indices.extend(session_to_indices[sid])

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    return train_indices, val_indices, sorted(train_sessions), sorted(val_sessions)


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
        - channel_var: (96,) per-channel variance from full session (for Kaggle-aligned loss)
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
                - channel_var: (C,) tensor, per-channel variance from full session
                - session_id: str
        """
        # Load pickle file
        with open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)
        
        # Convert to tensors (data already in correct format from preprocessing)
        return {
            "x_sbp": torch.from_numpy(sample["x_sbp"]),           # (W, 96) float32
            "y_sbp": torch.from_numpy(sample["y_sbp"]),           # (W, 96) float32
            "mask": torch.from_numpy(sample["mask"]),             # (W, 96) bool
            "channel_var": torch.from_numpy(sample["channel_var"]), # (96,) float32
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
        mask = batch["mask"].to(config.device)     # (B, W, C)
        channel_var = batch["channel_var"].to(config.device)  # (B, C) per-channel variance from session
        session_ids = batch["session_id"]  # list of session IDs for proper grouping
        
        batch_size = x_sbp.size(0)
        
        # ===== Forward pass =====
        pred = model(x_sbp, mask)  # (B, W, C)
        
        # ===== Compute loss =====
        # Loss is computed ONLY on masked positions
        # Uses Kaggle-aligned NMSE that groups by (session, channel)
        loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
        
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
            mask = batch["mask"].to(config.device)     # (B, W, C)
            channel_var = batch["channel_var"].to(config.device)  # (B, C)
            session_ids = batch["session_id"]  # list of session IDs
            
            batch_size = x_sbp.size(0)
            
            # ===== Forward pass (no grad tracking) =====
            pred = model(x_sbp, mask)  # (B, W, C)
            
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
    
    # Load full dataset
    full_dataset = SBPDataset(windows_dir=config.windows_dir)
    
    # Train/val split
    num_samples = len(full_dataset)
    metadata_csv_path = os.path.join(config.data_path, "metadata.csv")
    train_indices, val_indices, train_sessions, val_sessions = build_split_indices(
        sample_files=full_dataset.sample_files,
        metadata_csv_path=metadata_csv_path,
        val_ratio=config.val_ratio,
        split_mode=config.split_mode,
        seed=config.seed,
        proximity_temperature_days=config.proximity_temperature_days,
        proximity_randomness=config.proximity_randomness,
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    print(f"\nDataset split ({config.split_mode}):")
    print(f"  Total windows: {num_samples}")
    print(f"  Training windows: {len(train_dataset)} ({100*len(train_dataset)/num_samples:.1f}%)")
    print(f"  Validation windows: {len(val_dataset)} ({100*len(val_dataset)/num_samples:.1f}%)")
    print(f"  Training sessions: {len(train_sessions)}")
    print(f"  Validation sessions: {len(val_sessions)}")
    print(f"\nTrain DataLoader:")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"\nValidation DataLoader:")
    print(f"  Batches: {len(val_loader)}")
    
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
            best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
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
            print(f"✓ Checkpoint saved: {checkpoint_path}")

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
