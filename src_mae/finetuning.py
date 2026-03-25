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
from preprocessing import (
    sessionData, 
    compute_session_stats,
    sample_multi_span_lengths_and_starts,
    apply_multi_span_mask_to_window
)

# ============================================================================
# New Dataset Class for Finetuning
# ============================================================================
class SBPChannelMaskDataset(Dataset):
    """
    Dataset that mixes ENTIRE channel masking (Phase 2 style) 
    with temporal span masking (Phase 1 style).
    This prevents catastrophic forgetting of temporal features.
    """
    def __init__(self, sessions_data, is_train=True, window_size=200, config=None, channel_mask_prob=0.7):
        self.sessions_data = sessions_data
        self.is_train = is_train
        self.window_size = window_size
        self.windows = []
        self.config = config
        self.channel_mask_prob = channel_mask_prob

        # Precompute all non-overlapping windows
        for i, session in enumerate(sessions_data):
            N = session["N"]
            for w0 in range(0, N - self.window_size + 1, self.window_size):
                self.windows.append((i, w0))
                
        print(f"Prepared {len(self.windows)} windows for mixed-mask finetuning (is_train={is_train})")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Use deterministic RNG for validation, dynamic for training
        if self.is_train:
            rng = np.random.default_rng(random.getrandbits(32))
            sess_idx, w0 = self.windows[idx]
        else:
            rng = np.random.default_rng(idx + 42) 
            sess_idx, w0 = self.windows[idx % len(self.windows)]
            
        session = self.sessions_data[sess_idx]
        
        y_np = session["sbp"][w0:w0 + self.window_size].copy()
        kin_np = session["kin"][w0:w0 + self.window_size].copy()
        C = y_np.shape[1]
        
        x_np = y_np.copy()
        mask_np = np.zeros_like(y_np, dtype=bool)
        
        # Determine masking strategy
        strategy_roll = rng.random()
        
        if strategy_roll < self.channel_mask_prob:
            # --- CHANNEL MASKING (Phase 2 Style) ---
            num_channels = rng.integers(20, 40)
            channels = rng.choice(C, size=num_channels, replace=False)
            x_np[:, channels] = 0.0
            mask_np[:, channels] = True
            mask_type = 0 # 0 for Channel
        else:
            # --- SPAN MASKING (Phase 1 Style) ---
            num_spans = rng.integers(2, 4)
            spans = sample_multi_span_lengths_and_starts(rng, self.window_size, num_spans=num_spans, min_gap=10)
            x_np, mask_np = apply_multi_span_mask_to_window(y_np, spans, num_spans=num_spans, rng=rng)
            mask_type = 1 # 1 for Span

        return {
            "x_sbp": torch.from_numpy(x_np).float(),
            "y_sbp": torch.from_numpy(y_np).float(),
            "mask": torch.from_numpy(mask_np).float(),
            "kin": torch.from_numpy(kin_np).float(),
            "channel_mean": torch.from_numpy(session["channel_mean"]).float(),
            "channel_var": torch.from_numpy(session["channel_var"]).float(),
            "session_id": session["session_id"],
            "macro_timestamp": w0,
            "mask_type": mask_type
        }

# ============================================================================
# Dataloader Helper
# ============================================================================
def get_finetune_dataloaders(config, val_split=0.2, shuffle=True, num_workers=8):
    print("Loading full sessions into RAM for mixed-mask finetuning...")
    sessions, _ = sessionData(f"{config.data_path}/metadata.csv").generate_session_obj()
    
    all_sessions_data = []
    for session in tqdm(sessions, desc="Processing sessions"):
        if session.isTest(): continue
        sbp, kin, _, _ = session.load_data(config.data_path)
        if sbp is None or sbp.shape[0] < config.window_size: continue
            
        session_mean, session_var = compute_session_stats(sbp)
        session_dict = {
            "sbp": sbp,
            "kin": kin,
            "N": sbp.shape[0],
            "channel_mean": session_mean,
            "channel_var": session_var,
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
    
    # Track losses by type
    type_losses = {0: [], 1: []} # 0: Channel, 1: Span
    
    pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch}/{config.num_epochs}")
    
    for batch in pbar:
        x_sbp = batch["x_sbp"].to(config.device)
        y_sbp = batch["y_sbp"].to(config.device)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
        mask_float = batch["mask"].to(config.device).float()
        mask = batch["mask"].to(config.device)
        channel_mean = batch["channel_mean"].to(config.device)
        channel_var = batch["channel_var"].to(config.device)
        session_ids = batch["session_id"]
        mask_types = batch["mask_type"]
        
        batch_size = x_sbp.size(0)
        optimizer.zero_grad()
        
        pred = model(x_sbp, mask_float, macro_timestamp, channel_mean, channel_var)
        loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Track by type (unweighted for logging)
        with torch.no_grad():
            for i, m_type in enumerate(mask_types.tolist()):
                type_losses[m_type].append(loss.item())

        pbar.set_postfix({
            'L': f'{loss.item():.3f}', 
            'avg': f'{total_loss / total_samples:.3f}',
            'ch': f'{np.mean(type_losses[0]):.3f}' if type_losses[0] else 'N/A',
            'sp': f'{np.mean(type_losses[1]):.3f}' if type_losses[1] else 'N/A'
        })
    
    return total_loss / total_samples

def validate_one_epoch(model, dataloader, config, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    type_losses = {0: [], 1: []}

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")

    with torch.no_grad():
        for batch in pbar:
            x_sbp = batch["x_sbp"].to(config.device)
            y_sbp = batch["y_sbp"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            mask_float = batch["mask"].to(config.device).float()
            mask = batch["mask"].to(config.device)
            channel_mean = batch["channel_mean"].to(config.device)
            channel_var = batch["channel_var"].to(config.device)
            session_ids = batch["session_id"]
            mask_types = batch["mask_type"]
            
            batch_size = x_sbp.size(0)
            pred = model(x_sbp, mask_float, macro_timestamp, channel_mean, channel_var)
            loss = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for i, m_type in enumerate(mask_types.tolist()):
                type_losses[m_type].append(loss.item())

            pbar.set_postfix({
                'val_L': f'{loss.item():.3f}', 
                'ch': f'{np.mean(type_losses[0]):.3f}' if type_losses[0] else 'N/A',
                'sp': f'{np.mean(type_losses[1]):.3f}' if type_losses[1] else 'N/A'
            })
    
    avg_val = total_loss / total_samples
    print(f"\t[Breakdown] Channel NMSE: {np.mean(type_losses[0]):.4f} | Span NMSE: {np.mean(type_losses[1]):.4f}")
    return avg_val

# ============================================================================
# Main Loop
# ============================================================================
def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Finetune SBP reconstruction with mixed masking.")
    parser.add_argument("--window-size", type=int, default=config.window_size, help="Window size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of finetuning epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for finetuning")
    args = parser.parse_args()

    config.window_size = args.window_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.checkpoint_dir = f"checkpoints_{config.window_size}"

    print("=" * 70)
    print(f"Starting Mixed-Mask Finetuning (70% Channel, 30% Span)")
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
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
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
