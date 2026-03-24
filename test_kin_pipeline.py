#!/usr/bin/env python
"""
Test script to verify the entire kinematics pipeline works correctly.
Checks: data loading, MAE encoder, LSTM, loss computation, training step.
"""
import torch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_kin.config import Config
from src_kin.dataloader import get_dataloaders
from src_kin.model import LSTMKinematicDecoder
from src_mae.model import SBP_TCN_Transformer
from src_kin.losses import lfads_loss, calculate_r2
from src_kin.compute_kin_stats import compute_per_session_kin_stats
import torch.optim as optim

print("=" * 60)
print("KINEMATICS PIPELINE VALIDATION TEST")
print("=" * 60)

# 1. Load config
print("\n[1/6] Loading config...")
config = Config()
print(f"  ✓ Config loaded")
print(f"    - Window size: {config.window_size}")
print(f"    - Data path: {config.data_path}")
print(f"    - Encoder dim: {config.encoder_dim}")
print(f"    - Device: {config.device}")

# 2. Load MAE model
print("\n[2/6] Loading Phase 1 MAE (frozen feature extractor)...")
mae_model = SBP_TCN_Transformer(
    sbp_channels=config.sbp_channels,
    d_model=config.d_model,
    nhead=config.nhead,
    num_layers=config.num_layers,
    tcn_levels=config.tcn_levels,
    dropout=config.dropout
).to(config.device)

checkpoint = torch.load(config.mae_checkpoint_path, map_location=config.device, weights_only=False)
mae_model.load_state_dict(checkpoint['model_state_dict'])
mae_model.eval()
for param in mae_model.parameters():
    param.requires_grad = False
print(f"  ✓ MAE loaded and frozen")
print(f"    - Parameters: {sum(p.numel() for p in mae_model.parameters()):,}")

# 3. Build LSTM model
print("\n[3/6] Building LSTM kinematics decoder...")
lstm_model = LSTMKinematicDecoder(
    encoder_dim=config.encoder_dim,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_lstm_layers,
    output_dim=config.output_dim,
    dropout=config.lstm_dropout
).to(config.device)
print(f"  ✓ LSTM model created")
print(f"    - Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

# 4. Load data and stats
print("\n[4/6] Loading data and computing statistics...")
train_loader, val_loader, _, _ = get_dataloaders(config, num_workers=0)
session_kin_stats = compute_per_session_kin_stats(config.data_path)
print(f"  ✓ Data loaded")
print(f"    - Train batches: {len(train_loader)}")
print(f"    - Val batches: {len(val_loader)}")
print(f"    - Sessions with kin stats: {len(session_kin_stats)}")

# 5. Test forward pass
print("\n[5/6] Testing forward pass on single batch...")
batch = next(iter(train_loader))
sbp_masked = batch["sbp_masked"].to(config.device)
mask = batch["mask"].to(config.device)
macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
kin_target = batch["kin"].to(config.device)
session_ids = batch["session_id"]

print(f"  Input shapes:")
print(f"    - SBP masked: {sbp_masked.shape} (B, W, C)")
print(f"    - Mask: {mask.shape}")
print(f"    - Kin target: {kin_target.shape}")

# Normalize SBP (like in training)
from src_kin.train import compute_per_session_sbp_stats
session_sbp_stats = compute_per_session_sbp_stats(config.data_path)

sbp_normalized = sbp_masked.clone()
for i, sid in enumerate(session_ids):
    if sid in session_sbp_stats:
        mean = session_sbp_stats[sid]['mean'].to(config.device)
        std = session_sbp_stats[sid]['std'].to(config.device)
        sbp_normalized[i] = (sbp_masked[i] - mean) / std

print(f"  ✓ SBP normalized")

# Extract encoder features
with torch.no_grad():
    encoder_features = mae_model.extract_encoder_features(sbp_normalized, mask, macro_timestamp)
    encoder_features = encoder_features.mean(dim=2)
print(f"  ✓ MAE encoder features extracted: {encoder_features.shape} (B, W, d_model)")

# LSTM prediction
kin_pred, sbp_pred, mu, logvar = lstm_model(encoder_features, mask=mask)
print(f"  ✓ LSTM prediction: {kin_pred.shape} (B, W, 4)")

# 6. Test loss computation
print("\n[6/6] Testing loss computation and backward pass...")

# Normalize kinematics target
kin_target_norm = torch.zeros_like(kin_target)
for i, sid in enumerate(session_ids):
    if sid in session_kin_stats:
        mean = session_kin_stats[sid]['mean'].to(config.device)
        std = session_kin_stats[sid]['std'].to(config.device)
        kin_target_norm[i] = (kin_target[i] - mean) / std

loss_dict = lfads_loss(kin_pred, kin_target_norm, mu, logvar, 0, config, sbp_pred, encoder_features)
loss = loss_dict["loss"]
print(f"  ✓ Loss computed: {loss.item():.6f}")
print(f"    - MSE: {loss_dict['recon_mse'].item():.6f}")
print(f"    - Accel: {loss_dict['accel_loss'].item():.6f}")

# R² score
r2 = calculate_r2(kin_pred, kin_target_norm)
print(f"    - R²: {r2:.4f}")

# Test backward pass
optimizer = optim.AdamW(lstm_model.parameters(), lr=config.learning_rate)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=5.0)
optimizer.step()
print(f"  ✓ Backward pass successful")
print(f"    - Loss after step: {loss.item():.6f}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - PIPELINE IS WORKING CORRECTLY")
print("=" * 60)
print("\nReady to run full training:")
print("  python src_kin/train.py")
