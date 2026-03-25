#!/usr/bin/env python3
"""
Example: Using MAE encoder representation for kinematics prediction.

Workflow:
1. Load pre-trained MAE model
2. Extract transformer encoder representation (masking-agnostic)
3. Pass to kinematics decoder
4. Train kinematics decoder on downstream task
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_mae.model import SBP_TCN_Transformer
from src_kin.model import KinematicDecoderTransformer
from src_mae.config import Config as MAEConfig
from src_kin.config import Config as KinConfig

# ============================================================================
# 1. LOAD PRE-TRAINED MAE
# ============================================================================

mae_config = MAEConfig()

mae_model = SBP_TCN_Transformer(
    sbp_channels=mae_config.sbp_channels,
    d_model=mae_config.d_model,
    nhead=mae_config.nhead,
    num_layers=mae_config.num_layers,
    tcn_levels=mae_config.tcn_levels,
    dropout=mae_config.dropout
)

# Load checkpoint
checkpoint_path = "checkpoints_200/best_model_tcn_transformer.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=mae_config.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        mae_model.load_state_dict(checkpoint)
    print(f"✓ Loaded MAE checkpoint from {checkpoint_path}")
else:
    print(f"⚠️  Checkpoint not found at {checkpoint_path}")

mae_model.to(mae_config.device)
mae_model.eval()  # Freeze MAE for inference

# ============================================================================
# 2. BUILD KINEMATICS DECODER
# ============================================================================

kin_decoder = KinematicDecoderTransformer(
    d_model=mae_config.d_model,  # Must match MAE's d_model
    window_size=mae_config.window_size,  # 200
    hidden_dim=128,
    num_lstm_layers=2,
    output_dim=4,  # 4 kinematics channels
    dropout=0.1
)
kin_decoder.to(mae_config.device)

print(f"✓ Built kinematic decoder")
print(f"  MAE output: (B*W, C, d_model) = (B*200, 96, {mae_config.d_model})")
print(f"  Kinematic decoder output: (B, W, 4) = (B, 200, 4)")

# ============================================================================
# 3. EXAMPLE FORWARD PASS
# ============================================================================

batch_size = 4
window_size = 200
sbp_channels = 96

# Create dummy input
x_sbp = torch.randn(batch_size, window_size, sbp_channels).to(mae_config.device)
mask = torch.zeros(batch_size, window_size, sbp_channels).to(mae_config.device)
macro_time = torch.randn(batch_size, 1).to(mae_config.device)

# Randomly mask 30% of channels
mask_ratio = 0.3
num_masked_channels = int(sbp_channels * mask_ratio)
masked_channels = torch.randperm(sbp_channels)[:num_masked_channels]
x_sbp_masked = x_sbp.clone()
x_sbp_masked[:, :, masked_channels] = 0.0
mask[:, :, masked_channels] = 1.0

print(f"\n--- Example Forward Pass ---")
print(f"Input shape: {x_sbp.shape}")
print(f"Masked {mask_ratio*100:.0f}% of channels ({num_masked_channels}/{sbp_channels})")

# Extract encoder representation from MAE
with torch.no_grad():
    encoder_repr, sbp_mean, sbp_std = mae_model.extract_encoder_repr(x_sbp_masked, mask, macro_time)

print(f"Encoder representation shape: {encoder_repr.shape}")  # (B*W, C, d_model)
print(f"  Flattened: ({batch_size}*{window_size}, {sbp_channels}, {mae_config.d_model})")

# Predict kinematics
kin_pred = kin_decoder(encoder_repr)

print(f"Kinematics prediction shape: {kin_pred.shape}")  # (B, W, 4)
print(f"  Batch={batch_size}, Time={window_size}, Channels=4")

print(f"\n✓ End-to-end pipeline working!")

# ============================================================================
# 4. KEY INSIGHT
# ============================================================================

print(f"""
KEY DESIGN POINTS:
=================
1. MAE encoder learns masking-agnostic representation
   - Works regardless of masking level (30%, 50%, etc.)
   - Encodes visible channel information
   - Rich, learned features

2. Kinematics decoder doesn't need session conditioning
   - If encoder representation is good enough
   - No overfitting to session-specific artifacts
   - Clean separation: encoder (general) → decoder (task-specific)

3. Validation approach:
   - PCA check: encoder repr should be masking-agnostic
   - Kinematics R²: final metric of representation quality
   - Generalization: test on held-out sessions
""")
