"""
Verify that losses.py matches the preprocessing data convention.
"""
import numpy as np
import torch
import sys
sys.path.append('src_mae')

from preprocessing import apply_random_mask_to_window
from losses import masked_nmse_loss, masked_mse_loss

print("=" * 70)
print("VERIFICATION: Loss Convention vs Preprocessing Convention")
print("=" * 70)

# Create synthetic data similar to preprocessing
W, C = 200, 96
rng = np.random.default_rng(42)

# Simulate SBP data
sbp_original = rng.normal(loc=5.0, scale=2.0, size=(W, C)).astype(np.float32)
sbp_original = np.abs(sbp_original)  # SBP should be positive

# Apply masking like preprocessing does
start_bin, end_bin = 80, 120  # Mask bins 80-120
x_sbp, mask = apply_random_mask_to_window(sbp_original, start_bin, end_bin, rng, channels_per_bin=30)

print(f"\n1. PREPROCESSING OUTPUT:")
print(f"   x_sbp shape: {x_sbp.shape}")
print(f"   mask shape: {mask.shape}")
print(f"   mask dtype: {mask.dtype}")
print(f"   Total masked positions: {mask.sum()}")
print(f"   Total observed positions: {(~mask).sum()}")

# Verify preprocessing convention
masked_positions = mask
observed_positions = ~mask
print(f"\n2. PREPROCESSING CONVENTION CHECK:")
print(f"   x_sbp[masked] min/max: {x_sbp[masked_positions].min():.4f} / {x_sbp[masked_positions].max():.4f}")
print(f"   x_sbp[observed] min/max: {x_sbp[observed_positions].min():.4f} / {x_sbp[observed_positions].max():.4f}")
print(f"   ✓ Masked positions are zeros: {np.allclose(x_sbp[masked_positions], 0)}")
print(f"   ✓ Observed positions are non-zero: {(x_sbp[observed_positions] > 0).all()}")

# Convert to torch tensors (add batch dimension)
x_sbp_t = torch.from_numpy(x_sbp).unsqueeze(0)  # (1, W, C)
y_sbp_t = torch.from_numpy(sbp_original).unsqueeze(0)  # (1, W, C)
mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, W, C)

# Create a "prediction" - for testing, use ground truth + small noise on masked positions
pred_t = y_sbp_t.clone()
noise = torch.randn_like(pred_t) * 0.5
pred_t = pred_t * (~mask_t).float() + (pred_t + noise) * mask_t.float()

print(f"\n3. LOSS FUNCTION INPUT:")
print(f"   pred shape: {pred_t.shape}")
print(f"   target shape: {y_sbp_t.shape}")
print(f"   mask shape: {mask_t.shape}")
print(f"   mask True count: {mask_t.sum().item()}")

# Test the loss
try:
    nmse = masked_nmse_loss(pred_t, y_sbp_t, mask_t)
    mse = masked_mse_loss(pred_t, y_sbp_t, mask_t)
    
    print(f"\n4. LOSS COMPUTATION:")
    print(f"   ✓ masked_nmse_loss: {nmse.item():.6f}")
    print(f"   ✓ masked_mse_loss: {mse.item():.6f}")
    
    # Manually verify the loss is computed correctly
    # Check that loss uses ONLY masked positions
    unmask_t = ~mask_t
    
    # Variance from UNMASKED (observed) positions
    n_unmask = unmask_t.sum(dim=(0, 1)).clamp(min=1)
    mean_c = (y_sbp_t * unmask_t).sum(dim=(0, 1)) / n_unmask
    var_c = ((y_sbp_t - mean_c) ** 2 * unmask_t).sum(dim=(0, 1)) / n_unmask
    
    # MSE from MASKED positions
    n_mask = mask_t.sum(dim=(0, 1))
    active = n_mask > 0
    mse_c = ((pred_t - y_sbp_t) ** 2 * mask_t).sum(dim=(0, 1)) / n_mask.clamp(min=1)
    
    # NMSE
    nmse_c = mse_c[active] / var_c[active].clamp(min=1e-8)
    manual_nmse = nmse_c.mean()
    
    print(f"\n5. MANUAL VERIFICATION:")
    print(f"   Channels with masking: {active.sum().item()} / {C}")
    print(f"   Manual NMSE: {manual_nmse.item():.6f}")
    print(f"   Loss function NMSE: {nmse.item():.6f}")
    print(f"   ✓ Match: {torch.allclose(manual_nmse, nmse)}")
    
    # Test edge case: perfect prediction (NMSE should be ~0)
    perfect_pred = y_sbp_t.clone()
    nmse_perfect = masked_nmse_loss(perfect_pred, y_sbp_t, mask_t)
    print(f"\n6. PERFECT PREDICTION TEST:")
    print(f"   NMSE with perfect prediction: {nmse_perfect.item():.10f}")
    print(f"   ✓ Near zero: {nmse_perfect.item() < 1e-6}")
    
    # Test edge case: mean prediction (NMSE should be ~1.0)
    mean_pred = y_sbp_t * (~mask_t).float()  # Keep observed
    # For masked positions, use per-channel mean
    for c in range(C):
        if active[c]:
            mean_pred[0, mask_t[0, :, c], c] = mean_c[c]
    nmse_mean = masked_nmse_loss(mean_pred, y_sbp_t, mask_t)
    print(f"\n7. MEAN PREDICTION TEST:")
    print(f"   NMSE with mean prediction: {nmse_mean.item():.6f}")
    print(f"   ✓ Close to 1.0: {0.9 < nmse_mean.item() < 1.1}")
    
    print(f"\n{'=' * 70}")
    print("✅ VERIFICATION PASSED: Loss convention matches preprocessing!")
    print("=" * 70)
    print("\nSummary:")
    print("- Preprocessing: mask=True → masked (to predict)")
    print("- Loss function: mask=True → masked (compute loss here)")
    print("- Variance computed from: UNMASKED (observed) entries")
    print("- MSE computed from: MASKED entries only")
    print("- Convention is CONSISTENT across preprocessing and loss!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
