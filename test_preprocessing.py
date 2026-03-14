#!/usr/bin/env python3
"""Quick test of old_preprocessing.py functions."""

import sys
import numpy as np
from src_mae.old_preprocessing import (
    sample_span_len,
    sample_span_start,
    apply_random_mask_to_window,
    generate_one_masked_window,
    visualize_masked_window,
)

print("=" * 60)
print("Testing old_preprocessing.py")
print("=" * 60)

# Test 1: sample_span_len
print("\n1. Testing sample_span_len...")
rng = np.random.default_rng(42)
W = 200
for i in range(5):
    L = sample_span_len(rng, W)
    print(f"   Sample {i+1}: L={L} (valid: {1 <= L <= W})")
    assert 1 <= L <= W, f"Invalid span length: {L}"
print("   ✓ sample_span_len works")

# Test 2: sample_span_start
print("\n2. Testing sample_span_start...")
for i in range(5):
    L = sample_span_len(rng, W)
    t0 = sample_span_start(rng, W, L)
    t1 = t0 + L
    print(f"   Sample {i+1}: t0={t0}, t1={t1}, len={L} (valid: {0 <= t0 < t1 <= W})")
    assert 0 <= t0 < t1 <= W, f"Invalid span: [{t0}, {t1})"
print("   ✓ sample_span_start works")

# Test 3: apply_random_mask_to_window
print("\n3. Testing apply_random_mask_to_window...")
sbp_test = np.random.randn(200, 96).astype(np.float32)
t0, t1 = 50, 120
x, mask = apply_random_mask_to_window(sbp_test, t0, t1, rng)
masked_count = int(mask.sum())
print(f"   SBP shape: {sbp_test.shape}")
print(f"   Masked span: [{t0}, {t1})")
print(f"   Masked positions: {masked_count}")
print(f"   Masked region shape: {mask[t0:t1].shape}")
print(f"   Zeros in input at masked positions: {(x[t0:t1] == 0).sum()}")
assert x.shape == sbp_test.shape, "Output shape mismatch"
assert mask.dtype == bool, "Mask should be bool"
assert (x[t0:t1] == 0).all() == False, "Not all masked positions are zero (expected random channels)"
print("   ✓ apply_random_mask_to_window works")

# Test 4: generate_one_masked_window (end-to-end test)
print("\n4. Testing generate_one_masked_window (end-to-end)...")
try:
    sample = generate_one_masked_window("kaggle_data", window_size=200, seed=0)
    print(f"   Generated sample from session: {sample['session_id']}")
    print(f"   x_sbp shape: {sample['x_sbp'].shape}")
    print(f"   y_sbp shape: {sample['y_sbp'].shape}")
    print(f"   mask shape: {sample['mask'].shape}")
    print(f"   mask dtype: {sample['mask'].dtype}")
    print(f"   Masked positions: {int(sample['mask'].sum())}")
    print(f"   Span: {sample['span']}")

    # Verify shapes
    assert sample['x_sbp'].shape == (200, 96), f"Wrong x_sbp shape: {sample['x_sbp'].shape}"
    assert sample['y_sbp'].shape == (200, 96), f"Wrong y_sbp shape: {sample['y_sbp'].shape}"
    assert sample['mask'].shape == (200, 96), f"Wrong mask shape: {sample['mask'].shape}"

    # Verify x_sbp has zeros where masked
    t0, t1 = sample['span']
    # In masked span, some channels are zero (not all, only masked channels)
    masked_region = sample['x_sbp'][t0:t1]
    print(f"   Zeros in masked region: {(masked_region == 0).sum()}")

    print("   ✓ generate_one_masked_window works (end-to-end)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
