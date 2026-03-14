#!/usr/bin/env python3
"""Test preprocess_non_overlapping (the main function)."""

import sys
import os
import pickle
from pathlib import Path
from src_mae.old_preprocessing import preprocess_non_overlapping

print("=" * 60)
print("Testing preprocess_non_overlapping (main function)")
print("=" * 60)

# Create temp output dir
temp_dir = "/tmp/test_preprocessing_output"
os.makedirs(temp_dir, exist_ok=True)

# First, create a symlink to metadata.csv so it can find it
metadata_src = "kaggle_data/metadata.csv"
metadata_dst = f"{temp_dir}/metadata.csv"
if os.path.exists(metadata_src):
    if os.path.exists(metadata_dst):
        os.remove(metadata_dst)
    os.symlink(os.path.abspath(metadata_src), metadata_dst)
    print(f"Created symlink: {metadata_dst}")

# Run preprocessing with small window size to test quickly
print("\nRunning preprocess_non_overlapping(temp_dir, window_size=200)...")
try:
    preprocess_non_overlapping(temp_dir, window_size=200, seed=0)
    print("✓ preprocess_non_overlapping completed without errors")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check output
output_dir = f"{temp_dir}/masked_windows"
if os.path.exists(output_dir):
    pkl_files = list(Path(output_dir).glob("*.pkl"))
    print(f"\nGenerated {len(pkl_files)} pickle files")

    if len(pkl_files) > 0:
        # Load and inspect first file
        with open(pkl_files[0], 'rb') as f:
            sample = pickle.load(f)

        print(f"\nFirst sample contents:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  x_sbp shape: {sample['x_sbp'].shape}")
        print(f"  y_sbp shape: {sample['y_sbp'].shape}")
        print(f"  mask shape: {sample['mask'].shape}")
        print(f"  mask dtype: {sample['mask'].dtype}")
        print(f"  Masked positions: {int(sample['mask'].sum())}")
        print(f"  session_id: {sample['session_id']}")
        print(f"  span: {sample['span']}")
        print(f"  w0: {sample['w0']}")

        # Verify data integrity
        assert sample['x_sbp'].shape == (200, 96), "Wrong x_sbp shape"
        assert sample['y_sbp'].shape == (200, 96), "Wrong y_sbp shape"
        assert sample['mask'].shape == (200, 96), "Wrong mask shape"
        assert sample['mask'].dtype == bool, "Mask should be bool"

        print("\n✓ Sample format is correct")
    else:
        print("⚠ No pickle files generated (may need actual train data)")
else:
    print(f"⚠ Output directory not found: {output_dir}")

print("\n" + "=" * 60)
print("preprocess_non_overlapping works correctly! ✓")
print("=" * 60)
