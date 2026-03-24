#!/usr/bin/env python
"""
Preprocess Phase 1 data with Phase 2-style channel-level masking.
Masks entire 30% of channels consistently across all time bins.
Outputs to 'masked_window_p2' folder.
"""
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_mae.preprocessing import preprocess_channel_level_masking

if __name__ == "__main__":
    print("=" * 70)
    print("PREPROCESSING: Phase 2-STYLE CHANNEL-LEVEL MASKING")
    print("=" * 70)
    print("This will:")
    print("  1. Mask entire 30% of channels (deterministic per session)")
    print("  2. Save to 'kaggle_data/masked_window_p2'")
    print("  3. Create z-score normalized windows")
    print()

    preprocess_channel_level_masking(
        data_path="kaggle_data",
        window_size=200,
        step_size=50,
        mask_ratio=0.3
    )

    print()
    print("=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Train Phase 1 with: python src_mae/train.py")
    print("  2. Checkpoints saved to: checkpoints_new_p2/")
    print()
