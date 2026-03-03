#!/usr/bin/env python3
"""Visualize masked windows from preprocess_non_overlapping."""

import argparse
from src_mae.preprocessing import generate_one_masked_window, visualize_masked_window


def main():
    parser = argparse.ArgumentParser(description="Visualize a masked window from non-overlapping preprocessing")
    parser.add_argument("--data-path", "--data_path", type=str, default="kaggle_data",
                       help="Path to data directory")
    parser.add_argument("--session-id", "--session_id", type=str, default=None,
                       help="Train session ID (e.g. S008). If omitted, uses first train session.")
    parser.add_argument("--window-size", "--window_size", type=int, default=200,
                       help="Window size")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--save-path", "--save_path", type=str, default=None,
                       help="Save figure to this path instead of displaying")
    
    args = parser.parse_args()
    
    print(f"Generating masked window from {args.data_path}...")
    sample = generate_one_masked_window(
        data_path=args.data_path,
        window_size=args.window_size,
        seed=args.seed,
        session_id=args.session_id,
    )
    
    print(f"Generated sample from session {sample['session_id']}")
    print(f"  Masked span: [{sample['span'][0]}, {sample['span'][1]})")
    print(f"  Masked channels: {int(sample['mask'].sum())}")
    print(f"\nVisualizing...")
    
    visualize_masked_window(sample, save_path=args.save_path)


if __name__ == "__main__":
    main()
