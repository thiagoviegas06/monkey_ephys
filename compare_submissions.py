#!/usr/bin/env python
"""Compare two submission CSVs for differences."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare_csvs(csv1_path: str, csv2_path: str, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Compare two submission CSVs.
    
    Args:
        csv1_path: Path to first CSV
        csv2_path: Path to second CSV
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison
    
    Returns:
        True if CSVs are identical (within tolerance), False otherwise
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    print(f"CSV 1: {csv1_path}")
    print(f"  Shape: {df1.shape}")
    print(f"  Columns: {list(df1.columns)}")
    print()
    print(f"CSV 2: {csv2_path}")
    print(f"  Shape: {df2.shape}")
    print(f"  Columns: {list(df2.columns)}")
    print()
    
    all_same = True
    
    # Check columns
    if not df1.columns.equals(df2.columns):
        print("❌ COLUMNS DIFFER")
        print(f"  CSV1: {list(df1.columns)}")
        print(f"  CSV2: {list(df2.columns)}")
        all_same = False
    else:
        print("✅ Columns match")
    
    # Check shape
    if df1.shape[0] != df2.shape[0]:
        print(f"❌ ROW COUNT DIFFERS: {df1.shape[0]} vs {df2.shape[0]}")
        all_same = False
    else:
        print(f"✅ Row count matches: {df1.shape[0]}")
    
    # Check index columns (sample_id, session_id, time_bin, channel)
    key_cols = [col for col in ['sample_id', 'session_id', 'time_bin', 'channel'] if col in df1.columns]
    if key_cols:
        key_diff = ~(df1[key_cols] == df2[key_cols]).all(axis=1)
        if key_diff.any():
            print(f"❌ KEY COLUMNS DIFFER in {key_diff.sum()} rows")
            print(f"  First 5 differing rows:")
            print(df1[key_diff].head())
            all_same = False
        else:
            print(f"✅ Key columns ({', '.join(key_cols)}) match")
    
    # Check predicted values
    if 'predicted_sbp' in df1.columns and 'predicted_sbp' in df2.columns:
        pred1 = df1['predicted_sbp'].values
        pred2 = df2['predicted_sbp'].values
        
        # Use numpy's allclose for floating point comparison
        matches = np.isclose(pred1, pred2, rtol=rtol, atol=atol)
        
        if matches.all():
            print(f"✅ Predicted values match (rtol={rtol}, atol={atol})")
        else:
            n_diff = (~matches).sum()
            print(f"❌ PREDICTED VALUES DIFFER in {n_diff} rows ({100*n_diff/len(pred1):.2f}%)")
            
            diff = np.abs(pred1 - pred2)
            print(f"  Max difference: {diff[~matches].max():.10f}")
            print(f"  Mean difference: {diff[~matches].mean():.10f}")
            print(f"  Median difference: {np.median(diff[~matches]):.10f}")
            
            print(f"\n  First 10 differing rows:")
            diff_indices = np.where(~matches)[0][:10]
            for idx in diff_indices:
                print(f"    Row {idx}: {pred1[idx]:.10f} vs {pred2[idx]:.10f} (diff: {diff[idx]:.10f})")
            
            all_same = False
    
    print()
    if all_same:
        print("=" * 50)
        print("✅ CSVs are IDENTICAL")
        print("=" * 50)
        return True
    else:
        print("=" * 50)
        print("❌ CSVs are DIFFERENT")
        print("=" * 50)
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare two submission CSVs")
    parser.add_argument("csv1", type=str, help="First CSV file path")
    parser.add_argument("csv2", type=str, help="Second CSV file path")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for float comparison")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for float comparison")
    
    args = parser.parse_args()
    
    for path in [args.csv1, args.csv2]:
        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
    
    is_identical = compare_csvs(args.csv1, args.csv2, rtol=args.rtol, atol=args.atol)
    sys.exit(0 if is_identical else 1)


if __name__ == "__main__":
    main()
