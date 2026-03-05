"""
Quick comparison of different loss functions.
Trains for a few epochs with each loss to see which converges best.
"""

import sys
import subprocess
import re
from pathlib import Path

# Loss functions to test
LOSSES = [
    ("channel_nmse_loss", "Channel NMSE (no session grouping)"),
    ("variance_weighted_mse_loss", "Variance-weighted MSE (no division)"),
    ("huber_nmse_loss", "Huber NMSE (outlier-robust)"),
    ("kaggle_aligned_nmse_loss", "Kaggle-aligned NMSE (current)"),
]

# Quick test config (5 epochs each)
BASE_CONFIG = {
    "num_epochs": 5,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "split_mode": "test_proximity_session",
    "proximity_temperature_days": 60,
    "proximity_randomness": 0.3,
}


def run_with_loss(loss_name, loss_desc):
    """Run training with specified loss function."""
    print(f"\n{'='*60}")
    print(f"Testing: {loss_desc}")
    print(f"Loss function: {loss_name}")
    print(f"{'='*60}\n")
    
    # Modify train.py to use this loss
    train_file = Path("src_mae/train.py")
    content = train_file.read_text()
    
    # Replace loss function call in train_one_epoch
    original_content = content
    
    # Find the loss call pattern
    if "kaggle_aligned_nmse_loss" in content:
        content = content.replace(
            "from losses import masked_nmse_loss, kaggle_aligned_nmse_loss",
            f"from losses import masked_nmse_loss, kaggle_aligned_nmse_loss, {loss_name}"
        )
        
        # Replace in train function
        content = re.sub(
            r'loss = kaggle_aligned_nmse_loss\([^)]+\)',
            f'loss = {loss_name}(pred, y_sbp, mask, channel_var)',
            content
        )
        
        # Replace in validate function  
        content = re.sub(
            r'loss = kaggle_aligned_nmse_loss\([^)]+\)',
            f'loss = {loss_name}(pred, y_sbp, mask, channel_var)',
            content
        )
    
    # Write modified version
    train_file.write_text(content)
    
    try:
        # Run training
        result = subprocess.run(
            ["python", "src_mae/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 min max
        )
        
        # Parse results
        output = result.stdout + result.stderr
        
        # Extract final validation loss
        val_losses = re.findall(r'val_loss=([\d.]+)', output)
        best_match = re.search(r'Best model: epoch (\d+) with val_loss=([\d.]+)', output)
        
        if best_match:
            best_epoch = int(best_match.group(1))
            best_val_loss = float(best_match.group(2))
            print(f"\n✓ Best: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
            return best_val_loss
        elif val_losses:
            final_val = float(val_losses[-1])
            print(f"\n✓ Final val_loss={final_val:.6f}")
            return final_val
        else:
            print("\n✗ Could not parse validation loss")
            print(f"Output:\n{output[-500:]}")
            return float('inf')
            
    except subprocess.TimeoutExpired:
        print("\n✗ Training timed out (>10 min)")
        return float('inf')
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return float('inf')
    finally:
        # Restore original file
        train_file.write_text(original_content)


def main():
    results = []
    
    print("Loss Function Comparison")
    print(f"Config: {BASE_CONFIG}")
    
    for loss_name, loss_desc in LOSSES:
        val_loss = run_with_loss(loss_name, loss_desc)
        results.append((loss_name, loss_desc, val_loss))
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results.sort(key=lambda x: x[2])  # Sort by val_loss
    
    for rank, (loss_name, loss_desc, val_loss) in enumerate(results, 1):
        status = "✓" if val_loss < float('inf') else "✗"
        print(f"{rank}. {status} {loss_desc:45s} val_loss={val_loss:.6f}")
    
    print("\n" + "="*60)
    if results[0][2] < float('inf'):
        print(f"WINNER: {results[0][1]}")
        print(f"  → Use loss function: {results[0][0]}")
    print("="*60)


if __name__ == "__main__":
    main()
