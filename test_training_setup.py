"""
Test script to verify training setup works before running full training.
"""
import torch
import sys
sys.path.append('src_mae')

from train import SBPDataset, build_model, Config
from torch.utils.data import DataLoader

def test_dataset():
    """Test that dataset loads correctly."""
    print("=" * 70)
    print("Testing Dataset")
    print("=" * 70)
    
    config = Config()
    
    # Create dataset
    dataset = SBPDataset(
        data_path=config.data_path,
        window_size=config.window_size,
        split="train",
        seed=config.seed
    )
    
    print(f"✓ Dataset created: {len(dataset)} windows")
    
    # Get one sample
    sample = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  x_sbp shape: {sample['x_sbp'].shape}")
    print(f"  y_sbp shape: {sample['y_sbp'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  session_id: {sample['session_id']}")
    
    # Verify masking convention
    x_sbp = sample['x_sbp']
    y_sbp = sample['y_sbp']
    mask = sample['mask']
    
    n_masked = mask.sum().item()
    n_total = mask.numel()
    
    # Check masked positions are zero in x_sbp
    masked_vals = x_sbp[mask]
    observed_vals = x_sbp[~mask]
    
    print(f"\nMasking statistics:")
    print(f"  Total positions: {n_total}")
    print(f"  Masked: {n_masked} ({100*n_masked/n_total:.2f}%)")
    print(f"  Observed: {n_total - n_masked} ({100*(n_total-n_masked)/n_total:.2f}%)")
    print(f"  x_sbp[masked] all zeros: {torch.allclose(masked_vals, torch.zeros_like(masked_vals))}")
    print(f"  x_sbp[observed] range: [{observed_vals.min():.4f}, {observed_vals.max():.4f}]")
    print(f"  y_sbp range: [{y_sbp.min():.4f}, {y_sbp.max():.4f}]")
    
    assert sample['x_sbp'].shape == (config.window_size, 96)
    assert sample['y_sbp'].shape == (config.window_size, 96)
    assert sample['mask'].shape == (config.window_size, 96)
    
    print("\n✓ Dataset test passed!")
    return dataset


def test_dataloader(dataset):
    """Test that dataloader batches correctly."""
    print("\n" + "=" * 70)
    print("Testing DataLoader")
    print("=" * 70)
    
    config = Config()
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print(f"✓ DataLoader created: {len(loader)} batches")
    
    # Get one batch
    batch = next(iter(loader))
    
    print(f"\nBatch 0:")
    print(f"  x_sbp shape: {batch['x_sbp'].shape}")
    print(f"  y_sbp shape: {batch['y_sbp'].shape}")
    print(f"  mask shape: {batch['mask'].shape}")
    print(f"  Number of sessions in batch: {len(batch['session_id'])}")
    
    assert batch['x_sbp'].shape == (config.batch_size, config.window_size, 96)
    assert batch['y_sbp'].shape == (config.batch_size, config.window_size, 96)
    assert batch['mask'].shape == (config.batch_size, config.window_size, 96)
    
    print("\n✓ DataLoader test passed!")
    return loader


def test_model_forward(loader):
    """Test model forward pass."""
    print("\n" + "=" * 70)
    print("Testing Model Forward Pass")
    print("=" * 70)
    
    config = Config()
    
    # Build model
    model = build_model(config)
    model.eval()
    
    # Get a batch
    batch = next(iter(loader))
    
    x_sbp = batch['x_sbp']
    mask = batch['mask']
    
    print(f"Input shape: {x_sbp.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        pred = model(x_sbp, mask)
    
    print(f"Output shape: {pred.shape}")
    print(f"Output range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    assert pred.shape == x_sbp.shape
    
    print("\n✓ Model forward pass test passed!")
    return model


def test_loss_computation(loader, model):
    """Test loss computation."""
    print("\n" + "=" * 70)
    print("Testing Loss Computation")
    print("=" * 70)
    
    from losses import masked_nmse_loss, masked_mse_loss
    
    batch = next(iter(loader))
    
    x_sbp = batch['x_sbp']
    y_sbp = batch['y_sbp']
    mask = batch['mask']
    
    # Forward pass
    with torch.no_grad():
        pred = model(x_sbp, mask)
        
        # Compute losses
        nmse = masked_nmse_loss(pred, y_sbp, mask)
        mse = masked_mse_loss(pred, y_sbp, mask)
    
    print(f"NMSE Loss: {nmse.item():.6f}")
    print(f"MSE Loss: {mse.item():.6f}")
    
    assert nmse.item() >= 0
    assert mse.item() >= 0
    
    print("\n✓ Loss computation test passed!")


def test_backward_pass(loader, model):
    """Test backward pass and gradient flow."""
    print("\n" + "=" * 70)
    print("Testing Backward Pass")
    print("=" * 70)
    
    from losses import masked_nmse_loss
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    batch = next(iter(loader))
    
    x_sbp = batch['x_sbp']
    y_sbp = batch['y_sbp']
    mask = batch['mask']
    
    # Forward
    pred = model(x_sbp, mask)
    loss = masked_nmse_loss(pred, y_sbp, mask)
    
    print(f"Loss before backward: {loss.item():.6f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
    
    # Step
    optimizer.step()
    
    print("\n✓ Backward pass test passed!")


if __name__ == "__main__":
    print("\n🧪 Testing Training Setup\n")
    
    # Run tests
    dataset = test_dataset()
    loader = test_dataloader(dataset)
    model = test_model_forward(loader)
    test_loss_computation(loader, model)
    test_backward_pass(loader, model)
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Ready to train.")
    print("=" * 70)
    print("\nTo start training, run:")
    print("  python src_mae/train.py")
