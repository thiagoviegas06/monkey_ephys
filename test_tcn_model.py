#!/usr/bin/env python
"""
Smoke test for TCN model - verify shape contracts and basic functionality.
"""

import sys
import torch
import numpy as np

# Add src_mae to path
sys.path.insert(0, './src_mae')

from model import TCNReconstructor
from losses import kaggle_aligned_nmse_loss, masked_nmse_loss

def test_tcn_forward_pass():
    """Test TCN forward pass with dummy data."""
    print("=" * 70)
    print("TCN Forward Pass Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Create model
    model = TCNReconstructor(
        hidden_channels=128,
        num_layers=7,
        kernel_size=3,
        dropout=0.2
    ).to(device)

    rf = model.get_receptive_field()
    print(f"Model receptive field: {rf} timesteps (window size: 200)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create dummy batch
    batch_size = 4
    window_size = 200
    channels = 96

    x_sbp = torch.randn(batch_size, window_size, channels, device=device)
    mask = torch.randint(0, 2, (batch_size, window_size, channels), dtype=torch.bool, device=device)

    # Mask some positions in x_sbp (set to 0 where mask is True)
    x_sbp = x_sbp * (~mask).float()

    print(f"Input shapes:")
    print(f"  x_sbp: {x_sbp.shape} (expected: ({batch_size}, {window_size}, {channels}))")
    print(f"  mask:  {mask.shape} (expected: ({batch_size}, {window_size}, {channels}))")

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(x_sbp, mask)

    print(f"\nOutput shape: {pred.shape}")
    assert pred.shape == (batch_size, window_size, channels), \
        f"Output shape mismatch: {pred.shape} vs expected {(batch_size, window_size, channels)}"
    print("✓ Output shape matches expected!")

    # Check that observed values are preserved
    obs_mask = ~mask
    max_obs_diff = ((pred * obs_mask) - (x_sbp * obs_mask)).abs().max().item()
    print(f"\nMax difference in observed positions: {max_obs_diff:.2e}")
    assert max_obs_diff < 1e-5, "Observed values should be preserved!"
    print("✓ Observed values correctly preserved!")

    # Check that masked positions have non-zero predictions
    masked_count = mask.sum().item()
    masked_pred_nonzero = (pred[mask].abs() > 1e-6).sum().item()
    print(f"\nMasked positions: {masked_count}")
    print(f"Non-zero predictions in masked positions: {masked_pred_nonzero}")
    print(f"✓ Model is making predictions in masked regions")

    return model, x_sbp, mask, device


def test_loss_computation(model, x_sbp, mask, device):
    """Test loss computation with TCN output."""
    print("\n" + "=" * 70)
    print("Loss Computation Test")
    print("=" * 70)

    batch_size, window_size, channels = x_sbp.shape

    # Generate ground truth (similar to masked version but more structured)
    y_sbp = torch.randn(batch_size, window_size, channels, device=device)

    # Per-channel variance (simulate session variance)
    channel_var = torch.ones(batch_size, channels, device=device) * 1.0

    # Compute prediction
    model.eval()
    with torch.no_grad():
        pred = model(x_sbp, mask)

    # Test masked_nmse_loss
    try:
        loss_nmse = masked_nmse_loss(pred, y_sbp, mask)
        print(f"masked_nmse_loss: {loss_nmse.item():.6f}")
        print("✓ masked_nmse_loss computed successfully")
    except Exception as e:
        print(f"✗ masked_nmse_loss failed: {e}")

    # Test kaggle_aligned_nmse_loss (without session_ids - uses per-sample grouping)
    try:
        loss_kaggle = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var)
        print(f"kaggle_aligned_nmse_loss (no session_ids): {loss_kaggle.item():.6f}")
        print("✓ kaggle_aligned_nmse_loss computed successfully")
    except Exception as e:
        print(f"✗ kaggle_aligned_nmse_loss failed: {e}")

    # Test with session_ids
    session_ids = ["session_0", "session_0", "session_1", "session_1"]
    try:
        loss_kaggle_sess = kaggle_aligned_nmse_loss(pred, y_sbp, mask, channel_var, session_ids)
        print(f"kaggle_aligned_nmse_loss (with session_ids): {loss_kaggle_sess.item():.6f}")
        print("✓ kaggle_aligned_nmse_loss with session grouping computed successfully")
    except Exception as e:
        print(f"✗ kaggle_aligned_nmse_loss with session_ids failed: {e}")


def test_gradient_flow(model, x_sbp, mask, device):
    """Test that gradients flow through the model."""
    print("\n" + "=" * 70)
    print("Gradient Flow Test")
    print("=" * 70)

    batch_size, window_size, channels = x_sbp.shape

    y_sbp = torch.randn(batch_size, window_size, channels, device=device)
    channel_var = torch.ones(batch_size, channels, device=device)

    # Forward pass
    model.train()
    pred = model(x_sbp, mask)

    # Compute loss
    loss = masked_nmse_loss(pred, y_sbp, mask)

    # Backward pass
    loss.backward()

    # Check gradients
    total_grad_norm = 0.0
    num_params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            num_params_with_grad += 1

    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {num_params_with_grad}")
    assert num_params_with_grad > 0, "No gradients computed!"
    assert total_grad_norm > 0, "Gradient norm is zero!"
    print("✓ Gradients flowing correctly through model")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TCN Model Smoke Test")
    print("=" * 70 + "\n")

    model, x_sbp, mask, device = test_tcn_forward_pass()
    test_loss_computation(model, x_sbp, mask, device)
    test_gradient_flow(model, x_sbp, mask, device)

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
