# SpatialTCN: Quick Test of 2D Spatial Hypothesis

## What is SpatialTCN?

A hybrid model that treats (time, channel) as **2D spatial data** using Conv2d, addressing the architectural gap we identified:

```
Standard TCN:           U-Net:                SpatialTCN:
(B, 192, 200)          (B, 2, 200, 96)       (B, C, 200, 96)
    ↓                       ↓                      ↓
Conv1d on time      Conv2d on 2D grid      Conv2d on 2D grid
only                    (temporal +          (temporal +
                      channel neighbors)     channel neighbors)
    ↓                       ↓                      ↓
Misses channel      Exploits 2D          Exploits 2D
spatial structure   spatial structure     spatial structure
```

## Why This Test Matters

**Hypothesis**: The 16-19% performance gap between TCN (~0.91) and U-Net (~0.75) is because:
- TCN treats data as 1D sequence (time only)
- U-Net treats data as 2D spatial grid (time × channel)
- Masked reconstruction benefits from 2D spatial context

**Prediction**: SpatialTCN should close the gap by processing (W, C) as 2D spatial data like U-Net

## How to Test

### Step 1: Clear Old Checkpoints
```bash
rm -rf src_mae/checkpoints_tcn/*
```

### Step 2: Configure for SpatialTCN
Edit `src_mae/train.py` line 40:
```python
config.model_name = "spatial_tcn"  # Was "tcn"
```

### Step 3: Train
```bash
cd src_mae
python train.py
```

### Step 4: Compare Results

Expected validation metrics:
```
Standard TCN:       ~0.91-0.94 NMSE (baseline)
U-Net:              ~0.75 NMSE (target)
SpatialTCN:         ~0.80-0.85 NMSE (if hypothesis correct)
                    Gap would close by 50%+
```

## What SpatialTCN Does Differently

### Architecture Comparison

| Feature | Standard TCN | SpatialTCN | U-Net |
|---------|--------------|-----------|-------|
| Conv type | Conv1d | Conv2d | Conv2d |
| Input shape | (B, 192, 200) | (B, C, 200, 96) | (B, 2, 200, 96) |
| Kernel sees | [t±d] | [t±1, c±1] | [t±1, c±1] |
| Channel neighbor awareness | No | **Yes** | **Yes** |
| Multi-scale pooling | No | No | Yes |
| Skip connections | Per-block only | Per-block only | Multi-scale |
| Parameters | 1.1M | 335K | 7.8M |

### Code Implementation

```python
class SpatialTCN(nn.Module):
    """
    Key insight: Treat (time, channel) as 2D spatial grid
    """
    def forward(self, x_sbp, mask):
        B, W, C = x_sbp.shape

        # Stack: (B, 2, W, C) ← 2D spatial grid
        x = torch.stack([x_sbp, obs_mask], dim=1)

        # Conv2d processes both time and channel dimensions
        x = self.input_block(x)       # Conv2d sees (time, channel) neighbors
        for block in self.blocks:     # Each Conv2d block operates on (W, C) plane
            x = block(x)

        out = self.output_block(x)    # Conv2d output projection
        # ...
```

**Key difference**: Conv2d kernels now see `[t-1:t+2, c-1:c+2]` neighborhood, exploiting spatial structure

## Expected Outcomes

### If Hypothesis Correct (+8-12% improvement)
```
Train loss:  Similar or better
Val loss:    0.80-0.85 (vs 0.91-0.94 for standard TCN)
Convergence: Similar speed
Gap to U-Net: Closes to ~5-10% (vs current 16-19%)
```

### If Hypothesis Partially Correct (+3-5% improvement)
```
Val loss:    0.87-0.90 (vs 0.91-0.94)
Takeaway:    2D structure helps but other factors matter (multi-scale, skips)
```

### If Hypothesis Incorrect (<3% improvement)
```
Val loss:    ~0.91-0.94 (no change)
Takeaway:    Gap is from multi-scale or skip connections, not 2D structure
Next test:   Add temporal pooling to SpatialTCN
```

## Model Specifications

**Current Implementation**:
- base_channels: 64
- num_layers: 4
- dilations: [1, 2, 4, 8]
- Parameters: 335K (much smaller than U-Net!)
- Input: (B, 2, 200, 96) just like U-Net

## Quick Comparison Scripts

### See parameter count
```python
from model import SpatialTCN, SBP_Reconstruction_UNet

spatial_tcn = SpatialTCN()
unet = SBP_Reconstruction_UNet()
tcn = TCNReconstructor()

print(f"SpatialTCN: {sum(p.numel() for p in spatial_tcn.parameters()):,}")
print(f"U-Net: {sum(p.numel() for p in unet.parameters()):,}")
print(f"TCN: {sum(p.numel() for p in tcn.parameters()):,}")

# Expected:
# SpatialTCN: ~335K
# U-Net: ~7.8M
# TCN: ~1.1M
```

## Hypothesis Testing Framework

### Test 1: Does 2D help? (This test)
- SpatialTCN (2D spatial) vs Standard TCN (1D temporal)
- Expected: +8-12% if true

### Test 2: Does multi-scale help?
- SpatialTCN + temporal pooling
- Expected: Additional +5-8%

### Test 3: Do skip connections help?
- SpatialTCN + cross-scale skips
- Expected: Additional +3-5%

### Test 4: Combined effect
- SpatialTCN with temporal pooling + skips
- Expected: ~0.80-0.85 or better, approaching U-Net

## Diagnostic Insights

If SpatialTCN closes the gap:
✓ Confirms 2D spatial structure is critical for this task
✓ Supports redesigning temporal models with spatial awareness
✓ Suggests next improvements: multi-scale + skips

If SpatialTCN doesn't help much:
✓ Indicates multi-scale hierarchy is more important
✓ Multi-scale > 2D spatial processing
✓ Next: Add temporal pooling to SpatialTCN

## Summary

**Quick test to verify architectural hypothesis**:
- Change config to `model_name = "spatial_tcn"`
- Train and compare validation loss
- If +8-12% improvement → hypothesis confirmed
- If +3-5% → partial explanation
- If no improvement → multi-scale/skips are the issue

Simplest architecture change with highest confidence in outcome.
