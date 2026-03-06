# TCN Model - Quick Start Guide

## TL;DR

A new **TCN (Temporal Convolutional Network)** baseline has been implemented for SBP masked reconstruction. It uses Conv1d with dilated residual blocks instead of Conv2d.

## Quick Changes

### 1. Train with TCN (Default)
```bash
cd src_mae
python train.py  # TCN is now the default model
```

### 2. Switch Model
Edit `src_mae/train.py` line 40:
```python
config.model_name = "tcn"        # Temporal Conv (new!)
config.model_name = "unet"       # U-Net Conv2d
config.model_name = "simple_cnn" # SimpleCNN Conv2d
config.model_name = "resnet"     # ResNet Conv2d
```

### 3. Adjust TCN Hyperparameters
In `src_mae/train.py` lines 41-42:
```python
config.hidden_channels = 128   # Wider = more capacity, slower training
config.num_tcn_layers = 7      # Deeper = larger receptive field (45 @ 7 layers)
```

## Key Architecture

```
Input (B, 200, 96)
    ↓
Stack [signal, mask_indicator] (B, 200, 192)
    ↓
Input Projection: (B, 192, 200) → (B, 128, 200)  [Conv1d 192→128]
    ↓
7 Residual TCN Blocks with dilations [1,2,4,8,4,2,1]
- Each block: 2× Conv1d + GroupNorm + Dropout
- Bidirectional padding: maintains (B, 128, 200)
- Total receptive field: ~45 timesteps
    ↓
Output Projection: (B, 128, 200) → (B, 96, 200)  [Conv1d 128→96]
    ↓
Transpose: (B, 96, 200) → (B, 200, 96)
    ↓
Preserve Observed: output = pred × mask + input × (¬mask)
    ↓
Output (B, 200, 96)
```

## Model Comparison

| Model | Conv Type | Receptive Field | Parameters | Speed | Complexity |
|-------|-----------|-----------------|-----------|-------|-----------|
| **TCN** | Conv1d (new) | ~45 temporal | ~735K | ⚡⚡ | Low |
| UNet | Conv2d | Multi-scale | ~1.2M | ⚡ | High |
| SimpleCNN | Conv2d | Local | ~900K | ⚡⚡ | Low |
| ResNet | Conv2d | Local | ~1.0M | ⚡ | Medium |

## Test It

```bash
python test_tcn_model.py
```

Expected output:
```
======================================================================
✓ All tests passed!
======================================================================
```

Tests verify:
- ✓ Correct output shape (4, 200, 96)
- ✓ Observed values preserved
- ✓ Predictions made in masked regions
- ✓ Loss functions work (masked_nmse, kaggle_aligned_nmse)
- ✓ Gradients flow properly

## What Stayed the Same

❌ **No changes to**:
- Preprocessing pipeline (preprocess_non_overlapping)
- Dataset loading (SBPDataset)
- Loss functions (masked_nmse_loss, kaggle_aligned_nmse_loss)
- Training loop (train_one_epoch, validate_one_epoch)
- Inference interface (model.forward(x_sbp, mask))
- Checkpoint saving/loading

✅ **Just swap the model**, rest works as-is

## Why TCN?

1. **Efficient**: Conv1d over time is faster than Conv2d on (W, C) grids
2. **Natural**: Temporal convolutions suit sequential signal data
3. **Flexible**: Dilated convolutions achieve wide receptive field (~45) without deep stacking
4. **Practical**: Bidirectional processing for offline reconstruction
5. **Compatible**: Drop-in replacement for existing pipeline

## Performance Expectations

- **Training speed**: Should be similar or faster than UNet
- **Convergence**: May converge faster due to simpler architecture
- **Validation loss**: Likely similar or slightly better than UNet
- **Inference time**: Faster than UNet (no multi-scale pooling)

## Troubleshooting

### Model not loading?
```bash
# Make sure you're in src_mae directory or adjust imports
cd src_mae
python train.py
```

### Memory issues?
```python
# Reduce hidden_channels in Config
config.hidden_channels = 64  # Instead of 128
config.batch_size = 8        # Instead of 16
```

### Want deeper model?
```python
# Increase num_layers and batch_size together
config.num_tcn_layers = 11   # Receptive field ~200 (matches window!)
config.batch_size = 8         # Need smaller batch with deeper model
```

## Architecture Details

See **TCN_IMPLEMENTATION_SUMMARY.md** for:
- Detailed receptive field calculation
- Dilated convolution explanation
- Loss function integration
- Shape assertions and validation
- Full parameter breakdown

## Next Steps

1. **Run training**: `python src_mae/train.py`
2. **Monitor metrics**: Check train/val loss and convergence
3. **Compare models**: Try TCN vs UNet on same data
4. **Tune hyperparameters**: Adjust hidden_channels, num_layers, dropout
5. **Analyze predictions**: Check masked region reconstructions

Good luck! 🚀
