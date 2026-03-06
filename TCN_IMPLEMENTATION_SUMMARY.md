# TCN Baseline Implementation Summary

## Overview

Successfully implemented a **Temporal Convolutional Network (TCN)** baseline for masked SBP reconstruction, replacing the existing Conv2d-based models with efficient Conv1d temporal convolutions.

## Data & Model Setup

### Input Specifications
- **Shape**: (B, W, C) = (batch, 200 time_bins, 96 channels)
- **Input convention**: x_sbp zeros where masked (masked positions are 0)
- **Mask**: (B, W, C) boolean, True where position needs reconstruction
- **Tensor layout**: (Batch, Time, Channel) - batch-first, time-axis middle

### Target Output
- **Shape**: (B, W, C) same as input
- **Contract**: Model keeps observed values unchanged, predicts only masked positions

## TCN Architecture

### Design Rationale

1. **Conv1d over Time**: Processes temporal patterns along the W=200 dimension
   - Each filter operates over consecutive time bins
   - Natural for sequential temporal data
   - More efficient than Conv2d on (W, C) grids

2. **Bidirectional Processing**: Non-causal convolutions within the window
   - Uses symmetric padding: `padding = dilation * (kernel_size // 2)`
   - Model can attend to past and future within window context
   - Appropriate for offline reconstruction (not real-time)

3. **Dilated Convolutions**: Wide receptive field without depth
   - Exponential dilation schedule: [1, 2, 4, 8, 4, 2, 1] for 7 layers
   - Receptive field: ~45 timesteps (well within W=200)
   - Captures both local and longer-range temporal patterns

4. **Residual Blocks**: Stable gradient flow
   - Two Conv1d layers per block with skip connections
   - GroupNorm for stable training regardless of batch size
   - Dropout for regularization

### Model Specification

```python
class TCNReconstructor(nn.Module):
    """
    Temporal Convolutional Network for masked SBP reconstruction

    Args:
        hidden_channels: Number of channels in TCN processing (default: 128)
        num_layers: Number of residual blocks (default: 7)
        kernel_size: Conv1d kernel size (default: 3)
        dropout: Dropout probability (default: 0.2)
        dilation_multiplier: Base for exponential dilation (default: 2)
    """
```

### Receptive Field Analysis

**Calculation** (kernel_size=3):
- Each Conv1d layer with kernel_size=3 and dilation `d` spans `2*d+1` temporal positions
- Total RF = sum of individual layer receptive fields

**With default config** (num_layers=7, dilations=[1,2,4,8,4,2,1]):
- RF ≈ 45 timesteps
- Sufficient to capture meaningful temporal patterns
- Efficient use of W=200 window size

### Parameter Count
- ~735,000 parameters (similar to UNet baseline)
- Configurable via `hidden_channels` and `num_layers`

## I/O Contract

### Input Interface
```python
model(x_sbp, mask) -> pred
```
- `x_sbp`: (B, W, C) = (B, 200, 96) masked signal (zeros where masked)
- `mask`: (B, W, C) bool, True where masked
- Output: (B, W, C) reconstructed signal

### Internal Processing
1. **Channel stacking**: Concatenate x_sbp and obs_mask → (B, W, 2*C)
2. **Transpose to Conv1d format**: (B, W, 2*C) → (B, 2*C=192, W=200)
3. **Input projection**: 192 channels → hidden_channels (128)
4. **Residual blocks**: Process through 7 dilated convolution blocks
5. **Output projection**: hidden_channels → C=96 channels
6. **Transpose back**: (B, C, W) → (B, W, C)
7. **Masking**: Preserve observed values, keep predictions only for masked positions

### Preserve Observed Values
```python
out = out * mask.float() + x_sbp * (~mask).float()
```
- Ensures model output respects input constraints
- Critical for properly supervising only masked positions

## Loss Function Integration

Uses existing `kaggle_aligned_nmse_loss`:
- Grouped NMSE by (session, channel)
- NMSE = MSE / per_session_per_channel_variance
- Computed only on masked positions

## Implementation Changes

### Files Modified
1. **src_mae/model.py**
   - Added `TCNResidualBlock` class
   - Added `TCNReconstructor` class
   - Updated usage examples

2. **src_mae/train.py**
   - Imported TCNReconstructor
   - Updated Config: added `hidden_channels`, `num_tcn_layers`
   - Updated `build_model()`: TCN now default model (can switch via config)
   - Displays receptive field info at startup

3. **src_mae/eval.py**
   - Imported TCNReconstructor for compatibility

### Files Added
- **test_tcn_model.py**: Comprehensive smoke tests
  - Forward pass shape validation
  - Output value constraints (observed preservation)
  - Loss computation tests (both loss functions)
  - Gradient flow verification

## Training & Configuration

### Config Update
```python
class Config:
    # Model selection
    model_name = "tcn"  # Options: "tcn", "unet", "simple_cnn", "resnet"

    # TCN-specific
    hidden_channels = 128
    num_tcn_layers = 7

    # UNet-specific (kept for backward compatibility)
    base_channels = 64
```

### No Changes Required
- **Preprocessing**: Unchanged - reuse existing `preprocess_non_overlapping()`
- **Dataset**: Unchanged - SBPDataset compatible
- **Loss functions**: Unchanged - kaggle_aligned_nmse_loss works directly
- **Training loop**: Unchanged - train_one_epoch and validate_one_epoch work as-is
- **Inference**: Unchanged - model.forward(x_sbp, mask) interface identical

## Testing & Validation

### Smoke Test Results ✓

```
Model receptive field: 45 timesteps (window size: 200)
Total parameters: 735,361

Forward Pass Test:
✓ Output shape matches expected (4, 200, 96)
✓ Observed values correctly preserved
✓ Model making predictions in masked regions

Loss Computation:
✓ masked_nmse_loss: 1.078920
✓ kaggle_aligned_nmse_loss (no session_ids): 1.072948
✓ kaggle_aligned_nmse_loss (with session_ids): 1.073851

Gradient Flow:
✓ Total gradient norm: 3.883316 (across 66 trainable parameters)
✓ Gradients flowing correctly
```

### Shape Assertions
- Input validation: (B, W, C) shapes checked
- Dimension verification: C==96, W==200 asserted
- Output shape guaranteed to match input shape

## Why TCN Fits This Problem

1. **Temporal Nature**: Conv1d naturally operates over time dimension
   - SBP is a physiological signal with strong temporal structure
   - Masked regions require understanding temporal context

2. **Efficiency**: Fewer parameters than Conv2d alternatives
   - Only processes time dimension
   - No spatial pooling complexity
   - ~735K params vs UNet's multi-scale processing

3. **Receptive Field**: Dilated convolutions achieve wide RF efficiently
   - RF of 45 bins is informative but not excessive
   - Matches realistic temporal dependencies in neural data

4. **Bidirectional**: Non-causal processing appropriate for offline reconstruction
   - Can attend to both past and future time steps
   - Better for reconstruction quality vs. online prediction

5. **Composability**: Works seamlessly with existing pipeline
   - Same input/output interface as Conv2d models
   - Compatible with all existing loss functions
   - No preprocessing pipeline changes needed

## Usage

### Training with TCN
```python
from src_mae.train import main, Config

# Update config
config = Config()
config.model_name = "tcn"
config.hidden_channels = 128
config.num_tcn_layers = 7

# Run training
python src_mae/train.py  # Will use TCN by default
```

### Testing
```bash
python test_tcn_model.py  # Comprehensive model tests
```

### Switching Models
```python
# In train.py Config:
config.model_name = "unet"      # Switch to U-Net
config.model_name = "tcn"       # Switch back to TCN
config.model_name = "simple_cnn"  # Or SimpleCNN
```

## Future Improvements (Not Implemented)

1. **Causal padding option**: For online/streaming inference
2. **Adaptive dilation**: Learn or schedule dilations during training
3. **Multi-scale residuals**: Skip connections across non-contiguous blocks
4. **Channel-wise processing**: Separate TCN per channel vs. shared processing
5. **Depth tuning**: Experiment with num_layers in [5, 7, 9, 11]

## Code Quality

- **Type hints**: Clear in function signatures
- **Docstrings**: Comprehensive documentation
- **Assertions**: Input shape validation
- **Comments**: Explain key design decisions
- **Modular**: Easy to test and modify individual components
- **Backward compatible**: All existing code paths unchanged
