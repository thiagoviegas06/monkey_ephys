# Training Setup - Quick Reference Guide

## ✅ What Was Created

### 1. **train.py** - Main Training Script

#### Key Components:

**Config Class** (Lines 21-47)
```python
class Config:
    # Easy-to-modify parameters
    data_path = "kaggle_data"
    window_size = 200
    model_name = "unet"  # or "simple_cnn", "resnet"
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 10
```

**SBPDataset Class** (Lines 52-154)
- Loads training sessions from metadata
- Creates windows on-the-fly (28,062 windows from 226 sessions)
- Applies random masking for each window
- Returns: `{x_sbp, y_sbp, mask, session_id}`
- Convention: `mask=True` means masked (to predict)

**train_one_epoch()** (Lines 172-264)
- Clear training loop with progress bar
- Detailed logging every 10 batches
- Gradient clipping for stability
- Returns average NMSE loss

**main()** (Lines 270-360)
- Builds dataset & dataloader
- Creates model (U-Net by default)
- Sets up optimizer (AdamW)
- Runs training loop
- Saves checkpoints every epoch

---

## 🚀 How to Use

### Option 1: Run with defaults
```bash
python src_mae/train.py
```

### Option 2: Modify Config first
Edit `train.py` lines 21-47:
```python
class Config:
    batch_size = 32          # Larger batch
    learning_rate = 5e-4     # Faster learning
    num_epochs = 20          # More epochs
    model_name = "resnet"    # Try different model
```

---

## 📊 What You'll See During Training

```
======================================================================
Training Configuration
======================================================================
Model: unet
Window size: 200
Batch size: 16
Learning rate: 0.0001
Device: cpu
Epochs: 10
======================================================================

Building dataset...
Found 226 training sessions
Total windows: 28062

DataLoader created:
  Total windows: 28062
  Batches per epoch: 1754

Building model...
Built U-Net with base_channels=64
Total trainable parameters: 11,234,945

======================================================================
Starting Training
======================================================================

Epoch 1/10: 100%|████████| 1754/1754 [loss: 2.3456, avg_loss: 2.4123]

[Epoch 1, Batch 10/1754]
  NMSE Loss: 2.456789
  MSE Loss:  0.123456
  Masked: 30000/307200 (9.77%)
  Pred (masked): mean=3.1234, std=1.2345
  True (masked): mean=3.2345, std=1.3456
  
... (continues for all batches)

[Epoch 1] Average NMSE Loss: 2.412345
✓ Checkpoint saved: checkpoints/model_epoch_1.pt
```

---

## 🐛 Debugging Tips

### Check a single batch:
```python
from train import Config, SBPDataset, build_model
from torch.utils.data import DataLoader

config = Config()
dataset = SBPDataset(config.data_path, config.window_size)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(loader))
print("Batch shapes:", {k: v.shape for k, v in batch.items() if hasattr(v, 'shape')})
```

### Test model forward pass:
```python
model = build_model(config)
pred = model(batch['x_sbp'], batch['mask'])
print("Input:", batch['x_sbp'].shape)
print("Output:", pred.shape)
```

### Check loss values:
```python
from losses import masked_nmse_loss
loss = masked_nmse_loss(pred, batch['y_sbp'], batch['mask'])
print("NMSE:", loss.item())
```

---

## 📁 Output Structure

```
checkpoints/
├── model_epoch_1.pt
├── model_epoch_2.pt
└── ... (saved every epoch)
```

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Epoch number
- `loss`: Training loss
- `config`: Config dictionary

---

## 🔧 Key Design Decisions

1. **On-the-fly data generation**: More memory efficient than pre-generating all windows
2. **Random masking per epoch**: Different masks each time = data augmentation
3. **Gradient clipping**: Prevents gradient explosion (max_norm=1.0)
4. **AdamW optimizer**: Better generalization than Adam
5. **Progress bars**: Visual feedback with tqdm
6. **Detailed logging**: Monitor pred/true statistics every 10 batches

---

## 📈 Expected Performance

- **Initial NMSE**: ~20-30 (random predictions)
- **After 1 epoch**: ~5-10
- **After 5 epochs**: ~2-5
- **Target**: < 1.0 (better than mean prediction)

---

## ⚡ Performance Notes

Current setup (tested):
- **Dataset size**: 28,062 windows
- **Batches/epoch**: 1,754 (with batch_size=16)
- **Model params**: ~11M (U-Net)
- **Device**: CPU (slow) or CUDA (fast)

Speed tips:
- Use CUDA if available: `config.device = "cuda"`
- Increase `num_workers=4` in DataLoader
- Increase batch size if you have memory
- Use `simple_cnn` for faster iterations

---

## 🎯 Next Steps

1. ✅ Verify setup: `python test_training_setup.py`
2. 🚀 Start training: `python src_mae/train.py`
3. 📊 Monitor loss curves
4. 🔄 Experiment with hyperparameters
5. 💾 Load best checkpoint for inference
