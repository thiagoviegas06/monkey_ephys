# Drift Correction Strategy Analysis
## NYU Neuro Project 2 - SBP Masked Reconstruction

---

## 1. Current Pipeline Overview

### Data Flow
```
Raw SBP (200 sessions, 3 years)
    ↓
preprocess_non_overlapping()
    - 200-bin windows (2 seconds @ 100Hz)
    - ~30 random channels masked per bin in contiguous span
    - Output: pickle files with x_sbp, y_sbp, mask, channel_var, session_id, day
    ↓
Train Loop
    - Load preprocessed windows
    - Input: x_sbp (masked zeros), mask, kinematics
    - Target: y_sbp (ground truth)
    - Loss: kaggle_aligned_nmse_loss
      NMSE = MSE(masked positions) / per_session_per_channel_variance
    ↓
Model (TCN Transformer)
    - Forward: (B, W, C) → (B, W, C)
    - Output: predictions for masked positions
    ↓
Evaluation (eval.py)
    - Load test SBP (already masked by competition)
    - Predict masked positions window-by-window
    - Blend predictions into full array
    - Submit to Kaggle (original scale)
```

### Key Data Properties
- **Shape**: (B=batch, W=200 time_bins, C=96 channels)
- **Masking**: ~3000 positions per window (30 channels × ~100 time bins)
- **Variance**: Computed per-session per-channel from `y_sbp`
- **Session span**: 200 sessions over 3 years (up to 32% drift in raw data per memory)

---

## 2. The Drift Challenge

### Problem Statement
Sessions 1-50 have different signal baseline/amplitude than sessions 150-200, even for same measurement. This causes:

1. **Training bias**: Model trained on later sessions over-trained due to different signal statistics
2. **Loss scaling**: Kaggle loss groups by (session, channel), so each session's variance affects gradient flow differently
3. **Inference mismatch**: If you correct drift before training, you must reverse it before submission

### Current Drift Analysis (from your memory)
```
Across 226 sessions:
- Raw Pearson correlation: 0.417 decay (adjacent=0.828, distant=0.411)
- Global norm: 0.324 decay (adjacent=0.858, distant=0.533) - 23% improvement
- Adaptive norm: 0.205 decay (adjacent=0.668, distant=0.463) - 36% improvement
  BUT breaks short-term relationships (0.858→0.668)
```

**The core issue**: Adaptive/aggressive correction breaks local structure; global correction is middle-ground but still imperfect.

---

## 3. Five Possible Approaches

### **Approach A: No Explicit Correction (Baseline)**
**Idea**: Let the model learn to handle drift implicitly via regularization.

**Pros**:
- ✅ Simplest, no post-processing needed
- ✅ Model can learn session-specific offsets if helpful
- ✅ No risk of reversible transformation bugs

**Cons**:
- ❌ Model may waste capacity on learning per-session shifts
- ❌ Doesn't address loss scaling imbalance

**Implementation**: Just train as-is. Add session_id embedding as auxiliary input to model.

---

### **Approach B: Session-Wise Normalization (Reversible)**
**Idea**: Normalize each session to zero mean/unit variance, train, then denormalize at inference.

**Formula**:
```
Training:
  z_normalized = (y_sbp - μ_session) / σ_session
  Train on z_normalized

Inference:
  pred_normalized = model(x_normalized, ...)
  pred_original = pred_normalized * σ_session + μ_session
```

**Pros**:
- ✅ Reversible (no information loss)
- ✅ Standardizes variance across sessions → fairer loss
- ✅ Model trains on uniform statistics
- ✅ Easy to implement in preprocessing + eval

**Cons**:
- ⚠️ Breaks cross-session learning (model sees decoupled statistics)
- ❌ Each session has slightly different baseline → model can't learn temporal structure across boundaries
- ❌ Multi-window sessions: need consistent normalization within session

**When to use**: If drift is primarily in amplitude/offset, not temporal structure.

---

### **Approach C: Robust Global Normalization (Exponential Moving Average)**
**Idea**: Estimate a time-varying drift curve, subtract it, train on residuals.

**Formula**:
```
Per-channel per-session:
  drift_curve[session] = EMA(signal from sessions 1 to current)
  residual = signal - drift_curve
  Train on residual

Inference:
  pred_residual = model(...)
  pred_original = pred_residual + drift_curve[test_session]
```

**Pros**:
- ✅ Captures long-term trends while preserving local structure
- ✅ More accurate than global mean (adaptive to recent drift)
- ✅ Reversible

**Cons**:
- ⚠️ Requires careful hyperparameter tuning (EMA window)
- ⚠️ If test sessions are far from training, drift_curve estimate may be poor
- ❌ Assumes monotonic drift (not true if signal oscillates)

**When to use**: If you suspect gradual instrumental drift (like sensor calibration drift).

---

### **Approach D: Variance-Clamping (Loss-Level Fix)**
**Idea**: Cap per-session variance in loss computation to prevent outliers from dominating.

**Formula**:
```
In kaggle_aligned_nmse_loss():
  session_var = clip(session_var, percentile_10, percentile_90)
  nmse = mse / clipped_variance
```

**Pros**:
- ✅ No preprocessing changes needed
- ✅ Works with existing pipeline
- ✅ Prevents drift from skewing loss scale

**Cons**:
- ❌ Doesn't actually correct drift, just masks it
- ❌ May downweight important sessions
- ⚠️ Affects Kaggle metric alignment (loss ≠ Kaggle metric)

**When to use**: Quick fix if training is imbalanced but drift isn't severe.

---

### **Approach E: Learnable Drift + Auxiliary Head (Advanced)**
**Idea**: Separate the model into (1) drift predictor and (2) signal reconstruction.

**Architecture**:
```
Input: x_sbp (masked), session_id
  ↓
[Session Embedding] → drift_head → predicted_drift
                    ↓
              [TCN backbone] → reconstruction_head → predicted_signal
  ↓
Output: pred_signal + predicted_drift
```

**Pros**:
- ✅ Explicitly models drift as learnable parameter
- ✅ Can interpolate drift for test sessions
- ✅ Flexible: drift can be nonlinear (per-channel, per-time)

**Cons**:
- ⚠️ Significant architectural change
- ⚠️ Requires careful supervision (how to label "true drift"?)
- ❌ May overfit if not enough sessions with diverse drift

**When to use**: If you have good estimates of session-wise drift from external source.

---

## 4. Recommended Strategy: **Approach B + Early Stopping**

### Why This Makes Sense
1. **Drift is primarily amplitude/offset** (from your Pearson analysis: adjacent sessions similar, distant sessions different)
2. **Session-wise normalization** directly addresses this without breaking temporal structure within-session
3. **Reversible** → no loss of information or Kaggle metric misalignment
4. **Easy to implement** → minimal code changes

### Implementation Plan

#### Step 1: Preprocessing Modification
```python
# In preprocessing.py, add normalization tracking

def preprocess_non_overlapping_with_drift_correction(..., normalize=True):
    for session in sessions:
        sbp = load_session(session)

        if normalize:
            # Compute per-session per-channel statistics
            session_mean = sbp.mean(axis=0)  # (C,)
            session_std = sbp.std(axis=0)    # (C,)
            sbp_normalized = (sbp - session_mean) / session_std

            # Save normalization params for later reversal
            sample['session_mean'] = session_mean
            sample['session_std'] = session_std
            sample['y_sbp'] = sbp_normalized  # store normalized target
            sample['x_sbp'] = apply_mask(sbp_normalized)
        else:
            sample['y_sbp'] = sbp
            sample['x_sbp'] = apply_mask(sbp)
            sample['session_mean'] = None
            sample['session_std'] = None
```

#### Step 2: Training Loop (No changes needed)
- DataLoader already returns `session_mean`, `session_std` per batch
- Just store them: `sample['session_mean']`, `sample['session_std']`

#### Step 3: Evaluation Modification
```python
# In eval.py predict_sessions()

def predict_sessions(model, session_data, device):
    for session_id, info in session_data.items():
        # ... existing code ...

        masked_sbp = info["masked_sbp"]  # (N, C)
        session_mean = info.get("session_mean")
        session_std = info.get("session_std")

        # Normalize input if available
        if session_mean is not None and session_std is not None:
            x_normalized = (masked_sbp - session_mean) / session_std
        else:
            x_normalized = masked_sbp  # fallback: no normalization

        # ... run inference on x_normalized ...
        pred_normalized = model(...)

        # Denormalize predictions back to original scale
        if session_mean is not None and session_std is not None:
            pred_original = pred_normalized * session_std + session_mean
        else:
            pred_original = pred_normalized

        # Write original-scale predictions to submission
        submission[session_id] = pred_original
```

#### Step 4: Handle Test Set Edge Case
Test set sessions may not have been in training. Two options:
- **Option 1**: Compute session_mean/std from test set itself (if you have access to raw signal)
- **Option 2**: Use global mean/std as fallback
- **Option 3**: Train a small linear model to predict test session's statistics from metadata (e.g., day_from_nearest)

---

## 5. Alternative: Hybrid Approach (Approach B + Approach C)

If session-wise normalization isn't enough, combine it with EMA drift correction:

```
Per-channel:
  1. Compute EMA of session means: ema_mean[s] = 0.9*ema_mean[s-1] + 0.1*session_mean[s]
  2. Normalize within-session: z = (signal - session_mean) / session_std
  3. Subtract trend: residual = z - (ema_mean[s] - global_mean)
  4. Train on residual
  5. Inference: denormalize + add back trend
```

This gives you:
- ✅ Within-session consistency
- ✅ Long-term drift correction
- ✅ Reversible

---

## 6. Risk Analysis

| Approach | Implementation Risk | Kaggle Risk | Data Leakage Risk |
|----------|-------------------|------------|------------------|
| B (Session-wise norm) | Low | Low | None (fully reversible) |
| C (EMA drift) | Medium | Medium | None (fully reversible) |
| D (Variance clamp) | Low | **High** (loss ≠ metric) | None |
| E (Learnable drift) | High | Medium | None if properly supervised |

---

## 7. Quick Start: Implement Approach B

**Files to modify**:
1. `src_mae/preprocessing.py`: Add `normalize` parameter + save stats
2. `src_mae/dataloader.py`: Load stats from samples
3. `src_mae/eval.py`: Denormalize predictions before submission

**Estimated effort**: 30 minutes
**Expected improvement**: +2-5% (depending on drift severity)

---

## 8. Testing Strategy

Before full training:
1. **Smoke test**: Normalize → denormalize on random sample, check MSE(original, reconstructed) ≈ 0
2. **Validation check**: Train on normalized data, ensure validation loss is reasonable
3. **Submission sanity check**: Verify denormalized predictions have same scale as ground truth
4. **A/B test**: Train two models (one with drift correction, one without), compare Kaggle scores

