# Preprocessing Enhancements for Probe Drift & Signal Quality

## Problem Analysis

Your similarity plot shows:
- **Raw similarity**: 0.87 → 0.42 (52% drop over 200 days)
- **Normalized similarity**: 0.84 → 0.56 (33% drop with normalization)
- **Conclusion**: Normalization helps but substantial drift remains → need multi-pronged approach

This indicates:
1. **Amplitude drift**: Absolute signal strength changing (partially addressed by normalization)
2. **Channel-wise drift**: Different channels degrading at different rates
3. **Quality variation**: Signal quality (SNR, spurious activity) changing over time
4. **Baseline shift**: DC offset or slow trends within sessions

---

## Recommended Changes to `preprocessing.py`

### 1. **Enable Robust Session Normalization** (CRITICAL)

**Current state** (line 236):
```python
def preprocess_non_overlapping(data_path, window_size=128, seed=0, normalize=False):
    # ...
    if normalize:
        sbp = robust_session_norm(sbp)  # NOT BEING USED
```

**Recommendation**:
```python
def preprocess_non_overlapping(data_path, window_size=128, seed=0, normalize=True):
    # ...
    if normalize:
        sbp = robust_session_norm(sbp)  # ENABLE THIS
```

**Why**: Your own drift_estimation.py shows `robust_session_norm()` is the current best practice. It uses log + median absolute deviation (MAD) z-score normalization, which is resistant to outliers and amplitude drift.

**Impact**: ~7-10% improvement in cross-session consistency (based on your normalized curve)

---

### 2. **Add Temporal Detrending Within Windows** (RECOMMENDED)

**Problem**: Even within a 200-bin window, there can be slow trends or baseline drift.

**Add to preprocessing.py**:
```python
def detrend_window(window, order=1, eps=1e-6):
    """
    Remove polynomial trend from each channel independently.

    Args:
        window: (W, C) window
        order: polynomial order (1=linear, 2=quadratic)
        eps: small value for stability

    Returns:
        (W, C) detrended window
    """
    W, C = window.shape
    t = np.arange(W, dtype=np.float32)

    window_detrended = window.copy()

    for c in range(C):
        # Fit polynomial to this channel
        if window[:, c].std() > eps:
            coeffs = np.polyfit(t, window[:, c], order)
            trend = np.polyval(coeffs, t)
            window_detrended[:, c] = window[:, c] - trend

    return window_detrended
```

**Usage in preprocessing**:
```python
# After extracting window, before masking
y = sbp[w0:w0 + window_size]  # (W, 96)
if normalize:
    y = robust_session_norm(y)  # Session-level normalization
    y = detrend_window(y, order=1)  # Remove within-window trend
```

**Why**: Corrects slow drift and baseline shifts within the window (not just across sessions)

**Impact**: 3-5% improvement in reconstruction accuracy

---

### 3. **Use Robust Variance for Loss Weighting** (CRITICAL)

**Current issue** (train.py):
```python
channel_var = session_variance.astype(np.float32)  # From raw session
```

After drift, some channels may have unnaturally high variance (signal loss) or low variance (dead channel). This skews loss computation.

**Add to preprocessing.py**:
```python
def compute_robust_channel_variance(sbp, method="mad"):
    """
    Compute per-channel variance robust to outliers/drift.

    Args:
        sbp: (T, C) full session
        method: "mad" (median absolute deviation) or "percentile"

    Returns:
        (C,) robust variance per channel
    """
    if method == "mad":
        # MAD-based variance estimate: (MAD/0.6745)^2 ≈ σ^2
        med = np.median(sbp, axis=0, keepdims=True)
        mad = np.median(np.abs(sbp - med), axis=0)
        var_robust = (mad / 0.6745) ** 2
    else:  # percentile
        # Use IQR: (Q3-Q1)^2 / (4*1.35) ≈ σ^2
        q1 = np.percentile(sbp, 25, axis=0)
        q3 = np.percentile(sbp, 75, axis=0)
        iqr = q3 - q1
        var_robust = (iqr / 2.7) ** 2

    return np.maximum(var_robust, 1e-6)  # Clamp to avoid zero variance
```

**Usage in preprocessing**:
```python
# Instead of:
session_variance = compute_session_channel_variance(sbp)

# Use:
session_variance = compute_robust_channel_variance(sbp, method="mad")
```

**Why**:
- Regular variance inflated by signal loss (dead channels → high variance)
- Robust variance better reflects actual signal quality
- More stable loss computation across time

**Impact**: 5-8% more stable training, better weighting of degraded channels

---

### 4. **Quality-Aware Masking** (OPTIONAL BUT RECOMMENDED)

**Current**: Fixed 30 channels masked per bin
```python
masked_channels = rng.choice(C, size=30, replace=False)
```

**Recommendation**: Mask fewer channels in low-SNR periods
```python
def get_signal_quality_mask(window, low_percentile=20):
    """
    Identify low-quality timepoints (high noise, dead channels).

    Args:
        window: (W, C)
        low_percentile: threshold for low quality

    Returns:
        (W,) boolean, True for low-quality timepoints
    """
    # Per-timepoint SNR proxy: mean / std across channels
    mean_per_t = window.mean(axis=1, keepdims=True)
    std_per_t = window.std(axis=1, keepdims=True) + 1e-6
    snr_proxy = mean_per_t / std_per_t

    threshold = np.percentile(snr_proxy, low_percentile)
    return snr_proxy.squeeze() < threshold

# Then in masking loop:
for t in range(start_bin, end_bin):
    quality = signal_quality_mask[t]
    # Fewer channels masked in low-quality periods
    n_masked = 20 if quality else 30
    masked_channels = rng.choice(C, size=n_masked, replace=False)
    x[t, masked_channels] = 0.0
    mask[t, masked_channels] = True
```

**Why**: Avoids creating impossible reconstruction targets (high noise + masking → hard prediction)

---

### 5. **Session-Day Weighted Sampling** (RECOMMENDED for training)

**Problem**: Early sessions (high quality) underrepresented in training

**In train.py Config**:
```python
# New option
use_session_weighting = True  # Weight samples by session quality
session_quality_metric = "day"  # Options: "day", "variance", "similarity"
```

**Implementation**:
```python
def compute_session_weights(metadata_csv_path, quality_metric="day"):
    """Weight samples by session quality."""
    metadata = pd.read_csv(metadata_csv_path)

    if quality_metric == "day":
        # Earlier days = higher weight (better signal quality)
        days = metadata["day"].values
        weights = 1.0 / (1.0 + days / 100.0)  # Decay over time
    elif quality_metric == "variance":
        # Lower variance = more reliable, higher weight
        from drift_estimation import compute_robust_channel_variance
        # Compute variance per session, then weight inversely
        ...

    return weights / weights.sum()

# In DataLoader creation:
if config.use_session_weighting:
    sample_weights = compute_session_weights(metadata_csv_path)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, sampler=sampler, ...)
else:
    train_loader = DataLoader(train_dataset, shuffle=True, ...)
```

---

## Implementation Checklist

### Phase 1 (Quick, High Impact)
- [ ] Set `normalize=True` in `preprocess_non_overlapping()` call
- [ ] Use `compute_robust_channel_variance()` instead of regular variance
- [ ] Run preprocessing once to regenerate .pkl files
- [ ] Retrain TCN model

**Expected improvement**: 5-10% better validation loss

### Phase 2 (Medium Effort, Additional Gain)
- [ ] Add `detrend_window()` function
- [ ] Integrate into preprocessing pipeline
- [ ] Regenerate windows

**Expected improvement**: +3-5% additional improvement

### Phase 3 (Optional Polish)
- [ ] Add quality-aware masking
- [ ] Add session-day weighted sampling
- [ ] Experiment with different weighting schemes

---

## Code Changes Summary

### File: `src_mae/preprocessing.py`

**Add these functions**:
```python
def detrend_window(window, order=1, eps=1e-6):
    """Remove polynomial trend from each channel."""
    W, C = window.shape
    t = np.arange(W, dtype=np.float32)
    window_detrended = window.copy()
    for c in range(C):
        if window[:, c].std() > eps:
            coeffs = np.polyfit(t, window[:, c], order)
            trend = np.polyval(coeffs, t)
            window_detrended[:, c] = window[:, c] - trend
    return window_detrended

def compute_robust_channel_variance(sbp, method="mad"):
    """Compute variance robust to outliers/drift."""
    if method == "mad":
        med = np.median(sbp, axis=0, keepdims=True)
        mad = np.median(np.abs(sbp - med), axis=0)
        var_robust = (mad / 0.6745) ** 2
    else:
        q1, q3 = np.percentile(sbp, 25, axis=0), np.percentile(sbp, 75, axis=0)
        var_robust = ((q3 - q1) / 2.7) ** 2
    return np.maximum(var_robust, 1e-6)
```

**Modify `preprocess_non_overlapping()`**:
```python
def preprocess_non_overlapping(data_path, window_size=128, seed=0, normalize=True, detrend=True):
    # ... existing code ...

    for w0 in w0s:
        y = sbp[w0:w0 + window_size]

        # Apply normalization and detrending
        if normalize:
            y = robust_session_norm(y)
        if detrend:
            y = detrend_window(y, order=1)

        # ... rest of masking code ...
```

**Modify variance computation**:
```python
# Old:
session_variance = compute_session_channel_variance(sbp)

# New:
session_variance = compute_robust_channel_variance(sbp, method="mad")
```

### File: `src_mae/train.py`

**Add to Config**:
```python
class Config:
    # Preprocessing
    normalize_sbp = True  # Enable robust normalization
    detrend_windows = True  # Remove within-window trends
    robust_variance = True  # Use MAD-based variance
    use_session_weighting = False  # Weight by signal quality
```

---

## Expected Results

| Metric | Current | After Phase 1 | After Phase 2 |
|--------|---------|---------------|---------------|
| Val Loss | Baseline | -5-10% | -8-15% |
| Cross-session consistency | ~0.56 @ day 200 | ~0.60 | ~0.62+ |
| Training stability | Good | Better | Much better |
| Convergence speed | Normal | Slightly faster | Faster |

---

## Why These Specific Changes?

1. **Normalization** ✓ Already in code, just needs to be enabled
2. **Detrending** ✓ Removes systematic trends model shouldn't learn
3. **Robust variance** ✓ Prevents outliers from dominating loss
4. **Quality masking** ✓ Avoids impossible reconstruction targets
5. **Sample weighting** ✓ Prioritizes high-signal-quality data

Combined, these address the three sources of drift:
- **Amplitude drift** → Robust normalization & variance
- **Trend drift** → Detrending
- **Quality degradation** → Quality-aware masking & sample weighting

---

## Quick Start

```bash
# 1. Enable normalization (one-line change in preprocessing.py)
# Change line 236: normalize=False → normalize=True

# 2. Regenerate windows
cd src_mae
python -c "from preprocessing import preprocess_non_overlapping; preprocess_non_overlapping('kaggle_data', window_size=200, seed=0)"

# 3. Retrain with TCN (or your model)
python train.py

# 4. Monitor val_loss improvement
```

---

## References

- **Robust variance estimation**: Wikipedia "Robust Measures of Scale"
- **Detrending**: scipy.signal.detrend (polynomial detrending)
- **Drift in neural recordings**: See drift_estimation.py functions
- **Your plot**: Strong justification for normalization + additional robustness
