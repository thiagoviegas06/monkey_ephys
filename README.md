# Preprocessing Pipeline – SBP Masked Reconstruction

This document describes the preprocessing strategy used to generate training samples for **masked SBP (Spiking Band Power) reconstruction**.

**Goal:** Train a model that reconstructs masked neural channels within trials using surrounding neural context and kinematics.

---

## Dataset Structure

### Session Files

Each session contains:

- **`{session_id}_sbp.npy`** → shape `(N, 96)`  
  Spiking band power across 96 channels over N time bins.

- **`{session_id}_kinematics.npy`** → shape `(N, 4)`  
  Finger kinematics aligned with SBP bins.

- **`{session_id}_trial_info.npz`**
  - `start_bins`: shape `(n_trials,)`
  - `end_bins`: shape `(n_trials,)`
  - Each trial spans `[start_bins[i], end_bins[i])`.

### Metadata File

**`metadata.csv`** contains:
- `session_id`
- `split` (train/test)
- `day`
- `days_from_nearest_train`
- `n_bins`
- `n_trials`

---

## Objective

We generate fixed-length windows from each training session:

- **Window size:** `W = 128`
- **Windows per session:** `K = 500`

Each training sample contains:

- `x` → masked SBP window `(128, 96)`
- `y` → ground truth SBP window `(128, 96)`
- `kin` → kinematics window `(128, 4)`
- `mask_vec` → `(96,)` binary vector indicating masked channels
- `(a, b)` → trial span inside the window

---

## Why Trial-Based Windows?

Inspection of the test set shows:

- Masked SBP values are set to **0**
- Masking is **constant within a trial**
- Only a **small fraction** of trials are masked
- Most trials are **completely unmasked**

To match this structure during training, each sampled window is anchored around a complete trial, and masking is applied only within that trial.

---

## Window Sampling Strategy

For each training session:

1. Load SBP and kinematics arrays
2. Load trial boundaries
3. Sample `K = 500` windows

Each window:
- Has fixed size `W = 128`
- Contains at least one complete trial

### Selecting a Window

For a chosen trial `t`:

- **Trial span:** `[s_t, e_t)`
- **Trial length:** `L_t = e_t - s_t`
- **Requirement:** `L_t <= W`

Window start `w0` is sampled such that:

```
w0 <= s_t
e_t <= w0 + W
```

This guarantees the entire trial is inside the window.

**Window coverage:** `[w0, w0 + W)`

**Inside-window trial span:**
```
a = s_t - w0
b = e_t - w0
```

---

## Masking Strategy (Training Corruption)

For each sampled window:

Let:
```python
y = sbp[w0 : w0 + W]
x = y.copy()
```

**With probability** `p_mask_trial ≈ 0.03`:

1. Randomly select **30 channels** from 96
2. Zero those channels within the trial span:
   ```python
   x[a:b, masked_channels] = 0
   ```
3. Set:
   ```python
   mask_vec[channel] = 1
   ```

**Otherwise:**
- No masking is applied
- `mask_vec` remains zero

This mirrors test behavior:
- ✓ Masking is trial-consistent
- ✓ Masked values are zero
- ✓ Most trials remain unmasked

---

## Reproducibility

Each session uses a deterministic random seed derived from:

```python
seed + stable_hash(session_id)
```

This ensures:
- ✓ Reproducible window sampling
- ✓ Independent randomness per session
- ✓ No reliance on Python's unstable built-in hash

---

## Output Format

Each training sample contains:

| Component | Shape | Description |
|-----------|-------|-------------|
| Masked SBP window | `(128, 96)` | Input with masked channels |
| Ground truth SBP window | `(128, 96)` | Target for reconstruction |
| Kinematics window | `(128, 4)` | Behavioral context |
| Channel mask vector | `(96,)` | Binary mask indicator |
| Trial span | `(a, b)` | Trial boundaries within window |

These are passed into a PyTorch `Dataset` and `DataLoader` for training.

---

## Design Rationale

### Why 500 windows per session?

**Typical session length:** ~25,000–30,000 bins  
**Window size:** 128 bins

**500 windows provide:**
- ✓ Broad coverage of the session
- ✓ Trial diversity
- ✓ Balanced computational cost
- ✓ Stable batch sampling during training

---

## Future Extensions

Potential improvements:

- [ ] Add day embedding as conditioning input
- [ ] Match exact test masking probability from statistics
- [ ] Vary window sizes for ablations
- [ ] Use overlapping windows for stronger coverage
- [ ] Store preprocessed windows to disk for faster training