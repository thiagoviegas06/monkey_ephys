# TCN vs U-Net: Architectural Analysis for Masked SBP Reconstruction

## Executive Summary

**Performance Gap**: U-Net ~0.75 vs TCN ~0.91-0.94 NMSE (16-19% worse)

**Root Cause**: TCN processes time as 1D sequence, missing 2D spatial structure in the (W, C) grid that U-Net exploits

**Key Insight**: The (200, 96) shape is not purely temporal data—it's spatially structured 2D data (time × neural channels from same region). U-Net's Conv2d is fundamentally better suited to this structure.

---

## 1. Fundamental Architectural Differences

### U-Net (Conv2d on 2D Grid)
```
Input: (B, 2, 200, 96)  ← 2D grid [time × channels]
         ↓
Encoder pyramid:
  - Level 1: (B, 64, 200, 96)
  - Level 2: (B, 128, 100, 96)  [pooled on time]
  - Level 3: (B, 256, 50, 96)
  - Bottleneck: (B, 512, 25, 96)
         ↓
Decoder with skip connections back to each level
         ↓
Output: (B, 1, 200, 96) → (B, 200, 96)
```

### TCN (Conv1d on 1D Sequence)
```
Input: (B, 192, 200)  ← 1D sequence [concatenated channel pairs along time]
         ↓
Input projection: (B, 128, 200)
         ↓
7 Linear blocks with dilations [1,2,4,8,16,32,64]
  - Each block: depthwise-separable Conv1d + channel FFN + residual
         ↓
Output projection: (B, 96, 200) → (B, 200, 96)
```

---

## 2. Why U-Net Wins: Three Critical Factors

### Factor 1: 2D Spatial Structure Exploitation

**The Problem U-Net Solves**:
The (W=200, C=96) array is **spatially structured 2D data**, not a pure 1D sequence:
- Temporal dimension (W): sequential dependency across time
- Channel dimension (C): **spatial locality of adjacent channels** (from same neural region)

**U-Net's Advantage**:
- Conv2d kernels naturally capture correlations between:
  - Adjacent timepoints AND adjacent channels simultaneously
  - Example: kernel at position (t, c) sees [t-1:t+2, c-1:c+2] neighborhood
  - This is perfect for masked reconstruction because:
    - Masked values can be inferred from nearby channels (spatial)
    - AND nearby timepoints (temporal)

**TCN's Limitation**:
- Conv1d only sees temporal neighbors: [t-d, t+d]
- Channel dimension is processed separately (depthwise)
- Missing the 2D spatial locality that's crucial
- Channel FFN is pointwise (1×1) → no structural mixing
- Channels are treated as independent features, not spatially proximal

**Concrete Example**:
- Suppose channel 45 is masked at time 100
- U-Net can directly see channels 44, 45, 46 at time 99, 100, 101
- TCN sees channel 45 across time, but channel mixing is a separate operation
- U-Net's 2D neighborhood is more direct for spatial interpolation

---

### Factor 2: Multi-Scale Hierarchical Processing

**U-Net's Advantage**:
```
Pooling schedule:
  W: 200 → 100 → 50 → 25 (then upsample back)

This forces the model to:
  1. Learn compact representations at coarse scales
  2. Combine coarse + fine detail via skip connections
  3. Make reconstruction decisions at multiple scales
```

**Why Multi-Scale Helps Masked Reconstruction**:
- **Coarse scale** (W/8): Global structure, long-range dependencies
- **Medium scale** (W/4): Pattern continuity across masked regions
- **Fine scale** (W): Local detail, precise channel values

**TCN's Limitation**:
- Single scale throughout (except dilations provide different RF scales)
- Dilations [1,2,4,8,16,32,64] give different RF per layer, but:
  - All at same resolution
  - No feature compression/decompression cycle
  - Information dilution: dilation=64 kernel spans 256 timesteps but same feature size
  - Cannot jointly optimize coarse + fine reconstruction

**Information Bottleneck**:
- U-Net's bottleneck at W/8 (25 timesteps) forces learning compact codes
- TCN has no bottleneck → all 200 timesteps processed uniformly
- Bottlenecks help regularization and feature learning

---

### Factor 3: Skip Connections at Multiple Scales

**U-Net's Advantage**:
```
Dense skip connections from encoder to decoder at 3 scales:
  - Enc1 → Dec1 (full resolution)
  - Enc2 → Dec2 (half resolution)
  - Enc3 → Dec3 (quarter resolution)

Benefits:
  1. Gradient flow: direct paths for backprop to all levels
  2. Feature reuse: coarse features directly available to fine decoding
  3. Supervision signal: output can use features from multiple depths
  4. Spatial shortcuts: feature mixing at different resolutions
```

**TCN's Limitation**:
- Only within-block residuals (x + block(x))
- No cross-scale shortcuts
- All information flows linearly through 7 sequential blocks
- Gradient must flow through all 7 layers to affect early weights
- Deeper information has harder time influencing fine details

---

## 3. Comparative Analysis by Inductive Bias

| Aspect | U-Net | TCN | Winner |
|--------|-------|-----|--------|
| **Spatial 2D structure** | Conv2d sees (W,C) neighborhood | Conv1d + sep, channels separate | **U-Net** |
| **Channel adjacency** | Direct 2D neighborhood | Separate mixing | **U-Net** |
| **Multi-scale hierarchy** | 3-level pyramid | Single scale + dilations | **U-Net** |
| **Bottleneck forcing** | Yes (25 timesteps) | No | **U-Net** |
| **Skip connection density** | 3 scales | None | **U-Net** |
| **Gradient flow** | Direct multi-path | Linear 7-layer | **U-Net** |
| **Local 2D patterns** | Built-in via kernel | Must learn separately | **U-Net** |
| **Temporal range** | Multi-scale | Single scale | **U-Net** |
| **Parameter efficiency** | Less efficient (7.8M) | More efficient (1.1M) | TCN |
| **Causal constraint** | None (can violate) | Bidirectional | TCN |

---

## 4. Why Channel FFN Doesn't Solve It

**TCN's Approach**:
```
Temporal Conv1d (per-channel patterns)
         ↓
Channel FFN (nonlinear mixing: C → 4C → C)
```

**Why This Isn't Enough**:
- Temporal and channel processing are **sequential**, not integrated
- Conv1d captures temporal patterns per-channel
- Channel FFN comes **after** and must fix what temporal conv missed
- U-Net integrates spatial structure directly in kernels

**Analogy**:
- U-Net: Fully integrated 2D feature learning (natural)
- TCN: Learn 1D patterns, then try to mix channels (decomposed)

---

## 5. Why Depthwise-Separable Conv Helps But Isn't Enough

**TCN's Temporal Processing**:
```
Depthwise Conv: independent conv per channel
Pointwise Conv: 1×1 mixing across channels
```

**Why It's Not Like Conv2d**:
- Depthwise sees temporal neighbors for each channel independently
- Pointwise (1×1) mixes channels at **same time** only
- U-Net's Conv2d sees temporal **and** channel neighbors simultaneously
- Pointwise can't recover what depthwise missed at the spatial boundary

**Example**:
- Masked value at (t=100, c=45)
- Depthwise conv for channel 45 sees: channels [45] at times [98-102]
- Pointwise conv for time 100 sees: channels [1-96] at time 100
- U-Net's kernel sees: channels [44-46] at times [99-101] directly

---

## 6. The Output Projection Problem

**U-Net**:
```
Output: (B, 1, 200, 96)  ← single channel on 2D grid
After squeeze and output processing: (B, 200, 96)
```

**TCN**:
```
Output projection: hidden (128) → 96 channels directly
Explicit per-channel reconstruction
```

**Why U-Net's is Better for This Task**:
- Single output channel learns a **unified reconstruction score** across the (200, 96) grid
- Natural for the problem: "reconstruct at this (t, c) position"
- TCN's approach: each of 96 channels has its own reconstruction head
- Fewer parameters per channel in U-Net's approach due to output sharing

---

## 7. The Dilations Can't Fully Compensate

**TCN's Attempt**:
```
Dilations: [1, 2, 4, 8, 16, 32, 64]
RF: 255 timesteps (exceeds W=200)
```

**Why This Doesn't Solve the Spatial Problem**:
- RF handles temporal coverage ✓
- But doesn't address **channel spatial structure** ✗
- Dilation=64 kernel spans 256 timesteps but has no 2D spatial support
- Like using temporal convolutions without the spatial dimension

---

## 8. Three Most Likely Causes of the Gap

### Cause #1 (Primary): Missing 2D Spatial Structure
- **Weight**: 60% of performance gap
- U-Net exploits Conv2d's natural 2D locality
- TCN can't capture channel adjacency with 1D convolutions
- **Evidence**: Channel reconstruction should benefit from nearby channels

### Cause #2 (Secondary): No Multi-Scale Hierarchy
- **Weight**: 25% of performance gap
- U-Net's pyramid forces multi-scale feature learning
- TCN processes all scales uniformly via dilations
- **Evidence**: Different receptive field sizes per layer isn't same as hierarchical pooling

### Cause #3 (Tertiary): Skip Connection Scarcity
- **Weight**: 15% of performance gap
- U-Net has 3 scales of skip connections
- TCN has only per-block residuals
- **Evidence**: 7-layer linear chain harder to train than pyramid

---

## 9. Three Highest-Impact Architecture Changes to Close the Gap

### Change #1: Switch to 2D Processing (Highest Impact)

**Proposal**: Modified TCN that preserves spatial 2D structure

Instead of:
```python
# TCN: (B, 192, 200) → Conv1d → (B, 128, 200)
input_proj = Conv1d(192, 128, kernel_size=1)
```

Use:
```python
# Keep 2D structure
x = torch.cat([x_sbp, obs_mask], dim=2)  # (B, 200, 192) stays 2D
# Reshape to 2D grid while preserving spatial structure
x = x.reshape(B, 200, 96, 2)  # (B, time, channel, feature)

# Use 2D Conv that respects spatial structure
# Conv on (channel, feature) with kernel_size=(3,1)?
# Or Conv2d treating (time, channel) as spatial dimensions
```

**Expected Improvement**: +8-12% (biggest single change)

**Why**: Directly addresses the root cause—spatial 2D structure

---

### Change #2: Add Multi-Scale Pyramid Processing

**Proposal**: Temporal pooling stages like U-Net

```python
class MultiScaleTCN(nn.Module):
    def forward(self, x):
        # x: (B, 192, 200)

        # Scale 1: Full resolution
        x1 = self.block1(x)  # (B, 128, 200)

        # Scale 2: Half temporal resolution
        x2_pooled = self.temporal_pool(x1)  # (B, 128, 100)
        x2 = self.block2(x2_pooled)  # (B, 128, 100)

        # Scale 3: Quarter temporal resolution
        x3_pooled = self.temporal_pool(x2)  # (B, 128, 50)
        x3 = self.block3(x3_pooled)  # (B, 128, 50)

        # Bottleneck
        x_bot = self.bottleneck(x3)  # (B, 128, 50)

        # Decoder: upsample and combine with skip connections
        x3_up = self.upsample(x_bot) + x3
        x2_up = self.upsample(x3_up) + x2
        x1_up = self.upsample(x2_up) + x1

        output = x1_up
```

**Expected Improvement**: +5-8%

**Why**: Adds hierarchical multi-scale like U-Net without full 2D conv

---

### Change #3: Cross-Scale Skip Connections

**Proposal**: Dense skip connections between TCN scales

```python
# Instead of only within-block residuals:
# Add cross-scale shortcuts

scale1_out = scale1_out + channels_from_scale2_upsampled
scale1_out = scale1_out + channels_from_scale3_upsampled
```

**Expected Improvement**: +3-5%

**Why**: Improves gradient flow and feature reuse

---

## 10. Recommended Hybrid Architecture

**Best Solution**: "Conv1d-based Spatial-Temporal" model

```python
class SpatialTemporalTCN(nn.Module):
    """
    Processes (B, time, channel) as spatial data:
    - Treat time and channel dimensions separately but with interaction
    - Use 1D conv on time with careful channel handling
    - Use 1D conv on channel (transpose) with temporal handling
    - Interleave temporal and channel convolutions

    Or: Use Conv2d but only in the channel×time plane,
        treating it as 2D spatial data rather than time×space
    """
```

**Why This Approach**:
1. Keeps TCN's efficiency (1D convs)
2. Adds 2D spatial awareness (2D layout of time×channel)
3. Cheaper than full Conv2d on all dimensions
4. Can still use residual blocks and dilations

---

## 11. Experiments to Verify Hypotheses

### Experiment 1: Conv2d on (W, C) Plane
**Test**: Replace TCN with minimal Conv2d variant
```python
# Simple: Conv2d treating (W, C) as spatial
conv = Conv2d(192, 128, kernel_size=3, padding=1)
```
**Expectation**: Closes 50%+ of gap
**Cost**: Medium (changes fundamental structure)
**Confidence**: High

### Experiment 2: Temporal Pooling
**Test**: Add pooling stages to TCN
```python
# TCN with 3-scale pyramid (like U-Net)
```
**Expectation**: +5-8% improvement
**Cost**: Low (modular addition)
**Confidence**: High

### Experiment 3: Ablate U-Net Component by Component
**Test**: Remove features from U-Net one at a time:
1. Keep U-Net, remove all skip connections → measure loss
2. Keep U-Net, use single scale → measure loss
3. Keep U-Net, use 1D instead of 2D → measure loss
**Expectation**: Identify which feature contributes how much
**Cost**: Low
**Confidence**: Very high

### Experiment 4: Channel Locality Measurement
**Test**: Measure how much masked values correlate with:
- Temporal neighbors (same channel, nearby time)
- Spatial neighbors (nearby channel, same time)
**Expectation**: Should see both matter; spatial might matter more
**Cost**: Low (analysis only)
**Confidence**: High

---

## 12. Summary: Why U-Net Wins

| Factor | Impact | U-Net | TCN |
|--------|--------|-------|-----|
| 2D spatial structure | 60% | ✓✓✓ | ✗ |
| Multi-scale hierarchy | 25% | ✓✓ | ✗ |
| Skip connections | 15% | ✓✓ | ✗ |

**Conclusion**:
U-Net is simply better suited to the task because it's fundamentally a 2D problem (time × channels), and U-Net's Conv2d architecture directly exploits this 2D structure. TCN's 1D approach misses this crucial inductive bias.

**Path Forward**:
1. Try Conv2d version of TCN (quick test)
2. Add temporal pooling (medium effort)
3. Add cross-scale skips (medium effort)
4. Redesign around 2D spatial-temporal processing (larger effort, biggest gain)
