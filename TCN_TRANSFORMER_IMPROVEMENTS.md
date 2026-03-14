# SBP_TCN_Transformer: Architectural Bottlenecks & Improvements

## Current Architecture (Lines 368-482)

```
Input → Normalization → Channel-Independent TCN → Macro-Time Embedding
    ↓
Transformer (Cross-Channel, ONE layer per timestep) → Output Projection (d_model→1) → Denorm
```

### The Problem: Transformer is Capacity-Starved

You observe that increasing TCN layers improves performance, but this might mask a **transformer bottleneck**. Here's why:

1. **TCN dominates signal processing** (Phase 1):
   - Each of 96 channels gets its own independent TCN
   - Large TCN (e.g., 20 levels) learns rich temporal patterns per channel
   - Output: (B, C, d_model, W) with full temporal context

2. **Transformer has minimal capacity** (Phase 3):
   - Operates on reshaped (B*W, C, d_model) = processes 96 channels at each timestep
   - `d_model=64, nhead=8` → **8 dims per head** (very constrained)
   - Only **4 transformer layers** by default
   - **Abrupt output projection**: `Linear(d_model=64, 1)` with gain=0.1 initialization
   - No intermediate layers to "expand" before final prediction

3. **Information bottleneck**:
   ```
   TCN Rich Features (d_model, W)
       ↓
   Reshape to Process Cross-Channel (loses temporal context in attention)
       ↓
   Transformer (4 layers, 64 dims) = tiny capacity compared to TCN
       ↓
   Linear(64 → 1) = aggressive compression
   ```

---

## Root Causes

### Issue 1: Transformer Orientation
**Current**: Transformer processes **cross-channel** relationships at each timestep independently.
- Shape: (B*W, C, d_model) = treat each timestep separately
- ❌ Timesteps don't see each other's attention patterns
- ❌ Temporal patterns are learned by TCN, not reinforced by transformer

**Better**: Transformer should process **temporal** relationships with channels as features.
- Shape: (B*C, W, d_model) = treat each channel separately in transformer
- ✅ Self-attention captures long-range temporal dependencies
- ✅ Complements TCN's local temporal modeling

### Issue 2: Capacity Constraints
**Current**:
```python
d_model = 64
nhead = 8
num_layers = 4
output_proj = Linear(64, 1)  # gain=0.1 (very conservative init)
dim_feedforward = d_model * 4 = 256
```

- Total transformer params: ~4 layers × (d_model² + ffn) ≈ **100K params**
- TCN (with 20 levels): ~500K+ params
- **Ratio: TCN is 5-10x larger than transformer**

**Better**: Balance capacity.
```python
d_model = 128-256
nhead = 16
num_layers = 8-12
output_proj = intermediate projections (64 → 32 → 1)
dim_feedforward = d_model * 8 = 1024-2048
```

- Transformer params: **500K-2M**
- Roughly equal to TCN, better feature utilization

### Issue 3: Output Projection is Too Aggressive
**Current**:
```python
self.output_proj = nn.Linear(d_model, 1)  # 64 → 1 with gain=0.1
```

- Collapses 64-dimensional learned representation to scalar in one step
- Heavy initialization gain (0.1) suppresses signal early in training
- No opportunity for the model to refine predictions before output

**Better**: Multi-layer projection.
```python
self.output_proj = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1)
)
```

### Issue 4: TCN-Transformer Communication is One-Way
**Current**:
- TCN outputs fixed features
- Transformer processes them with no gradient feedback to TCN initialization
- No explicit gating between TCN and transformer pathways

**Better**: Allow complementary processing.
```python
# Option A: Learnable blending
gate = sigmoid(mlp(tcn_out))
output = gate * tcn_out + (1 - gate) * transformer_out

# Option B: Early channel mixing
# Add an intermediate cross-channel layer BEFORE transformer to boost signal
x = cross_channel_conv(tcn_out)  # 96 channels → 96 channels
x = transformer(x)
```

### Issue 5: Kinematics Are Silently Used
**Current** (line 448):
```python
x_tcn = torch.cat([sbp_exp, mask_exp, kin_exp], dim=-1)
```

- Kinematics are just concatenated features in TCN
- No separate attention or fusion mechanism
- Transformer never directly sees kinematics

**Better**: Explicit multi-modal fusion.
```python
# Kinematics stream: independent processing
kin_tcn = tcn_kin(kin_exp)  # (B, C, d_model, W)

# Cross-modal fusion
fused = transformer_with_cross_attention(
    query=sbp_features,
    key=kin_features,
    value=kin_features
)
```

---

## Recommended Improvements (Ranked by Impact)

### **Tier 1: High Impact, Low Risk** ⭐⭐⭐

#### Improvement 1A: Increase Transformer Capacity
```python
class SBP_TCN_Transformer(nn.Module):
    def __init__(self, sbp_channels=96, kin_channels=4,
                 d_model=256,        # ← 64 → 256
                 nhead=16,           # ← 8 → 16
                 num_layers=8,       # ← 4 → 8
                 tcn_levels=4,
                 dropout=0.1):
        ...
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 6,  # ← 4 → 6 (more FFN capacity)
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True
            ),
            num_layers=num_layers
        )
        ...
        # Better output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1)
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.zeros_(self.output_proj[0].bias)
        nn.init.xavier_uniform_(self.output_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.output_proj[3].bias)
```

**Why this works**:
- ✅ Matches TCN capacity for proper gradient flow
- ✅ Multi-step output projection allows model to "think" before deciding
- ✅ Minimal implementation change

**Expected impact**: +3-5% improvement (based on capacity ratios)

---

#### Improvement 1B: Reorient Transformer for Temporal Processing
**Instead of**: Process cross-channel at each timestep
**Do this**: Process temporal at each channel

```python
# In forward(), replace Phase 3:

# PHASE 3: TEMPORAL TRANSFORMER (not cross-channel)
# Current shape: (B, W, C, d_model)
x = x.permute(0, 2, 1, 3)  # (B, C, W, d_model)
x = x.reshape(B * C, W, self.d_model)  # ← Process time dimension
x = self.pre_transformer_norm(x)
x = self.transformer_encoder(x)  # ← Learns temporal patterns
x = x.reshape(B, C, W, self.d_model)

# PHASE 4: CROSS-CHANNEL REFINEMENT (optional, for fusion)
# If keeping cross-channel, do it AFTER temporal processing
x = x.reshape(B * W, C, self.d_model)  # (B*W, C, d_model)
x = self.cross_channel_attn(x)  # NEW: lightweight cross-channel
x = x.reshape(B, W, C, self.d_model)
```

**Why this works**:
- ✅ Transformer learns long-range temporal dependencies (what TCN's receptive field might miss)
- ✅ Keeps local temporal context separate from cross-channel relationships
- ✅ More interpretable: "which timepoints are similar across channels"

**Expected impact**: +2-4% improvement (captures complementary patterns)

---

### **Tier 2: Medium Impact, Medium Risk** ⭐⭐

#### Improvement 2A: Add Learnable Gate Between TCN and Transformer
```python
# In __init__:
self.gate_fcn = nn.Sequential(
    nn.Linear(d_model, d_model // 4),
    nn.GELU(),
    nn.Linear(d_model // 4, 1),
    nn.Sigmoid()
)

# In forward():
# After transformer (line 469):
gate_weights = self.gate_fcn(x)  # (B*W, C, 1)
x_gated = gate_weights * x  # Emphasize transformer contribution
# Or blend:
# x = x_tcn_features + gate_weights * (transformer_out - x_tcn_features)
```

**Why this works**:
- ✅ Let the model learn when transformer is useful vs. trusting TCN
- ✅ Soft skip connection prevents gradient vanishing

**Expected impact**: +1-2% improvement

---

#### Improvement 2B: Stronger Channel Embeddings
```python
# Current (line 391-393):
self.channel_embeddings = nn.Parameter(torch.zeros(1, sbp_channels, d_model))
nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)  # Very small

# Better:
self.channel_embeddings = nn.Parameter(torch.randn(sbp_channels, d_model) * 0.1)

# AND use them in TCN phase:
# After TCN output (line 451):
x = tcn_out.view(B, C, self.d_model, W)  # (B, C, d_model, W)
x = x + self.channel_embeddings.view(1, C, self.d_model, 1)  # Broadcast
```

**Why this works**:
- ✅ Channels have learnable identity from the start
- ✅ TCN already knows "which channel am I processing"
- ✅ Helps with generalization across channels

**Expected impact**: +0.5-1% improvement

---

### **Tier 3: High Impact, Higher Risk** ⭐⭐⭐ (Experimental)

#### Improvement 3A: Dual-Stream Architecture (TCN + Transformer)
```python
# Process kinematics SEPARATELY
self.kin_encoder = nn.Sequential(
    nn.Linear(kin_channels, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, d_model)
)

# In forward, PHASE 1:
kin_features = self.kin_encoder(kin_norm)  # (B, W, d_model)

# PHASE 3: Multi-modal fusion
sbp_feat = tcn_out  # (B, C, d_model, W)
kin_feat = kin_features.unsqueeze(1)  # (B, 1, W, d_model)

# Broadcast kinematics across all channels
x = sbp_feat + kin_feat  # Gating fusion
x = x.reshape(B*W, C, d_model)
x = self.transformer_encoder(x)
```

**Why this works**:
- ✅ Explicitly models kinematics as a cross-channel signal
- ✅ Prevents kinematics from being "lost" in channel-independent processing

**Expected impact**: +2-3% improvement

---

#### Improvement 3B: Add Residual Blocks in Transformer
```python
# Instead of pure TransformerEncoder, use:
class TransformerWithResiduals(nn.Module):
    def __init__(self, d_model, nhead, num_layers, ...):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(...) for _ in range(num_layers)
        ])
        # Learnable residual strength
        self.residual_strength = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, strength in zip(self.layers, self.residual_strength):
            x_new = layer(x)
            x = x + strength * (x_new - x)  # Learnable residual
        return x
```

**Why this works**:
- ✅ Helps with training stability in deeper networks
- ✅ Model learns optimal skip connection strength

**Expected impact**: +1-2% (mainly for stability)

---

## Implementation Priority

### Phase 1 (Week 1): Low-Risk High-Gain
```
1. Improvement 1A (Capacity)
   - Change: d_model=256, nhead=16, num_layers=8, multi-layer output_proj
   - Train & validate

2. Improvement 2B (Channel Embeddings)
   - Use in TCN phase
   - Retrain
```
**Expected**: +3-5% improvement

### Phase 2 (Week 2): Medium Complexity
```
3. Improvement 1B (Temporal Transformer)
   - Reorient transformer to process time
   - Careful reshaping to avoid bugs

4. Improvement 2A (Gating)
   - Add learnable blend
```
**Expected**: +2-4% more improvement

### Phase 3 (Week 3+): Experimental
```
5. Improvement 3A/3B if gains plateau
```

---

## Quick Wins (Do First)

If you want **minimal changes** for **quick improvement**:

```python
# Edit lines 369-414 in model.py

class SBP_TCN_Transformer(nn.Module):
    def __init__(self, sbp_channels=96, kin_channels=4,
                 d_model=256,      # ← UP from 64
                 nhead=16,         # ← UP from 8
                 num_layers=8,     # ← UP from 4
                 tcn_levels=4,
                 dropout=0.1):
        ...
        # Better output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
```

**Changes needed**: ~5 lines in `__init__`
**Training overhead**: ~2-3x more compute (larger d_model)
**Expected gain**: +3-5%

---

## Why These Improvements Work

| Problem | Improvement | Why It Helps |
|---------|------------|-------------|
| TCN dominates | Increase transformer capacity | Fair competition between pathways |
| Temporal lost in cross-channel | Reorient to temporal | Recover long-range time patterns |
| Output too aggressive | Multi-layer projection | More capacity for final refinement |
| Channels don't interact early | Use embeddings in TCN | Channel identity established early |
| Kinematics ignored | Explicit fusion | Proper multi-modal integration |

---

## Testing Strategy

1. **Baseline**: Current model, note NMSE loss
2. **Test 1A**: Just capacity increase (d_model, nhead, num_layers)
   - If +2-5%, good signal that transformer was bottleneck ✓
   - If no gain, transformer wasn't the problem (maybe need Improvement 1B)
3. **Test 1B**: Reorient transformer + 1A
   - Verify no crashes from reshaping
   - Check NMSE (should beat 1A alone)
4. **Test combinations**: 1A + 1B + 2A + 2B together
   - Should see cumulative gains

---

## Summary

**The bottleneck**: Your transformer is **tiny** (~100K params) compared to TCN (~500K+), with minimal output projection.

**The fix**:
- ✅ **Increase d_model**: 64→256 (capacity)
- ✅ **Increase layers**: 4→8+ (depth)
- ✅ **Multi-layer output**: 64→1 becomes 64→32→1
- ✅ **Consider temporal reorientation**: Process time, not just cross-channel

**Expected total gain**: **5-10%** if you implement Improvements 1A + 1B + 2B together.

