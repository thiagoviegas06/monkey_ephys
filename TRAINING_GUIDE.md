# Intracortical Decoding - Training & Implementation Guide

This project is divided into two phases: **Phase 1 (Masked Autoencoder)** for neural signal reconstruction and **Phase 2 (Kinematic Decoder)** for finger kinematic prediction from neural activity.

---

## Phase 1: Neural Reconstruction (`src_mae`)

Phase 1 focuses on reconstructing masked Spiking Band Power (SBP) channels using an interleaved Axial TCN + Transformer architecture.

### Quick Start
```bash
# Train the MAE model (standard window size 200)
python src_mae/train.py --window-size 200

# Evaluate and generate submission for Phase 1
python src_mae/eval.py
```

### Core Components
- **`src_mae/config.py`**: Central "source of truth" for MAE hyperparameters.
- **`src_mae/model.py`**: Contains the `SBP_TCN_Transformer`. It alternates between Temporal Attention (across time) and Spatial Attention (across channels).
- **`src_mae/dataloader.py`**: Implements dynamic windowing and a mixed masking strategy (50% Channel Masking, 50% Span Masking). 
- **`src_mae/losses.py`**: Implements `kaggle_aligned_nmse_loss`, which computes NMSE per (session, channel).

### Key Design Decisions
1. **Interleaved Axial Attention**: Decouples temporal and spatial processing, allowing the model to learn complex dependencies across both dimensions iteratively.
2. **Local TCN Feature Extraction**: Uses dilated convolutions to capture local temporal context per channel before applying global attention.
3. **Mixed Masking Strategy**: Training with both full-channel dropouts and temporal spans ensures the model is robust to both sensor failure and transient signal loss.

---

## Phase 2: Kinematic Decoding (`src_kin`)

Phase 2 uses the trained Phase 1 MAE model as a feature extractor. The decoder predicts finger kinematics by processing internal neural representations.

### Quick Start
```bash
# Ensure you have a Phase 1 checkpoint (e.g., in checkpoints_200/)
# Update src_kin/config.py: mae_checkpoint_path = "checkpoints_200/best_model_tcn_transformer.pt"

# Train the Kinematic Decoder
python src_kin/train.py

# Evaluate and generate Phase 2 submission
python src_kin/eval.py
```

### Core Components
- **`src_kin/model.py`**: `KinematicDecoderTransformer`.
    - **Channel Attention**: A Squeeze-and-Excitation style module that learns which SBP channels are most relevant for movement. 
    - **Temporal Attention**: A Transformer Encoder that models movement dynamics over the time window.
- **`src_kin/train.py`**: Processes neural data through the **frozen** Phase 1 MAE to extract embeddings (`use_mae_embeddings = True`).
- **`src_kin/losses.py`**: Multi-objective loss function:
    $$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_1 \cdot \mathcal{L}_{\text{Pearson}} + \lambda_2 \cdot \mathcal{L}_{\text{Acceleration\_Penalty}}$$
    The acceleration penalty ensures smooth, realistic finger trajectories.
- **`src_kin/dataloader.py`**: Automatically identifies active channels (non-zero columns) and handles kinematic normalization.   

### Key Design Decisions
1. **MAE as Feature Extractor**: Instead of just using the imputed SBP, the decoder can utilize the high-dimensional latent representations (embeddings) from the MAE, which capture richer neural context.
2. **Learned Channel Weighting**: Since ~30% of channels are dropped per session, the `ChannelAttention` mechanism allows the model to dynamically prioritize reliable neural inputs.
3. **Smoothness Regularization**: Penalizing high acceleration in the output helps prevent "jittery" predictions common in high-frequency neural decoding.

---

## Monitoring & Performance

| Phase | Metric | Initial | Target | Note |
|-------|--------|---------|--------|------|
| **1 (MAE)** | NMSE | ~20.0 | < 1.0 | Lower is better (session-grouped). |
| **2 (KIN)** | Pearson R | ~0.40 | > 0.85 | Correlation between prediction and ground truth. |

### Debugging & Visualization
- Use `visualize_prediction.py` to inspect MAE reconstruction quality.
- Use `visualize_kin_prediction.py` to compare predicted vs. actual finger positions.

---

## Output Structure

```
checkpoints_200/           # Phase 1 Weights
|-- best_model_tcn_transformer.pt
+-- ...

checkpoints_kin/           # Phase 2 Weights
|-- best_model_perceiver.pt
+-- ...

submission_eval.csv        # Phase 1 Submission
submission_phase2.csv      # Phase 2 Submission
```
