# Intracortical Decoding - Training & Implementation Guide

This project is divided into two phases: **Phase 1 (Masked Autoencoder)** for neural signal reconstruction and **Phase 2 (LFADS Decoder)** for finger kinematic prediction from neural activity.

---

## Phase 1: Neural Reconstruction (`src_mae`)

Phase 1 focuses on reconstructing masked Spiking Band Power (SBP) channels using a hybrid TCN + Cross-Channel Transformer.

### Quick Start
```bash
# Train the MAE model
python src_mae/train.py --window-size 200

# Evaluate and generate submission for Phase 1
python src_mae/eval.py --window-size 200
```

### Core Components
- **`src_mae/config.py`**: Central "source of truth" for MAE hyperparameters.
- **`src_mae/model.py`**: Contains the `SBP_TCN_Transformer`. Uses Reversible Instance Normalization (RevIN) for drift robustness.
- **`src_mae/dataloader.py`**: Dynamic windowing and on-the-fly masking of neural data.
- **`src_mae/losses.py`**: Implements `kaggle_aligned_nmse_loss`, matching the competition metric.

### Key Design Decisions
1. **Hybrid Architecture**: TCN extracts temporal features independently per channel; Transformer models cross-channel correlations.
2. **RevIN**: Normalizes signals based on visible (unmasked) channels only, making the model robust to absolute power drifts.
3. **Session-Grouped Loss**: NMSE is computed per (session, channel) to align with Kaggle's scoring.

---

## Phase 2: Kinematic Decoding (`src_kin`)

Phase 2 uses the best Phase 1 model as a pre-processor to "fix" neural data before decoding kinematics using a Latent Factor Analysis via Dynamical Systems (LFADS) approach.

### Quick Start
```bash
# Ensure you have a Phase 1 checkpoint at:
# checkpoints_200/best_model_tcn_transformer.pt

# Train the Kinematic Decoder
python src_kin/train.py

# Evaluate and generate Phase 2 submission
python src_kin/eval.py
```

### Core Components
- **`src_kin/model.py`**: `LFADSKinematicDecoder`. A VAE-based architecture:
    - **Encoder**: Bidirectional GRU reading the sequence.
    - **Generator**: Autonomous GRUCell modeling neural dynamics.
    - **Decoders**: Maps latent factors to both Kinematics (4-ch) and SBP (96-ch).
- **`src_kin/train.py`**: Implements a **two-stage pipeline**:
    1. **Impute**: Pass masked SBP through the frozen Phase 1 MAE.
    2. **Decode**: Pass the "clean" SBP through the Phase 2 LFADS.
- **`src_kin/losses.py`**: Complex VAE loss function:
    $$\mathcal{L} = \mathcal{L}_{\text{Kin\_MSE}} + \beta \cdot D_{\text{KL}} + \alpha \cdot \mathcal{L}_{\text{SBP\_MSE}}$$
- **`src_kin/dataloader.py`**: Handles Phase 2 data paths. Automatically infers channel dropout masks by identifying all-zero columns.

### Key Design Decisions
1. **Multi-Task Regularization**: The model is forced to reconstruct the imputed neural data alongside kinematics. This "self-supervision" helps the latent factors capture the true underlying manifold.
2. **Beta Annealing**: $\beta$ starts at 0.0 and increases linearly over 2000 steps to prevent **Posterior Collapse** (where the generator ignores the encoder).
3. **Velocity-Inclusive**: Although only positions are scored, we decode all 4 kinematic variables to provide richer feedback for the dynamical system.

---

## Monitoring & Performance

| Phase | Metric | Initial | Target | Note |
|-------|--------|---------|--------|------|
| **1 (MAE)** | NMSE | ~20.0 | < 1.0 | Lower is better. |
| **2 (KIN)** | MSE | ~0.50 | < 0.15 | Pearson R is often used for validation. |

### Debugging & Visualization
- Use `visualize_prediction.py` to see Phase 1 reconstructions.
- Check `beta` values in the `src_kin/train.py` progress bar; if $\beta$ reaches `max_beta` too early, increase `beta_anneal_steps` in `config.py`.

---

## Output Structure

```
checkpoints_<window_size>/       # Phase 1 Weights
├── best_model_tcn_transformer.pt
└── ...

checkpoints_kin/       # Phase 2 Weights
├── best_model_lfads.pt
└── ...

submission_phase2.csv  # Final Output for Kaggle
```
