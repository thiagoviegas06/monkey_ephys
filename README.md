# Intracortical Decoding - Phase 1 & 2

This repository contains a two-phase pipeline for intracortical neural decoding. 
- **Phase 1**: Masked Autoencoder (MAE) for reconstructing missing/masked Spiking Band Power (SBP) neural channels.
- **Phase 2**: Kinematic Decoder for predicting finger positions from neural activity (with drift and channel dropout).

---

## Architectures

### Phase 1: MAE Reconstruction (`SBP_TCN_Transformer`)
The goal is to recover 30 randomly masked channels in 96-channel SBP data.
- **Local TCN Feature Extraction**: A Temporal Convolutional Network with dilated layers extracts local temporal features per channel independently.
- **Interleaved Axial Attention**: Alternates between:
    - **Temporal Attention**: Self-attention over time bins (across $W=200$).
    - **Spatial Attention**: Self-attention across channels (across $C=96$).
- **Asymmetric Decoder**: A lightweight transformer decoder that reconstructs the unmasked SBP values from the encoded representations.

### Phase 2: Kinematics Decoding (`KinematicDecoderTransformer`)
The goal is to predict finger positions from SBP data that has ~30% channel dropout.
- **MAE Embeddings**: Instead of raw SBP, the decoder utilizes the high-dimensional latent representations from a frozen Phase 1 model.
- **Channel Attention**: A Squeeze-and-Excitation style module learns per-channel importance weights, allowing the model to focus on reliable (non-dropped) channels.
- **Temporal Attention**: A transformer encoder models the dynamics of the aggregated channel features over the time window.      
- **Linear Output**: A final projection to kinematic coordinates (index and MRP positions/velocities).

---

## Visualization Tools

Visualization is key to debugging neural reconstructions and decoding performance.

### 1. MAE Reconstruction (Phase 1)
Visualize how the MAE fills in missing neural data.
```bash
python visualize_prediction.py --session-id S201
```
- Displays: **Ground Truth** vs. **Reconstruction** vs. **Error Map**.
- Highlights masked time-spans and computes NMSE for the window.

### 2. Phase 2 Data Inspection
Inspect the raw Phase 2 training/test windows, including channel dropout masks.
```bash
python visualize_phase2_window.py --session-id D001
```
- Shows SBP heatmaps (marking inactive channels in white) and aligned kinematics.

### 3. Kinematic Prediction (Phase 2)
Compare predicted finger positions against ground truth for validation samples.
```bash
python visualize_kin_prediction.py --session-id D042
```
- Plots predicted vs. actual trajectories for **Index Position** and **MRP Position**.

### 4. Submission Evaluation
Visualize the predictions stored in a final Kaggle submission CSV.
```bash
python visualize_eval_submission.py --submission-csv submission_eval.csv --session-id S008
```
- Reconstructs the full neural session from the CSV entries to visualize the "patched" SBP.

---

## Project Structure

- `src_mae/`: Source code for Phase 1 (Model, Train, Eval, Dataloader).
- `src_kin/`: Source code for Phase 2 (Model, Train, Eval, Dataloader).
- `kaggle_data/`: Root directory for Phase 1 data.
- `kaggle_data_phase2/`: Root directory for Phase 2 data.
- `checkpoints_200/`: Storage for Phase 1 model weights.
- `checkpoints_kin/`: Storage for Phase 2 model weights.

---

## Running the Pipeline

For detailed instructions on training and hyperparameters, please refer to the [TRAINING_GUIDE.md](TRAINING_GUIDE.md).
