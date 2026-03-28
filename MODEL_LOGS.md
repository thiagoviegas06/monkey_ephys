# Model Training Logs

## Run 1: Initial Baseline
- **Date**: March 25, 2026
- **Configuration**:
    - `d_model`: 128
    - `nhead`: 8
    - `num_encoder_layers`: 4
    - `num_temporal_layers`: 2
    - `num_decoder_layers`: 2
    - `tcn_levels`: 6
    - `learning_rate`: 1e-3
    - `batch_size`: 32
    - `early_stopping_patience`: 5
    - **Masking**: Span lengths 45-85 (Buggy for window 200)
- **Results**:
    - **Best Val NMSE**: 0.771650 (Epoch 20)
    - **Final Val NMSE**: 0.776108 (Epoch 25 - Early Stopping)
    - **Note**: Model was straining GPU memory and likely over-masked due to the span length bug.

---

## Run 2: Optimized Configuration
- **Date**: March 25, 2026
- **Configuration**:
    - `d_model`: 128
    - `nhead`: 8
    - `num_encoder_layers`: 5 (+1 for better spatial context)
    - `num_temporal_layers`: 2
    - `num_decoder_layers`: 2
    - `tcn_levels`: 5 (Reduced to save memory, RF ~125)
    - `learning_rate`: 7e-4 (Lower for stability)
    - `batch_size`: 32
    - `early_stopping_patience`: 8 (Increased for slower convergence)
    - **Masking**: Fixed span lengths 20-50 (Corrected for window 200)
- **Results**:
    - **Best Val NMSE**: 0.761353 (Epoch 27)
    - **Train NMSE**: 0.746852 (Epoch 27)
    - **Status**: Plateauing after ~20 epochs and slowly improving after.
- **Key Insights**:
    - **Underfitting**: Very tight Train/Val gap (< 0.02) indicates the model has capacity to grow.
    - **Performance Ceiling**: Baseline NMSE (~0.77) is dominated by Phase 2-style channel dropout (full channel masking).
- **Next Steps**:
    - Increase `d_model` to 192 and `num_axial_layers` to 6.
    - Log Channel vs. Span NMSE separately to isolate the bottleneck.

---

## Run 3: Kinematics Decoder Phase 2 (Perceiver-style Decoder)
- **Date**: March 28, 2026
- **Model**: `KinematicDecoderTransformer` with MAE Embeddings
- **MAE Base**: `best_model_tcn_transformer.pt` (Phase 1)
- **Configuration**:
    - `window_size`: 200
    - `d_model`: 192
    - `nhead`: 8 (Axial)
    - `num_axial_layers`: 4
    - `num_decoder_layers`: 2
    - `tcn_levels`: 4
    - `decoder_num_temporal_layers`: 2
    - `decoder_num_heads`: 8
    - `decoder_dropout`: 0.1
    - `use_mae_embeddings`: True (`mae_embedding_type='full'`)
    - `learning_rate`: 3e-4
    - `batch_size`: 128
    - `weight_decay`: 1e-4
    - `kin_recon_weight`: 1.0
    - `correlation_weight`: 0.1
    - `acceleration_weight`: 0.01
- **Results**:
    - **Best Val MSE**: 0.1892 (Epoch 50)
    - **Val R2**: 0.8041
    - **Train MSE**: 0.0710 (Epoch 50)
    - **Train R2**: 0.9306
- **Key Insights**:
    - **Strong Performance**: R2 of 0.80 on validation is a very strong baseline, significantly outperforming the initial Phase 1 targets.
    - **Overfitting**: There is a noticeable gap between Train R2 (0.93) and Val R2 (0.80), suggesting the model is starting to memorize training session specifics.
    - **Smooth Convergence**: Loss decreased consistently throughout 50 epochs without hitting a hard plateau.
- **Next Steps**
  - Incerase number of decoder temporal layers from 2 to 6
  - Increase dropout or add weight decay to combat the emerging overfitting gap.
- **Other Potential Next Steps**:
    - Implement a "Velocity-Consistency" loss to link position and velocity predictions.
    - Explore Cross-Attention for channel aggregation instead of a large linear projection.
    