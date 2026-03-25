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
    - `batch_size`: 128
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
    - `batch_size`: 128
    - `early_stopping_patience`: 8 (Increased for slower convergence)
    - **Masking**: Fixed span lengths 20-50 (Corrected for window 200)
- **Results**:
    - *Awaiting training results*
- **Improvements**: Corrected masking logic, slightly deeper spatial transformer, and optimized TCN receptive field for better memory/performance balance.
