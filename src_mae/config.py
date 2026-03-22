import torch

# ============================================================================
# Configuration file for training - modify these parameters as needed
# ============================================================================
class Config:
    """Training configuration - modify these parameters"""
    
    # Data
    preprocess_type = "non_overlapping"  # Options: "non_overlapping", "overlapping_dynamic"
    data_path = "kaggle_data"
    window_size = 200
    seed = 42
    
    # Preprocessing
    preprocess = False  # Data is now loaded dynamically to improve training times. Same proces and seed are utilzied for reproducibility.
    windows_dir = f"kaggle_data/masked_windows_{window_size}"  # Where preprocessed windows are saved
    lag_bins = 0
    
    # Model
    model_name = "tcn_transformer"
    sbp_channels = 96
    
    # SBP_TCN_Transformer Hyperparameters
    d_model = 64  # Embedding dimension for transformer
    nhead = 8  # Number of attention heads
    num_layers = 8 # Number of transformer encoder blocks stacked on top of each other
    tcn_levels = 8  # Number of TCN dilation layers
    dropout = 0.1  # Dropout rate in transformer for regularization
    
    # Training
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 50
    early_stopping_patience = 5  # Stop if val loss doesn't improve for this many epochs
    early_stopping_min_delta = 5e-5  # Minimum change in val loss to qualify as an improvement
    
    # Masking
    # 3 fixed sets of 28 channels to be masked (approx 30% of 96)
    # These are used instead of random bin-level masking.
    fixed_mask_sets = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
        [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    ]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = f"checkpoints_{window_size}"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches
