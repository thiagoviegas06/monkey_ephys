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
    d_model = 192  # Embedding dimension for transformer
    nhead = 8  # Number of attention heads
    num_axial_layers = 4 # Number of interleaved temporal-spatial blocks
    num_decoder_layers = 2 # Number of transformer decoder blocks
    tcn_levels = 4  # Number of TCN dilation layers
    dropout = 0.15  # Dropout rate for regularization
    channel_mask_prob = 0.5 # Probability of full channel mask during training

    
    # Training
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 80
    early_stopping_patience = 8  # Stop if val loss doesn't improve for this many epochs
    early_stopping_min_delta = 5e-5  # Minimum change in val loss to qualify as an improvement
    
    # Stats
    STATS = [
        (51, 58.5, 82), (59, 64.0, 183), (56, 59.0, 70),
        (47, 54.5, 117), (52, 57.5, 152), (46, 59.0, 108),
        (63, 68.0, 104), (51, 55.5, 108), (54, 61.0, 114),
        (54, 61.5, 123), (54, 57.5, 86), (51, 54.0, 103),
        (38, 59.0, 119), (53, 60.5, 167), (51, 53.0, 58),
        (54, 59.0, 131), (53, 65.5, 72), (54, 65.5, 129),
        (55, 72.0, 101), (53, 69.5, 122), (37, 57.0, 128),
        (54, 62.5, 117), (55, 68.0, 121), (37, 58.0, 90),
    ]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = f"checkpoints_{window_size}"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches
