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
    preprocess = True  # Set True to run preprocessing (only needed once)
    windows_dir = f"kaggle_data/masked_windows_{window_size}"  # Where preprocessed windows are saved
    lag_bins = 0
    
    # Model
    model_name = "tcn_transformer"   # Options: "unet", "simple_cnn", "resnet", "transformer", "tcn_transformer"
    base_channels = 64   # For UNet/ResNet
    sbp_channels = 96
    kin_channels = 4
    
    # For transformer model
    d_model = 128  # Embedding dimension for transformer
    nhead = 8  # Number of attention heads
    num_layers = 6 # Number of transformer encoder blocks stacked on top of each other
    tcn_levels = 8  # Number of TCN dilation layers
    dropout = 0.05  # Dropout rate in transformer for regularization
    use_prior = True  # Whether to use the prior in the TCN-Transformer model

    # Training
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 80
    early_stopping_patience = 10  # Stop if val loss doesn't improve for this many epochs
    early_stopping_min_delta = 1e-4  # Minimum change in val loss to qualify as an improvement
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = f"checkpoints_{window_size}"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches
