import torch

# ============================================================================
# Configuration file for training - modify these parameters as needed
# ============================================================================
class Config:
    """Training configuration - modify these parameters"""
    
    # Data
    data_path = "kaggle_data"
    window_size = 400
    seed = 42
    
    # Preprocessing
    preprocess = False  # Set True to run preprocessing (only needed once)
    windows_dir = f"kaggle_data/masked_windows_{window_size}"  # Where preprocessed windows are saved
    
    # Model
    model_name = "tcn_transformer"   # Options: "unet", "simple_cnn", "resnet", "transformer", "tcn_transformer"
    base_channels = 64   # For UNet/ResNet
    sbp_channels = 96
    kin_channels = 4
    
    # For transformer model
    d_model = 64  # Embedding dimension for transformer
    nhead = 8  # Number of attention heads
    num_layers = 6 # Number of transformer encoder blocks stacked on top of each other
    tcn_levels = 8  # Number of TCN dilation layers
    dropout = 0.1  # Dropout rate in transformer for regularization
    
    # Training
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    early_stopping_patience = 5  # Stop if val loss doesn't improve for this many epochs
    early_stopping_min_delta = 5e-4  # Minimum change in val loss to qualify as an improvement
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = f"checkpoints_{window_size}"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches
