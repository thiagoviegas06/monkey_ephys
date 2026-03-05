import torch

# ============================================================================
# Configuration file for training - modify these parameters as needed
# ============================================================================
class Config:
    """Training configuration - modify these parameters"""
    
    # Data
    data_path = "kaggle_data"
    window_size = 200
    seed = 42
    
    # Preprocessing
    preprocess = False  # Set True to run preprocessing (only needed once)
    windows_dir = "kaggle_data/masked_windows"  # Where preprocessed windows are saved
    
    # Model
    model_name = "transformer"  # Options: "unet", "simple_cnn", "resnet", "transformer"
    base_channels = 64   # For UNet/ResNet
    sbp_channels = 96
    kin_channels = 4
    
    # For transformer model
    d_model = 256  # 96 SBP + 4 kin + 96 mask = 196 total input channels, so d_model should be >= 196)
    nhead = 8  # Number of attention heads
    num_layers = 4 # Number of transformer encoder blocks stacked on top of each other
    dropout = 0.1  # Dropout rate in transformer for regularization
    
    # Training
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 20
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = "checkpoints"
    save_every = 5  # Save checkpoint every N epochs
    
    # Logging
    log_every = 10  # Log every N batches