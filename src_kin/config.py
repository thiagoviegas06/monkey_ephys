import torch
import os

# Get the project root directory (one level up from src_kin)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configuration file for Phase 2 Kinematic Decoder training
# ============================================================================
class Config:
    """Training configuration - modify these parameters as needed"""
    
    # Data Paths
    data_path = os.path.join(ROOT_DIR, "kaggle_data_phase2")
    phase1_data_path = os.path.join(ROOT_DIR, "kaggle_data")
    
    # SBP_TCN_Transformer Hyperparameters (must match the saved checkpoint)
    window_size = 200
    sbp_channels = 96
    d_model = 192  
    nhead = 8  
    num_axial_layers = 4 
    num_decoder_layers = 2
    tcn_levels = 4  
    dropout = 0.15  
    
    # Pre-trained MAE Model for imputation
    mae_checkpoint_path = os.path.join(ROOT_DIR, f"checkpoints_{window_size}", "best_model_tcn_transformer.pt")
    use_mae_embeddings = True # Set to True to use MAE embeddings instead of raw SBP
    mae_embedding_type = 'full' # 'full' is required for the new Transformer decoder
    
    # Seed
    seed = 42
    
    # Kinematic Decoder Transformer Hyperparameters
    decoder_num_temporal_layers = 2
    decoder_num_heads = 8
    decoder_dropout = 0.1
    output_dim = 4 # Outputs 4 kinematics channels
    
    # Training
    batch_size = 128
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 50
    early_stopping_patience = 7  
    early_stopping_min_delta = 1e-5  
    
    # Loss Hyperparameters
    kin_recon_weight = 1.0 # Weight for the kinematic reconstruction loss
    correlation_weight = 0.1 # Weight for the Pearson correlation loss
    acceleration_weight = 0.01 # Weight for the acceleration penalty (smoothing)
    
    # Smoothing
    smoothing_kernel_size = 5 # Kernel size for moving average smoothing
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints_kin")
    save_every = 5  
    
    # Logging
    log_every = 10  
