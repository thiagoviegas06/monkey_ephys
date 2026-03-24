import torch

# ============================================================================
# Configuration file for Phase 2 Kinematic Decoder training
# ============================================================================
class Config:
    """Training configuration - modify these parameters as needed"""
    
    # Data Paths
    data_path = "phase2_v2_kaggle_data"  # Contains D001, D007, etc. with full kinematics
    
    # Pre-trained MAE Model for imputation
    mae_checkpoint_path = "checkpoints_200/best_model_tcn_transformer.pt"

    # SBP_TCN_Transformer Hyperparameters (must match the saved checkpoint)
    window_size = 300  # Middle ground: captures ~1.5 movement cycles, better temporal context than 200
    sbp_channels = 96
    d_model = 64  
    nhead = 8  
    num_layers = 8 
    tcn_levels = 8  
    dropout = 0.1  
    
    # Seed
    seed = 42
    
    # LSTM Model Hyperparameters (uses MAE encoder features, not raw SBP)
    encoder_dim = d_model  # Match MAE's d_model (64) - encoder feature dimension
    hidden_dim = 128  # LSTM hidden size
    num_lstm_layers = 2  # Number of stacked LSTM layers
    output_dim = 4  # Outputs 4 kinematics channels (though only 2 are evaluated)
    lstm_dropout = 0.1  # Dropout between LSTM layers
    
    # Training
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 50
    early_stopping_patience = 7  
    early_stopping_min_delta = 1e-5  
    
    # LFADS Loss Hyperparameters (Simplified for R² optimization)
    # Loss = MSE(kin) + acceleration_weight * Acceleration_Penalty
    acceleration_weight = 0.1 # Light smoothing penalty (prevents jitter, doesn't over-constrain)
    
    # Window stride for training (sliding window step size)
    window_step_size = 75  # 25% overlap, proportional to window_size=300

    # Smoothing
    smoothing_kernel_size = 5 # Kernel size for moving average smoothing
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = "checkpoints_kin"
    save_every = 5  
    
    # Logging
    log_every = 10  
