import torch

# ============================================================================
# Configuration file for Phase 2 Kinematic Decoder training
# ============================================================================
class Config:
    """Training configuration - modify these parameters as needed"""
    
    # Data Paths
    data_path = "kaggle_data_phase2"
    
    # Pre-trained MAE Model for imputation
    mae_checkpoint_path = "checkpoints_200/best_model_tcn_transformer.pt"
    
    # SBP_TCN_Transformer Hyperparameters (must match the saved checkpoint)
    window_size = 200
    sbp_channels = 96
    d_model = 64  
    nhead = 8  
    num_layers = 8 
    tcn_levels = 8  
    dropout = 0.1  
    
    # Seed
    seed = 42
    
    # LFADS Model Hyperparameters
    hidden_dim = 128
    gen_dim = 128
    factor_dim = 40
    output_dim = 4 # Outputs 4 kinematics channels (though only 2 are evaluated)
    lfads_dropout = 0.3 # Dropout for LFADS Decoder

    
    # Training
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 50
    early_stopping_patience = 7  
    early_stopping_min_delta = 1e-5  
    
    # LFADS Loss Hyperparameters
    beta_anneal_steps = 2000 # Number of batches to linearly anneal beta
    max_beta = 0.01 # Maximum weight for the KL divergence penalty
    kin_recon_weight = 50.0 # Weight for the kinematic reconstruction loss
    correlation_weight = 10.0 # Weight for the Pearson correlation loss
    acceleration_weight = 1.0 # Weight for the acceleration penalty (smoothing)
    sbp_recon_weight = 0.0 # Weight for the multi-task SBP reconstruction (disabled as requested)
    
    # Smoothing
    smoothing_kernel_size = 5 # Kernel size for moving average smoothing
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    checkpoint_dir = "checkpoints_kin"
    save_every = 5  
    
    # Logging
    log_every = 10  
