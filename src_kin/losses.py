import torch
import torch.nn.functional as F

def pearson_correlation_loss(y_pred, y_true):
    """
    Calculates the Pearson Correlation loss for the first 2 position channels.
    Loss = 1 - average_correlation across channels.
    y_pred, y_true: (Batch, Time, Channels)
    """
    # Extract only the first 2 channels (index_pos, mrp_pos)
    y_pred = y_pred[..., :2]
    y_true = y_true[..., :2]
    
    # Mean across time dimension
    mu_pred = torch.mean(y_pred, dim=1, keepdim=True)
    mu_true = torch.mean(y_true, dim=1, keepdim=True)
    
    # Center the data
    y_pred_c = y_pred - mu_pred
    y_true_c = y_true - mu_true
    
    # Calculate correlation: sum(x_c * y_c) / sqrt(sum(x_c^2) * sum(y_c^2))
    num = torch.sum(y_pred_c * y_true_c, dim=1)
    den = torch.sqrt(torch.sum(y_pred_c**2, dim=1) * torch.sum(y_true_c**2, dim=1) + 1e-8)
    
    corr = num / den
    # Loss is 1 - average correlation across batch and channels
    return 1 - corr.mean()

def acceleration_penalty(y_pred):
    """
    Penalizes high-frequency jitter by calculating the second derivative (acceleration)
    of the predicted positions.
    y_pred: (Batch, Time, Channels)
    """
    # Use only evaluated channels
    y_pred = y_pred[..., :2]
    
    # Second derivative: acc[t] = y[t+2] - 2*y[t+1] + y[t]
    # We can use finite differences
    accel = y_pred[:, 2:, :] - 2 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
    return torch.mean(accel**2)

def lfads_loss(kin_pred, kin_target, mu, logvar, step, config, sbp_pred=None, sbp_target=None):
    """
    Computes the enhanced loss for the LFADS Kinematic Decoder.
    Loss = kin_recon_weight * MSE(Kinematics) 
           + correlation_weight * Pearson_Loss
           + acceleration_weight * Acceleration_Penalty
           + beta * KL_Divergence
    """
    # 1. Kinematic Reconstruction (MSE)
    # Scaled up to focus gradients on trajectory
    recon_loss = F.mse_loss(kin_pred, kin_target)
    
    # 2. Pearson Correlation Loss
    corr_loss = pearson_correlation_loss(kin_pred, kin_target)
    
    # 3. Acceleration Penalty (In-network smoothing)
    accel_loss = acceleration_penalty(kin_pred)
    
    # 4. KL Divergence (Latent Regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 5. Beta Annealing
    beta = min(config.max_beta, config.max_beta * (step / max(1, config.beta_anneal_steps)))
    
    # Total weighted loss
    loss = (config.kin_recon_weight * recon_loss) + \
           (config.correlation_weight * corr_loss) + \
           (config.acceleration_weight * accel_loss) + \
           (beta * kl_loss)
    
    # 6. Optional Multi-Task SBP Regularization (usually set to 0.0)
    sbp_loss = torch.tensor(0.0, device=kin_pred.device)
    if sbp_pred is not None and sbp_target is not None and config.sbp_recon_weight > 0:
        sbp_loss = F.mse_loss(sbp_pred, sbp_target)
        loss = loss + config.sbp_recon_weight * sbp_loss
        
    return {
        "loss": loss,
        "recon_mse": recon_loss,
        "corr_loss": corr_loss,
        "accel_loss": accel_loss,
        "kl_loss": kl_loss,
        "sbp_loss": sbp_loss,
        "beta": beta
    }

def calculate_r2(y_pred, y_true):
    """
    Calculates the R^2 (Coefficient of Determination) score for the first 2 position channels.
    y_pred, y_true: (Batch, Time, Channels)
    """
    # Extract only the first 2 channels (index_pos, mrp_pos)
    y_pred = y_pred[..., :2]
    y_true = y_true[..., :2]
    
    # Flatten Batch and Time dimensions
    y_pred_flat = y_pred.reshape(-1, 2)
    y_true_flat = y_true.reshape(-1, 2)
    
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2, dim=0)
    
    # Calculate mean per channel across all samples in the flattened array
    mean_true = torch.mean(y_true_flat, dim=0)
    ss_tot = torch.sum((y_true_flat - mean_true) ** 2, dim=0)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.mean().item()
