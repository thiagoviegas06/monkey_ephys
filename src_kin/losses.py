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

def kinematic_perceiver_loss(kin_pred, kin_target, config):
    """
    Computes the enhanced loss for the Perceiver IO Kinematic Decoder.
    Loss = kin_recon_weight * MSE(Kinematics) 
           + correlation_weight * Pearson_Loss
           + acceleration_weight * Acceleration_Penalty
    """
    # 1. Kinematic Reconstruction (MSE)
    recon_loss = F.mse_loss(kin_pred, kin_target)
    
    # 2. Pearson Correlation Loss
    corr_loss = pearson_correlation_loss(kin_pred, kin_target)
    
    # 3. Acceleration Penalty (In-network smoothing)
    accel_loss = acceleration_penalty(kin_pred)
    
    # Total weighted loss
    loss = (config.kin_recon_weight * recon_loss) + \
           (config.correlation_weight * corr_loss) + \
           (config.acceleration_weight * accel_loss)
           
    return {
        "loss": loss,
        "recon_mse": recon_loss,
        "corr_loss": corr_loss,
        "accel_loss": accel_loss
    }

def get_r2_components(y_pred, y_true):
    """
    Returns components needed to compute a mathematically correct global R^2 
    score for the first 2 position channels over the entire epoch.
    y_pred, y_true: (Batch, Time, Channels)
    """
    # Extract only the first 2 channels (index_pos, mrp_pos)
    y_pred = y_pred[..., :2]
    y_true = y_true[..., :2]
    
    # Flatten Batch and Time dimensions
    y_pred_flat = y_pred.reshape(-1, 2)
    y_true_flat = y_true.reshape(-1, 2)
    
    ss_res_batch = torch.sum((y_true_flat - y_pred_flat) ** 2, dim=0)
    sum_y_batch = torch.sum(y_true_flat, dim=0)
    sum_y_sq_batch = torch.sum(y_true_flat ** 2, dim=0)
    count_batch = torch.tensor(y_true_flat.shape[0], device=y_true.device, dtype=torch.float32)
    
    return ss_res_batch, sum_y_batch, sum_y_sq_batch, count_batch

def calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count):
    """
    Calculates the final R^2 score from accumulated components.
    """
    mean_y = total_sum_y / total_count
    ss_tot = total_sum_y_sq - (total_sum_y ** 2) / total_count
    r2 = 1 - (total_ss_res / (ss_tot + 1e-8))
    return r2.mean().item()
