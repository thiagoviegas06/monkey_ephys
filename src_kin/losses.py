import torch
import torch.nn.functional as F
import numpy as np

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
    Simplified loss for LFADS Kinematic Decoder, optimized for R² metric.
    Loss = MSE(Kinematics) + acceleration_weight * Acceleration_Penalty

    Focuses on fitting data well rather than learning a VAE prior.
    """
    # 1. Kinematic Reconstruction (MSE) - PRIMARY LOSS
    recon_loss = F.mse_loss(kin_pred, kin_target)

    # 2. Light Acceleration Penalty (In-network smoothing)
    # Prevents jitter but doesn't constrain dynamics too much
    accel_loss = acceleration_penalty(kin_pred)

    # Total weighted loss (MSE + light smoothing)
    loss = recon_loss + config.acceleration_weight * accel_loss

    # Return dict for logging (KL and correlation removed for simplicity)
    corr_loss = torch.tensor(0.0, device=kin_pred.device)
    kl_loss = torch.tensor(0.0, device=kin_pred.device)
    sbp_loss = torch.tensor(0.0, device=kin_pred.device)

    return {
        "loss": loss,
        "recon_mse": recon_loss,
        "corr_loss": corr_loss,
        "accel_loss": accel_loss,
        "kl_loss": kl_loss,
        "sbp_loss": sbp_loss,
        "beta": 0.0
    }

def calculate_per_session_r2(all_preds, all_targets):
    """
    Computes mean R² per session per channel — matches Kaggle's evaluation.

    Args:
        all_preds:   list of (N_i, 2) numpy arrays, one per session
        all_targets: list of (N_i, 2) numpy arrays, one per session

    Returns:
        mean R² across sessions and channels (scalar)
    """
    r2s = []
    for pred, target in zip(all_preds, all_targets):
        for ch in range(pred.shape[1]):
            ss_res = np.sum((target[:, ch] - pred[:, ch]) ** 2)
            ss_tot = np.sum((target[:, ch] - target[:, ch].mean()) ** 2) + 1e-8
            r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s))


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

def get_r2_components(y_pred, y_true):
    """
    Extracts R² components for accumulation across batches.
    Returns: (ss_res, sum_y, sum_y_sq, count)

    y_pred, y_true: (Batch, Time, Channels)
    """
    # Extract only the first 2 channels (index_pos, mrp_pos)
    y_pred = y_pred[..., :2]
    y_true = y_true[..., :2]

    # Flatten Batch and Time dimensions
    y_pred_flat = y_pred.reshape(-1, 2)
    y_true_flat = y_true.reshape(-1, 2)

    # Sum of squared residuals
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2).item()

    # Sum of y values and sum of squared y values (for computing global mean and variance)
    sum_y = torch.sum(y_true_flat).item()
    sum_y_sq = torch.sum(y_true_flat ** 2).item()
    count = y_true_flat.numel()  # N * num_channels, must match sum_y/sum_y_sq totals

    return ss_res, sum_y, sum_y_sq, count

def calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count):
    """
    Calculates global R² from accumulated components.

    Args:
        total_ss_res: sum of squared residuals across all batches
        total_sum_y: sum of y values across all batches
        total_sum_y_sq: sum of y² values across all batches
        total_count: total number of samples

    Returns:
        r2: R² score (float)
    """
    if total_count == 0:
        return 0.0

    # Compute global mean
    mean_y = total_sum_y / total_count

    # Compute total sum of squares
    ss_tot = total_sum_y_sq - (total_sum_y ** 2) / total_count

    # Compute R²
    r2 = 1.0 - (total_ss_res / (ss_tot + 1e-8))
    return r2
