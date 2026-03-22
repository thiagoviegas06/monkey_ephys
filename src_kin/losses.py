import torch
import torch.nn.functional as F

def lfads_loss(kin_pred, kin_target, mu, logvar, step, config, sbp_pred=None, sbp_target=None):
    """
    Computes the loss for the LFADS Kinematic Decoder.
    Loss = MSE(Kinematics) + beta * KL_Divergence + (optional) sbp_recon_weight * MSE(SBP)
    """
    # 1. Reconstruction Loss (MSE) - using only the first 2 channels if evaluated, but since we predict 4, we optimize for 4.
    # In practice, only the position channels are evaluated, but velocity helps regularize the model.
    recon_loss = F.mse_loss(kin_pred, kin_target)
    
    # 2. KL Divergence for the latent space (forcing g0 ~ N(0, 1))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Beta Annealing to prevent posterior collapse
    beta = min(config.max_beta, config.max_beta * (step / max(1, config.beta_anneal_steps)))
    
    loss = recon_loss + beta * kl_loss
    
    # 4. Multi-Task Regularization (SBP Reconstruction)
    sbp_loss = torch.tensor(0.0, device=kin_pred.device)
    if sbp_pred is not None and sbp_target is not None and config.sbp_recon_weight > 0:
        sbp_loss = F.mse_loss(sbp_pred, sbp_target)
        loss = loss + config.sbp_recon_weight * sbp_loss
        
    return loss, recon_loss, kl_loss, sbp_loss, beta

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
