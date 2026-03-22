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
