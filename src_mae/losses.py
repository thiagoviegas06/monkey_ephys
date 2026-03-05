import torch

def masked_nmse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    pred/target/mask: (B, W, C)
    mask: bool, True where entry is masked (we supervise there)
    """
    assert pred.shape == target.shape == mask.shape

    B, W, C = target.shape

    # ---- per-channel variance from UNMASKED entries ----
    unmask = ~mask  # True where observed

    # Count unmasked per channel
    n_unmask = unmask.sum(dim=(0, 1)).clamp(min=1)  # (C,)

    # Mean per channel over unmasked entries
    mean_c = (target * unmask).sum(dim=(0, 1)) / n_unmask  # (C,)

    # Var per channel over unmasked entries
    var_c = ((target - mean_c) ** 2 * unmask).sum(dim=(0, 1)) / n_unmask  # (C,)
    var_c = var_c.clamp(min=eps)

    # ---- per-channel masked MSE ----
    n_mask = mask.sum(dim=(0, 1))  # (C,)
    # avoid divide by zero: only compute for channels that are masked at least once
    active = n_mask > 0

    mse_c = ((pred - target) ** 2 * mask).sum(dim=(0, 1)) / n_mask.clamp(min=1)  # (C,)

    nmse_c = mse_c[active] / var_c[active]
    return nmse_c.mean()

def masked_mse_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor,
                    eps: float = 1e-8):
    """
    pred, target, mask: (B, W, C)
    mask: bool tensor, True where entry was masked

    Returns:
        scalar masked MSE
    """
    assert pred.shape == target.shape == mask.shape

    diff_sq = (pred - target) ** 2

    # only masked entries contribute
    masked_diff = diff_sq * mask

    denom = mask.sum().clamp(min=eps)

    return masked_diff.sum() / denom


def stable_masked_nmse_loss(pred, target, mask, global_var_c):
    """
    global_var_c: (C,) tensor of pre-calculated variances for each channel
    """
    # squared error only on masked parts
    sq_error = ((pred - target) ** 2) * mask
    
    # MSE per channel in this batch
    mse_c = sq_error.sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)
    
    # Normalize by the fixed global variance, not the batch variance
    nmse_c = mse_c / global_var_c.clamp(min=1e-8)
    
    # Only average over channels that actually had masks in this batch
    active_channels = mask.sum(dim=(0, 1)) > 0
    return nmse_c[active_channels].mean()


def kaggle_aligned_nmse_loss(pred: torch.Tensor, 
                             target: torch.Tensor, 
                             mask: torch.Tensor,
                             channel_var: torch.Tensor,
                             session_ids: list = None,
                             eps: float = 1e-8):
    """
    Kaggle-aligned NMSE loss that matches the competition metric.
    
    Groups predictions by (session, channel) and computes:
        For each (session, channel): NMSE = MSE / var_session_channel
    Then averages across all (session, channel) groups.
    
    This creates proper (session, channel) grouping similar to Kaggle metric.
    
    Args:
        pred: (B, W, C) predictions
        target: (B, W, C) ground truth 
        mask: (B, W, C) bool, True where entry was masked (supervised)
        channel_var: (B, C) per-channel variance from each sample's session
        session_ids: list of B session IDs for proper grouping. If None, uses per-sample grouping.
        eps: small value for numerical stability
    
    Returns:
        scalar loss
    """
    assert pred.shape == target.shape == mask.shape, \
        f"Shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}"
    assert channel_var.shape[0] == pred.shape[0] and channel_var.shape[1] == pred.shape[2], \
        f"channel_var shape {channel_var.shape} incompatible with batch {pred.shape}"
    
    B, W, C = pred.shape
    device = pred.device
    
    # Squared error at masked positions: (B, W, C)
    sq_error = ((pred - target) ** 2) * mask
    
    if session_ids is None:
        # Fallback: per-sample grouping if no session_ids provided
        # This is useful for compatibility but less accurate
        
        # Count masked positions per sample and channel: (B, C)
        n_masked = mask.sum(dim=1)  # sum over W dimension, result: (B, C)
        
        # Compute MSE per (sample, channel): (B, C)
        mse_bc = sq_error.sum(dim=1) / n_masked.clamp(min=1)
        
        # Normalize by per-session channel variance: (B, C)
        channel_var_clamped = channel_var.clamp(min=eps)
        nmse_bc = mse_bc / channel_var_clamped
        
        # Only consider (sample, channel) pairs that were actually masked
        active = n_masked > 0
        
        if active.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=pred.dtype)
        
        return nmse_bc[active].mean()
    
    else:
        # Proper session-based grouping
        # For each (session, channel) group in the batch, compute NMSE
        
        unique_sessions = list(set(session_ids))
        nmse_list = []
        weight_list = []
        
        for sess_id in unique_sessions:
            # Find samples from this session (avoid creating new tensor in autograd)
            sess_indices = [i for i, sid in enumerate(session_ids) if sid == sess_id]
            if len(sess_indices) == 0:
                continue
            
            # Get data for this session: (N_sess, W, C)
            sess_pred = pred[sess_indices]  # (N_sess, W, C)
            sess_target = target[sess_indices]  # (N_sess, W, C)
            sess_mask_data = mask[sess_indices]  # (N_sess, W, C)
            sess_var = channel_var[sess_indices]  # (N_sess, C)
            
            # Sum squared errors and count for this session across all samples
            # Shape results: (C,)
            sess_sq_error = ((sess_pred - sess_target) ** 2) * sess_mask_data
            sess_n_masked = sess_mask_data.sum(dim=(0, 1))  # (C,)
            
            # Compute MSE per channel for this session: (C,)
            sess_mse_c = sess_sq_error.sum(dim=(0, 1)) / sess_n_masked.clamp(min=1)
            
            # Normalize by session's channel variance: (C,)
            # All samples from same session have identical channel_var, so take first
            sess_var_c = sess_var[0].clamp(min=eps)  # (C,)
            sess_nmse_c = sess_mse_c / sess_var_c
            
            # Only include channels that were masked in this session
            sess_active = sess_n_masked > 0
            
            if sess_active.sum() > 0:
                # Weighted averaging: weight by number of masked positions
                session_weight = sess_n_masked[sess_active].sum()
                nmse_list.append(sess_nmse_c[sess_active].mean())
                weight_list.append(session_weight)
        
        if len(nmse_list) == 0:
            return torch.tensor(0.0, device=device, dtype=pred.dtype)
        
        # Weighted average across sessions (more stable gradients)
        weights = torch.stack(weight_list)
        weights = weights / weights.sum()  # normalize
        nmses = torch.stack(nmse_list)
        return (nmses * weights).sum()


def variance_weighted_mse_loss(pred: torch.Tensor, 
                                target: torch.Tensor, 
                                mask: torch.Tensor,
                                channel_var: torch.Tensor,
                                eps: float = 1e-8):
    """
    Simpler alternative: MSE weighted inversely by channel variance.
    More stable gradients than NMSE (no division, just weighting).
    Correlates well with Kaggle NMSE while being easier to optimize.
    
    Args:
        pred: (B, W, C) predictions
        target: (B, W, C) ground truth 
        mask: (B, W, C) bool, True where masked
        channel_var: (B, C) per-channel variance
        eps: numerical stability
    
    Returns:
        scalar weighted MSE
    """
    # Squared error at masked positions: (B, W, C)
    sq_error = ((pred - target) ** 2) * mask
    
    # Inverse variance weights (higher weight for low-variance channels)
    # Broadcast channel_var from (B,C) to (B,W,C)
    weights = 1.0 / channel_var.clamp(min=eps).unsqueeze(1)  # (B, 1, C)
    
    # Weighted MSE
    weighted_sq_error = sq_error * weights
    
    # Average over all masked positions
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    return weighted_sq_error.sum() / n_masked


def huber_nmse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: torch.Tensor,
                   channel_var: torch.Tensor,
                   delta: float = 1.0,
                   eps: float = 1e-8):
    """
    Huber-smoothed NMSE: clips large errors for more stable gradients.
    Less sensitive to outliers than pure NMSE.
    
    Args:
        pred: (B, W, C) predictions
        target: (B, W, C) ground truth 
        mask: (B, W, C) bool, True where masked
        channel_var: (B, C) per-channel variance
        delta: Huber threshold (errors above this are L1-smoothed)
        eps: numerical stability
    
    Returns:
        scalar Huber NMSE
    """
    # Raw errors
    errors = (pred - target) * mask  # (B, W, C)
    
    # Normalize errors by sqrt(variance) to get standardized residuals
    channel_std = channel_var.clamp(min=eps).sqrt().unsqueeze(1)  # (B, 1, C)
    normalized_errors = errors / channel_std
    
    # Huber loss on normalized errors
    abs_errors = normalized_errors.abs()
    quadratic = torch.where(abs_errors <= delta, 
                           0.5 * normalized_errors ** 2,
                           delta * abs_errors - 0.5 * delta ** 2)
    
    # Only consider masked positions
    masked_loss = quadratic * mask
    
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    return masked_loss.sum() / n_masked


def channel_nmse_loss(pred: torch.Tensor, 
                      target: torch.Tensor, 
                      mask: torch.Tensor,
                      channel_var: torch.Tensor,
                      eps: float = 1e-8):
    """
    Channel-only NMSE: averages across channels without session grouping.
    Much simpler than kaggle_aligned but still variance-normalized.
    
    Often trains more stably while correlating well with Kaggle metric.
    
    Args:
        pred: (B, W, C) predictions
        target: (B, W, C) ground truth 
        mask: (B, W, C) bool, True where masked
        channel_var: (B, C) per-channel variance
        eps: numerical stability
    
    Returns:
        scalar channel-averaged NMSE
    """
    B, W, C = pred.shape
    
    # Squared error at masked positions: (B, W, C)
    sq_error = ((pred - target) ** 2) * mask
    
    # MSE per channel across full batch: (C,)
    mse_per_channel = sq_error.sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)
    
    # Average variance per channel across batch: (C,)
    var_per_channel = channel_var.mean(dim=0).clamp(min=eps)
    
    # NMSE per channel: (C,)
    nmse_per_channel = mse_per_channel / var_per_channel
    
    # Only average over channels that had masked data
    active_channels = mask.sum(dim=(0, 1)) > 0
    
    if active_channels.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    return nmse_per_channel[active_channels].mean()
