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