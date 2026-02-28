from __future__ import annotations

import torch


def masked_mse_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    y_hat: (B, 96, T)
    y_true: (B, 96, T)
    mask: (B, 96) where 1 = masked channel
    """
    mask_exp = mask.unsqueeze(-1).to(dtype=y_hat.dtype)  # (B,96,1)

    diff_sq = (y_hat - y_true).pow(2)
    masked_diff_sq = diff_sq * mask_exp

    denom = mask_exp.sum().clamp(min=eps) * y_hat.shape[-1]  # masked channels * T
    return masked_diff_sq.sum() / denom


def nmse_masked(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NMSE = sum((y_hat-y)^2 * mask) / sum((y^2) * mask), over all masked channels and time.
    """
    mask_exp = mask.unsqueeze(-1).to(dtype=y_hat.dtype)  # (B,96,1)

    num = ((y_hat - y_true).pow(2) * mask_exp).sum()
    den = (y_true.pow(2) * mask_exp).sum().clamp(min=eps)

    return num / den
