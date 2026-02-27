from __future__ import annotations

import torch


def masked_mse_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    diff_sq = (y_hat - y_true).pow(2)
    masked_diff_sq = diff_sq * mask
    denom = mask.sum().clamp(min=eps)
    return masked_diff_sq.sum() / denom


def nmse_masked(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mse = masked_mse_loss(y_hat, y_true, mask, eps=eps)
    signal_power = ((y_true.pow(2)) * mask).sum() / mask.sum().clamp(min=eps)
    return mse / signal_power.clamp(min=eps)
