from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 196,   # 96 SBP + 4 kin + 96 obs_mask
        hidden_dim: int = 128,
        output_dim: int = 96,
        num_layers: int = 4,
        kernel_size: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        blocks = []
        in_channels = input_dim
        padding = kernel_size // 2

        for _ in range(num_layers):
            blocks.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_sbp: torch.Tensor, x_kin: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        # x_sbp:   (B, T, 96)
        # x_kin:   (B, T, 4)
        # obs_mask:(B, T, 96) 1=observed, 0=masked
        x = torch.cat([x_sbp, x_kin, obs_mask], dim=-1)  # (B, T, 196)
        x = x.transpose(1, 2)                            # (B, 196, T)
        x = self.encoder(x)                              # (B, hidden_dim, T)

        center_idx = x.shape[-1] // 2
        x_center = x[:, :, center_idx]                   # (B, hidden_dim)
        return self.head(x_center)                       # (B, 96)