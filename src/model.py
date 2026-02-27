from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 100,
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

    def forward(self, x_sbp: torch.Tensor, x_kin: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_sbp, x_kin], dim=-1)
        x = x.transpose(1, 2)
        x = self.encoder(x)

        center_idx = x.shape[-1] // 2
        x_center = x[:, :, center_idx]
        return self.head(x_center)
