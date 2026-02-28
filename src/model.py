from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 196,   # 96 SBP + 4 kin + 96 obs_mask
        hidden_dim: int = 128,
        output_dim: int = 96,
        num_layers: int = 6,
        kernel_size: int = 9,
        dropout: float = 0.2,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()

        blocks = []
        in_channels = input_dim

        for i in range(num_layers):
            d = dilation_base ** i  # 1,2,4,8,16,32...
            padding = (kernel_size // 2) * d  # keeps length T

            blocks.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                )
            )
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x_sbp: torch.Tensor, x_kin: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_sbp, x_kin, obs_mask], dim=-1)  # (B,T,196)
        x = x.transpose(1, 2)                            # (B,196,T)
        x = self.encoder(x)                              # (B,H,T)
        y_seq = self.head(x)                             # (B,96,T)
        return y_seq