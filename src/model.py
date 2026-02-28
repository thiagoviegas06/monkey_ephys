from __future__ import annotations

import torch
import torch.nn as nn


def _choose_gn_groups(channels: int, preferred: int) -> int:
    for g in (preferred, 16, 8, 4, 2, 1):
        if g > 0 and channels % g == 0:
            return g
    return 1


class TemporalCNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 196,  # 96 SBP + 4 kin + 96 obs_mask
        hidden_dim: int = 128,
        output_dim: int = 96,
        num_layers: int = 6,
        kernel_size: int = 9,
        dropout: float = 0.2,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = input_dim

        for i in range(num_layers):
            dilation = dilation_base**i
            padding = (kernel_size // 2) * dilation
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x_sbp: torch.Tensor, x_kin: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_sbp, x_kin, obs_mask], dim=-1)  # (B,T,196)
        x = x.transpose(1, 2)  # (B,196,T)
        x = self.encoder(x)
        return self.head(x)  # (B,96,T)


class TCNResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        gn_groups: int,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        groups = _choose_gn_groups(channels, gn_groups)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Explicit channel-mixing path
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.mix_norm = nn.GroupNorm(groups, channels)
        self.mix_act = nn.GELU()
        self.mix_drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.drop1(self.act1(self.norm1(self.conv1(x))))
        y = self.mix_drop(self.mix_act(self.mix_norm(self.mix(y))))
        y = self.drop2(self.act2(self.norm2(self.conv2(y))))
        return y + residual


class TemporalTCNResidual(nn.Module):
    def __init__(
        self,
        input_dim: int = 196,
        hidden_dim: int = 192,
        output_dim: int = 96,
        num_layers: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.15,
        dilation_base: int = 2,
        dilation_cap: int = 64,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = _choose_gn_groups(hidden_dim, gn_groups)

        self.pre = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        dilation = 1
        for _ in range(num_layers):
            blocks.append(
                TCNResBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    gn_groups=groups,
                )
            )
            dilation = min(dilation * dilation_base, dilation_cap)

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x_sbp: torch.Tensor, x_kin: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_sbp, x_kin, obs_mask], dim=-1)  # (B,T,196)
        x = x.transpose(1, 2)  # (B,196,T)
        x = self.pre(x)
        x = self.encoder(x)
        delta = self.head(x)  # (B,96,T)
        return x_sbp.transpose(1, 2) + delta


def build_model(model_name: str, **model_kwargs: int | float) -> nn.Module:
    name = model_name.lower()
    if name == "cnn":
        return TemporalCNNBaseline(**model_kwargs)
    if name == "tcn":
        return TemporalTCNResidual(**model_kwargs)
    raise ValueError(f"Unsupported model_name={model_name!r}. Expected one of ['cnn', 'tcn'].")

