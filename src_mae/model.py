import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation  # same-length padding

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)

        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        return x + h  # residual


class SBPInpaintingTCN(nn.Module):
    """
    Input:
      x_sbp: (B, W, 96)  masked
      kin:   (B, W, 4)
      obs_mask: (B, W, 96) 1 where observed, 0 where masked (float or bool)

    Output:
      y_hat: (B, W, 96)
    """
    def __init__(
        self,
        in_dim: int = 196,     # 96 sbp + 4 kin + 96 obs_mask
        hidden: int = 256,
        out_dim: int = 96,
        depth: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.1,
        dilation_base: int = 2,
        dilation_cap: int = 128,
    ):
        super().__init__()

        self.in_proj = nn.Conv1d(in_dim, hidden, kernel_size=1)

        blocks = []
        d = 1
        for _ in range(depth):
            blocks.append(ResDilatedBlock(hidden, kernel_size, dilation=d, dropout=dropout))
            d = min(d * dilation_base, dilation_cap)
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv1d(hidden, out_dim, kernel_size=1)

    def forward(self, x_sbp, kin, obs_mask):
        # x_sbp: (B,W,96), kin: (B,W,4), obs_mask:(B,W,96)
        if obs_mask.dtype != x_sbp.dtype:
            obs_mask = obs_mask.to(dtype=x_sbp.dtype)

        x = torch.cat([x_sbp, kin, obs_mask], dim=-1)  # (B,W,196)

        # to (B,C,T)
        x = x.transpose(1, 2)  # (B,196,W)

        x = self.in_proj(x)    # (B,hidden,W)
        for blk in self.blocks:
            x = blk(x)
        y = self.out_proj(x)   # (B,96,W)

        return y.transpose(1, 2)  # (B,W,96)