import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel attention: Learn which SBP channels are important for kinematics.

    Different channels have different relationships to movement.
    This module learns per-channel importance weights.
    """
    def __init__(self, d_model, num_channels=96, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Squeeze-and-excitation style: learn channel importance
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // reduction, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B*W, C, d_model)

        Returns:
            x_weighted: (B*W, C, d_model) - channel-weighted features
            attn_weights: (B*W, C, 1) - learned channel importance
        """
        # Channel-wise attention: (B*W, C, d_model)
        # Compute importance per channel
        x_transpose = x.transpose(1, 2)  # (B*W, d_model, C)

        avg_pool = self.avg_pool(x_transpose)  # (B*W, d_model, 1)
        max_pool = self.max_pool(x_transpose)  # (B*W, d_model, 1)

        # Apply FC layers to learn channel importance
        avg_attn = self.fc(avg_pool.squeeze(-1))  # (B*W, d_model)
        max_attn = self.fc(max_pool.squeeze(-1))  # (B*W, d_model)

        attn = avg_attn + max_attn  # (B*W, d_model)
        attn = self.sigmoid(attn).unsqueeze(-1)  # (B*W, d_model, 1)

        # Apply attention to features
        x_weighted = x * attn  # (B*W, C, d_model) broadcast attention across channels

        return x_weighted, attn


class KinematicDecoderTransformer(nn.Module):
    """
    Advanced kinematics decoder with channel and temporal attention.

    Takes the transformer encoder output from MAE and learns kinematics
    with learned channel importance and temporal attention.

    Input: (B*W, C, d_model) transformer encoder output
    Output: (B, W, 4) predicted kinematics

    Architecture:
    1. Channel Attention: Learn per-channel importance weights
       (B*W, C, d_model) → (B*W, C, d_model) weighted

    2. Channel Aggregation with learned weights:
       (B*W, C, d_model) → (B*W, d_model)

    3. Reshape to temporal: (B*W, d_model) → (B, W, d_model)

    4. Temporal Transformer Attention:
       (B, W, d_model) → (B, W, d_model) with self-attention over time

    5. Output projection: (B, W, d_model) → (B, W, 4)
    """
    def __init__(self, d_model=192, window_size=200, num_channels=96,
                 num_temporal_layers=2, num_heads=8, output_dim=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_channels = num_channels

        # 1. Channel Attention: Learn which channels matter
        self.channel_attn = ChannelAttention(d_model, num_channels)

        # 2. Learned channel projection (weighted aggregation)
        # Instead of mean pooling, learn how to combine channels
        self.channel_proj = nn.Sequential(
            nn.Linear(d_model * num_channels, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 3. Temporal Attention: Self-attention over time steps
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_attention = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_temporal_layers
        )

        # 4. Output projection
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, transformer_repr, **kwargs):
        """
        Args:
            transformer_repr: (B, W, C, d_model) or (B*W, C, d_model)
            **kwargs: ignored, kept for compatibility

        Returns:
            kin_pred: (B, W, 4) predicted kinematics
        """
        # Handle 4D input (B, W, C, d_model)
        if transformer_repr.dim() == 4:
            B, W, C, d_model = transformer_repr.shape
            transformer_repr = transformer_repr.reshape(B * W, C, d_model)
        else:
            B_W, C, d_model = transformer_repr.shape
            B = B_W // self.window_size
            W = self.window_size

        # 1. Channel Attention: Learn which channels are important
        x_weighted, channel_weights = self.channel_attn(transformer_repr)
        # x_weighted: (B*W, C, d_model)

        # 2. Channel Aggregation with learned projection
        # Reshape for projection: (B*W, C*d_model)
        x_flat = x_weighted.reshape(B * W, -1)  # (B*W, C*d_model)
        x_proj = self.channel_proj(x_flat)  # (B*W, d_model)

        # 3. Reshape for temporal attention: (B, W, d_model)
        x_temporal = x_proj.reshape(B, W, d_model)

        # 4. Temporal attention: Learn dependencies across time
        x_attended = self.temporal_attention(x_temporal)  # (B, W, d_model)

        # 5. Project to kinematics
        kin_pred = self.fc_out(x_attended)  # (B, W, 4)

        return kin_pred
