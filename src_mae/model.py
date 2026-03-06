import torch
import torch.nn as nn
import torch.nn.functional as F

class SBP_Reconstruction_UNet(nn.Module):
    """
    U-Net for masked SBP reconstruction.
    Optimized for (W=200, C=96) shape - pools only in time dimension.
    
    Mask convention: True = masked (to be predicted), False = observed
    """
    def __init__(self, base_channels=64, in_ch=2, out_ch=1, gn_groups=8):
        super().__init__()

        def norm(c):
            g = min(gn_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            return nn.GroupNorm(g, c)

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                norm(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                norm(cout),
                nn.ReLU(inplace=True),
            )

        # Pool only in time dimension to preserve channel structure
        # (W, C) -> (W/2, C)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.up = lambda x: F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)

        # Encoder - progressively downsample time
        self.enc1 = conv_block(in_ch, base_channels)           # 64
        self.enc2 = conv_block(base_channels, base_channels*2) # 128
        self.enc3 = conv_block(base_channels*2, base_channels*4) # 256

        # Bottleneck
        self.bottleneck = conv_block(base_channels*4, base_channels*8) # 512

        # Decoder - progressively upsample time with skip connections
        self.dec3 = conv_block(base_channels*8 + base_channels*4, base_channels*4)
        self.dec2 = conv_block(base_channels*4 + base_channels*2, base_channels*2)
        self.dec1 = conv_block(base_channels*2 + base_channels, base_channels)

        # Output projection
        self.out = nn.Conv2d(base_channels, out_ch, kernel_size=1)

    def forward(self, x_sbp, mask):
        """
        x_sbp: (B, W, C)  - masked SBP input (0s where masked)
        mask:  (B, W, C)  - bool tensor, True where masked, False where observed
        
        Returns: (B, W, C) - reconstructed SBP
        """
        B, W, C = x_sbp.shape
        
        # Create observed mask (inverse of mask)
        obs_mask = ~mask  # True where observed
        obs = obs_mask.float()

        # 2-channel input: [masked_signal, observation_indicator]
        # This tells the model which positions are already known
        x = torch.stack([x_sbp, obs], dim=1)  # (B, 2, W, C)

        # Encoder path
        # Input: 200x96 -> 100x96 -> 50x96 -> 25x96
        e1 = self.enc1(x)            # (B, 64, W, C)
        e2 = self.enc2(self.pool(e1))# (B, 128, W/2, C)
        e3 = self.enc3(self.pool(e2))# (B, 256, W/4, C)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 512, W/8, C)

        # Decoder path with skip connections
        # 25x96 -> 50x96 -> 100x96 -> 200x96
        d3 = self.up(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Project to output
        out = self.out(d1).squeeze(1)  # (B, W, C)

        # Keep observed entries unchanged; only predict masked positions
        # This ensures the model only changes what needs to be reconstructed
        out = out * mask.float() + x_sbp * obs
        
        return out


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for masked SBP reconstruction.
    No downsampling - processes at full (200, 96) resolution throughout.
    Good baseline to compare against U-Net.
    """
    def __init__(self, hidden_channels=128, num_layers=6):
        super().__init__()
        
        layers = []
        # Input: 2 channels (signal + mask indicator)
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(8, hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers with residual-like connections
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(8, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Output: 1 channel (reconstructed signal)
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_sbp, mask):
        """
        x_sbp: (B, W, C) - masked SBP input
        mask:  (B, W, C) - True where masked
        """
        obs_mask = ~mask
        obs = obs_mask.float()
        
        # Stack signal and observation indicator
        x = torch.stack([x_sbp, obs], dim=1)  # (B, 2, W, C)
        
        # Process through CNN
        out = self.net(x).squeeze(1)  # (B, W, C)
        
        # Keep observed values, only predict masked
        out = out * mask.float() + x_sbp * obs
        
        return out


class ResNetBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResNetReconstructor(nn.Module):
    """
    ResNet-style architecture for SBP reconstruction.
    Uses residual connections for better gradient flow.
    """
    def __init__(self, hidden_channels=128, num_blocks=8):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, 1, 1)
        
    def forward(self, x_sbp, mask):
        """
        x_sbp: (B, W, C) - masked SBP input
        mask:  (B, W, C) - True where masked
        """
        obs_mask = ~mask
        obs = obs_mask.float()
        
        # Stack inputs
        x = torch.stack([x_sbp, obs], dim=1)  # (B, 2, W, C)
        
        # Process
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        out = self.output_proj(x).squeeze(1)  # (B, W, C)
        
        # Keep observed values
        out = out * mask.float() + x_sbp * obs
        
        return out


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise-separable convolution for efficient temporal processing.

    First applies depthwise conv (per-channel), then pointwise (1x1) for mixing.
    More efficient than standard conv and allows better channel interactions.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)

        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=channels
        )
        # Pointwise: mix across channels
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """x: (B, C, T)"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelFFN(nn.Module):
    """
    Channel-wise feedforward network for mixing spatial/channel information.

    Applies point-wise (1x1) convolutions with expansion:
    C -> 4*C -> C
    This allows nonlinear mixing of channel information.
    """
    def __init__(self, channels, expansion_factor=4, dropout=0.2):
        super().__init__()
        hidden = int(channels * expansion_factor)

        self.up = nn.Conv1d(channels, hidden, kernel_size=1)
        self.gn_up = nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden)
        self.down = nn.Conv1d(hidden, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T) - operates independently per timestep"""
        residual = x

        # Expand channels
        out = self.up(x)
        out = self.gn_up(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Contract back
        out = self.down(out)
        out = out + residual
        return out


class TCNResidualBlock(nn.Module):
    """
    Enhanced TCN residual block with temporal + channel mixing.

    Uses depthwise-separable convolutions for temporal processing,
    then channel FFN for explicit channel interactions.
    Provides both local temporal patterns AND inter-channel mixing.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()

        # Temporal processing: depthwise-separable conv
        self.temporal_conv = DepthwiseSeparableConv1d(
            channels, kernel_size=kernel_size, dilation=dilation
        )
        self.gn_temporal = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout1 = nn.Dropout(dropout)

        # Channel mixing: FFN-style pointwise operations
        self.channel_ffn = ChannelFFN(channels, expansion_factor=4, dropout=dropout)

    def forward(self, x):
        """
        x: (B, C, T) - tensor in Conv1d format
        Returns: (B, C, T)

        Process flow:
        1. Temporal: depthwise-separable conv (per-channel temporal, then mix)
        2. Channel: FFN for inter-channel interactions
        3. Residual connection throughout
        """
        residual = x

        # Temporal processing
        out = self.temporal_conv(x)
        out = self.gn_temporal(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # Channel mixing
        out = self.channel_ffn(out)

        # Residual
        out = out + residual
        out = F.relu(out)
        return out


class TCNReconstructor(nn.Module):
    """
    Enhanced Temporal Convolutional Network (TCN) for masked SBP reconstruction.

    Architecture combines:
    1. **Temporal Processing**: Depthwise-separable Conv1d for efficient per-channel temporal patterns
    2. **Channel Mixing**: FFN-style pointwise operations (C -> 4C -> C) for inter-channel interactions
    3. **Dilated Receptive Field**: Exponential dilation [1,2,4,8,4,2,1] for 45-timestep RF
    4. **Bidirectional**: Non-causal convolutions for offline reconstruction

    This design allows the model to learn:
    - Local temporal patterns within each SBP channel
    - Nonlinear dependencies between channels (biologically realistic)
    - Both fine (dilation=1) and coarse (dilation=8) temporal scales

    Receptive field calculation (kernel_size=3):
    - Layer with dilation d spans 2*d+1 temporal positions
    - With dilations [1,2,4,8,4,2,1]: RF ~ 45 timesteps
    - Well within W=200 window while capturing meaningful patterns

    Args:
        hidden_channels: Number of hidden channels in TCN layers (default: 128)
        num_layers: Number of residual blocks (default: 7, odd for symmetric dilations)
        kernel_size: Temporal Conv1d kernel size (default: 3)
        dropout: Dropout probability (default: 0.2)
        dilation_multiplier: Base for exponential dilation schedule (default: 2)
    """
    def __init__(
        self,
        hidden_channels=128,
        num_layers=7,
        kernel_size=3,
        dropout=0.2,
        dilation_multiplier=2
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Input projection: 2*C-channel input [x_sbp, obs_mask] -> hidden_channels
        # C=96, so input has 192 channels (96 signal + 96 mask indicator)
        self.input_proj = nn.Sequential(
            nn.Conv1d(2 * 96, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Build dilation schedule: exponential growth then symmetry
        # E.g., for num_layers=7: [1, 2, 4, 8, 4, 2, 1]
        half = num_layers // 2
        dilations = [dilation_multiplier ** i for i in range(half)]
        dilations.append(dilation_multiplier ** half)  # Peak dilation
        dilations = dilations + dilations[-2::-1]  # Symmetric: reverse without repeating peak
        assert len(dilations) == num_layers, f"Expected {num_layers} dilations, got {len(dilations)}"

        # Build residual blocks with increasing then decreasing dilations
        self.blocks = nn.ModuleList([
            TCNResidualBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        # Output projection: hidden_channels -> 96 channels (one reconstruction per SBP channel)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 96, kernel_size=1)  # FIXED: output 96 channels, not 1!
        )

        # Store dilations for documentation
        self.dilations = dilations

    def forward(self, x_sbp, mask):
        """
        x_sbp: (B, W, C) - masked SBP input (0s where masked)
        mask:  (B, W, C) - bool tensor, True where masked

        Returns:
            (B, W, C) - reconstructed SBP with observed values preserved
        """
        B, W, C = x_sbp.shape

        # Validate shapes
        assert x_sbp.shape == (B, W, C), f"x_sbp shape {x_sbp.shape} doesn't match expected (B, W, C)"
        assert mask.shape == (B, W, C), f"mask shape {mask.shape} doesn't match x_sbp"
        assert C == 96, f"Expected 96 channels, got {C}"
        assert W == 200, f"Expected 200 time steps, got {W}"

        # Create observation mask (inverse of mask)
        obs_mask = (~mask).float()  # True where observed

        # Stack inputs along channel dimension: [x_sbp, obs_mask]
        # (B, W, C) + (B, W, C) -> (B, W, 2*C)
        x = torch.cat([x_sbp, obs_mask], dim=2)  # (B, W, 2*C)

        # Transpose to Conv1d format: (B, W, 2*C) -> (B, 2*C, W)
        x = x.transpose(1, 2)  # (B, 2*C, W)

        # Input projection: 2*C channels -> hidden_channels
        x = self.input_proj(x)  # (B, hidden_channels, W)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)  # (B, hidden_channels, W)

        # Output projection: hidden_channels -> C (one per channel)
        out = self.output_proj(x)  # (B, C, W)

        # Transpose back to (B, W, C)
        out = out.transpose(1, 2)  # (B, W, C)

        # Keep observed entries unchanged; only predict masked positions
        out = out * mask.float() + x_sbp * obs_mask

        return out

    def get_receptive_field(self):
        """
        Estimate receptive field size based on kernel size and dilations.
        RF = 1 + sum(2 * dilation_i) for kernel_size=3
        """
        rf = 1
        for dilation in self.dilations:
            rf += 2 * dilation
        return rf


"""
Usage Example:
--------------

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor, TCNReconstructor
from losses import masked_nmse_loss

# Choose a model
model = TCNReconstructor(hidden_channels=128, num_layers=7)  # New TCN baseline
# model = SBP_Reconstruction_UNet(base_channels=64)  # Conv2d U-Net
# model = SimpleCNN(hidden_channels=128, num_layers=6)  # Fastest
# model = ResNetReconstructor(hidden_channels=128, num_blocks=8)  # Good middle ground

# In training loop:
for batch in dataloader:
    x_sbp = batch['x_sbp']  # (B, 200, 96) - zeros where masked
    y_sbp = batch['y_sbp']  # (B, 200, 96) - ground truth
    mask = batch['mask']     # (B, 200, 96) - True where masked

    # Forward pass
    pred = model(x_sbp, mask)

    # Compute loss only on masked positions
    loss = masked_nmse_loss(pred, y_sbp, mask)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Model Comparison:
- TCNReconstructor: Conv1d temporal convolutions, bidirectional, dilated, efficient
- SBP_Reconstruction_UNet: Best performance, captures multi-scale patterns
- SimpleCNN: Fastest training, good baseline
- ResNetReconstructor: Better gradient flow than SimpleCNN, competitive with U-Net

All models handle (200, 96) shape and use the convention: mask=True means masked.
"""
