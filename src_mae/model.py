import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def forward(self, x_sbp, kin, mask):
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
    
    def forward(self, x_sbp, kin, mask):
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
        
    def forward(self, x_sbp, kin, mask):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class SBPImputer(nn.Module):
    def __init__(self, sbp_channels=96, kin_channels=4, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(SBPImputer, self).__init__()
        
        # The input consists of:
        # 1. The masked SBP data (96 channels)
        # 2. The kinematic data (4 channels)
        # 3. The Boolean mask indicator (96 channels) - Lets the model know exactly what needs filling
        in_features = sbp_channels + kin_channels + sbp_channels
        
        # Projection layer to map raw inputs to transformer hidden dimensions
        self.input_projection = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer maps back to the 96 SBP channels
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, sbp_channels),
            nn.ReLU() # SBP is generally non-negative, ReLU guarantees positive predictions
        )

    def forward(self, x_sbp, kin, mask):
        """
        Args:
            x_sbp: [Batch, Time, 96]
            kin: [Batch, Time, 4]
            mask: [Batch, Time, 96]
        Returns:
            predictions: [Batch, Time, 96]
        """
        # Concatenate features along the channel dimension
        x = torch.cat([x_sbp, kin, mask], dim=-1)
        
        # Project and add position encodings
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        encoded = self.transformer(x)
        
        # Predict missing values
        out = self.output_projection(encoded)
        
        return out


"""
Usage Example:
--------------

from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor
from losses import masked_nmse_loss

# Choose a model
model = SBP_Reconstruction_UNet(base_channels=64)  # Best overall
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
- SBP_Reconstruction_UNet: Best performance, captures multi-scale patterns
- SimpleCNN: Fastest training, good baseline
- ResNetReconstructor: Better gradient flow than SimpleCNN, competitive with U-Net

All models handle (200, 96) shape and use the convention: mask=True means masked.
"""
