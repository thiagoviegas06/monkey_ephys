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
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model) for batch_first=True

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # A small network to project a single scalar into the high-dimensional latent space
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, macro_time):
        """
        Args:
            macro_time: (batch_size, 1) - Normalized days/sessions since start
        Returns:
            time_emb: (batch_size, 1, d_model)
        """
        # Pass through MLP and add a sequence dimension for broadcasting
        time_emb = self.mlp(macro_time)
        return time_emb.unsqueeze(1)


class SBPImputer(nn.Module):
    def __init__(self, sbp_channels=96, kin_channels=4, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        input_dim = sbp_channels + sbp_channels + kin_channels  # masked SBP + mask indicator + kinematics
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Micro-time (intra-sequence) and Macro-time (inter-sequence) encoders
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.macro_time_encoder = ContinuousTimeEmbedding(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, sbp_channels)

    def forward(self, sbp_masked, kinematics, mask, macro_time):
        """
        Args:
            sbp_masked: (batch, seq_len, 96) - Original data with missing values as 0
            mask: (batch, seq_len, 96) - 1/True for masked (missing), 0/False for visible
            kinematics: (batch, seq_len, 4)
            macro_time: (batch, 1)
        """
        # 1. Concatenate raw features
        x = torch.cat([sbp_masked, mask.float(), kinematics], dim=-1)
        
        # 2. Project and inject temporal embeddings (Micro + Macro)
        x = self.input_proj(x)
        x = self.pos_encoder(x) 
        time_emb = self.macro_time_encoder(macro_time)
        x = x + time_emb 
        x = self.norm(x)
        
        # 3. Pass through Transformer
        x = self.transformer_encoder(x)
        
        # 4. Project back to SBP channels
        raw_predictions = self.output_proj(x)
        
        # 5. Blend: Only apply predictions over the masked values
        # torch.where(condition, if_true, if_false)
        # If mask is 1 (True), use our prediction. If 0 (False), keep original SBP.
        final_output = torch.where(mask.bool(), raw_predictions, sbp_masked)
        
        return final_output

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
