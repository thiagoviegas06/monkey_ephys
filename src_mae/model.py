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

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Initialize the final layer with small weights so it doesn't 
        # heavily bias the network at the very beginning of training.
        nn.init.uniform_(self.mlp[2].weight, -0.05, 0.05)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, macro_time):
        time_emb = self.mlp(macro_time)
        return time_emb.unsqueeze(1)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
            # Scale down the residual projection to prevent variance explosion
            nn.init.xavier_uniform_(self.res_conv.weight, gain=0.5)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        res = self.res_conv(x)
        
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out).transpose(1, 2)
        out = self.drop1(self.act1(out))
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out).transpose(1, 2)
        out = self.drop2(self.act2(out))
        
        return out + res


class SBP_TCN_Transformer(nn.Module):
    def __init__(self, sbp_channels=96, kin_channels=4, d_model=64, nhead=8, num_layers=4, tcn_levels=4, dropout=0.1):
        super().__init__()
        self.sbp_channels = sbp_channels
        self.d_model = d_model
        
        # --- 0. Input Normalization ---
        self.macro_bn = nn.BatchNorm1d(1)
        
        # --- 1. Channel-Independent TCN ---
        in_features = 2 + kin_channels
        
        tcn_layers = []
        for i in range(tcn_levels):
            dilation = 2 ** i 
            in_ch = in_features if i == 0 else d_model
            tcn_layers.append(TemporalBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        
        # --- 2. Macro-Time Embedding ---
        self.macro_time_encoder = ContinuousTimeEmbedding(d_model)
        
        # --- 3. Spatial / Channel Embeddings ---
        self.channel_embeddings = nn.Parameter(torch.zeros(1, sbp_channels, d_model))
        # Initialize with very small variance (0.02) instead of standard 1.0
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)
        
        # A bridging norm to stabilize features before hitting the Transformer
        self.pre_transformer_norm = nn.LayerNorm(d_model)
        
        # --- 4. Cross-Channel Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-Norm stabilizes deep transformers immediately
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 5. Output Projection ---
        self.output_proj = nn.Linear(d_model, 1)
        # Initialize output close to 0 so the model starts by guessing the mean
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sbp_masked, kinematics, mask, macro_time):
        B, W, C = sbp_masked.shape
        
        # ==========================================
        # PHASE 0: REVERSIBLE INSTANCE NORMALIZATION
        # ==========================================
        # A. Normalize SBP (Only compute stats on VISIBLE values so 0s don't skew it)
        visible_mask = (~mask.bool()).float()  # 1.0 if visible, 0.0 if masked
        num_visible = visible_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        sbp_mean = (sbp_masked * visible_mask).sum(dim=1, keepdim=True) / num_visible
        sbp_var = (((sbp_masked - sbp_mean) * visible_mask) ** 2).sum(dim=1, keepdim=True) / num_visible
        sbp_std = torch.sqrt(sbp_var + 1e-5)
        
        # Normalize SBP, keeping masked values safely at exactly 0
        sbp_norm = ((sbp_masked - sbp_mean) / sbp_std) * visible_mask
        
        # B. Normalize Kinematics (Fully visible, so standard instance norm over time)
        kin_mean = kinematics.mean(dim=1, keepdim=True)
        kin_std = kinematics.std(dim=1, keepdim=True) + 1e-5
        kin_norm = (kinematics - kin_mean) / kin_std
        
        # C. Normalize Macro Time across the batch
        macro_time_norm = self.macro_bn(macro_time)
        
        # ==========================================
        # PHASE 1: INDEPENDENT TCN
        # ==========================================
        kin_exp = kin_norm.unsqueeze(1).expand(B, C, W, -1)
        sbp_exp = sbp_norm.transpose(1, 2).unsqueeze(-1)
        mask_exp = mask.transpose(1, 2).unsqueeze(-1)       
        
        x_tcn = torch.cat([sbp_exp, mask_exp, kin_exp], dim=-1)
        x_tcn = x_tcn.reshape(B * C, W, -1).transpose(1, 2)
        
        tcn_out = self.tcn(x_tcn) 
        x = tcn_out.view(B, C, self.d_model, W)
        
        # ==========================================
        # PHASE 2: MACRO TIME EMBEDDINGS
        # ==========================================
        time_emb = self.macro_time_encoder(macro_time_norm).unsqueeze(-1)
        x = x + time_emb
        
        # ==========================================
        # PHASE 3: SPATIAL PREP & TRANSFORMER
        # ==========================================
        x = x.permute(0, 3, 1, 2) 
        x = x.reshape(B * W, C, self.d_model)
        
        x = x + self.channel_embeddings
        x = self.pre_transformer_norm(x)  # Standardize before Attention
        
        x = self.transformer_encoder(x)
        
        # ==========================================
        # PHASE 4: PROJECTION & REVERSIBLE BLENDING
        # ==========================================
        pred_norm = self.output_proj(x)
        pred_norm = pred_norm.view(B, W, C)
        
        # Un-normalize the predictions back to the original signal's distribution
        pred_unnorm = (pred_norm * sbp_std) + sbp_mean
        
        final_output = torch.where(mask.bool(), pred_unnorm, sbp_masked)
        
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
