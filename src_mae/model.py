import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
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
    """
    Interleaved Axial TCN + Transformer for masked SBP reconstruction.
    Alternates between Temporal and Spatial mixing layers to iteratively refine 
    representations, allowing spatial context to inform temporal processing in later layers.
    """
    def __init__(self, sbp_channels=96, d_model=192, nhead=8, num_axial_layers=4, 
                 num_decoder_layers=2, tcn_levels=4, dropout=0.15):
        super().__init__()
        self.sbp_channels = sbp_channels
        self.d_model = d_model
        
        # --- 0. Input Normalization ---
        self.macro_bn = nn.BatchNorm1d(1)
        
        # --- 1. Initial Local Feature Extraction (TCN) ---
        in_features = 2 # sbp_masked + mask_indicator
        tcn_layers = []
        for i in range(tcn_levels):
            dilation = 2 ** i 
            in_ch = in_features if i == 0 else d_model
            tcn_layers.append(TemporalBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        
        # --- 2. Macro-Time & Positional Encodings ---
        self.macro_time_encoder = ContinuousTimeEmbedding(d_model)
        self.temp_pos_encoding = PositionalEncoding(d_model, dropout)
        self.channel_embeddings = nn.Parameter(torch.zeros(1, sbp_channels, d_model))
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)
        
        # --- 3. Interleaved Axial Encoder Blocks ---
        self.axial_layers = nn.ModuleList()
        for _ in range(num_axial_layers):
            # A. Temporal Attention (Across W)
            temp_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation='gelu', norm_first=True
            )
            # B. Spatial Attention (Across C)
            spat_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation='gelu', norm_first=True
            )
            self.axial_layers.append(nn.ModuleDict({
                'temp': nn.TransformerEncoder(temp_layer, num_layers=1),
                'spat': nn.TransformerEncoder(spat_layer, num_layers=1)
            }))
            
        self.enc_norm = nn.LayerNorm(d_model)
        
        # --- 4. Asymmetric Decoder ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        
        # --- 5. Output Projection ---
        self.output_proj = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sbp_masked, mask, macro_time, channel_mean=None, channel_var=None, return_embeddings=False):
        B, W, C = sbp_masked.shape
        
        # PHASE 0: NORMALIZATION (Session-Aware)
        visible_mask = (1.0 - mask)
        num_visible_c = visible_mask.sum(dim=1, keepdim=True) # (B, 1, C)
        
        local_mean = (sbp_masked * visible_mask).sum(dim=1, keepdim=True) / num_visible_c.clamp(min=1.0)
        local_var = (((sbp_masked - local_mean) * visible_mask) ** 2).sum(dim=1, keepdim=True) / num_visible_c.clamp(min=1.0)
        local_std = torch.sqrt(local_var + 1e-5)
        
        if channel_mean is not None and channel_var is not None:
            sess_mean = channel_mean.unsqueeze(1)
            sess_std = torch.sqrt(channel_var.unsqueeze(1) + 1e-5)
            is_fully_masked = (num_visible_c == 0).float()
            sbp_mean = local_mean * (1 - is_fully_masked) + sess_mean * is_fully_masked
            sbp_std = local_std * (1 - is_fully_masked) + sess_std * is_fully_masked
        else:
            sbp_mean, sbp_std = local_mean, local_std

        sbp_norm = ((sbp_masked - sbp_mean) / sbp_std) * visible_mask
        macro_time_norm = self.macro_bn(macro_time)
        
        # PHASE 1: LOCAL TCN (Temporal Feature Extraction per channel)
        sbp_exp = sbp_norm.transpose(1, 2).unsqueeze(-1) # (B, C, W, 1)
        mask_exp = mask.transpose(1, 2).unsqueeze(-1)       
        x_tcn = torch.cat([sbp_exp, mask_exp], dim=-1).reshape(B * C, W, 2).transpose(1, 2)
        
        x = self.tcn(x_tcn).transpose(1, 2).reshape(B, C, W, self.d_model) # (B, C, W, d_model)
        
        # PHASE 2: INTERLEAVED AXIAL ENCODER
        spat_pad_mask = mask.reshape(B * W, C).bool()
        
        # Prevent NaNs if a time bin is fully masked by unmasking at least one channel
        # We check this once as the mask is constant across axial layers
        all_masked = spat_pad_mask.all(dim=1)
        if all_masked.any():
            spat_pad_mask = spat_pad_mask.clone()
            spat_pad_mask[all_masked, 0] = False
            
        for layer in self.axial_layers:
            # A. Temporal Attention (Across W)
            x = x.reshape(B * C, W, self.d_model)
            x = self.temp_pos_encoding(x)
            x = layer['temp'](x)
            
            # B. Spatial Attention (Across C)
            x = x.reshape(B, C, W, self.d_model).permute(0, 2, 1, 3).contiguous().reshape(B * W, C, self.d_model)
            x = x + self.channel_embeddings
            x = layer['spat'](x, src_key_padding_mask=spat_pad_mask)
            
            # Reshape back to (B, C, W, d_model) for next temporal pass
            x = x.reshape(B, W, C, self.d_model).permute(0, 2, 1, 3).contiguous()
            
        # For the final output, we need (B * W, C, d_model)
        enc_out = self.enc_norm(x.permute(0, 2, 1, 3).reshape(B * W, C, self.d_model))
        
        # PHASE 3: ASYMMETRIC DECODER
        mask_tokens = self.mask_token.expand(B * W, C, -1) + self.channel_embeddings
        time_emb = self.macro_time_encoder(macro_time_norm).repeat_interleave(W, dim=0)
        mask_tokens = mask_tokens + time_emb
        
        dec_in = torch.where(spat_pad_mask.unsqueeze(-1), mask_tokens, enc_out)
        dec_out = self.decoder(dec_in)
        
        if return_embeddings:
            return dec_out.reshape(B, W, C, self.d_model)
        
        # PHASE 4: PROJECTION & UN-NORMALIZATION
        pred_norm = self.output_proj(dec_out).reshape(B, W, C)
        pred_unnorm = (pred_norm * sbp_std) + sbp_mean
        
        return torch.where(mask.bool(), pred_unnorm, sbp_masked)
