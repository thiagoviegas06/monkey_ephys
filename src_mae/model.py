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
    Hybrid TCN + Axial Transformer (Temporal then Spatial) for masked SBP reconstruction.
    Uses Session-level statistics to handle fully masked channels.
    """
    def __init__(self, sbp_channels=96, d_model=128, nhead=8, num_encoder_layers=4, 
                 num_temporal_layers=2, num_decoder_layers=2, tcn_levels=6, dropout=0.1):
        super().__init__()
        self.sbp_channels = sbp_channels
        self.d_model = d_model
        
        # --- 0. Input Normalization ---
        self.macro_bn = nn.BatchNorm1d(1)
        
        # --- 1. Channel-Independent TCN ---
        in_features = 2 # sbp_masked + mask_indicator
        
        tcn_layers = []
        for i in range(tcn_levels):
            dilation = 2 ** i 
            in_ch = in_features if i == 0 else d_model
            tcn_layers.append(TemporalBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        
        # --- 2. Macro-Time Embedding ---
        self.macro_time_encoder = ContinuousTimeEmbedding(d_model)
        
        # --- 3. Positional Encodings ---
        self.temp_pos_encoding = PositionalEncoding(d_model, dropout)
        self.channel_embeddings = nn.Parameter(torch.zeros(1, sbp_channels, d_model))
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)
        
        # --- 4. Axial Transformer Layers ---
        # A. Temporal Transformer (Across W)
        temp_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temp_layer, num_layers=num_temporal_layers)
        
        # B. Spatial Transformer (Across C)
        spat_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(spat_layer, num_layers=num_encoder_layers)
        
        self.enc_norm = nn.LayerNorm(d_model)
        
        # C. Decoder (Full Spatio-Temporal context)
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

    def forward(self, sbp_masked, mask, macro_time, channel_mean=None, channel_var=None):
        """
        sbp_masked: (B, W, C)
        mask: (B, W, C) - 1.0 where masked
        macro_time: (B, 1)
        channel_mean/var: (B, C) Session-level stats
        """
        B, W, C = sbp_masked.shape
        
        # ==========================================
        # PHASE 0: NORMALIZATION (Session-Aware)
        # ==========================================
        visible_mask = (1.0 - mask)
        num_visible_c = visible_mask.sum(dim=1, keepdim=True) # (B, 1, C)
        
        # Calculate local stats for unmasked pixels
        local_mean = (sbp_masked * visible_mask).sum(dim=1, keepdim=True) / num_visible_c.clamp(min=1.0)
        local_var = (((sbp_masked - local_mean) * visible_mask) ** 2).sum(dim=1, keepdim=True) / num_visible_c.clamp(min=1.0)
        local_std = torch.sqrt(local_var + 1e-5)
        
        # If a channel is fully masked in the window, fallback to session stats
        if channel_mean is not None and channel_var is not None:
            # Reshape session stats: (B, C) -> (B, 1, C)
            sess_mean = channel_mean.unsqueeze(1)
            sess_std = torch.sqrt(channel_var.unsqueeze(1) + 1e-5)
            
            is_fully_masked = (num_visible_c == 0).float()
            sbp_mean = local_mean * (1 - is_fully_masked) + sess_mean * is_fully_masked
            sbp_std = local_std * (1 - is_fully_masked) + sess_std * is_fully_masked
        else:
            sbp_mean = local_mean
            sbp_std = local_std

        sbp_norm = ((sbp_masked - sbp_mean) / sbp_std) * visible_mask
        macro_time_norm = self.macro_bn(macro_time)
        
        # ==========================================
        # PHASE 1: INDEPENDENT TCN (Temporal Feature Extraction)
        # ==========================================
        sbp_exp = sbp_norm.transpose(1, 2).unsqueeze(-1) # (B, C, W, 1)
        mask_exp = mask.transpose(1, 2).unsqueeze(-1)       
        
        x_tcn = torch.cat([sbp_exp, mask_exp], dim=-1) # (B, C, W, 2)
        x_tcn = x_tcn.reshape(B * C, W, 2).transpose(1, 2) # (B*C, 2, W)
        
        tcn_out = self.tcn(x_tcn) # (B*C, d_model, W)
        x = tcn_out.transpose(1, 2).view(B, C, W, self.d_model) # (B, C, W, d_model)
        
        # ==========================================
        # PHASE 2: TEMPORAL ATTENTION (Across W)
        # ==========================================
        # (B, C, W, d_model) -> (B*C, W, d_model)
        x_temp = x.view(B * C, W, self.d_model)
        x_temp = self.temp_pos_encoding(x_temp)
        x_temp = self.temporal_encoder(x_temp) # Intra-channel temporal mixing
        
        # ==========================================
        # PHASE 3: SPATIAL ATTENTION (Across C)
        # ==========================================
        # (B*C, W, d_model) -> (B, C, W, d_model) -> (B, W, C, d_model) -> (B*W, C, d_model)
        x_spat = x_temp.view(B, C, W, self.d_model).permute(0, 2, 1, 3).reshape(B * W, C, self.d_model)
        x_spat = x_spat + self.channel_embeddings
        
        # Spatial encoder only sees visible channels to build pristine representation
        # padding_mask for spatial transformer: (B*W, C)
        spat_pad_mask = mask.view(B * W, C).bool()
        enc_out = self.spatial_encoder(x_spat, src_key_padding_mask=spat_pad_mask)
        enc_out = self.enc_norm(enc_out)
        
        # ==========================================
        # PHASE 4: ASYMMETRIC DECODER (In-painting)
        # ==========================================
        # Construct Decoder input: encoded visible tokens + learned mask tokens for missing
        mask_tokens = self.mask_token.expand(B * W, C, -1)
        # Add channel embeddings to mask tokens
        mask_tokens = mask_tokens + self.channel_embeddings
        
        # Combine macro-time embedding
        time_emb = self.macro_time_encoder(macro_time_norm) # (B, 1, d_model)
        time_emb_expanded = time_emb.repeat_interleave(W, dim=0) # (B*W, 1, d_model)
        
        mask_tokens = mask_tokens + time_emb_expanded
        
        dec_in = torch.where(spat_pad_mask.unsqueeze(-1), mask_tokens, enc_out)
        dec_out = self.decoder(dec_in)
        
        # ==========================================
        # PHASE 5: PROJECTION & UN-NORMALIZATION
        # ==========================================
        pred_norm = self.output_proj(dec_out)
        pred_norm = pred_norm.view(B, W, C)
        
        pred_unnorm = (pred_norm * sbp_std) + sbp_mean
        
        # Blend with original observed data
        final_output = torch.where(mask.bool(), pred_unnorm, sbp_masked)
        
        return final_output
