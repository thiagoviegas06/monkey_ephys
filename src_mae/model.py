import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    """
    Hybrid TCN + Cross-Channel Transformer for masked SBP reconstruction.
    Operates without kinematics data.
    Uses True Asymmetric MAE architecture (Encoder for visible only, Decoder for imputation).
    """
    def __init__(self, sbp_channels=96, d_model=64, nhead=8, num_encoder_layers=6, num_decoder_layers=2, tcn_levels=4, dropout=0.1):
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
        
        # --- 3. Spatial / Channel Embeddings ---
        self.channel_embeddings = nn.Parameter(torch.zeros(1, sbp_channels, d_model))
        # Initialize with very small variance (0.02) instead of standard 1.0
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)
        
        # A bridging norm to stabilize features before hitting the Transformer
        self.pre_transformer_norm = nn.LayerNorm(d_model)
        
        # --- 4. Cross-Channel Asymmetric Transformer (Encoder-Decoder) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-Norm stabilizes deep transformers immediately
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        
        # --- 5. Output Projection ---
        self.output_proj = nn.Linear(d_model, 1)
        # Initialize output close to 0 so the model starts by guessing the mean
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sbp_masked, mask, macro_time):
        """
        sbp_masked: (B, W, C)
        mask: (B, W, C) - 1.0 where masked, 0.0 where visible
        macro_time: (B, 1)
        """
        B, W, C = sbp_masked.shape
        
        # ==========================================
        # PHASE 0: REVERSIBLE INSTANCE NORMALIZATION (PER-CHANNEL)
        # ==========================================
        # A. Normalize SBP (Only compute stats on VISIBLE values per channel)
        visible_mask = (~mask.bool()).float()  # 1.0 if visible, 0.0 if masked
        # num_visible: (B, 1, C) - total unmasked pixels per channel
        num_visible_c = visible_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Temporal Mean/Std per channel
        sbp_mean = (sbp_masked * visible_mask).sum(dim=1, keepdim=True) / num_visible_c
        sbp_var = (((sbp_masked - sbp_mean) * visible_mask) ** 2).sum(dim=1, keepdim=True) / num_visible_c
        sbp_std = torch.sqrt(sbp_var + 1e-5)
        
        # Normalize SBP, keeping masked values safely at exactly 0
        sbp_norm = ((sbp_masked - sbp_mean) / sbp_std) * visible_mask
        
        # C. Normalize Macro Time across the batch
        macro_time_norm = self.macro_bn(macro_time)
        
        # ==========================================
        # PHASE 1: INDEPENDENT TCN
        # ==========================================
        sbp_exp = sbp_norm.transpose(1, 2).unsqueeze(-1)
        mask_exp = mask.transpose(1, 2).unsqueeze(-1)       
        
        x_tcn = torch.cat([sbp_exp, mask_exp], dim=-1) # (B, C, W, 2)
        x_tcn = x_tcn.reshape(B * C, W, -1).transpose(1, 2)
        
        tcn_out = self.tcn(x_tcn) 
        x = tcn_out.view(B, C, self.d_model, W)
        
        # ==========================================
        # PHASE 2: MACRO TIME EMBEDDINGS
        # ==========================================
        time_emb = self.macro_time_encoder(macro_time_norm).unsqueeze(-1)
        x = x + time_emb
        
        # ==========================================
        # PHASE 3: SPATIAL PREP & ASYMMETRIC TRANSFORMER
        # ==========================================
        x = x.permute(0, 3, 1, 2) 
        x = x.reshape(B * W, C, self.d_model)
        
        x = x + self.channel_embeddings
        x = self.pre_transformer_norm(x)  # Standardize before Attention
        
        # Padding mask for the Encoder: True where masked
        # Shape: (B*W, C). PyTorch's src_key_padding_mask prevents attending TO True positions.
        padding_mask = mask.view(B * W, C).bool()
        
        # The Encoder strictly processes visible channels to build a pristine spatial representation.
        enc_out = self.encoder(x, src_key_padding_mask=padding_mask)
        
        # Construct Decoder input: 
        # For visible tokens, use the deeply encoded pristine representation.
        # For masked tokens, inject the original TCN spatial embedding back in as a "query" token.
        decoder_in = torch.where(padding_mask.unsqueeze(-1), x, enc_out)
        
        # The Decoder processes the full sequence, allowing masked tokens to query visible tokens.
        dec_out = self.decoder(decoder_in)
        
        # ==========================================
        # PHASE 4: PROJECTION & REVERSIBLE BLENDING
        # ==========================================
        pred_norm = self.output_proj(dec_out)
        pred_norm = pred_norm.view(B, W, C)
        
        # Un-normalize the predictions back to the original signal's distribution
        pred_unnorm = (pred_norm * sbp_std) + sbp_mean
        
        final_output = torch.where(mask.bool(), pred_unnorm, sbp_masked)
        
        return final_output
