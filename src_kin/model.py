import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):
        # positions: (Batch, SeqLen)
        # return: (Batch, SeqLen, D_model)
        pe_lookup = self.pe[positions.long()]
        return pe_lookup

class PerceiverIOKinematicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.perceiver_d_model
        self.latent_dim = config.perceiver_latent_dim
        
        # FiLM Generator MLP
        # Takes normalized session_num (1D) and outputs gamma, beta (each size sbp_channels)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, config.sbp_channels * 2)
        )
        
        # Maps 96 channels to D_model
        self.input_proj = nn.Linear(config.sbp_channels, self.d_model)
        
        # Positional Encoding for time
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Latent array
        self.latents = nn.Parameter(torch.randn(1, config.perceiver_num_latents, self.latent_dim))
        
        # Encoder (Cross-Attention)
        self.encoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            kdim=self.d_model,
            vdim=self.d_model,
            num_heads=config.perceiver_cross_heads,
            batch_first=True
        )
        self.encoder_norm1 = nn.LayerNorm(self.latent_dim)
        self.encoder_norm2 = nn.LayerNorm(self.d_model)
        
        # Processor (Self-Attention)
        processor_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=config.perceiver_latent_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=config.perceiver_dropout,
            activation='gelu',
            batch_first=True
        )
        self.processor = nn.TransformerEncoder(processor_layer, num_layers=config.perceiver_latent_layers)
        
        # Output queries: We use time embedding as output query.
        self.query_proj = nn.Linear(self.d_model, self.d_model)
        
        # Decoder (Cross-Attention)
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            kdim=self.latent_dim,
            vdim=self.latent_dim,
            num_heads=config.perceiver_cross_heads,
            batch_first=True
        )
        self.decoder_norm1 = nn.LayerNorm(self.d_model)
        self.decoder_norm2 = nn.LayerNorm(self.latent_dim)
        
        # Final Readout
        self.readout = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, config.output_dim)
        )
        
        self.max_session_num = config.max_session_num
        self.window_size = config.window_size

    def forward(self, x, mask=None, session_num=None, macro_timestamp=None, sbp_mean=None, sbp_std=None):
        batch_size, seq_len, _ = x.size()
        
        if mask is None:
            mask = (x == 0.0).float()
            
        # Normalize SBP 
        visible_mask = (~mask.bool()).float()
        
        if sbp_mean is not None and sbp_std is not None:
            mean = sbp_mean.to(x.device).view(1, 1, -1)
            std = sbp_std.to(x.device).view(1, 1, -1)
        else:
            num_visible = visible_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            mean = (x * visible_mask).sum(dim=(1, 2), keepdim=True) / num_visible
            var = (((x - mean) * visible_mask) ** 2).sum(dim=(1, 2), keepdim=True) / num_visible
            std = torch.sqrt(var + 1e-4)
            
        x_norm = ((x - mean) / std) * visible_mask
        
        # FiLM Modulation
        if session_num is None:
            session_num = torch.zeros(batch_size, 1, device=x.device)
        
        session_norm = session_num / self.max_session_num
        film_params = self.film_mlp(session_norm) # (Batch, 96*2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        # gamma, beta: (Batch, 96). Expand to (Batch, 1, 96)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # Apply FiLM: 1.0 + gamma acts as scale centered at 1
        x_mod = x_norm * (1.0 + gamma) + beta
        
        # Map modulated input to D_model
        x_proj = self.input_proj(x_mod) # (Batch, SeqLen, D_model)
        
        # Absolute Time Positional Encodings
        if macro_timestamp is None:
            macro_timestamp = torch.zeros(batch_size, device=x.device)
            
        if macro_timestamp.dim() == 2:
            macro_timestamp = macro_timestamp.squeeze(-1) # (Batch,)
            
        time_offsets = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        absolute_time = macro_timestamp.unsqueeze(-1) + time_offsets # (Batch, SeqLen)
        
        pos_emb = self.pos_encoder(x_proj, absolute_time) # (Batch, SeqLen, D_model)
        
        # Input Array (M)
        input_array = x_proj + pos_emb # (Batch, SeqLen, D_model)
        
        # --- PERCEIVER IO CORE ---
        # Expand latents to batch size
        latents = self.latents.expand(batch_size, -1, -1) # (Batch, NumLatents, D_latent)
        
        # Encode (Cross-Attention)
        q = self.encoder_norm1(latents)
        kv = self.encoder_norm2(input_array)
        encoded_latents, _ = self.encoder_cross_attn(q, kv, kv)
        latents = latents + encoded_latents
        
        # Process (Self-Attention)
        latents = self.processor(latents)
        
        # Decode (Cross-Attention)
        output_queries = self.query_proj(pos_emb) # (Batch, SeqLen, D_model)
        
        q_dec = self.decoder_norm1(output_queries)
        kv_dec = self.decoder_norm2(latents)
        decoded_out, _ = self.decoder_cross_attn(q_dec, kv_dec, kv_dec)
        
        final_repr = output_queries + decoded_out
        
        # Map to Kinematics
        kinematic_preds = self.readout(final_repr)
        
        # Return dummy values for compatibility with evaluating / training loops
        dummy_sbp = torch.zeros_like(x)
        dummy_mu = torch.zeros(batch_size, 1, device=x.device)
        dummy_logvar = torch.zeros(batch_size, 1, device=x.device)
        
        return kinematic_preds, dummy_sbp, dummy_mu, dummy_logvar