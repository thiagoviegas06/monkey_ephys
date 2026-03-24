import torch
import torch.nn as nn


class LSTMKinematicDecoder(nn.Module):
    """
    LSTM-based kinematics decoder using MAE encoder features.
    Input: (B, W, C, d_model) learned representations from MAE encoder
    Output: (B, W, 4) predicted kinematics

    Spatial pooling: (B, W, C*d_model) → (B, W, 256) via learnable linear layer
    This preserves per-channel information instead of averaging.
    """
    def __init__(self, num_channels=96, encoder_dim=64, spatial_dim=256, hidden_dim=128, num_layers=2, output_dim=4, dropout=0.1):
        super(LSTMKinematicDecoder, self).__init__()

        self.num_channels = num_channels
        self.encoder_dim = encoder_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Spatial pooling: (B, W, C*d_model) → (B, W, spatial_dim)
        # Learns weighted combination of all channel features
        self.spatial_pool = nn.Linear(num_channels * encoder_dim, spatial_dim)

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection: hidden_dim*2 (bidirectional) → output_dim
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, encoder_features, mask=None):
        """
        Args:
            encoder_features: (B, W, C, d_model) learned features from MAE encoder
            mask: unused, kept for compatibility

        Returns:
            kin_pred: (B, W, 4) kinematics predictions
            sbp_pred: dummy (unused, for compatibility)
            mu: dummy (unused, for compatibility)
            logvar: dummy (unused, for compatibility)
        """
        # Flatten channels and d_model: (B, W, C, d_model) → (B, W, C*d_model)
        batch_size, seq_len, num_channels, d_model = encoder_features.size()
        encoder_features_flat = encoder_features.reshape(batch_size, seq_len, num_channels * d_model)

        # Spatial pooling: (B, W, C*d_model) → (B, W, spatial_dim)
        spatial_features = self.spatial_pool(encoder_features_flat)

        # LSTM forward pass on spatially pooled features
        # lstm_out: (B, W, hidden_dim*2)
        lstm_out, _ = self.lstm(spatial_features)

        # Project to kinematics
        # (B, W, hidden_dim*2) → (B, W, 4)
        kin_pred = self.fc_out(lstm_out)

        # Return dummy values for compatibility with training code
        sbp_pred = torch.zeros(batch_size, seq_len, 1, device=encoder_features.device)
        mu = torch.zeros(batch_size, self.hidden_dim, device=encoder_features.device)
        logvar = torch.zeros(batch_size, self.hidden_dim, device=encoder_features.device)

        return kin_pred, sbp_pred, mu, logvar


class LFADSKinematicDecoder(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, gen_dim=128, factor_dim=40, output_dim=4):
        super(LFADSKinematicDecoder, self).__init__()
        
        # 1. Encoder: Bidirectional GRU to read the imputed SBP sequence
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Map the final bidirectional hidden states to mu and logvar for g0
        self.fc_mu = nn.Linear(hidden_dim * 2, gen_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, gen_dim)
        
        # 2. Generator: Autonomous GRUCell representing the dynamical system
        self.generator = nn.GRUCell(input_size=1, hidden_size=gen_dim) 
        
        # 3. Decoders: Map generator state to latent factors, then to kinematics and SBP
        self.fc_factors = nn.Linear(gen_dim, factor_dim)
        self.fc_kinematics = nn.Linear(factor_dim, output_dim)
        self.fc_sbp = nn.Linear(factor_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # --- NORMALIZATION ---
        # Spatial-Temporal Mask-Aware Normalization
        # In Phase 2, certain channels are zeroed out for the whole session.
        # We calculate global stats for the window across active channels only.
        if mask is None:
            # If no mask provided, assume values that are exactly 0 are masked
            mask = (x == 0.0).float()
            
        visible_mask = (~mask.bool()).float()  # 1.0 if visible, 0.0 if masked
        num_visible = visible_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        
        # Calculate global mean and std across ALL active channels in this window
        mean = (x * visible_mask).sum(dim=(1, 2), keepdim=True) / num_visible
        var = (((x - mean) * visible_mask) ** 2).sum(dim=(1, 2), keepdim=True) / num_visible
        std = torch.sqrt(var + 1e-5)
        
        x_norm = ((x - mean) / std) * visible_mask
        
        # --- ENCODER ---
        _, h_n = self.encoder(x_norm)
        # Concatenate final forward and backward hidden states
        h_n_concat = torch.cat((h_n[0], h_n[1]), dim=1) 
        
        # Get distribution for initial generator state
        mu = self.fc_mu(h_n_concat)
        logvar = self.fc_logvar(h_n_concat)
        
        # Sample initial state g_0
        g_t = self.reparameterize(mu, logvar)
        
        # --- GENERATOR & DECODER ---
        kinematic_preds = []
        sbp_preds = []
        
        # Dummy input for the autonomous generator (it runs purely on its hidden state)
        dummy_input = torch.zeros(batch_size, 1, device=x.device)
        
        for t in range(seq_len):
            # Evolve the dynamical system
            g_t = self.generator(dummy_input, g_t)
            
            # Map to latent factors
            f_t = self.fc_factors(g_t)
            
            # Map to kinematics
            kin_t = self.fc_kinematics(f_t)
            kinematic_preds.append(kin_t.unsqueeze(1))
            
            # Map to SBP
            sbp_t = self.fc_sbp(f_t)
            
            # Un-normalize SBP prediction back to original scale 
            # (matches sbp_imputed in loss calculation)
            # mean and std are (B, 1, 1), so they broadcast correctly across (B, C)
            sbp_t_unnorm = (sbp_t * std.squeeze(-1)) + mean.squeeze(-1)
            sbp_preds.append(sbp_t_unnorm.unsqueeze(1))
            
        kinematic_preds = torch.cat(kinematic_preds, dim=1)
        sbp_preds = torch.cat(sbp_preds, dim=1)
        
        # Note: Since positions are normalized to [0, 1], you could apply a Sigmoid 
        # to the first two indices of kinematic_preds here, but it's often better 
        # to leave it linear and handle the bounding via the loss function or data scaling.
        
        return kinematic_preds, sbp_preds, mu, logvar