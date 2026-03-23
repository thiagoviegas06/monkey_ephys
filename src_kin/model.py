import torch
import torch.nn as nn

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