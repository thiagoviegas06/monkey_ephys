import os
import argparse
import sys
import torch
from tqdm import tqdm

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.dataloader import get_dataloaders
from src_kin.model import LSTMKinematicDecoder
from src_kin.losses import lfads_loss, calculate_r2
from src_mae.model import SBP_TCN_Transformer
from src_kin.compute_kin_stats import compute_per_session_kin_stats
from src_kin.preprocessing import SessionDataPhase2

def build_mae_model(config):
    # Builds the SBP_TCN_Transformer with Phase 1 config
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    checkpoint = torch.load(config.mae_checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded Phase 1 MAE model from {config.mae_checkpoint_path}")
    return model

def compute_per_session_sbp_stats(data_path):
    """
    Compute SBP normalization statistics per session (z-score).
    Returns: dict {session_id: {'mean': (96,), 'std': (96,)}}
    """
    import numpy as np

    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()

    session_stats = {}

    for session in sessions:
        sbp, _ = session.load_data()
        if sbp is not None and sbp.shape[0] > 0:
            mean = np.mean(sbp, axis=0)  # (96,)
            std = np.std(sbp, axis=0) + 1e-8  # (96,) prevent division by zero

            session_stats[session.session_id] = {
                'mean': torch.tensor(mean, dtype=torch.float32),
                'std': torch.tensor(std, dtype=torch.float32)
            }

    print(f"✓ Computed per-session SBP stats for {len(session_stats)} sessions")
    return session_stats


def build_lstm_model(config):
    model = LSTMKinematicDecoder(
        num_channels=config.sbp_channels,  # 96
        encoder_dim=config.encoder_dim,  # MAE encoder feature dimension (64)
        spatial_dim=config.spatial_dim,  # 256 - learnable pooling dimension
        hidden_dim=config.hidden_dim,
        num_layers=config.num_lstm_layers,
        output_dim=config.output_dim,
        dropout=config.lstm_dropout
    )
    return model.to(config.device)

def train_one_epoch(mae_model, lstm_model, dataloader, optimizer, config, epoch, step, session_kin_stats=None, session_sbp_stats=None):
    lstm_model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_r2 = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    for batch in pbar:
        sbp_masked = batch["sbp_masked"].to(config.device) # (B, W, C)
        mask = batch["mask"].to(config.device) # (B, W, C)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float() # (B, 1)
        kin_target = batch["kin"].to(config.device) # (B, W, 4)
        session_ids = batch["session_id"]  # List of session IDs

        # Normalize SBP per-session before passing to MAE
        sbp_normalized = sbp_masked.clone()
        if session_sbp_stats is not None:
            for i, sid in enumerate(session_ids):
                if sid in session_sbp_stats:
                    mean = session_sbp_stats[sid]['mean'].to(config.device)
                    std = session_sbp_stats[sid]['std'].to(config.device)
                    sbp_normalized[i] = (sbp_masked[i] - mean) / std

        # Normalize targets per-session (Z-Score Normalization)
        if session_kin_stats is not None:
            kin_target_norm = torch.zeros_like(kin_target)
            for i, sid in enumerate(session_ids):
                if sid in session_kin_stats:
                    mean = session_kin_stats[sid]['mean'].to(config.device)
                    std = session_kin_stats[sid]['std'].to(config.device)
                    kin_target_norm[i] = (kin_target[i] - mean) / std
                else:
                    kin_target_norm[i] = kin_target[i]  # Fallback: no normalization
            kin_target = kin_target_norm

        batch_size = sbp_masked.size(0)
        optimizer.zero_grad()

        # Phase 1: Extract learned representations from MAE encoder
        # Use full (unmasked) SBP since kinematics shouldn't depend on masking patterns
        # Create zero mask (all visible) for encoder
        zero_mask = torch.zeros_like(sbp_normalized, dtype=torch.bool)

        with torch.no_grad():
            encoder_features = mae_model.extract_encoder_features(sbp_normalized, zero_mask, macro_timestamp)
            # encoder_features: (B, W, C, d_model) - keep per-channel features
            # Don't average - let spatial_pool layer learn per-channel importance

        # Phase 2: LSTM Kinematics Decoder (uses encoder features, not SBP)
        kin_pred, sbp_pred, mu, logvar = lstm_model(encoder_features, mask=mask)
        
        # Loss
        loss_dict = lfads_loss(
            kin_pred, kin_target, mu, logvar, step, config
        )
        loss = loss_dict["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Metrics (not for backprop)
        with torch.no_grad():
            r2_score = calculate_r2(kin_pred, kin_target)
        
        step += 1
        total_loss += loss.item() * batch_size
        total_recon += loss_dict["recon_mse"].item() * batch_size
        total_r2 += r2_score * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'loss': f"{loss.item():.2f}",
            'mse': f"{loss_dict['recon_mse'].item():.4f}",
            'corr': f"{loss_dict['corr_loss'].item():.4f}",
            'accel': f"{loss_dict['accel_loss'].item():.4f}",
            'R2': f"{r2_score:.4f}",
            'beta': f"{loss_dict['beta']:.4f}"
        })
        
    return total_loss / total_samples, total_recon / total_samples, total_r2 / total_samples, step

def validate_one_epoch(mae_model, lstm_model, dataloader, config, epoch, step, session_kin_stats=None, session_sbp_stats=None):
    lstm_model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_r2 = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")
    with torch.no_grad():
        for batch in pbar:
            sbp_masked = batch["sbp_masked"].to(config.device)
            mask = batch["mask"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            kin_target = batch["kin"].to(config.device)
            session_ids = batch["session_id"]  # List of session IDs

            # Normalize SBP per-session before passing to MAE
            sbp_normalized = sbp_masked.clone()
            if session_sbp_stats is not None:
                for i, sid in enumerate(session_ids):
                    if sid in session_sbp_stats:
                        mean = session_sbp_stats[sid]['mean'].to(config.device)
                        std = session_sbp_stats[sid]['std'].to(config.device)
                        sbp_normalized[i] = (sbp_masked[i] - mean) / std

            # Normalize targets per-session (Z-Score Normalization)
            if session_kin_stats is not None:
                kin_target_norm = torch.zeros_like(kin_target)
                for i, sid in enumerate(session_ids):
                    if sid in session_kin_stats:
                        mean = session_kin_stats[sid]['mean'].to(config.device)
                        std = session_kin_stats[sid]['std'].to(config.device)
                        kin_target_norm[i] = (kin_target[i] - mean) / std
                    else:
                        kin_target_norm[i] = kin_target[i]  # Fallback: no normalization
                kin_target = kin_target_norm

            batch_size = sbp_masked.size(0)

            # Extract encoder features from MAE using full (unmasked) SBP
            zero_mask = torch.zeros_like(sbp_normalized, dtype=torch.bool)
            encoder_features = mae_model.extract_encoder_features(sbp_normalized, zero_mask, macro_timestamp)
            # encoder_features: (B, W, C, d_model) - keep per-channel features
            kin_pred, sbp_pred, mu, logvar = lstm_model(encoder_features, mask=mask)
            
            loss_dict = lfads_loss(
                kin_pred, kin_target, mu, logvar, step, config
            )
            loss = loss_dict["loss"]
            
            # Metrics
            r2_score = calculate_r2(kin_pred, kin_target)
            
            total_loss += loss.item() * batch_size
            total_recon += loss_dict["recon_mse"].item() * batch_size
            total_r2 += r2_score * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'val_loss': f"{loss.item():.2f}",
                'val_mse': f"{loss_dict['recon_mse'].item():.4f}",
                'val_R2': f"{r2_score:.4f}"
            })
            
    return total_loss / total_samples, total_recon / total_samples, total_r2 / total_samples

def main():
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("Building models...")
    mae_model = build_mae_model(config)
    lstm_model = build_lstm_model(config)

    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(config, num_workers=4)

    # Compute PER-SESSION SBP statistics (Z-score normalization for MAE input)
    print("Computing per-session SBP statistics...")
    session_sbp_stats = compute_per_session_sbp_stats(config.data_path)

    # Compute PER-SESSION kinematics statistics (Z-score normalization)
    print("Computing per-session kinematics statistics...")
    print("  (Project requirement: 'per-session z-score normalization is THE key ingredient for drift handling')")
    session_kin_stats = compute_per_session_kin_stats(config.data_path)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    step = 0

    # =========================================================================
    # STAGE 1: Frozen MAE (Train LSTM only) - 20 epochs
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 1: Frozen MAE - Training LSTM only")
    print("="*70)

    for epoch in range(1, config.frozen_epochs + 1):
        train_loss, train_recon, train_r2, step = train_one_epoch(
            mae_model, lstm_model, train_loader, optimizer, config, epoch, step,
            session_kin_stats=session_kin_stats, session_sbp_stats=session_sbp_stats
        )
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f} R2={train_r2:.4f}")

        val_loss, val_recon, val_r2 = validate_one_epoch(
            mae_model, lstm_model, val_loader, config, epoch, step,
            session_kin_stats=session_kin_stats, session_sbp_stats=session_sbp_stats
        )
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f} R2={val_r2:.4f}")

        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_lstm.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': lstm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'session_kin_stats': session_kin_stats,
                'stage': 'frozen'
            }, best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1

    # =========================================================================
    # STAGE 2: Unfreeze MAE + LSTM (Fine-tuning) - 10 epochs
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning - Training MAE + LSTM")
    print("="*70)

    # Unfreeze MAE
    for param in mae_model.parameters():
        param.requires_grad = True
    print("✓ Unfroze MAE model")

    # Create new optimizer for both models with lower learning rate
    optimizer_finetune = torch.optim.AdamW(
        list(mae_model.parameters()) + list(lstm_model.parameters()),
        lr=config.finetune_lr,
        weight_decay=config.weight_decay
    )
    print(f"✓ Created new optimizer with lr={config.finetune_lr}")

    epochs_without_improvement = 0  # Reset counter for finetuning stage

    for epoch in range(config.frozen_epochs + 1, config.frozen_epochs + config.finetune_epochs + 1):
        train_loss, train_recon, train_r2, step = train_one_epoch(
            mae_model, lstm_model, train_loader, optimizer_finetune, config, epoch, step,
            session_kin_stats=session_kin_stats, session_sbp_stats=session_sbp_stats
        )
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f} R2={train_r2:.4f}")

        val_loss, val_recon, val_r2 = validate_one_epoch(
            mae_model, lstm_model, val_loader, config, epoch, step,
            session_kin_stats=session_kin_stats, session_sbp_stats=session_sbp_stats
        )
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f} R2={val_r2:.4f}")

        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_lstm.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': lstm_model.state_dict(),
                'mae_state_dict': mae_model.state_dict(),
                'optimizer_state_dict': optimizer_finetune.state_dict(),
                'val_loss': val_loss,
                'session_kin_stats': session_kin_stats,
                'stage': 'finetuned'
            }, best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)

if __name__ == "__main__":
    main()
