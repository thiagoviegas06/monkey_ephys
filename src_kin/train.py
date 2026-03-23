import os
import argparse
import sys
import torch
from tqdm import tqdm

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.dataloader import get_dataloaders
from src_kin.model import LFADSKinematicDecoder
from src_kin.losses import lfads_loss, calculate_r2
from src_mae.model import SBP_TCN_Transformer
from src_kin.compute_kin_stats import compute_per_session_kin_stats

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

def build_lfads_model(config):
    model = LFADSKinematicDecoder(
        input_dim=config.sbp_channels,
        hidden_dim=config.hidden_dim,
        gen_dim=config.gen_dim,
        factor_dim=config.factor_dim,
        output_dim=config.output_dim
    )
    return model.to(config.device)

def train_one_epoch(mae_model, lfads_model, dataloader, optimizer, config, epoch, step, session_kin_stats=None):
    lfads_model.train()

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
        
        # Phase 1: Impute missing channels
        with torch.no_grad():
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            # The imputed signal shouldn't have gradients flowing back to MAE
            sbp_imputed = sbp_imputed.detach()
            
        # Phase 2: LFADS Decoder
        kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed, mask=mask)
        
        # Loss
        loss_dict = lfads_loss(
            kin_pred, kin_target, mu, logvar, step, config, sbp_pred, sbp_imputed
        )
        loss = loss_dict["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lfads_model.parameters(), max_norm=5.0)
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

def validate_one_epoch(mae_model, lfads_model, dataloader, config, epoch, step, session_kin_stats=None):
    lfads_model.eval()

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
            
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed, mask=mask)
            
            loss_dict = lfads_loss(
                kin_pred, kin_target, mu, logvar, step, config, sbp_pred, sbp_imputed
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
    lfads_model = build_lfads_model(config)
    
    optimizer = torch.optim.AdamW(lfads_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(config, num_workers=4)

    # Compute PER-SESSION kinematics statistics (Z-score normalization)
    print("Computing per-session kinematics statistics...")
    print("  (Project requirement: 'per-session z-score normalization is THE key ingredient for drift handling')")
    session_kin_stats = compute_per_session_kin_stats(config.data_path)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    step = 0

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_recon, train_r2, step = train_one_epoch(mae_model, lfads_model, train_loader, optimizer, config, epoch, step, session_kin_stats)
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f} R2={train_r2:.4f}")

        val_loss, val_recon, val_r2 = validate_one_epoch(mae_model, lfads_model, val_loader, config, epoch, step, session_kin_stats)
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f} R2={val_r2:.4f}")
        
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_lfads.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': lfads_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'kin_mean': kin_mean, # Save stats with model
                'kin_std': kin_std,
            }, best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()
