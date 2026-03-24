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
from src_kin.losses import lfads_loss, get_r2_components, calculate_global_r2
from src_mae.model import SBP_TCN_Transformer
from src_kin.compute_kin_stats import compute_kin_stats, compute_sbp_stats

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
        output_dim=config.output_dim,
        dropout=config.lfads_dropout
    )
    return model.to(config.device)

def train_one_epoch(mae_model, lfads_model, dataloader, optimizer, config, epoch, step, kin_mean=None, kin_std=None, sbp_mean=None, sbp_std=None):
    lfads_model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_samples = 0
    
    total_ss_res = 0.0
    total_sum_y = 0.0
    total_sum_y_sq = 0.0
    total_count = 0.0
    
    # Ensure stats are on the correct device
    if kin_mean is not None:
        kin_mean = kin_mean.to(config.device)
        kin_std = kin_std.to(config.device)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    for batch in pbar:
        sbp_masked = batch["sbp_masked"].to(config.device) # (B, W, C)
        mask = batch["mask"].to(config.device) # (B, W, C)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float() # (B, 1)
        kin_target = batch["kin"].to(config.device) # (B, W, 4)
        
        # Normalize targets (Global Z-Score)
        if kin_mean is not None:
            kin_target = (kin_target - kin_mean) / kin_std
        
        batch_size = sbp_masked.size(0)
        optimizer.zero_grad()
        
        # Phase 1: Impute missing channels
        with torch.no_grad():
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            # The imputed signal shouldn't have gradients flowing back to MAE
            sbp_imputed = sbp_imputed.detach()
            
        # Phase 2: LFADS Decoder
        kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed, mask=mask, sbp_mean=sbp_mean, sbp_std=sbp_std)
        
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
            ss_res, sum_y, sum_y_sq, count = get_r2_components(kin_pred, kin_target)
            total_ss_res += ss_res
            total_sum_y += sum_y
            total_sum_y_sq += sum_y_sq
            total_count += count
            
            # Running global R2
            running_ss_tot = total_sum_y_sq - (total_sum_y ** 2) / total_count
            running_r2 = 1 - (total_ss_res / (running_ss_tot + 1e-8))
            running_r2_score = running_r2.mean().item()
        
        step += 1
        total_loss += loss.item() * batch_size
        total_recon += loss_dict["recon_mse"].item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'loss': f"{loss.item():.2f}",
            'mse': f"{loss_dict['recon_mse'].item():.4f}",
            'corr': f"{loss_dict['corr_loss'].item():.4f}",
            'accel': f"{loss_dict['accel_loss'].item():.4f}",
            'R2': f"{running_r2_score:.4f}",
            'beta': f"{loss_dict['beta']:.4f}"
        })
        
    final_r2 = calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count)
    return total_loss / total_samples, total_recon / total_samples, final_r2, step

def validate_one_epoch(mae_model, lfads_model, dataloader, config, epoch, step, kin_mean=None, kin_std=None, sbp_mean=None, sbp_std=None):
    lfads_model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_samples = 0
    
    total_ss_res = 0.0
    total_sum_y = 0.0
    total_sum_y_sq = 0.0
    total_count = 0.0
    
    # Ensure stats are on the correct device
    if kin_mean is not None:
        kin_mean = kin_mean.to(config.device)
        kin_std = kin_std.to(config.device)
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")
    with torch.no_grad():
        for batch in pbar:
            sbp_masked = batch["sbp_masked"].to(config.device)
            mask = batch["mask"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            kin_target = batch["kin"].to(config.device)
            
            # Normalize targets
            if kin_mean is not None:
                kin_target = (kin_target - kin_mean) / kin_std
            
            batch_size = sbp_masked.size(0)
            
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed, mask=mask, sbp_mean=sbp_mean, sbp_std=sbp_std)
            
            loss_dict = lfads_loss(
                kin_pred, kin_target, mu, logvar, step, config, sbp_pred, sbp_imputed
            )
            loss = loss_dict["loss"]
            
            # Metrics
            ss_res, sum_y, sum_y_sq, count = get_r2_components(kin_pred, kin_target)
            total_ss_res += ss_res
            total_sum_y += sum_y
            total_sum_y_sq += sum_y_sq
            total_count += count
            
            running_ss_tot = total_sum_y_sq - (total_sum_y ** 2) / total_count
            running_r2 = 1 - (total_ss_res / (running_ss_tot + 1e-8))
            running_r2_score = running_r2.mean().item()
            
            total_loss += loss.item() * batch_size
            total_recon += loss_dict["recon_mse"].item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'val_loss': f"{loss.item():.2f}",
                'val_mse': f"{loss_dict['recon_mse'].item():.4f}",
                'val_R2': f"{running_r2_score:.4f}"
            })
            
    final_r2 = calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count)
    return total_loss / total_samples, total_recon / total_samples, final_r2

def main():
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("Building models...")
    mae_model = build_mae_model(config)
    lfads_model = build_lfads_model(config)
    
    optimizer = torch.optim.AdamW(lfads_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(config, num_workers=4)
    
    # Compute global statistics for kinematics
    print("Computing kinematics statistics...")
    kin_mean, kin_std = compute_kin_stats(config.data_path)
    
    # Compute global statistics for SBP
    print("Computing SBP statistics...")
    sbp_mean, sbp_std = compute_sbp_stats(config.data_path)
    
    best_val_recon = float('inf')
    epochs_without_improvement = 0
    step = 0
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_recon, train_r2, step = train_one_epoch(mae_model, lfads_model, train_loader, optimizer, config, epoch, step, kin_mean, kin_std, sbp_mean, sbp_std)
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f} R2={train_r2:.4f}")
        
        val_loss, val_recon, val_r2 = validate_one_epoch(mae_model, lfads_model, val_loader, config, epoch, step, kin_mean, kin_std, sbp_mean, sbp_std)
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f} R2={val_r2:.4f}")
        
        # Save best model based on Recon (MSE) as it's more stable and correlated with R2
        if val_recon < best_val_recon - config.early_stopping_min_delta:
            best_val_recon = val_recon
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_lfads.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': lfads_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon': val_recon,
                'val_r2': val_r2,
                'kin_mean': kin_mean, # Save stats with model
                'kin_std': kin_std,
                'sbp_mean': sbp_mean,
                'sbp_std': sbp_std,
            }, best_path)
            print(f"✓ Saved best model (MSE: {val_recon:.4f}) to {best_path}")

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()
