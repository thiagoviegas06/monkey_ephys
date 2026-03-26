import os
import argparse
import sys
import torch
from tqdm import tqdm

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.dataloader import get_dataloaders
from src_kin.model import PerceiverIOKinematicDecoder
from src_kin.losses import kinematic_perceiver_loss, get_r2_components, calculate_global_r2
from src_mae.model import SBP_TCN_Transformer
from src_kin.compute_kin_stats import compute_kin_stats, compute_sbp_stats

def build_mae_model(config):
    # Builds the SBP_TCN_Transformer with Phase 1 config
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    checkpoint = torch.load(config.mae_checkpoint_path, map_location=config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(config.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded Phase 1 MAE model from {config.mae_checkpoint_path}")
    return model

def build_kinematic_model(config):
    model = PerceiverIOKinematicDecoder(config)
    return model.to(config.device)

def train_one_epoch(mae_model, kinematic_model, dataloader, optimizer, config, epoch, step, kin_mean=None, kin_std=None, sbp_mean_global=None, sbp_std_global=None):
    kinematic_model.train()
    
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
    if sbp_mean_global is not None:
        sbp_mean_global = sbp_mean_global.to(config.device)
        sbp_std_global = sbp_std_global.to(config.device)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    for batch in pbar:
        sbp_masked = batch["sbp_masked"].to(config.device) # (B, W, C)
        mask = batch["mask"].to(config.device) # (B, W, C)
        macro_timestamp = batch["macro_timestamp"].to(config.device).float() # (B,)
        session_num = batch["session_num"].to(config.device).float() # (B, 1)
        kin_target = batch["kin"].to(config.device) # (B, W, 4)
        
        # Normalize targets (Global Z-Score)
        if kin_mean is not None:
            kin_target = (kin_target - kin_mean) / kin_std
        
        batch_size = sbp_masked.size(0)
        optimizer.zero_grad()
        
        # Phase 1: Impute missing channels
        with torch.no_grad():
            # MAE expects macro_timestamp as (B, 1) and optional session stats
            # For simplicity in Kin decoder, we pass global stats if provided, or None
            # The MAE model will use window-local stats if channel_mean is None.
            # But better to provide the sbp_mean_global to handle fully masked channels.
            
            # Correctly expand global stats to (batch_size, sbp_channels)
            channel_mean = sbp_mean_global.view(1, 1).expand(batch_size, config.sbp_channels) if sbp_mean_global is not None else None
            channel_var = (sbp_std_global**2).view(1, 1).expand(batch_size, config.sbp_channels) if sbp_std_global is not None else None
            
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp.unsqueeze(-1), 
                                   channel_mean=channel_mean,
                                   channel_var=channel_var)
            # The imputed signal shouldn't have gradients flowing back to MAE
            sbp_imputed = sbp_imputed.detach()
            
        # Phase 2: Perceiver IO Decoder
        kin_pred, _, _, _ = kinematic_model(sbp_imputed, mask=mask, session_num=session_num, macro_timestamp=macro_timestamp, sbp_mean=sbp_mean_global, sbp_std=sbp_std_global)
        
        # Loss
        loss_dict = kinematic_perceiver_loss(
            kin_pred, kin_target, config
        )
        loss = loss_dict["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kinematic_model.parameters(), max_norm=5.0)
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
            'R2': f"{running_r2_score:.4f}"
        })
        
    final_r2 = calculate_global_r2(total_ss_res, total_sum_y, total_sum_y_sq, total_count)
    return total_loss / total_samples, total_recon / total_samples, final_r2, step

def validate_one_epoch(mae_model, kinematic_model, dataloader, config, epoch, step, kin_mean=None, kin_std=None, sbp_mean_global=None, sbp_std_global=None):
    kinematic_model.eval()
    
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
    if sbp_mean_global is not None:
        sbp_mean_global = sbp_mean_global.to(config.device)
        sbp_std_global = sbp_std_global.to(config.device)
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")
    with torch.no_grad():
        for batch in pbar:
            sbp_masked = batch["sbp_masked"].to(config.device)
            mask = batch["mask"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].to(config.device).float()
            session_num = batch["session_num"].to(config.device).float()
            kin_target = batch["kin"].to(config.device)
            
            # Normalize targets
            if kin_mean is not None:
                kin_target = (kin_target - kin_mean) / kin_std
            
            batch_size = sbp_masked.size(0)
            
            # Correctly expand global stats to (batch_size, sbp_channels)
            channel_mean = sbp_mean_global.view(1, 1).expand(batch_size, config.sbp_channels) if sbp_mean_global is not None else None
            channel_var = (sbp_std_global**2).view(1, 1).expand(batch_size, config.sbp_channels) if sbp_std_global is not None else None
            
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp.unsqueeze(-1),
                                   channel_mean=channel_mean,
                                   channel_var=channel_var)
            
            kin_pred, _, _, _ = kinematic_model(sbp_imputed, mask=mask, session_num=session_num, macro_timestamp=macro_timestamp, sbp_mean=sbp_mean_global, sbp_std=sbp_std_global)
            
            loss_dict = kinematic_perceiver_loss(
                kin_pred, kin_target, config
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
    kinematic_model = build_kinematic_model(config)
    
    optimizer = torch.optim.AdamW(kinematic_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
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
        train_loss, train_recon, train_r2, step = train_one_epoch(mae_model, kinematic_model, train_loader, optimizer, config, epoch, step, kin_mean, kin_std, sbp_mean, sbp_std)
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f} R2={train_r2:.4f}")
        
        val_loss, val_recon, val_r2 = validate_one_epoch(mae_model, kinematic_model, val_loader, config, epoch, step, kin_mean, kin_std, sbp_mean, sbp_std)
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f} R2={val_r2:.4f}")
        
        # Save best model based on Recon (MSE) as it's more stable and correlated with R2
        if val_recon < best_val_recon - config.early_stopping_min_delta:
            best_val_recon = val_recon
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_perceiver.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': kinematic_model.state_dict(),
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
