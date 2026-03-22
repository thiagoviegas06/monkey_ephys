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
from src_kin.losses import lfads_loss
from src_mae.model import SBP_TCN_Transformer

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

def train_one_epoch(mae_model, lfads_model, dataloader, optimizer, config, epoch, step):
    lfads_model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    for batch in pbar:
        sbp_masked = batch["sbp_masked"].to(config.device) # (B, W, C)
        mask = batch["mask"].to(config.device) # (B, W, C)
        macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float() # (B, 1)
        kin_target = batch["kin"].to(config.device) # (B, W, 4)
        
        batch_size = sbp_masked.size(0)
        optimizer.zero_grad()
        
        # Phase 1: Impute missing channels
        with torch.no_grad():
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            # The imputed signal shouldn't have gradients flowing back to MAE
            sbp_imputed = sbp_imputed.detach()
            
        # Phase 2: LFADS Decoder
        kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed)
        
        # Loss
        loss, recon_loss, kl_loss, sbp_loss, beta = lfads_loss(
            kin_pred, kin_target, mu, logvar, step, config, sbp_pred, sbp_imputed
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lfads_model.parameters(), max_norm=5.0)
        optimizer.step()
        
        step += 1
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon(MSE)': f"{recon_loss.item():.4f}",
            'beta': f"{beta:.4f}"
        })
        
    return total_loss / total_samples, total_recon / total_samples, step

def validate_one_epoch(mae_model, lfads_model, dataloader, config, epoch, step):
    lfads_model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}/{config.num_epochs}")
    with torch.no_grad():
        for batch in pbar:
            sbp_masked = batch["sbp_masked"].to(config.device)
            mask = batch["mask"].to(config.device)
            macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
            kin_target = batch["kin"].to(config.device)
            
            batch_size = sbp_masked.size(0)
            
            sbp_imputed = mae_model(sbp_masked, mask, macro_timestamp)
            kin_pred, sbp_pred, mu, logvar = lfads_model(sbp_imputed)
            
            loss, recon_loss, kl_loss, sbp_loss, beta = lfads_loss(
                kin_pred, kin_target, mu, logvar, step, config, sbp_pred, sbp_imputed
            )
            
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'val_recon': f"{recon_loss.item():.4f}"
            })
            
    return total_loss / total_samples, total_recon / total_samples

def main():
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("Building models...")
    mae_model = build_mae_model(config)
    lfads_model = build_lfads_model(config)
    
    optimizer = torch.optim.AdamW(lfads_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(config, num_workers=4)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    step = 0
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_recon, step = train_one_epoch(mae_model, lfads_model, train_loader, optimizer, config, epoch, step)
        print(f"Epoch {epoch} Train: Loss={train_loss:.4f} Recon(MSE)={train_recon:.4f}")
        
        val_loss, val_recon = validate_one_epoch(mae_model, lfads_model, val_loader, config, epoch, step)
        print(f"Epoch {epoch} Val:   Loss={val_loss:.4f} Recon(MSE)={val_recon:.4f}")
        
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model_lfads.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': lfads_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()
