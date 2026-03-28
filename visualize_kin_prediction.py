import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add root directory to path to import src_mae module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_kin.config import Config
from src_kin.dataloader import get_dataloaders
from src_kin.model import KinematicDecoderTransformer
from src_mae.model import SBP_TCN_Transformer

def build_models(config):
    # MAE
    mae = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"Loading MAE model from {config.mae_checkpoint_path}...")
    mae_checkpoint = torch.load(config.mae_checkpoint_path, map_location=config.device)
    if 'model_state_dict' in mae_checkpoint:
        mae.load_state_dict(mae_checkpoint['model_state_dict'])
    else:
        mae.load_state_dict(mae_checkpoint)
    mae.eval()

    # Kinematic Decoder
    kinematic_model = KinematicDecoderTransformer(
        d_model=config.d_model,
        window_size=config.window_size,
        num_channels=config.sbp_channels,
        num_temporal_layers=config.decoder_num_temporal_layers,
        num_heads=config.decoder_num_heads,
        output_dim=config.output_dim,
        dropout=config.decoder_dropout
    ).to(config.device)
    
    best_model_path = os.path.join(config.checkpoint_dir, "best_model_perceiver.pt")
    if os.path.exists(best_model_path):
        print(f"Loading kinematic model from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=config.device)
        kinematic_model.load_state_dict(checkpoint['model_state_dict'])
        kin_mean = checkpoint.get('kin_mean', None)
        kin_std = checkpoint.get('kin_std', None)
        sbp_mean = checkpoint.get('sbp_mean', None)
        sbp_std = checkpoint.get('sbp_std', None)
    else:
        print(f"Warning: {best_model_path} not found. Using untrained model and computing stats...")
        from src_kin.compute_kin_stats import compute_kin_stats, compute_sbp_stats
        kin_mean, kin_std = compute_kin_stats(config.data_path, getattr(config, 'phase1_data_path', None))
        sbp_mean, sbp_std = compute_sbp_stats(config.data_path, getattr(config, 'phase1_data_path', None))

    kinematic_model.eval()
    return mae, kinematic_model, kin_mean, kin_std, sbp_mean, sbp_std

def visualize_kin_prediction(config, session_id=None, sample_idx=0):
    mae, kinematic_model, kin_mean, kin_std, sbp_mean_global, sbp_std_global = build_models(config)
    
    # Load data
    print("Loading validation data...")
    # Use small val_split just to get some data quickly if needed, but get_dataloaders handles it
    _, val_loader, _, _ = get_dataloaders(config, num_workers=0)
    
    # Find a batch
    found_batch = None
    for batch in val_loader:
        if session_id is None:
            found_batch = batch
            break
        else:
            if session_id in batch["session_id"]:
                found_batch = batch
                break
    
    if found_batch is None:
        print(f"Session {session_id} not found in validation set.")
        return

    # Process one sample from the batch
    idx = sample_idx
    if session_id:
        try:
            idx = found_batch["session_id"].index(session_id)
        except ValueError:
            idx = 0

    sbp_masked = found_batch["sbp_masked"][idx:idx+1].to(config.device)
    mask = found_batch["mask"][idx:idx+1].to(config.device)
    macro_timestamp = found_batch["macro_timestamp"][idx:idx+1].to(config.device)
    kin_target = found_batch["kin"][idx:idx+1].to(config.device)
    actual_session_id = found_batch["session_id"][idx]

    with torch.no_grad():
        # Correctly expand global stats
        B = 1
        channel_mean = sbp_mean_global.view(1, 1).expand(B, config.sbp_channels).to(config.device) if sbp_mean_global is not None else None
        channel_var = (sbp_std_global**2).view(1, 1).expand(B, config.sbp_channels).to(config.device) if sbp_std_global is not None else None
        
        if getattr(config, 'use_mae_embeddings', False):
            mae_out = mae(sbp_masked, mask, macro_timestamp.unsqueeze(-1), 
                               channel_mean=channel_mean,
                               channel_var=channel_var,
                               return_embeddings=True)
        else:
            mae_out = mae(sbp_masked, mask, macro_timestamp.unsqueeze(-1), 
                               channel_mean=channel_mean,
                               channel_var=channel_var)
        
        kin_pred = kinematic_model(mae_out)
        
        # Un-normalize
        if kin_mean is not None:
            # Check if kin_pred is normalized (it usually is if trained with normalized targets)
            # In train.py: kin_target = (kin_target - kin_mean) / kin_std
            kin_pred_unnorm = kin_pred * kin_std.to(config.device) + kin_mean.to(config.device)
        else:
            kin_pred_unnorm = kin_pred

    # Plot
    kin_target_np = kin_target.squeeze().cpu().numpy()
    kin_pred_np = kin_pred_unnorm.squeeze().cpu().numpy()
    
    # Clip predictions to [0, 1] for positions
    kin_pred_np[:, :2] = np.clip(kin_pred_np[:, :2], 0, 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Kinematic Prediction: Session {actual_session_id} | Window size {config.window_size}", fontsize=16)
    
    channels = ["Index Position", "MRP Position"]
    for i in range(2):
        axes[i].plot(kin_target_np[:, i], 'k-', label="Ground Truth", alpha=0.7, linewidth=2)
        axes[i].plot(kin_pred_np[:, i], 'r--', label="Prediction", linewidth=1.5)
        axes[i].set_title(channels[i], fontweight='bold')
        axes[i].set_ylabel("Normalized Position [0, 1]")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.1, 1.1)
        
    axes[1].set_xlabel("Time Bin (20ms units)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_name = f"prediction_{actual_session_id}.png"
    plt.savefig(save_name, dpi=150)
    print(f"Saved visualization to {save_name}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    config = Config()
    visualize_kin_prediction(config, session_id=args.session_id, sample_idx=args.sample_idx)
