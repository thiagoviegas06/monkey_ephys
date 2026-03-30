import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src_mae to path to allow imports from dataloader, model, etc.
src_mae_path = os.path.abspath("src_mae")
if src_mae_path not in sys.path:
    sys.path.append(src_mae_path)

from model import SBP_TCN_Transformer
from config import Config
from dataloader import get_dataloaders

def main():
    # 1. Load Config
    config = Config()
    config.batch_size = 1
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.device}")
    
    # 2. Initialize Model
    # Explicitly using hyperparameters from config
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    
    # 3. Load Checkpoint
    checkpoint_path = "checkpoints_200/best_model_tcn_transformer.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
    
    model.eval()
    
    # 4. Get Data
    print("Loading data from validation set...")
    # We only need one sample, but get_dataloaders will load the sessions first.
    # Note: This might take a few seconds as it loads sessions into RAM.
    _, val_loader, _, _ = get_dataloaders(config, batch_size=1, val_split=0.1, shuffle=True)
    
    # 5. Sample a window from val_loader
    print("Sampling a window...")
    batch = next(iter(val_loader))
    
    # Print session ID and macro timestamp for reference
    print(f"Session ID: {batch['session_id'][0]}, Macro Timestamp: {batch['macro_timestamp'][0].item()}")

    y_sbp = batch["y_sbp"].to(config.device) # Ground Truth (B, W, C)
    channel_mean = batch["channel_mean"].to(config.device)
    channel_var = batch["channel_var"].to(config.device)
    # macro_timestamp needs to be (B, 1) for the model's BatchNorm1d
    macro_timestamp = batch["macro_timestamp"].unsqueeze(-1).to(config.device).float()
    
    B, W, C = y_sbp.shape
    
    # 6. Apply 30% channel masking manually
    # The prompt asks for 30% masked channels (zeroed out)
    num_masked_channels = int(C * 0.3)
    masked_indices = torch.randperm(C)[:num_masked_channels]
    
    mask = torch.zeros((B, W, C), device=config.device)
    mask[:, :, masked_indices] = 1.0
    
    x_sbp = y_sbp.clone()
    x_sbp = x_sbp * (1.0 - mask) # Zero out masked channels
    
    # 7. Forward Pass for Reconstruction
    print("Performing forward pass...")
    with torch.no_grad():
        reconstruction = model(
            sbp_masked=x_sbp,
            mask=mask,
            macro_time=macro_timestamp,
            channel_mean=channel_mean,
            channel_var=channel_var
        )
    
    # 8. Plotting Comparison
    print("Generating plots...")
    gt_plot = y_sbp[0].cpu().numpy()        # (W, C)
    masked_plot = x_sbp[0].cpu().numpy()   # (W, C)
    recon_plot = reconstruction[0].cpu().numpy() # (W, C)
    
    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Use common color scale based on Ground Truth
    vmin = gt_plot.min()
    vmax = gt_plot.max()
    
    # Transpose to show Channels on y-axis, Time on x-axis (W, C) -> (C, W)
    im1 = axes[0].imshow(gt_plot.T, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='magma')
    axes[0].set_title("Ground Truth SBP Heatmap")
    axes[0].set_ylabel("Channels")
    axes[0].set_xlabel("Time (Bins)")
    
    im2 = axes[1].imshow(masked_plot.T, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='magma')
    axes[1].set_title(f"Masked Input ({num_masked_channels} Ch Masked)")
    axes[1].set_xlabel("Time (Bins)")
    
    im3 = axes[2].imshow(recon_plot.T, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='magma')
    axes[2].set_title("Model Reconstruction")
    axes[2].set_xlabel("Time (Bins)")
    
    # Add a single colorbar for the figure
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im3, cax=cbar_ax, label='SBP Intensity')
    
    save_path = "mae_reconstruction.png"
    plt.savefig(save_path, dpi=150)
    print(f"Successfully saved reconstruction plot to {save_path}")

if __name__ == "__main__":
    main()
