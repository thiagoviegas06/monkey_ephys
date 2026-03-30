import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import os

from src_mae.config import Config
from src_mae.dataloader import get_dataloaders
from src_mae.model import SBP_TCN_Transformer

def main():
    parser = argparse.ArgumentParser(description="Diagnostic PCA of SBP Embeddings")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_200/best_model_tcn_transformer.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="kaggle_data", help="Path to data directory")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to process for analysis")
    parser.add_argument("--save_path", type=str, default="pca_visualization.png", help="Path to save the visualization")
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.data_path = args.data_dir
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load dataloaders
    print("Loading validation data...")
    _, val_loader, _, _ = get_dataloaders(config, batch_size=config.batch_size, num_workers=4)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Extract embeddings
    print("Extracting embeddings...")
    masked_embeddings = []
    unmasked_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Batches", total=args.num_batches)):
            if i >= args.num_batches:
                break
            
            x_sbp = batch["x_sbp"].to(device)
            mask = batch["mask"].to(device)
            macro_time = batch["macro_timestamp"].float().to(device).unsqueeze(1)
            channel_mean = batch["channel_mean"].to(device)
            channel_var = batch["channel_var"].to(device)
            
            # Forward pass to get embeddings
            # Embeddings shape: (B, W, C, d_model)
            embeddings = model(x_sbp, mask, macro_time, channel_mean, channel_var, return_embeddings=True)
            
            B, W, C, D = embeddings.shape
            embeddings = embeddings.reshape(-1, D)
            mask_flat = mask.reshape(-1)
            
            # Separate based on mask
            m_emb = embeddings[mask_flat == 1].cpu().numpy()
            u_emb = embeddings[mask_flat == 0].cpu().numpy()
            
            masked_embeddings.append(m_emb)
            unmasked_embeddings.append(u_emb)
            
    masked_embeddings = np.concatenate(masked_embeddings, axis=0)
    unmasked_embeddings = np.concatenate(unmasked_embeddings, axis=0)
    
    # Subsample if too many points for PCA/Visualization
    max_points = 50000
    if len(masked_embeddings) > max_points:
        idx = np.random.choice(len(masked_embeddings), max_points, replace=False)
        masked_embeddings = masked_embeddings[idx]
    if len(unmasked_embeddings) > max_points:
        idx = np.random.choice(len(unmasked_embeddings), max_points, replace=False)
        unmasked_embeddings = unmasked_embeddings[idx]

    print(f"Collected {len(masked_embeddings)} masked and {len(unmasked_embeddings)} unmasked tokens for PCA.")

    # PCA
    print("Running PCA...")
    all_embeddings = np.concatenate([masked_embeddings, unmasked_embeddings], axis=0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)
    
    pca_masked = pca_result[:len(masked_embeddings)]
    pca_unmasked = pca_result[len(masked_embeddings):]
    
    # Spread Analysis
    # Spread defined as the sum of variances along PCs
    masked_spread = np.var(pca_masked, axis=0).sum()
    unmasked_spread = np.var(pca_unmasked, axis=0).sum()
    print(f"Masked Spread (Trace of Cov): {masked_spread:.4f}")
    print(f"Unmasked Spread (Trace of Cov): {unmasked_spread:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_unmasked[:, 0], pca_unmasked[:, 1], alpha=0.2, label='Unmasked (Visible)', c='blue', s=2)
    plt.scatter(pca_masked[:, 0], pca_masked[:, 1], alpha=0.2, label='Masked (Reconstructed)', c='red', s=2)
    
    plt.title(f"PCA of SBP Decoder Embeddings\nVariance Explained: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add stats text box
    stats_text = (f"Masked Points: {len(masked_embeddings)}\n"
                  f"Unmasked Points: {len(unmasked_embeddings)}\n"
                  f"Masked Spread: {masked_spread:.4f}\n"
                  f"Unmasked Spread: {unmasked_spread:.4f}\n"
                  f"Spread Ratio (M/U): {masked_spread/unmasked_spread:.4f}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.save_path}")

if __name__ == "__main__":
    main()
