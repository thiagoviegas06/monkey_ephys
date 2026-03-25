#!/usr/bin/env python3
"""
Diagnostic script: Extract transformer embeddings and check for representation collapse.
Uses PCA to visualize if the model is learning diverse representations.
"""

import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from model import SBP_TCN_Transformer
from config import Config
from torch.utils.data import Dataset, DataLoader


class PreprocessedMaskedDataset(Dataset):
    """Load pre-generated masked windows from pickle files."""
    def __init__(self, data_dir, is_train=True, train_split=0.9, seed=42):
        self.data_dir = data_dir
        self.is_train = is_train
        self.seed = seed

        all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        if not all_files:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")

        all_files.sort()
        import random
        random.seed(seed)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * train_split)

        if is_train:
            self.file_list = all_files[:split_idx]
        else:
            self.file_list = all_files[split_idx:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        y_sbp = torch.from_numpy(sample["y_sbp"]).float()
        kin = torch.from_numpy(sample["kin"]).float()
        channel_var = torch.from_numpy(sample["channel_var"]).float()
        session_id = sample["session_id"]

        if "x_sbp" in sample:
            x_sbp = torch.from_numpy(sample["x_sbp"]).float()
        else:
            x_sbp = y_sbp.clone()
            if "mask" in sample:
                x_sbp[torch.from_numpy(sample["mask"]).bool()] = 0.0

        if "mask" in sample:
            mask = torch.from_numpy(sample["mask"]).bool()
        else:
            W, C = y_sbp.shape
            mask = torch.zeros((W, C), dtype=torch.bool)

        day = sample.get("day", 0.0)
        macro_timestamp = torch.tensor([day], dtype=torch.float32)

        return {
            "x_sbp": x_sbp,
            "y_sbp": y_sbp,
            "mask": mask,
            "kin": kin,
            "channel_var": channel_var,
            "macro_timestamp": macro_timestamp,
            "session_id": session_id,
        }


def extract_transformer_embeddings(model, dataloader, config, num_batches=5):
    """
    Extract transformer embeddings from model.
    Returns: (embeddings, labels) where embeddings is (N, d_model) and labels is (N,)
    """
    model.eval()
    embeddings_list = []
    session_labels = []

    # Hook to capture transformer output
    transformer_out = {}

    def hook_fn(module, input, output):
        # output shape: (B*W, C, d_model)
        # We want to flatten to (B*W*C, d_model)
        transformer_out['embeddings'] = output.detach().cpu()

    # Register hook on transformer encoder
    hook = model.transformer_encoder.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                x_sbp = batch["x_sbp"].to(config.device)
                mask = batch["mask"].to(config.device).float()
                macro_timestamp = batch["macro_timestamp"].to(config.device).float()
                session_ids = batch["session_id"]

                # Forward pass (triggers hook)
                _ = model(x_sbp, mask, macro_timestamp)

                # Extract embeddings from hook
                if 'embeddings' in transformer_out:
                    emb = transformer_out['embeddings']  # (B*W, C, d_model)
                    B_W, C, d_model = emb.shape

                    # Flatten to (B*W*C, d_model)
                    emb_flat = emb.reshape(-1, d_model)
                    embeddings_list.append(emb_flat)

                    # Create labels for each sample (encode session_id)
                    # session_ids is a list of B strings (one per sample in batch)
                    B = x_sbp.size(0)
                    for i, sid in enumerate(session_ids):
                        # Each sample in batch contributes W*C embeddings
                        sample_labels = [sid] * (len(emb) // B)
                        session_labels.extend(sample_labels)

    finally:
        hook.remove()

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings_list, dim=0).numpy()  # (N, d_model)

    return embeddings, session_labels


def plot_pca_visualization(embeddings, labels, output_path='pca_visualization.png'):
    """
    Apply PCA and visualize embeddings in 2D/3D.
    """
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Number of unique sessions: {len(set(labels))}")

    # Apply PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained (2D): {sum(pca.explained_variance_ratio_):.4f}")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Color by session
    unique_sessions = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))
    session_to_color = {sid: colors[i] for i, sid in enumerate(unique_sessions)}

    for session_id in unique_sessions:
        mask = np.array([label == session_id for label in labels])
        axes[0].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            alpha=0.6,
            s=20,
            label=session_id,
            color=session_to_color[session_id]
        )

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('Transformer Embeddings - PCA (colored by session)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)

    # Plot 2: Density/spread
    axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.3, s=10, c='blue')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1].set_title('Transformer Embeddings - PCA (all points)')
    axes[1].grid(alpha=0.3)

    # Compute and display collapse metrics
    center = embeddings_2d.mean(axis=0)
    distances = np.linalg.norm(embeddings_2d - center, axis=1)

    print(f"\n=== REPRESENTATION COLLAPSE ANALYSIS ===")
    print(f"Mean distance from center: {distances.mean():.4f}")
    print(f"Std distance from center:  {distances.std():.4f}")
    print(f"Min distance: {distances.min():.4f}")
    print(f"Max distance: {distances.max():.4f}")
    print(f"Spread ratio (max/mean): {distances.max() / distances.mean():.2f}")

    if distances.std() < distances.mean() * 0.1:
        print("⚠️  WARNING: Low spread detected - possible representation collapse!")
    else:
        print("✓ Healthy spread - model is learning diverse representations")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize transformer embeddings")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with preprocessed data')
    parser.add_argument('--num_batches', type=int, default=5, help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_file', type=str, default='pca_visualization.png', help='Output image path')

    args = parser.parse_args()
    config = Config()

    print("="*70)
    print("TRANSFORMER EMBEDDING ANALYSIS")
    print("="*70)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )

    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    print("✓ Model loaded")

    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    dataset = PreprocessedMaskedDataset(args.data_dir, is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"✓ Data loaded ({len(dataset)} samples)")

    # Extract embeddings
    print(f"\nExtracting transformer embeddings (analyzing {args.num_batches} batches)...")
    embeddings, labels = extract_transformer_embeddings(model, dataloader, config, num_batches=args.num_batches)

    # Visualize
    print("\nGenerating PCA visualization...")
    plot_pca_visualization(embeddings, labels, output_path=args.output_file)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
