from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import SBPWindowDataset
from src.eval import evaluate_model
from src.io import load_metadata, load_sessions
from src.losses import masked_mse_loss
from src.model import TemporalCNNBaseline
from src.preprocess import compute_train_session_stats
from src.utils import ensure_dir, resolve_device, seed_everything, split_train_val_sessions


def _build_loaders(
    data_dir: str,
    metadata_path: str,
    window_size: int,
    mask_channels: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
) -> Tuple[DataLoader, DataLoader, List[str], List[str], np.ndarray, np.ndarray]:
    metadata = load_metadata(metadata_path)
    train_ids, val_ids = split_train_val_sessions(metadata, val_fraction=val_fraction, seed=seed)

    train_sessions = load_sessions(data_dir, train_ids, split="train")
    val_sessions = load_sessions(data_dir, val_ids, split="train")

    # Save channel-level fallback stats for test-time normalization.
    train_means = []
    train_stds = []
    for s in train_sessions:
        mean, std = compute_train_session_stats(s.sbp)
        train_means.append(mean)
        train_stds.append(std)
    global_mean = np.mean(np.stack(train_means, axis=0), axis=0).astype(np.float32)
    global_std = np.mean(np.stack(train_stds, axis=0), axis=0).astype(np.float32)

    train_ds = SBPWindowDataset(
        sessions=train_sessions,
        window_size=window_size,
        split="train",
        seed=seed,
        mask_channels=mask_channels,
        deterministic_masks=False,
    )
    val_ds = SBPWindowDataset(
        sessions=val_sessions,
        window_size=window_size,
        split="val",
        seed=seed,
        mask_channels=mask_channels,
        deterministic_masks=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, train_ids, val_ids, global_mean, global_std


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    metadata_path = str(Path(args.data_dir) / args.metadata_file)
    train_loader, val_loader, train_ids, val_ids, global_mean, global_std = _build_loaders(
        data_dir=args.data_dir,
        metadata_path=metadata_path,
        window_size=args.window_size,
        mask_channels=args.mask_channels,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )

    model = TemporalCNNBaseline(
        input_dim=100,
        hidden_dim=args.hidden_dim,
        output_dim=96,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_cosine_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ensure_dir(args.checkpoint_dir)
    best_path = ckpt_dir / args.checkpoint_name

    best_val_nmse = float("inf")

    print(f"Device: {device}")
    print(f"Train sessions: {len(train_ids)} | Val sessions: {len(val_ids)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x_sbp = batch["x_sbp"].to(device)
            x_kin = batch["x_kin"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_sbp, x_kin)
            loss = masked_mse_loss(y_hat, y, mask)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(1, n_batches)
        val_metrics = evaluate_model(model, val_loader, device=device)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_nmse={val_metrics['nmse']:.6f}"
        )

        if val_metrics["nmse"] < best_val_nmse:
            best_val_nmse = val_metrics["nmse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {
                        "input_dim": 100,
                        "hidden_dim": args.hidden_dim,
                        "output_dim": 96,
                        "num_layers": args.num_layers,
                        "kernel_size": args.kernel_size,
                        "dropout": args.dropout,
                    },
                    "window_size": args.window_size,
                    "train_global_mean": global_mean,
                    "train_global_std": global_std,
                    "train_session_ids": train_ids,
                    "val_session_ids": val_ids,
                    "best_val_nmse": best_val_nmse,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path} (val_nmse={best_val_nmse:.6f})")

    print(f"Training finished. Best val NMSE: {best_val_nmse:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Temporal CNN baseline for masked SBP reconstruction.")
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--window-size", type=int, default=201)
    parser.add_argument("--mask-channels", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-cosine-scheduler", dest="use_cosine_scheduler", action="store_true")
    parser.add_argument("--no-cosine-scheduler", dest="use_cosine_scheduler", action="store_false")
    parser.set_defaults(use_cosine_scheduler=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    return args


if __name__ == "__main__":
    train(parse_args())
