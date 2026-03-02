from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.dataset import SBPWindowDataset
from src.io import load_metadata, load_sessions
from src.losses import masked_mse_loss, nmse_masked
from src.model import build_model
from src.utils import resolve_device, seed_everything, split_train_val_sessions


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    nmse_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        x_sbp = batch["x_sbp"].to(device, non_blocking=True)
        x_kin = batch["x_kin"].to(device, non_blocking=True)
        obs_mask = batch["obs_mask"].to(device, non_blocking=True)
        y = batch["y_seq"].to(device, non_blocking=True).transpose(1, 2)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            y_hat = model(x_sbp, x_kin, obs_mask)
            loss = masked_mse_loss(y_hat, y, mask)
            nmse = nmse_masked(y_hat, y, mask)

        loss_sum += float(loss.item())
        nmse_sum += float(nmse.item())
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "nmse": float("nan")}

    return {
        "loss": loss_sum / n_batches,
        "nmse": nmse_sum / n_batches,
    }


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    try:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "cnn")
    model = build_model(model_name, **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metadata = load_metadata(Path(args.data_dir) / args.metadata_file)
    _, val_ids = split_train_val_sessions(metadata, val_fraction=args.val_fraction, seed=args.seed)

    val_sessions = load_sessions(args.data_dir, val_ids, split="train")
    val_ds = SBPWindowDataset(
        sessions=val_sessions,
        window_size=args.window_size,
        split="val",
        seed=args.seed,
        mask_channels=args.mask_channels,
        mask_channels_min=args.mask_channels,
        mask_channels_max=args.mask_channels,
        deterministic_masks=True,
        max_centers_per_session=args.max_val_centers_per_session,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_model(model, val_loader, device=device)
    print(
        f"Eval on {len(val_ids)} val sessions | "
        f"loss={metrics['loss']:.6f} | nmse={metrics['nmse']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation split.")
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--window-size", type=int, default=201)
    parser.add_argument("--mask-channels", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-centers-per-session", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    if args.max_val_centers_per_session is not None and args.max_val_centers_per_session < 1:
        raise ValueError("--max-val-centers-per-session must be >= 1 when set")
    return args


if __name__ == "__main__":
    run_eval(parse_args())
