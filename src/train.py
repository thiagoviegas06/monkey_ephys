from __future__ import annotations

import argparse
import os
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
from src.model import build_model
from src.preprocess import compute_train_session_stats
from src.utils import ensure_dir, resolve_device, seed_everything, split_train_val_sessions


def _build_loaders(
    data_dir: str,
    metadata_path: str,
    window_size: int,
    train_mask_min: int,
    train_mask_max: int,
    val_mask_channels: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    prefetch_factor: int,
    persistent_workers: bool,
    max_train_centers_per_session: int | None,
    max_val_centers_per_session: int | None,
) -> Tuple[DataLoader, DataLoader, List[str], List[str], np.ndarray, np.ndarray]:
    metadata = load_metadata(metadata_path)
    train_ids, val_ids = split_train_val_sessions(metadata, val_fraction=val_fraction, seed=seed)

    train_sessions = load_sessions(data_dir, train_ids, split="train")
    val_sessions = load_sessions(data_dir, val_ids, split="train")

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
        mask_channels=train_mask_max,
        mask_channels_min=train_mask_min,
        mask_channels_max=train_mask_max,
        deterministic_masks=False,
        max_centers_per_session=max_train_centers_per_session,
    )
    val_ds = SBPWindowDataset(
        sessions=val_sessions,
        window_size=window_size,
        split="val",
        seed=seed,
        mask_channels=val_mask_channels,
        mask_channels_min=val_mask_channels,
        mask_channels_max=val_mask_channels,
        deterministic_masks=True,
        max_centers_per_session=max_val_centers_per_session,
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_ids, val_ids, global_mean, global_std


def _resolve_model_kwargs(args: argparse.Namespace) -> Dict[str, int | float]:
    base_kwargs: Dict[str, int | float] = {
        "input_dim": 196,
        "hidden_dim": args.hidden_dim,
        "output_dim": 96,
        "num_layers": args.num_layers,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "dilation_base": args.dilation_base,
    }
    if args.model == "tcn":
        base_kwargs["dilation_cap"] = args.dilation_cap
        base_kwargs["gn_groups"] = args.gn_groups
    return base_kwargs


def _resolve_amp_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "auto":
        return torch.bfloat16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        if args.matmul_precision != "default":
            torch.set_float32_matmul_precision(args.matmul_precision)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = args.benchmark_cudnn

    metadata_path = str(Path(args.data_dir) / args.metadata_file)
    train_loader, val_loader, train_ids, val_ids, global_mean, global_std = _build_loaders(
        data_dir=args.data_dir,
        metadata_path=metadata_path,
        window_size=args.window_size,
        train_mask_min=args.mask_min,
        train_mask_max=args.mask_max,
        val_mask_channels=args.mask_channels,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        max_train_centers_per_session=args.max_train_centers_per_session,
        max_val_centers_per_session=args.max_val_centers_per_session,
    )

    model_kwargs = _resolve_model_kwargs(args)
    model = build_model(args.model, **model_kwargs).to(device)
    if args.compile_model and device.type == "cuda":
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode=args.compile_mode)
        else:
            print("torch.compile not available in this PyTorch version; continuing without compilation.")
    amp_dtype = _resolve_amp_dtype(args.precision, device)
    use_grad_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_cosine_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ensure_dir(args.checkpoint_dir)
    best_path = ckpt_dir / args.checkpoint_name
    best_val_nmse = float("inf")

    print(f"Device: {device}")
    print(
        f"Precision: {args.precision} (resolved={amp_dtype}) | "
        f"compile={args.compile_model and device.type == 'cuda'}"
    )
    if device.type == "cuda":
        print(
            f"TF32={args.allow_tf32} | cudnn.benchmark={torch.backends.cudnn.benchmark} | "
            f"deterministic={torch.backends.cudnn.deterministic}"
        )
    print(f"Model: {args.model} | kwargs={model_kwargs}")
    print(f"Train mask range: [{args.mask_min}, {args.mask_max}] | Val mask channels: {args.mask_channels}")
    print(f"Train sessions: {len(train_ids)} | Val sessions: {len(val_ids)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Eval every: {args.eval_every} epoch(s)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        printed = False  # print mask stats once per epoch

        for batch in train_loader:
            x_sbp = batch["x_sbp"].to(device, non_blocking=True)
            x_kin = batch["x_kin"].to(device, non_blocking=True)
            obs_mask = batch["obs_mask"].to(device, non_blocking=True)
            y = batch["y_seq"].to(device, non_blocking=True).transpose(1, 2)
            mask = batch["mask"].to(device, non_blocking=True)  # (B,96)

            if not printed:
                # Basic sanity
                print("mask unique:", torch.unique(mask).tolist())
                masked_per_sample = mask.sum(dim=1)
                print("masked chans per sample (first 8):", masked_per_sample[:8].tolist())
                print(
                    "masked chans stats:",
                    {
                        "min": int(masked_per_sample.min().item()),
                        "max": int(masked_per_sample.max().item()),
                        "mean": float(masked_per_sample.float().mean().item()),
                    },
                )

                # Hard checks (tune these if you expect variation)
                assert mask.ndim == 2 and mask.shape[1] == 96, f"mask shape wrong: {tuple(mask.shape)}"
                assert torch.all((mask == 0) | (mask == 1)), "mask must be binary 0/1"
                assert int(masked_per_sample.min().item()) >= args.mask_min, "too few masked channels"
                assert int(masked_per_sample.max().item()) <= args.mask_max, "too many masked channels"

                printed = True

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                y_hat = model(x_sbp, x_kin, obs_mask)
                loss = masked_mse_loss(y_hat, y, mask)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(1, n_batches)
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            val_metrics = evaluate_model(model, val_loader, device=device, amp_dtype=amp_dtype)
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
                        "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "model_kwargs": model_kwargs,
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
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | val_skipped")

    print(f"Training finished. Best val NMSE: {best_val_nmse:.6f}")
    return {"best_val_nmse": best_val_nmse, "best_path": str(best_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked SBP reconstruction models.")
    parser.add_argument("--data-dir", type=str, default="kaggle_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv")
    parser.add_argument("--model", type=str, choices=["cnn", "tcn"], default="tcn")

    parser.add_argument("--window-size", type=int, default=201)
    parser.add_argument("--mask-channels", type=int, default=30, help="Fixed mask channels used for validation")
    parser.add_argument("--mask-min", type=int, default=10, help="Train-time minimum masked channels per window")
    parser.add_argument("--mask-max", type=int, default=60, help="Train-time maximum masked channels per window")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument(
        "--max-train-centers-per-session",
        type=int,
        default=None,
        help="Optional cap on train windows per session to control epoch runtime.",
    )
    parser.add_argument(
        "--max-val-centers-per-session",
        type=int,
        default=200,
        help="Optional cap on validation windows per session during training.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs (always runs on final epoch).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-cosine-scheduler", dest="use_cosine_scheduler", action="store_true")
    parser.add_argument("--no-cosine-scheduler", dest="use_cosine_scheduler", action="store_false")
    parser.set_defaults(use_cosine_scheduler=True)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--dilation-base", type=int, default=2)
    parser.add_argument("--dilation-cap", type=int, default=64)
    parser.add_argument("--gn-groups", type=int, default=8)
    parser.add_argument("--precision", type=str, choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--compile-model", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile-model", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    parser.add_argument("--compile-mode", type=str, choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune")
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true")
    parser.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument("--benchmark-cudnn", dest="benchmark_cudnn", action="store_true")
    parser.add_argument("--no-benchmark-cudnn", dest="benchmark_cudnn", action="store_false")
    parser.set_defaults(benchmark_cudnn=True)
    parser.add_argument("--deterministic", action="store_true", help="Deterministic mode (slower; disables cudnn benchmark).")
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["default", "high", "medium"],
        default="high",
        help="torch.set_float32_matmul_precision setting for CUDA.",
    )

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")
    if args.mask_min < 1 or args.mask_max > 96 or args.mask_min > args.mask_max:
        raise ValueError("Require 1 <= --mask-min <= --mask-max <= 96")
    if args.mask_channels < 1 or args.mask_channels > 96:
        raise ValueError("--mask-channels must be in [1,96]")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.max_train_centers_per_session is not None and args.max_train_centers_per_session < 1:
        raise ValueError("--max-train-centers-per-session must be >= 1 when set")
    if args.max_val_centers_per_session is not None and args.max_val_centers_per_session < 1:
        raise ValueError("--max-val-centers-per-session must be >= 1 when set")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    return args


if __name__ == "__main__":
    train(parse_args())
