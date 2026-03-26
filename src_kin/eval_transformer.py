#!/usr/bin/env python3
"""
Eval script for KinematicDecoderTransformer with optional checkpoint ensembling.

Supports weighted ensemble where each checkpoint's contribution is proportional
to its saved validation R² score.

Usage (single checkpoint):
    python src_kin/eval_transformer.py \
      --mae_checkpoint checkpoints_200/best_model_tcn_transformer.pt \
      --kin_checkpoints checkpoints_kin_decoder/best_kin_decoder.pt \
      --data_dir phase2_v2_kaggle_data \
      --output submission_transformer.csv

Usage (weighted ensemble — weights derived from saved val R²):
    python src_kin/eval_transformer.py \
      --mae_checkpoint checkpoints_200/best_model_tcn_transformer.pt \
      --kin_checkpoints checkpoints_kin_decoder/best_kin_decoder.pt \
                        checkpoints_kin_decoder/kin_decoder_epoch21.pt \
                        checkpoints_kin_decoder/kin_decoder_epoch24.pt \
      --data_dir phase2_v2_kaggle_data \
      --output submission_ensemble.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from glob import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_mae.model import SBP_TCN_Transformer
from src_kin.model import KinematicDecoderTransformer
from src_mae.config import Config as MAEConfig
from src_kin.config import Config as KinConfig


def build_mae_model(checkpoint_path, config, device):
    model = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"✓ Loaded MAE from {checkpoint_path}")
    return model


def build_kin_decoder(checkpoint_path, d_model, window_size, device):
    decoder = KinematicDecoderTransformer(
        d_model=d_model,
        window_size=window_size,
        num_channels=96,
        num_temporal_layers=2,
        num_heads=8,
        output_dim=4,
        dropout=0.0
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    decoder.load_state_dict(state)
    decoder.to(device).eval()

    # Read val R² saved in checkpoint — used for weighted ensembling
    val_r2 = checkpoint.get('val_r2', None) if isinstance(checkpoint, dict) else None
    print(f"✓ Loaded kin decoder from {checkpoint_path} (val R²={val_r2:.4f})" if val_r2 is not None
          else f"✓ Loaded kin decoder from {checkpoint_path} (val R² not found, weight=1.0)")
    return decoder, val_r2


def compute_ensemble_weights(val_r2_scores):
    """
    Convert val R² scores into normalized weights.
    Clips negative R² to 0 before normalizing so bad checkpoints
    don't get negative weight.
    """
    weights = np.array([max(r2, 0.0) for r2 in val_r2_scores], dtype=np.float32)
    total = weights.sum()
    if total == 0:
        # All scores zero or missing — fall back to uniform weights
        weights = np.ones(len(val_r2_scores), dtype=np.float32)
        total = weights.sum()
    return weights / total


@torch.no_grad()
def predict_session(mae, decoders, weights, sbp, window_size, device):
    """
    Run inference over a session with weighted ensemble across decoders.

    Args:
        decoders: list of KinematicDecoderTransformer
        weights:  (num_decoders,) numpy array — normalized weights summing to 1
        sbp:      (N, 96) numpy array — per-session z-scored, masked channels zeroed

    Returns:
        preds: (N, 2) numpy array — [index_pos, mrp_pos] clipped to [0, 1]
    """
    N, C = sbp.shape
    pred_accum = np.zeros((N, 4), dtype=np.float32)
    count_accum = np.zeros(N, dtype=np.float32)

    w0 = 0
    while w0 < N:
        w1 = min(w0 + window_size, N)
        actual_len = w1 - w0

        sbp_w = sbp[w0:w1].copy()
        if actual_len < window_size:
            pad = np.zeros((window_size - actual_len, C), dtype=np.float32)
            sbp_w = np.concatenate([sbp_w, pad], axis=0)

        channel_zero = (sbp_w == 0.0).all(axis=0)
        mask = np.zeros_like(sbp_w, dtype=np.float32)
        mask[:, channel_zero] = 1.0

        sbp_t   = torch.from_numpy(sbp_w).unsqueeze(0).to(device)
        mask_t  = torch.from_numpy(mask).unsqueeze(0).to(device)
        macro_t = torch.tensor([[float(w0)]], dtype=torch.float32, device=device)

        encoder_repr, _, _ = mae.extract_encoder_repr(sbp_t, mask_t, macro_t)

        # Weighted average across decoder checkpoints
        window_pred = np.zeros((window_size, 4), dtype=np.float32)
        for decoder, w in zip(decoders, weights):
            kin_pred = decoder(encoder_repr).squeeze(0).cpu().numpy()  # (W, 4)
            window_pred += w * kin_pred

        pred_accum[w0:w1] += window_pred[:actual_len]
        count_accum[w0:w1] += 1.0
        w0 += window_size

    preds = pred_accum / np.maximum(count_accum[:, None], 1.0)
    return np.clip(preds[:, :2], 0.0, 1.0)


def get_test_sessions(data_dir):
    test_dir = os.path.join(data_dir, "test")
    sbp_files = sorted(glob(os.path.join(test_dir, "*_sbp.npy")))
    sessions = [Path(f).stem.replace("_sbp", "") for f in sbp_files]
    return sessions, test_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mae_checkpoint', type=str,
                        default='checkpoints_200/best_model_tcn_transformer.pt')
    parser.add_argument('--kin_checkpoints', nargs='+',
                        default=['checkpoints_kin_decoder/best_kin_decoder.pt'],
                        help='One or more kin decoder checkpoints to ensemble')
    parser.add_argument('--data_dir', type=str, default='phase2_v2_kaggle_data')
    parser.add_argument('--output', type=str, default='submission_transformer.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mae_config = MAEConfig()
    kin_config = KinConfig()

    mae = build_mae_model(args.mae_checkpoint, mae_config, device)

    decoders, val_r2_scores = [], []
    for ckpt in args.kin_checkpoints:
        decoder, val_r2 = build_kin_decoder(ckpt, mae_config.d_model, kin_config.window_size, device)
        decoders.append(decoder)
        val_r2_scores.append(val_r2 if val_r2 is not None else 1.0)

    weights = compute_ensemble_weights(val_r2_scores)

    print(f"\nEnsemble weights:")
    for ckpt, r2, w in zip(args.kin_checkpoints, val_r2_scores, weights):
        print(f"  {Path(ckpt).name}: val R²={r2:.4f} → weight={w:.4f}")

    sessions, test_dir = get_test_sessions(args.data_dir)
    print(f"\nFound {len(sessions)} test sessions")

    sub_path = os.path.join(args.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    for session_id in tqdm(sessions, desc="Predicting"):
        sbp_path = os.path.join(test_dir, f"{session_id}_sbp.npy")
        sbp = np.load(sbp_path).astype(np.float32)

        # Per-session z-score on active channels only
        channel_active = (sbp != 0.0).any(axis=0)
        if channel_active.any():
            sbp_mean = np.where(channel_active, sbp.mean(axis=0), 0.0)
            sbp_std  = np.where(channel_active, sbp.std(axis=0) + 1e-5, 1.0)
            sbp = (sbp - sbp_mean) / sbp_std
            sbp[:, ~channel_active] = 0.0

        preds = predict_session(mae, decoders, weights, sbp, kin_config.window_size, device)

        idx = sub["session_id"] == session_id
        if not idx.any():
            print(f"  Warning: {session_id} not found in submission template, skipping")
            continue

        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)
        valid = time_bins < len(preds)
        if not valid.all():
            print(f"  Warning: {session_id} has {(~valid).sum()} time_bins beyond session length")

        sub.loc[idx & sub["time_bin"].isin(time_bins[valid]), "index_pos"] = preds[time_bins[valid], 0]
        sub.loc[idx & sub["time_bin"].isin(time_bins[valid]), "mrp_pos"]   = preds[time_bins[valid], 1]

    sub.to_csv(args.output, index=False)
    print(f"\n✓ Submission saved to {args.output}")
    print(f"  Rows: {len(sub)} | Sessions: {sub['session_id'].nunique()}")


if __name__ == '__main__':
    main()
