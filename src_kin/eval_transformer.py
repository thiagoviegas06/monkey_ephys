#!/usr/bin/env python3
"""
Eval script for KinematicDecoderTransformer.

Loads a frozen MAE encoder + trained transformer decoder, runs inference on
test sessions, and writes a submission CSV matching sample_submission.csv.

Usage:
    python src_kin/eval_transformer.py \
      --mae_checkpoint checkpoints_200/best_model_tcn_transformer.pt \
      --kin_checkpoint checkpoints_kin_decoder/best_kin_decoder.pt \
      --data_dir phase2_v2_kaggle_data \
      --output submission_transformer.csv
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
        dropout=0.0  # no dropout at inference
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    decoder.load_state_dict(state)
    decoder.to(device).eval()
    print(f"✓ Loaded kin decoder from {checkpoint_path}")
    return decoder


@torch.no_grad()
def predict_session(mae, decoder, sbp, window_size, device):
    """
    Run sliding non-overlapping windows over a session and return predictions.

    Args:
        sbp: (N, 96) numpy array — test SBP (channels already zeroed by kaggle)

    Returns:
        preds: (N, 2) numpy array — [index_pos, mrp_pos] clipped to [0, 1]
    """
    N, C = sbp.shape
    # Accumulate predictions and counts for averaging overlapping windows
    pred_accum = np.zeros((N, 4), dtype=np.float32)
    count_accum = np.zeros(N, dtype=np.float32)

    # Use non-overlapping windows; handle tail with padding
    w0 = 0
    while w0 < N:
        w1 = min(w0 + window_size, N)
        actual_len = w1 - w0

        sbp_w = sbp[w0:w1].copy()  # (actual_len, 96)

        # Pad to window_size if needed (last window)
        if actual_len < window_size:
            pad = np.zeros((window_size - actual_len, C), dtype=np.float32)
            sbp_w = np.concatenate([sbp_w, pad], axis=0)

        # Infer mask from zero-valued channels (test data is pre-masked by kaggle)
        # A channel is masked if it is entirely zero across the window
        channel_zero = (sbp_w == 0.0).all(axis=0)  # (96,) bool
        mask = np.zeros_like(sbp_w, dtype=np.float32)
        mask[:, channel_zero] = 1.0

        sbp_t = torch.from_numpy(sbp_w).unsqueeze(0).to(device)    # (1, W, 96)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)     # (1, W, 96)
        macro_t = torch.tensor([[float(w0)]], dtype=torch.float32, device=device)  # (1, 1)

        encoder_repr, _, _ = mae.extract_encoder_repr(sbp_t, mask_t, macro_t)
        # encoder_repr: (1*W, C, d_model) = (W, 96, 64)

        kin_pred = decoder(encoder_repr)  # (1, W, 4)
        kin_pred = kin_pred.squeeze(0).cpu().numpy()  # (W, 4)

        pred_accum[w0:w1] += kin_pred[:actual_len]
        count_accum[w0:w1] += 1.0

        w0 += window_size

    preds = pred_accum / np.maximum(count_accum[:, None], 1.0)
    return np.clip(preds[:, :2], 0.0, 1.0)  # return only index_pos, mrp_pos


def get_test_sessions(data_dir):
    test_dir = os.path.join(data_dir, "test")
    sbp_files = sorted(glob(os.path.join(test_dir, "*_sbp.npy")))
    sessions = [Path(f).stem.replace("_sbp", "") for f in sbp_files]
    return sessions, test_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mae_checkpoint', type=str,
                        default='checkpoints_200/best_model_tcn_transformer.pt')
    parser.add_argument('--kin_checkpoint', type=str,
                        default='checkpoints_kin_decoder/best_kin_decoder.pt')
    parser.add_argument('--data_dir', type=str,
                        default='phase2_v2_kaggle_data')
    parser.add_argument('--output', type=str,
                        default='submission_transformer.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mae_config = MAEConfig()
    kin_config = KinConfig()

    mae = build_mae_model(args.mae_checkpoint, mae_config, device)
    decoder = build_kin_decoder(
        args.kin_checkpoint,
        d_model=mae_config.d_model,
        window_size=kin_config.window_size,
        device=device
    )

    sessions, test_dir = get_test_sessions(args.data_dir)
    print(f"\nFound {len(sessions)} test sessions")

    # Load and fill submission template
    sub_path = os.path.join(args.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    for session_id in tqdm(sessions, desc="Predicting"):
        sbp_path = os.path.join(test_dir, f"{session_id}_sbp.npy")
        sbp = np.load(sbp_path).astype(np.float32)  # (N, 96)

        # Per-session z-score — must match MAE training preprocessing
        # Compute stats only on visible (non-zero) channels to avoid masked channels
        # pulling the mean/std toward zero
        channel_active = (sbp != 0.0).any(axis=0)  # (96,) — False for fully-masked channels
        if channel_active.any():
            sbp_mean = np.where(channel_active, sbp.mean(axis=0), 0.0)
            sbp_std  = np.where(channel_active, sbp.std(axis=0) + 1e-5, 1.0)
            sbp = (sbp - sbp_mean) / sbp_std
            sbp[:, ~channel_active] = 0.0  # keep masked channels at zero

        preds = predict_session(mae, decoder, sbp, kin_config.window_size, device)

        idx = sub["session_id"] == session_id
        if not idx.any():
            print(f"  Warning: {session_id} not found in submission template, skipping")
            continue

        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)

        # Guard against time_bins exceeding session length
        valid = time_bins < len(preds)
        if not valid.all():
            print(f"  Warning: {session_id} has {(~valid).sum()} time_bins beyond session length")

        sub.loc[idx & sub["time_bin"].isin(time_bins[valid]), "index_pos"] = preds[time_bins[valid], 0]
        sub.loc[idx & sub["time_bin"].isin(time_bins[valid]), "mrp_pos"] = preds[time_bins[valid], 1]

    sub.to_csv(args.output, index=False)
    print(f"\n✓ Submission saved to {args.output}")
    print(f"  Rows: {len(sub)} | Sessions: {sub['session_id'].nunique()}")


if __name__ == '__main__':
    main()
