import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.model import KinematicDecoderTransformer
from src_mae.model import SBP_TCN_Transformer
from src_kin.preprocessing import SessionDataPhase2
from src_kin.compute_kin_stats import compute_kin_stats, compute_sbp_stats

def build_mae_model(config):
    mae = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_axial_layers=config.num_axial_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    ckpt = torch.load(config.mae_checkpoint_path, map_location=config.device)
    mae.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False
    return mae

def build_kinematic_model(config):
    return KinematicDecoderTransformer(
        d_model=config.d_model,
        window_size=config.window_size,
        num_channels=config.sbp_channels,
        num_temporal_layers=config.decoder_num_temporal_layers,
        num_heads=config.decoder_num_heads,
        output_dim=config.output_dim,
        dropout=config.decoder_dropout
    ).to(config.device)

def load_decoders(checkpoint_paths, config):
    """Load decoder models and extract their val_r2 weights."""
    decoders = []
    weights = []
    stats_list = []

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=config.device)
        model = build_kinematic_model(config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        decoders.append(model)

        val_r2 = ckpt.get('val_r2', 0.0)
        weights.append(max(val_r2, 0.0))  # clip negative R² to 0

        stats_list.append({
            'kin_mean': ckpt.get('kin_mean'),
            'kin_std':  ckpt.get('kin_std'),
            'sbp_mean': ckpt.get('sbp_mean'),
            'sbp_std':  ckpt.get('sbp_std'),
        })
        print(f"  Loaded {os.path.basename(path)}  val_r2={val_r2:.4f}")

    weights = np.array(weights, dtype=np.float32)
    if weights.sum() == 0:
        weights = np.ones(len(decoders), dtype=np.float32)
    weights /= weights.sum()
    print(f"Ensemble weights: {weights}")

    # Use stats from the best checkpoint (highest val_r2)
    best_idx = int(np.argmax(weights))
    best_stats = stats_list[best_idx]
    return decoders, weights, best_stats

def smooth_predictions(preds, kernel_size=5):
    if kernel_size <= 1:
        return preds
    smoothed = np.zeros_like(preds)
    for c in range(preds.shape[1]):
        pad = kernel_size // 2
        padded = np.pad(preds[:, c], (pad, pad), mode='edge')
        conv = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')
        smoothed[:, c] = conv[:preds.shape[0]]
    return smoothed

@torch.no_grad()
def predict_session_ensemble(mae, decoders, weights, sbp, config,
                              kin_mean, kin_std, sbp_mean, sbp_std, session_id):
    N = sbp.shape[0]
    W = config.window_size
    preds = np.zeros((N, config.output_dim), dtype=np.float32)

    try:
        num_id = float(''.join(filter(str.isdigit, session_id)))
    except ValueError:
        num_id = 0.0

    kin_mean = kin_mean.to(config.device)
    kin_std  = kin_std.to(config.device)
    sbp_mean = sbp_mean.to(config.device) if sbp_mean is not None else None
    sbp_std  = sbp_std.to(config.device)  if sbp_std  is not None else None

    for w0 in range(0, N, W):
        w1 = min(w0 + W, N)
        sbp_w = torch.from_numpy(sbp[w0:w1]).unsqueeze(0)  # (1, len, C)

        pad_len = W - (w1 - w0)
        if pad_len > 0:
            last_bin = sbp_w[:, -1:, :]
            sbp_w = torch.cat([sbp_w, last_bin.repeat(1, pad_len, 1)], dim=1)

        sbp_w = sbp_w.to(config.device)
        mask = (sbp_w == 0.0).float()
        macro_ts = torch.tensor([[w0]], dtype=torch.float32, device=config.device)

        channel_mean = sbp_mean.view(1, 1).expand(1, config.sbp_channels) if sbp_mean is not None else None
        channel_var  = (sbp_std**2).view(1, 1).expand(1, config.sbp_channels) if sbp_std is not None else None

        if getattr(config, 'use_mae_embeddings', False):
            mae_out = mae(sbp_w, mask, macro_ts,
                         channel_mean=channel_mean, channel_var=channel_var,
                         return_embeddings=True)
        else:
            mae_out = mae(sbp_w, mask, macro_ts,
                         channel_mean=channel_mean, channel_var=channel_var)
        mae_out = mae_out.detach()

        # Weighted ensemble over decoders
        window_pred = np.zeros((W, config.output_dim), dtype=np.float32)
        for decoder, w in zip(decoders, weights):
            kin_pred = decoder(mae_out)
            kin_pred = (kin_pred * kin_std + kin_mean).cpu().numpy()[0]  # (W, 4)
            window_pred += w * kin_pred

        actual_len = w1 - w0
        preds[w0:w1] = window_pred[:actual_len]

    return preds

def run_eval(checkpoint_paths, output_csv="submission_ensemble.csv"):
    config = Config()

    print("Building MAE model...")
    mae = build_mae_model(config)

    print("Loading decoder checkpoints...")
    decoders, weights, stats = load_decoders(checkpoint_paths, config)

    # Resolve stats: use checkpoint or recompute
    kin_mean = stats['kin_mean']
    kin_std  = stats['kin_std']
    sbp_mean = stats['sbp_mean']
    sbp_std  = stats['sbp_std']

    if kin_mean is None or kin_std is None:
        print("Recomputing kinematics stats...")
        kin_mean, kin_std = compute_kin_stats(config.data_path, getattr(config, 'phase1_data_path', None))
    if sbp_mean is None or sbp_std is None:
        print("Recomputing SBP stats...")
        sbp_mean, sbp_std = compute_sbp_stats(config.data_path, getattr(config, 'phase1_data_path', None))

    kin_mean = kin_mean.to(config.device)
    kin_std  = kin_std.to(config.device)
    sbp_mean = sbp_mean.to(config.device)
    sbp_std  = sbp_std.to(config.device)

    print("Loading test sessions...")
    session_manager = SessionDataPhase2(config.data_path, is_train=False)
    sessions = session_manager.generate_session_obj()

    predictions = {}
    for session in tqdm(sessions, desc="Predicting"):
        sbp, _ = session.load_data()
        if sbp is None:
            continue
        preds = predict_session_ensemble(
            mae, decoders, weights, sbp, config,
            kin_mean, kin_std, sbp_mean, sbp_std, session.session_id
        )
        preds = smooth_predictions(preds, kernel_size=config.smoothing_kernel_size)
        predictions[session.session_id] = preds

    print("Constructing submission...")
    sub_path = os.path.join(config.data_path, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    for session_id, preds in predictions.items():
        idx = sub["session_id"] == session_id
        if not idx.any():
            continue
        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)
        sub.loc[idx, "index_pos"] = np.clip(preds[time_bins, 0], 0, 1)
        sub.loc[idx, "mrp_pos"]   = np.clip(preds[time_bins, 1], 0, 1)

    sub.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", required=False,
        help="Paths to decoder checkpoint .pt files. If omitted, auto-discovers all kin_decoder_epoch*.pt + best_model_perceiver.pt in checkpoint_dir."
    )
    parser.add_argument(
        "--output", default="submission_ensemble.csv",
        help="Output CSV filename"
    )
    args = parser.parse_args()

    if args.checkpoints:
        ckpt_paths = args.checkpoints
    else:
        config = Config()
        ckpt_paths = sorted(glob(os.path.join(config.checkpoint_dir, "kin_decoder_epoch*.pt")))
        best = os.path.join(config.checkpoint_dir, "best_model_perceiver.pt")
        if os.path.exists(best):
            ckpt_paths.append(best)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints found in {config.checkpoint_dir}")
        print(f"Auto-discovered {len(ckpt_paths)} checkpoints.")

    run_eval(ckpt_paths, output_csv=args.output)
