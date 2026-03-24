import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.model import LSTMKinematicDecoder
from src_mae.model import SBP_TCN_Transformer
from src_kin.preprocessing import SessionDataPhase2
from src_kin.compute_kin_stats import compute_per_session_kin_stats

def compute_per_session_sbp_stats(data_path):
    """Compute SBP normalization statistics per session (z-score)."""
    session_manager = SessionDataPhase2(data_path, is_train=True)
    sessions = session_manager.generate_session_obj()

    session_stats = {}

    for session in sessions:
        sbp, _ = session.load_data()
        if sbp is not None and sbp.shape[0] > 0:
            mean = np.mean(sbp, axis=0)
            std = np.std(sbp, axis=0) + 1e-8

            session_stats[session.session_id] = {
                'mean': torch.tensor(mean, dtype=torch.float32),
                'std': torch.tensor(std, dtype=torch.float32)
            }

    print(f"✓ Computed per-session SBP stats for {len(session_stats)} sessions")
    return session_stats


def build_models(config):
    # MAE
    mae = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    mae.load_state_dict(torch.load(config.mae_checkpoint_path, map_location=config.device)['model_state_dict'])
    mae.eval()

    # LSTM Kinematics Decoder (uses MAE encoder features)
    lstm = LSTMKinematicDecoder(
        encoder_dim=config.encoder_dim,  # MAE encoder feature dimension (64)
        hidden_dim=config.hidden_dim,
        num_layers=config.num_lstm_layers,
        output_dim=config.output_dim,
        dropout=config.lstm_dropout
    ).to(config.device)
    best_lstm_path = os.path.join(config.checkpoint_dir, "best_model_lstm.pt")
    checkpoint = torch.load(best_lstm_path, map_location=config.device)
    lstm.load_state_dict(checkpoint['model_state_dict'])
    lstm.eval()

    # Load per-session kinematics stats for denormalization
    session_kin_stats = checkpoint.get('session_kin_stats', None)

    if session_kin_stats is None:
        print("Per-session kinematics stats not found in checkpoint. Recomputing from training data...")
        session_kin_stats = compute_per_session_kin_stats(config.data_path)
    else:
        # Move to device if needed
        for sid in session_kin_stats:
            session_kin_stats[sid]['mean'] = session_kin_stats[sid]['mean'].to(config.device)
            session_kin_stats[sid]['std'] = session_kin_stats[sid]['std'].to(config.device)

    # Load per-session SBP stats for normalization before MAE
    print("Loading per-session SBP statistics...")
    session_sbp_stats = compute_per_session_sbp_stats(config.data_path)
    for sid in session_sbp_stats:
        session_sbp_stats[sid]['mean'] = session_sbp_stats[sid]['mean'].to(config.device)
        session_sbp_stats[sid]['std'] = session_sbp_stats[sid]['std'].to(config.device)

    return mae, lstm, session_kin_stats, session_sbp_stats

def smooth_predictions(preds, kernel_size=5):
    """
    Applies a simple moving average filter to the predictions.
    preds: (N, C) numpy array
    """
    if kernel_size <= 1:
        return preds
    
    smoothed = np.zeros_like(preds)
    for c in range(preds.shape[1]):
        # Pad to handle edge effects
        pad_size = kernel_size // 2
        padded = np.pad(preds[:, c], (pad_size, pad_size), mode='edge')
        conv = np.convolve(padded, np.ones(kernel_size)/kernel_size, mode='valid')
        # Handle case where conv might be slightly different length due to rounding
        smoothed[:, c] = conv[:preds.shape[0]]
    return smoothed

@torch.no_grad()
def predict_session(mae, lstm, sbp, config, session_id, session_kin_stats, session_sbp_stats):
    N = sbp.shape[0]
    W = config.window_size
    preds = np.zeros((N, config.output_dim), dtype=np.float32)

    # Get per-session normalization stats for SBP
    if session_id in session_sbp_stats:
        sbp_mean = session_sbp_stats[session_id]['mean'].cpu().numpy()
        sbp_std = session_sbp_stats[session_id]['std'].cpu().numpy()
    else:
        print(f"Warning: Session {session_id} not in SBP stats, using identity normalization")
        sbp_mean = np.zeros(config.sbp_channels, dtype=np.float32)
        sbp_std = np.ones(config.sbp_channels, dtype=np.float32)

    # Get per-session denormalization stats for kinematics
    if session_id in session_kin_stats:
        kin_mean = session_kin_stats[session_id]['mean'].cpu().numpy()
        kin_std = session_kin_stats[session_id]['std'].cpu().numpy()
    else:
        print(f"Warning: Session {session_id} not in kinematics stats, using identity denormalization")
        kin_mean = np.zeros(config.output_dim, dtype=np.float32)
        kin_std = np.ones(config.output_dim, dtype=np.float32)

    # Process in windows
    for w0 in range(0, N, W):
        w1 = min(w0 + W, N)
        sbp_w = torch.from_numpy(sbp[w0:w1]).unsqueeze(0) # (1, w_len, C)

        # Pad if necessary for the final window
        pad_len = W - (w1 - w0)
        if pad_len > 0:
            sbp_w = torch.cat([sbp_w, torch.zeros(1, pad_len, config.sbp_channels)], dim=1)

        sbp_w = sbp_w.to(config.device)

        # Normalize SBP before passing to MAE
        sbp_mean_tensor = torch.tensor(sbp_mean, dtype=torch.float32, device=config.device)
        sbp_std_tensor = torch.tensor(sbp_std, dtype=torch.float32, device=config.device)
        sbp_w_norm = (sbp_w - sbp_mean_tensor) / sbp_std_tensor

        mask = (sbp_w == 0.0).float().to(config.device)
        macro_timestamp = torch.tensor([[w0]], dtype=torch.float32, device=config.device)

        # Extract encoder features from MAE (frozen feature extractor)
        encoder_features = mae.extract_encoder_features(sbp_w_norm, mask, macro_timestamp)
        # encoder_features: (1, W, C, d_model) → average across channels
        encoder_features = encoder_features.mean(dim=2)

        # Decode kinematics from encoder features
        kin_pred, _, _, _ = lstm(encoder_features, mask=mask)

        # Denormalize kinematics (Per-Session Z-Score)
        kin_pred = kin_pred * torch.tensor(kin_std, dtype=torch.float32, device=config.device) + torch.tensor(kin_mean, dtype=torch.float32, device=config.device)

        kin_pred = kin_pred.cpu().numpy()[0]

        # Store non-padded segment
        actual_len = w1 - w0
        preds[w0:w1] = kin_pred[:actual_len]

    return preds

def run_eval():
    config = Config()
    print("Building models...")
    mae, lstm, session_kin_stats, session_sbp_stats = build_models(config)

    print("Loading test data...")
    session_manager = SessionDataPhase2(config.data_path, is_train=False)
    sessions = session_manager.generate_session_obj()

    predictions = {}
    for session in tqdm(sessions, desc="Predicting Test Sessions"):
        sbp, _ = session.load_data()
        if sbp is None: continue
        preds = predict_session(mae, lstm, sbp, config, session.session_id, session_kin_stats, session_sbp_stats)
        
        # Apply smoothing
        preds = smooth_predictions(preds, kernel_size=config.smoothing_kernel_size)
        
        predictions[session.session_id] = preds
        
    print("Constructing submission...")
    sub_path = os.path.join(config.data_path, "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    
    # Positions are physically bounded to [0, 1]
    for session_id, preds in predictions.items():
        idx = sub["session_id"] == session_id
        if not idx.any(): continue
        
        time_bins = sub.loc[idx, "time_bin"].to_numpy(dtype=np.int64)
        
        # Clip to [0, 1] after smoothing and un-normalization
        sub.loc[idx, "index_pos"] = np.clip(preds[time_bins, 0], 0, 1)
        sub.loc[idx, "mrp_pos"] = np.clip(preds[time_bins, 1], 0, 1)
        
    out_csv = "submission_phase2.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    run_eval()
