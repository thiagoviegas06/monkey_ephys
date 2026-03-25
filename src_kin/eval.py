import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_kin.config import Config
from src_kin.model import PerceiverIOKinematicDecoder
from src_mae.model import SBP_TCN_Transformer
from src_kin.preprocessing import SessionDataPhase2
from src_kin.compute_kin_stats import compute_kin_stats

def build_models(config):
    # MAE
    mae = SBP_TCN_Transformer(
        sbp_channels=config.sbp_channels,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        tcn_levels=config.tcn_levels,
        dropout=config.dropout
    ).to(config.device)
    mae.load_state_dict(torch.load(config.mae_checkpoint_path, map_location=config.device)['model_state_dict'])
    mae.eval()

    # Perceiver IO Kinematic Decoder
    kinematic_model = PerceiverIOKinematicDecoder(config).to(config.device)
    best_model_path = os.path.join(config.checkpoint_dir, "best_model_perceiver.pt")
    
    # Need to handle case if checkpoint doesn't exist yet gracefully
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.device)
        kinematic_model.load_state_dict(checkpoint['model_state_dict'])
        # Get kinematics and SBP stats from checkpoint or recompute
        kin_mean = checkpoint.get('kin_mean', None)
        kin_std = checkpoint.get('kin_std', None)
        sbp_mean = checkpoint.get('sbp_mean', None)
        sbp_std = checkpoint.get('sbp_std', None)
    else:
        print(f"Warning: {best_model_path} not found. Using untrained model.")
        kin_mean, kin_std, sbp_mean, sbp_std = None, None, None, None

    kinematic_model.eval()

    
    if kin_mean is None or kin_std is None:
        print("Kinematics stats not found in checkpoint. Recomputing...")
        from src_kin.compute_kin_stats import compute_kin_stats
        kin_mean, kin_std = compute_kin_stats(config.data_path)
    
    if sbp_mean is None or sbp_std is None:
        print("SBP stats not found in checkpoint. Recomputing...")
        from src_kin.compute_kin_stats import compute_sbp_stats
        sbp_mean, sbp_std = compute_sbp_stats(config.data_path)

    kin_mean = kin_mean.to(config.device)
    kin_std = kin_std.to(config.device)
    sbp_mean = sbp_mean.to(config.device)
    sbp_std = sbp_std.to(config.device)

    return mae, kinematic_model, kin_mean, kin_std, sbp_mean, sbp_std

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
def predict_session(mae, kinematic_model, sbp, config, kin_mean, kin_std, sbp_mean, sbp_std, session_id):
    N = sbp.shape[0]
    W = config.window_size
    preds = np.zeros((N, config.output_dim), dtype=np.float32)
    
    try:
        num_id = float(''.join(filter(str.isdigit, session_id)))
    except ValueError:
        num_id = 0.0
    
    # Process in windows
    for w0 in range(0, N, W):
        w1 = min(w0 + W, N)
        sbp_w = torch.from_numpy(sbp[w0:w1]).unsqueeze(0) # (1, w_len, C)
        
        # Pad if necessary for the final window
        pad_len = W - (w1 - w0)
        if pad_len > 0:
            sbp_w = torch.cat([sbp_w, torch.zeros(1, pad_len, config.sbp_channels)], dim=1)
            
        sbp_w = sbp_w.to(config.device)
        mask = (sbp_w == 0.0).float().to(config.device)
        
        # Format timestamps and session
        macro_timestamp = torch.tensor([[w0]], dtype=torch.float32, device=config.device)
        session_num = torch.tensor([[num_id]], dtype=torch.float32, device=config.device)
        
        # Impute missing neural activity
        # Expand global stats to (1, sbp_channels) for the window
        channel_mean = sbp_mean.view(1, 1).expand(1, config.sbp_channels) if sbp_mean is not None else None
        channel_var = (sbp_std**2).view(1, 1).expand(1, config.sbp_channels) if sbp_std is not None else None
        
        sbp_imputed = mae(sbp_w, mask, macro_timestamp, channel_mean=channel_mean, channel_var=channel_var)
        
        # Decode kinematics
        kin_pred, _, _, _ = kinematic_model(
            sbp_imputed, 
            mask=mask, 
            session_num=session_num, 
            macro_timestamp=macro_timestamp.squeeze(-1), 
            sbp_mean=sbp_mean, 
            sbp_std=sbp_std
        )
        
        # Un-normalize kinematics (Global Z-Score)
        kin_pred = kin_pred * kin_std + kin_mean
        
        kin_pred = kin_pred.cpu().numpy()[0]
        
        # Store non-padded segment
        actual_len = w1 - w0
        preds[w0:w1] = kin_pred[:actual_len]
        
    return preds

def run_eval():
    config = Config()
    print("Building models...")
    mae, kinematic_model, kin_mean, kin_std, sbp_mean, sbp_std = build_models(config)
    
    print("Loading test data...")
    session_manager = SessionDataPhase2(config.data_path, is_train=False)
    sessions = session_manager.generate_session_obj()
    
    predictions = {}
    for session in tqdm(sessions, desc="Predicting Test Sessions"):
        sbp, _ = session.load_data()
        if sbp is None: continue
        preds = predict_session(mae, kinematic_model, sbp, config, kin_mean, kin_std, sbp_mean, sbp_std, session.session_id)
        
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
