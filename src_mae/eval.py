import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob
from preprocessing import sample_span_start
from config import Config
from model import SBP_Reconstruction_UNet, SimpleCNN, ResNetReconstructor, SBPImputer

config = Config()

# ============================================================================
# Model Builder
# ============================================================================
def build_model(config):
    """Build model based on config."""
    if config.model_name == "unet":
        model = SBP_Reconstruction_UNet(base_channels=config.base_channels)
        print(f"Built U-Net with base_channels={config.base_channels}")
    elif config.model_name == "simple_cnn":
        model = SimpleCNN(hidden_channels=128, num_layers=6)
        print("Built SimpleCNN")
    elif config.model_name == "resnet":
        model = ResNetReconstructor(hidden_channels=128, num_blocks=8)
        print("Built ResNet Reconstructor")
    elif config.model_name == "transformer":
        model = SBPImputer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            sbp_channels=config.sbp_channels,
            kin_channels=config.kin_channels
        )
        print(f"Built Transformer Reconstructor with d_model={config.d_model}, nhead={config.nhead}, num_layers={config.num_layers}, dropout={config.dropout}")
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model.to(config.device)


def mask_segments(mask_2d: np.ndarray):
    """
    mask_2d: (N, C) bool, True where masked.
    Returns list of (start,end) segments over time indices where any channel is masked.
    end is exclusive.
    """
    t = np.where(mask_2d.any(axis=1))[0]
    if len(t) == 0:
        return []
    # contiguous runs
    cuts = np.where(np.diff(t) > 1)[0]
    starts = np.r_[t[0], t[cuts + 1]]
    ends   = np.r_[t[cuts] + 1, t[-1] + 1]
    return list(zip(starts, ends))

def windows_covering_segment(seg_start, seg_end, N, W):
    """
    Build NON-overlapping windows of length W that cover [seg_start, seg_end) in time.
    Returns list of (w0,w1).
    """
    windows = []
    cur = seg_start
    while cur < seg_end:
        w0 = cur
        w1 = min(w0 + W, N)
        # ensure window length W when possible (shift left if we hit the end)
        if w1 - w0 < W and N >= W:
            w0 = N - W
            w1 = N
        windows.append((w0, w1))
        cur = w1  # <-- makes them non-overlapping
    return windows

def randomized_window_covering_segment(seg_start, seg_end, N, W, rng):
    """
    Build a single window of length W that fully covers [seg_start, seg_end),
    with randomized placement of the masked segment inside the window.
    """
    L = seg_end - seg_start

    if N <= W:
        return (0, N)

    lo = max(0, seg_end - W)
    hi = min(seg_start, N - W)
    if lo > hi:
        w0 = max(0, min(seg_start, N - W))
        return (w0, w0 + W)

    t0 = sample_span_start(rng, W=W, L=L)
    w0 = seg_start - t0
    w0 = int(np.clip(w0, lo, hi))
    return (w0, w0 + W)

def build_randomized_windows_from_mask(mask_2d: np.ndarray, window_size: int, rng):
    N = mask_2d.shape[0]
    segs = mask_segments(mask_2d)
    if not segs:
        return []

    windows = []
    if N <= window_size:
        return [(0, N)] * len(segs)

    for seg_start, seg_end in segs:
        L = seg_end - seg_start
        if L > window_size:
            raise ValueError(
                f"Masked segment length ({L}) exceeds window_size ({window_size}); "
                "cannot cover this region with a single window."
            )

        lo = max(0, seg_end - window_size)
        hi = min(seg_start, N - window_size)
        if lo > hi:
            w0 = max(0, min(seg_start, N - window_size))
        else:
            t0 = sample_span_start(rng, W=window_size, L=L)
            w0 = int(np.clip(seg_start - t0, lo, hi))

        windows.append((w0, w0 + window_size))

    return windows


def preprocess_test(data_path, window_size, metadata_csv):
    masked_files = os.path.join(data_path, "test/*_sbp_masked.npy")
    session_objs = []
    expected_regions = 10
    df = pd.read_csv(metadata_csv)
    for file in glob(masked_files):
        session_id = Path(file).stem.split("_")[0]
        rng = np.random.default_rng(42 + (hash(session_id) & 0xFFFFFFFF))
        masked_sbp = np.load(file)  # (W, C)
        mask  = np.load(file.replace("sbp_masked", "mask"))  # (W, C)
        kin   = np.load(file.replace("sbp_masked", "kinematics"))  # (W, 3)

        segs = mask_segments(mask)
        windows = build_randomized_windows_from_mask(mask, window_size, rng)

        if len(segs) != expected_regions:
            print(
                f"WARNING: Session {session_id} has {len(segs)} masked regions "
                f"(expected {expected_regions})."
            )

        if len(windows) != len(segs):
            raise RuntimeError(
                f"Session {session_id}: expected {len(segs)} windows (one per region), "
                f"got {len(windows)}."
            )

        for w0, w1 in windows:
            print(f"Session {session_id} | window ({w0},{w1}) covers masked segment with {mask[w0:w1].sum()} masked positions.")
            session_info = df[df["session_id"] == session_id].iloc[0]
            session_objs.append({
                "session_id": session_id,
                "x_sbp": masked_sbp[w0:w1],
                "mask": mask[w0:w1],
                "kinematics": kin[w0:w1],
            })
    return session_objs

#DIEGO INCORRECT ATTEMPT TO RUN EVALUATION
def evaluate_and_submit(data_path, model_path, output_csv="submission.csv", window_size=200):
    """
    Runs the trained model on the masked test data and generates the Kaggle submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    print(f"Loading model weights from {model_path}...")
    model = build_model(config)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found! Please train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Get the list of test sessions from metadata
    metadata_path = os.path.join(data_path, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    test_sessions = metadata[metadata["split"] == "test"]["session_id"].tolist()
    
    if not test_sessions:
        print("No test sessions found in metadata.csv. Please check your splits.")
        return

    # 3. Iterate over test sessions to generate dense predictions
    session_preds = {}
    print("Running inference on test sessions...")
    
    with torch.no_grad():
        for sess_id in tqdm(test_sessions, desc="Processing Sessions"):
            sbp_path = os.path.join(data_path, "test", f"{sess_id}_sbp_masked.npy")
            mask_path = os.path.join(data_path, "test", f"{sess_id}_mask.npy")
            kin_path = os.path.join(data_path, "test", f"{sess_id}_kinematics.npy")
            
            if not os.path.exists(sbp_path):
                print(f"Missing data for {sess_id}, skipping...")
                continue
                
            x_sbp = np.load(sbp_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
            kin = np.load(kin_path).astype(np.float32)
            
            seq_len = x_sbp.shape[0]
            pred_sbp_full = np.zeros_like(x_sbp)
            
            # Process in chunks of `window_size` to prevent Out-of-Memory (OOM) errors
            # and to match the sequence lengths seen during training.
            for i in range(0, seq_len, window_size):
                end = min(i + window_size, seq_len)
                
                # Extract chunk and add batch dimension (B=1)
                x_chunk = torch.tensor(x_sbp[i:end]).unsqueeze(0).to(device)
                kin_chunk = torch.tensor(kin[i:end]).unsqueeze(0).to(device)
                mask_chunk = torch.tensor(mask[i:end]).unsqueeze(0).to(device)
                
                # Forward pass
                out_chunk = model(x_chunk, kin_chunk, mask_chunk)
                
                # Store chunk predictions back into the full session array
                pred_sbp_full[i:end] = out_chunk.squeeze(0).cpu().numpy()
                
            session_preds[sess_id] = pred_sbp_full

    # 4. Map the dense predictions to the required submission format
    print("Mapping predictions to Kaggle format...")
    test_mask_df = pd.read_csv(os.path.join(data_path, "test_mask.csv"))
    
    predictions_list = []
    
    # We iterate over the index file to pick out the exact row/col that was masked
    # Note: If your CSV uses different headers for time bins and channels (e.g., 'row_idx', 'col_idx'),
    # you will need to update the keys inside this loop accordingly.
    for _, row in tqdm(test_mask_df.iterrows(), total=len(test_mask_df), desc="Formatting Submission"):
        sample_id = int(row['sample_id'])
        sess_id = row['session_id']
        time_bin = int(row['time_bin'])
        channel = int(row['channel'])
        
        # Retrieve the predicted value for this specific bin and channel
        pred_val = session_preds[sess_id][time_bin, channel]
        
        predictions_list.append({
            'sample_id': sample_id,
            'predicted_sbp': pred_val
        })
        
    # 5. Save the final submission file
    submission_df = pd.DataFrame(predictions_list)
    
    # Ensure it's sorted by sample_id exactly as requested by Kaggle
    submission_df = submission_df.sort_values(by='sample_id')
    submission_df.to_csv(output_csv, index=False)
    
    print(f"\nEvaluation complete! Saved ready-to-submit predictions to: {output_csv}")
    print(submission_df.head())

if __name__ == "__main__":
    # Point this to the root of your kaggle_data folder
    DATA_DIR = "kaggle_data" 
    MODEL_WEIGHTS = "checkpoints/best_transformer.pth"
    OUTPUT_FILE = "submission.csv"
    
    evaluate_and_submit(
        data_path=DATA_DIR, 
        model_path=MODEL_WEIGHTS, 
        output_csv=OUTPUT_FILE,
        window_size=200
    )


# if __name__ == "__main__":
#     data_path = "kaggle_data"
#     metadata_csv = f"{data_path}/metadata.csv"
#     window_size = 200
#     session_objs = preprocess_test(data_path, window_size, metadata_csv)
#     print(f"Total windows created: {len(session_objs)}")


