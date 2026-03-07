import os
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from src_mae.model import SBPImputer
from src_mae.losses import masked_nmse_loss

def visualize_prediction(windows_dir, model_path, session_id=None, save_path=None):
    """
    Visualizes the ground truth, prediction, and error map for a given session.
    
    Args:
        windows_dir: Directory containing the preprocessed .pkl files.
        model_path: Path to the trained model weights (.pth).
        session_id: Specific session to visualize (e.g., 'S101'). If None, picks the first available.
        save_path: Optional path to save the resulting plot as an image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    print(f"Loading model weights from {model_path}...")
    model = SBPImputer(d_model=256, nhead=8, num_layers=4).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found! Please train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Find a matching pickle file for the session
    if session_id:
        # Search for any window belonging to this session
        pattern = os.path.join(windows_dir, f"{session_id}_*.pkl")
        # Fallback if the naming convention in preprocessing was slightly different
        if not glob.glob(pattern): 
            pattern = os.path.join(windows_dir, f"*{session_id}*.pkl")
    else:
        pattern = os.path.join(windows_dir, "*.pkl")
        
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise ValueError(f"No .pkl data found for session: {session_id} in {windows_dir}")
        
    pkl_file = files[15] # Just grab the first available window for this session
    print(f"Loading data from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        sample = pickle.load(f)

    #Calculate the macro timestamp for this sample based on the filename (if not already in the sample)
    if "macro_timestamp" not in sample:
        filename = os.path.basename(pkl_file)
        if "_" in filename:
            macro_timestamp_str = filename.split("_")[1].split(".")[0]
            try:
                sample["macro_timestamp"] = int(macro_timestamp_str)
            except ValueError:
                sample["macro_timestamp"] = 0
        else:
            sample["macro_timestamp"] = 0

    # 3. Format inputs for the model (Add Batch dimension B=1)
    x_sbp = torch.from_numpy(sample["x_sbp"]).float().unsqueeze(0).to(device)
    y_sbp = torch.from_numpy(sample["y_sbp"]).float().unsqueeze(0).to(device)
    kin = torch.from_numpy(sample["kin"]).float().unsqueeze(0).to(device)
    macro_timestamp = torch.tensor([sample["macro_timestamp"]]).float().unsqueeze(0).to(device)
    mask_float = torch.from_numpy(sample["mask"]).float().unsqueeze(0).to(device)
    
    # Nd array for mask_bool
    mask_bool = sample["mask"]

    # 4. Run Inference
    with torch.no_grad():
        pred = model(x_sbp, kin, mask_float, macro_timestamp)
    
    mask_tensor = torch.from_numpy(mask_bool).unsqueeze(0).to(device)
    # Ensure it is boolean for the ~mask operation in losses.py
    mask_tensor = mask_tensor.bool()
    # Calculate Mask NMSE for this window (only on masked positions)
    window_nmse = masked_nmse_loss(pred, y_sbp, mask_tensor).item()

    # 5. Extract to numpy for plotting
    y_true_np = y_sbp.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    mask_np = mask_float.squeeze(0).cpu().numpy()
    
    # Calculate Masked Error Map
    error_np = np.abs(y_true_np - pred_np) * mask_np  # Absolute error only in masked regions

    # Identify masked span (for plotting highlights)
    # Find rows (time bins) where at least one channel was masked
    time_mask = mask_np.any(axis=1)
    masked_bins = np.where(time_mask)[0]
    
    if len(masked_bins) > 0:
        t0, t1 = masked_bins[0], masked_bins[-1]
    else:
        t0, t1 = 0, 0
        

    # 6. Create Plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    actual_session_id = sample.get("session_id", session_id)
    fig.suptitle(f"Session: {actual_session_id} | Window Start: {sample.get('w0', 'N/A')} | NMSE: {window_nmse:.4f}", fontsize=16)

    # Plot A: Ground Truth
    im0 = axes[0].imshow(y_true_np.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0].set_title("Ground Truth (y_sbp)")
    axes[0].set_ylabel("Channel")
    if t0 != t1: axes[0].axvspan(t0, t1, color='red', alpha=0.15, label="Masked Region")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    # Plot B: Predicted
    # (Optional: If you want to show the original unmasked regions seamlessly stitched with the predictions, 
    # you can use: display_pred = x_sbp.squeeze().cpu().numpy() + (pred_np * mask_np) )
    display_pred = pred_np # We'll just show the raw predictions everywhere for transparency
    
    im1 = axes[1].imshow(display_pred.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[1].set_title("Model Prediction (pred)")
    axes[1].set_ylabel("Channel")
    if t0 != t1: axes[1].axvspan(t0, t1, color='red', alpha=0.15)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    # Plot C: Error Map
    # We apply the mask to the error map to isolate where the model was actively predicting missing data
    isolated_error = error_np * mask_np
    
    im2 = axes[2].imshow(isolated_error.T, aspect="auto", interpolation="nearest", origin="lower", cmap='magma')
    axes[2].set_title("Error Map (|Ground Truth - Predicted|) * Mask")
    axes[2].set_ylabel("Channel")
    axes[2].set_xlabel("Time Bin")
    if t0 != t1: axes[2].axvspan(t0, t1, color='red', alpha=0.15)
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.02)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Ensure matplotlib is installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install matplotlib: pip install matplotlib")
        exit()
        
    WINDOWS_DIR = "kaggle_data/masked_windows" # Path to your .pkl files
    MODEL_WEIGHTS = "checkpoints/best_transformer.pth"
    
    # Change 'S101' to the session ID you want to inspect, or leave None for the first file
    TARGET_SESSION = "S201" 
    
    visualize_prediction(
        windows_dir=WINDOWS_DIR, 
        model_path=MODEL_WEIGHTS, 
        session_id=TARGET_SESSION,
        save_path=None
    )