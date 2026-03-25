
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def check_variances(data_path="kaggle_data"):
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    all_sessions = metadata["session_id"].tolist()
    
    all_vars = []
    max_vals = []
    low_var_details = []
    
    print(f"Checking {len(all_sessions)} sessions...")
    for sess_id in tqdm(all_sessions):
        split = metadata[metadata["session_id"] == sess_id]["split"].iloc[0]
        folder = "train" if split == "train" else "test"
        suffix = "" if split == "train" else "_masked"
        path = os.path.join(data_path, folder, f"{sess_id}_sbp{suffix}.npy")
        
        if not os.path.exists(path):
            continue
        sbp = np.load(path)
        var = np.var(sbp, axis=0)
        all_vars.extend(var.tolist())
        max_vals.append(np.max(sbp))
        
        if np.max(sbp) > 500:
             print(f"Session {sess_id} ({split}) has max SBP: {np.max(sbp)}")

    all_vars = np.array(all_vars)
    max_vals = np.array(max_vals)
    print("\nGlobal Statistics:")
    print(f"Max SBP value across all sessions: {np.max(max_vals)}")
    print(f"Max variance across all sessions: {np.max(all_vars)}")
    print(f"Min variance (non-zero): {all_vars[all_vars > 0].min()}")
    print(f"Mean variance: {all_vars.mean()}")

if __name__ == "__main__":
    check_variances()
