import pandas as pd
import numpy as np
import zlib
import pickle
import os
from pathlib import Path
from tqdm import tqdm

def sample_span_start(rng, W, L, p_uniform=0.6, skew_strength=3.0):
    """
    Sample t0 in [0, W-L] with a mixture of:
      - uniform (p_uniform)
      - left-skew beta (remaining/2)
      - right-skew beta (remaining/2)
    skew_strength > 1 makes it hug edges more.
    """
    max_start = W - L
    if max_start <= 0:
        return 0

    u = rng.random()
    if u < p_uniform:
        return int(rng.integers(0, max_start + 1))

    # skewed
    left = (u < p_uniform + (1 - p_uniform) / 2)

    a = 1.0
    b = skew_strength
    x = rng.beta(a, b)  # concentrates near 0
    if not left:
        x = 1.0 - x      # concentrates near 1

    return int(np.round(x * max_start))

def _read_metadata_file(metadata_file):
    """Reads the metadata file and returns a DataFrame."""
    return pd.read_csv(metadata_file)

class sessionData:
    """Class to hold session data and metadata."""

    def __init__(self, metadata_file):
        self.metadata = _read_metadata_file(metadata_file)
        self.sessions = self.metadata["session_id"].tolist()
    
    def generate_session_obj(self):
        session_objects = []
        max_bin_count = self.metadata["n_bins"].max()
        print(f"Max bin count across sessions: {max_bin_count}")
        for session_id in self.sessions:
            session_metadata = self.metadata[self.metadata["session_id"] == session_id]
            split_data = session_metadata["split"].iloc[0]
            if split_data == "train":
                train = True
            else:
                train = False
            day = session_metadata["day"].iloc[0]
            day_from_nearest = session_metadata["days_from_nearest_train"].iloc[0]
            n_bins = session_metadata["n_bins"].iloc[0]
            n_trials = session_metadata["n_trials"].iloc[0]
            session_obj = sessionObj(train=train, session_id=session_id, day=day, day_from_nearest=day_from_nearest, n_bins=n_bins, n_trials=n_trials)
            session_objects.append(session_obj)
        return session_objects, max_bin_count


class sessionObj:
    """Class to hold individual session data."""

    def __init__(self, train=True, session_id=None, day=None, day_from_nearest=None, n_bins=-1, n_trials=-1):
        self.train = train
        self.session_id = session_id
        self.day = day
        self.day_from_nearest = day_from_nearest
        self.n_bins = n_bins
        self.n_trials = n_trials

    def get_sbp_path(self, data_path):
        if self.train:
            sbp_path = f"{data_path}/train/{self.session_id}_sbp.npy"
        else:
            sbp_path = f"{data_path}/test/{self.session_id}_sbp_masked.npy"
        return sbp_path
    
    def get_kin_path(self, data_path):
        if self.train:
            kin_path = f"{data_path}/train/{self.session_id}_kinematics.npy"
        else:
            kin_path = f"{data_path}/test/{self.session_id}_kinematics.npy"
        return kin_path
    
    def get_trial_info(self, data_path):
        if self.train:
            trial_info_path = f"{data_path}/train/{self.session_id}_trial_info.npz"
        else:
            trial_info_path = f"{data_path}/test/{self.session_id}_trial_info.npz"
        return trial_info_path

    def load_data(self, data_path):
        sbp_path = self.get_sbp_path(data_path)
        kin_path = self.get_kin_path(data_path)
        trial_info_path = self.get_trial_info(data_path)

        try:
            sbp_norm = np.load(sbp_path).astype(np.float32)
            kinematics = np.load(kin_path).astype(np.float32)
            trial_info = np.load(trial_info_path)

            trials_start = trial_info["start_bins"]
            trials_end = trial_info["end_bins"]
            return sbp_norm, kinematics, trials_start, trials_end

        except Exception as e:
            print(f"Error loading data for session {self.session_id}: {e}")
            return None, None, None, None
        
    def isTest(self):
        return not self.train

def sample_window_start_containing_trial(starts, ends, N, W, rng):
    # choose a trial that can fit inside the window
    lengths = ends - starts
    ok = np.where(lengths <= W)[0]
    if len(ok) == 0:
        return None  # fallback needed

    ti = int(rng.choice(ok))
    s_t, e_t = int(starts[ti]), int(ends[ti])

    lo = max(0, e_t - W)
    hi = min(s_t, N - W)
    if lo > hi:
        return None  # rare edge case near boundaries

    w0 = int(rng.integers(lo, hi + 1))
    return w0, ti

def non_overlapping_windows(N, W):
    """Generate non-overlapping window start indices."""
    w0s = []
    w0 = 0
    while w0 + W <= N:
        w0s.append(w0)
        w0 += W
    return w0s

def apply_elementwise_random_mask(sbp, rng, mask_fraction=0.4):
    """
    Randomly masks `mask_fraction` of ALL individual data points 
    in the window (not full channels or specific time spans).
    """
    W, C = sbp.shape
    x = sbp.copy()
    mask = np.zeros((W, C), dtype=np.bool_)

    total_elements = W * C
    num_masked = int(np.round(total_elements * mask_fraction))

    # Pick random 1D indices representing the locations to mask
    masked_indices_1d = rng.choice(total_elements, size=num_masked, replace=False)
    
    # Convert back to 2D indices
    t_idx, c_idx = np.unravel_index(masked_indices_1d, (W, C))

    x[t_idx, c_idx] = 0.0
    mask[t_idx, c_idx] = True

    return x, mask

def compute_session_channel_variance(sbp):
    session_variance = np.var(sbp, axis=0)
    return session_variance


def preprocess_non_overlapping(data_path, window_size=128, seed=0):
    out_dir = os.path.join(data_path, f"masked_windows_{window_size}")
    os.makedirs(out_dir, exist_ok=True)
    sessions, max_bin_count = sessionData(f"{data_path}/metadata.csv").generate_session_obj()

    # Add tqdm progress bar for sessions
    sessions = tqdm(sessions, desc="Processing sessions")

    for session in sessions:
        if session.isTest():
            continue
        sbp, kin, starts_bins, end_bins = session.load_data(data_path)
        if sbp is None:
            continue
        
        N = sbp.shape[0]
        if N < window_size:
            continue
            
        rng = np.random.default_rng(seed + (hash(session.session_id) & 0xFFFFFFFF))
        w0s = non_overlapping_windows(N, window_size)
        
        # tqdm.write is better than print when using tqdm
        tqdm.write(f"{session.session_id} | N={N} | windows={len(w0s)}")

        session_variance = compute_session_channel_variance(sbp)

        for w0 in w0s:
            y = sbp[w0:w0 + window_size]          # (W, C)
            kin_w = kin[w0:w0 + window_size]      # (W, 4)

            # Generate 3 independent random 40% masks over the same window
            for mask_idx in range(3):
                
                # Dynamically generate an entirely random 40% spread mask
                x, M = apply_elementwise_random_mask(y, rng, mask_fraction=0.4)

                sample = {
                    "x_sbp": x.astype(np.float32),
                    "y_sbp": y.astype(np.float32),
                    "mask": M,
                    "kin": kin_w.astype(np.float32),
                    "channel_var": session_variance.astype(np.float32),
                    "session_id": session.session_id,
                    "w0": int(w0),
                    "span": (0, window_size), # Entire window is subjected to masking
                    "day": float(session.day),
                    "day_from_nearest": float(session.day_from_nearest),
                    
                    # Save the augmentation metadata for downstream tracking
                    "mask_idx": mask_idx, 
                    "mask_fraction": 0.4
                }

                # Update filename to prevent overwrites and identify the augmentation
                filename = f"{session.session_id}_{w0}_mask{mask_idx}.pkl"
                sample_path = os.path.join(out_dir, filename)
                
                with open(sample_path, "wb") as f:
                    pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Print a status update for the first window to confirm it's working
            if w0 == w0s[0]:  
                tqdm.write(f"  Saved 3 augmented variations (40% masking) for window {w0}")


def generate_one_masked_window(data_path, window_size=200, seed=0, session_id=None):
    """
    Generate a single masked window for visualization from preprocess_non_overlapping.
    
    Returns:
        dict with keys: x_sbp, y_sbp, mask, kin, session_id, w0, span
    """
    sessions, _ = sessionData(f"{data_path}/metadata.csv").generate_session_obj()
    
    if session_id is None:
        # Find first train session
        for s in sessions:
            if s.train:
                session_id = s.session_id
                session = s
                break
    else:
        # Find specific session
        session = None
        for s in sessions:
            if s.session_id == session_id:
                session = s
                break
        if session is None:
            raise ValueError(f"Session {session_id} not found")
    
    if session.isTest():
        raise ValueError(f"Session {session_id} is a test session, need train session")
    
    sbp, kin, starts_bins, end_bins = session.load_data(data_path)
    if sbp is None:
        raise RuntimeError(f"Failed to load data for session {session.session_id}")
    
    N = sbp.shape[0]
    if N < window_size:
        raise ValueError(f"Session {session.session_id} has {N} bins, smaller than window_size={window_size}")
    
    rng = np.random.default_rng(seed + (hash(session.session_id) & 0xFFFFFFFF))
    
    # Get first non-overlapping window
    w0s = non_overlapping_windows(N, window_size)
    if not w0s:
        raise RuntimeError("No windows generated")
    
    w0 = w0s[0]
    y = sbp[w0:w0 + window_size].copy()
    kin_w = kin[w0:w0 + window_size].copy()
    
    x, M = apply_elementwise_random_mask(y, rng, mask_fraction=0.4)

    print(f"Generated masked window for session {session.session_id}, w0={w0}, masked positions={int(M.sum())}")
    
    return {
        "x_sbp": x.astype(np.float32),
        "y_sbp": y.astype(np.float32),
        "mask": M,
        "kin": kin_w.astype(np.float32),
        "session_id": session.session_id,
        "w0": int(w0),
        "span": (0, window_size),
    }


def visualize_masked_window(sample, save_path=None):
    """
    Visualize a masked window from preprocess_non_overlapping.
    
    Args:
        sample: dict with x_sbp, y_sbp, mask, kin, session_id, w0, span
        save_path: optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc
    
    x_sbp = sample["x_sbp"]
    y_sbp = sample["y_sbp"]
    kin = sample["kin"]
    mask = sample["mask"]  # (W, 96) 2D mask
    session_id = sample["session_id"]
    w0 = sample["w0"]
    W, C = y_sbp.shape
    
    # Which channels are masked anywhere in the window
    mask_vec = mask.any(axis=0)  # (96,) boolean - True if channel masked in any bin
    masked_channels = np.flatnonzero(mask_vec)
    
    # Create display copy where masked values are NaN (will show as white)
    x_sbp_display = x_sbp.copy()
    x_sbp_display[x_sbp == 0] = np.nan
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    
    # Ground truth SBP
    im0 = axes[0, 0].imshow(y_sbp.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 0].set_title("Ground-truth SBP window")
    axes[0, 0].set_xlabel("Window bin")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Masked input SBP - NaN values appear white
    im1 = axes[0, 1].imshow(x_sbp_display.T, aspect="auto", interpolation="nearest", origin="lower", cmap='viridis')
    axes[0, 1].set_title("Masked input SBP window (white = masked)")
    axes[0, 1].set_xlabel("Window bin")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Time profile
    mean_all = y_sbp.mean(axis=1)
    axes[1, 0].plot(mean_all, label="mean SBP (all channels)", linewidth=2)
    if masked_channels.size > 0:
        mean_masked_true = y_sbp[:, masked_channels].mean(axis=1)
        mean_masked_in = x_sbp[:, masked_channels].mean(axis=1)
        axes[1, 0].plot(mean_masked_true, label="mean masked channels (true)", linewidth=1.7)
        axes[1, 0].plot(mean_masked_in, label="mean masked channels (input)", linewidth=1.7)
    axes[1, 0].set_title("SBP time profile in window")
    axes[1, 0].set_xlabel("Window bin")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(loc="upper right")
    
    # Kinematics
    axes[1, 1].plot(kin)
    axes[1, 1].set_title("Kinematics (4 channels)")
    axes[1, 1].set_xlabel("Window bin")
    axes[1, 1].set_ylabel("Value")
    
    mask_count_positions = int(mask.sum())  # Total masked (bin, channel) positions
    mask_pct = (mask_count_positions / (W * C)) * 100
    
    fig.suptitle(
        (
            f"session={session_id}  w0={w0} | "
            f"Total window size={W}x{C} | Masked points={mask_count_positions} ({mask_pct:.1f}%)"
        ),
        fontsize=12,
    )
    
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Saved plot to: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    preprocess_non_overlapping("kaggle_data", window_size=200, seed=0)
