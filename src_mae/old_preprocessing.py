import pandas as pd
import numpy as np
import pickle
import os

# paste your rows here once; or load from csv
STATS = [
    (51, 58.5, 82), (59, 64.0, 183), (56, 59.0, 70),
    (47, 54.5, 117), (52, 57.5, 152), (46, 59.0, 108),
    (63, 68.0, 104), (51, 55.5, 108), (54, 61.0, 114),
    (54, 61.5, 123), (54, 57.5, 86), (51, 54.0, 103),
    (38, 59.0, 119), (53, 60.5, 167), (51, 53.0, 58),
    (54, 59.0, 131), (53, 65.5, 72), (54, 65.5, 129),
    (55, 72.0, 101), (53, 69.5, 122), (37, 57.0, 128),
    (54, 62.5, 117), (55, 68.0, 121), (37, 58.0, 90),
]

def sample_span_len(rng: np.random.Generator, W: int, stats=STATS) -> int:
    """
    Sample a contiguous masked span length, clipped to window size W.
    Uses a mixture of per-session triangular distributions.
    """
    mn, med, mx = stats[int(rng.integers(0, len(stats)))]
    # triangular expects left, mode, right
    L = rng.triangular(left=mn, mode=med, right=mx)
    L = int(np.round(L))
    return max(1, min(W, L))

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
            train = True if split_data == "train" else False
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
            return f"{data_path}/train/{self.session_id}_sbp.npy"
        else:
            return f"{data_path}/test/{self.session_id}_sbp_masked.npy"
    
    def get_kin_path(self, data_path):
        if self.train:
            return f"{data_path}/train/{self.session_id}_kinematics.npy"
        else:
            return f"{data_path}/test/{self.session_id}_kinematics.npy"
    
    def get_trial_info(self, data_path):
        if self.train:
            return f"{data_path}/train/{self.session_id}_trial_info.npz"
        else:
            return f"{data_path}/test/{self.session_id}_trial_info.npz"

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

def non_overlapping_windows(N, W):
    """Generate non-overlapping window start indices."""
    w0s = []
    w0 = 0
    while w0 + W <= N:
        w0s.append(w0)
        w0 += W
    return w0s

def sample_multi_span_lengths_and_starts(rng, W, num_spans=2, min_gap=10, stats=STATS):                                        
      """                                                                                                                      
      Sample K non-overlapping span lengths AND positions.                                                                     
      All spans fit within [0, W) with at least min_gap between them.
   
      Returns: list of (t0, t1) tuples
      """
      total_gap_space = (num_spans - 1) * min_gap
      total_budget = W - total_gap_space

      if total_budget < num_spans:
          # Can't fit K spans, fall back to 1
          L = sample_span_len(rng, W, stats=stats)
          t0 = sample_span_start(rng, W, L)
          return [(t0, t0 + L)]

      # Sample lengths for each span
      lengths = []
      remaining_budget = total_budget

      for i in range(num_spans):
          # Reserve minimum budget for remaining spans
          min_reserved = (num_spans - i - 1) * 20
          max_possible = remaining_budget - min_reserved

          # Sample from triangular, but clip to what's available
          L = sample_span_len(rng, max_possible, stats=stats)
          lengths.append(L)
          remaining_budget -= L

      # Now sample positions with guaranteed gaps
      spans = []
      pos = 0

      for i, length in enumerate(lengths):
          # Random offset within this span's available range
          available_for_offset = W - sum(lengths) - total_gap_space
          offset = rng.integers(0, available_for_offset + 1) if available_for_offset > 0 else 0

          t0 = pos + offset
          t1 = t0 + length
          spans.append((t0, t1))

          pos = t1 + min_gap  # move past this span + gap

      return spans

def apply_multi_span_mask_to_window(sbp, spans, num_spans=2, rng=None, min_gap=10):
    W, C = sbp.shape
    x = sbp.copy()
    mask = np.zeros((W, C), dtype=bool)

    for t0, t1 in spans:
        channels = rng.choice(C, size=rng.integers(20, 40), replace=False)
        x[t0:t1, channels] = 0.0
        mask[t0:t1, channels] = True

    return x, mask

def compute_session_channel_variance(sbp):
    session_variance = np.var(sbp, axis=0)
    return session_variance

def preprocess_non_overlapping(data_path, window_size=128, seed=0):
    out_dir = os.path.join(data_path, "masked_windows")
    os.makedirs(out_dir, exist_ok=True)
    sessions, max_bin_count = sessionData(f"{data_path}/metadata.csv").generate_session_obj()

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
        print(f"{session.session_id} | N={N} | windows={len(w0s)}")

        session_variance = compute_session_channel_variance(sbp)
        variance_shape = session_variance.shape
        print(f"  Session channel variance shape: {variance_shape}")
        print(f"  Session channel variance (mean across channels): {session_variance.mean():.4f}")

        for w0 in w0s:
            y = sbp[w0:w0 + window_size]          # (W,96)
            kin_w = kin[w0:w0 + window_size]      # (W,4)

            random_two_three = rng.integers(2, 4)  # 2 or 3 spans
            spans = sample_multi_span_lengths_and_starts(rng, window_size, num_spans=random_two_three, min_gap=10)
            x, M = apply_multi_span_mask_to_window(y, spans, num_spans=random_two_three, rng=rng, min_gap=10)

            sample = {
                "x_sbp": x.astype(np.float32),
                "y_sbp": y.astype(np.float32),
                "mask": M,
                "kin": kin_w.astype(np.float32),
                "channel_var": session_variance.astype(np.float32),  # (96,) per-channel variance from full session
                "session_id": session.session_id,
                "w0": int(w0),
                "spans": spans,
                "day": float(session.day),
                "day_from_nearest": float(session.day_from_nearest),
            }

            sample_path = os.path.join(out_dir, f"{session.session_id}_{w0}.pkl")
            with open(sample_path, "wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if len(w0s) <= 5 or w0 == w0s[0]:  # Print first window or if few windows
                print(f"  Saved: {session.session_id}_{w0}.pkl | spans={spans} | masked={int(M.sum())} positions")

if __name__ == "__main__":
    preprocess_non_overlapping("kaggle_data", window_size=200, seed=0)