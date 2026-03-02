import pandas as pd
import numpy as np


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
        return session_objects


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

def apply_mask_to_window(sbp, bin_mask):
    masked_sbp_window = sbp.copy()
    masked_sbp_window[:, bin_mask] = 0
    return masked_sbp_window


import zlib
import numpy as np

def preprocess(data_path, window_size=128, K=500, seed=0, p_mask_trial=0.03):
    sessions = sessionData(f"{data_path}/metadata.csv").generate_session_obj()

    for session in sessions:
        if not session.train:
            print(f"Skipping test session {session.session_id} for now...")
            continue

        sbp, kin, starts_bins, end_bins = session.load_data(data_path)
        if sbp is None:
            continue
        N = sbp.shape[0]
        if N < window_size:
            continue

        seed_sess = seed + zlib.adler32(session.session_id.encode("utf-8"))
        rng = np.random.default_rng(seed_sess)

        samples = []
        w0s = []

        for _ in range(K):
            out = sample_window_start_containing_trial(starts_bins, end_bins, N, window_size, rng)
            if out is None:
                continue
            w0, ti = out

            window_sbp = sbp[w0:w0 + window_size]
            window_kin = kin[w0:w0 + window_size]

            s_t, e_t = int(starts_bins[ti]), int(end_bins[ti])
            assert w0 <= s_t and e_t <= w0 + window_size

            a = s_t - w0
            b = e_t - w0

            x = window_sbp.copy()
            mask_vec = np.zeros(96, dtype=np.float32)

            if rng.random() < p_mask_trial:
                masked_channels = rng.choice(96, size=30, replace=False)
                x[a:b, masked_channels] = 0.0
                mask_vec[masked_channels] = 1.0

            samples.append((x, window_kin, window_sbp, mask_vec, a, b))
            w0s.append(w0)

        if w0s:
            print(f"{session.session_id} | N={N} | samples={len(samples)} | w0[min,max]=({min(w0s)},{max(w0s)})")

if __name__ == "__main__":
    preprocess("kaggle_data")

    