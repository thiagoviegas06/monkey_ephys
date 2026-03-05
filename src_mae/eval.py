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


if __name__ == "__main__":
    data_path = "kaggle_data"
    metadata_csv = f"{data_path}/metadata.csv"
    window_size = 200
    session_objs = preprocess_test(data_path, window_size, metadata_csv)
    print(f"Total windows created: {len(session_objs)}")


