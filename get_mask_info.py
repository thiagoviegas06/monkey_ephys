import numpy as np

def inspect_mask(test_dir, session_id):
    sbp_m = np.load(f"{test_dir}/{session_id}_sbp_masked.npy")      # (N,96)
    mask  = np.load(f"{test_dir}/{session_id}_mask.npy")            # (N,96) bool
    info  = np.load(f"{test_dir}/{session_id}_trial_info.npz")
    starts = info["start_bins"].astype(int)
    ends   = info["end_bins"].astype(int)

    N, C = mask.shape
    print("N,C:", (N, C))
    print("masked chans per bin: min/mean/max =",
          mask.sum(axis=1).min(), mask.sum(axis=1).mean(), mask.sum(axis=1).max())

    # Check masked entries are actually zeroed
    masked_vals = sbp_m[mask]
    print("masked sbp values: min/max/mean =", masked_vals.min(), masked_vals.max(), masked_vals.mean())

    # Is mask constant over time?
    const_over_time = np.all(mask == mask[0:1], axis=1).all()
    print("mask constant over entire session?:", const_over_time)

    # How many unique mask rows exist? (sample to avoid huge cost)
    idx = np.linspace(0, N-1, num=min(5000, N), dtype=int)
    unique_rows = np.unique(mask[idx], axis=0).shape[0]
    print("unique mask rows (sampled):", unique_rows)

    # Is mask constant within each trial?
    trial_const = 0
    for s,e in zip(starts, ends):
        m = mask[s:e]
        if m.shape[0] == 0:
            continue
        if np.all(m == m[0:1], axis=1).all():
            trial_const += 1
    print("trials with constant mask:", trial_const, "/", len(starts))

# example:
inspect_mask("kaggle_data/test", "S008")
inspect_mask("kaggle_data/test", "S147")
inspect_mask("kaggle_data/test", "S230")