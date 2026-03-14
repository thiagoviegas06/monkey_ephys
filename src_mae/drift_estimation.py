import numpy as np
import matplotlib.pyplot as plt
import os

def chunk_rms_profile(X, chunk_len):
    # X: (T, C)
    T, C = X.shape
    n_chunks = T // chunk_len
    Xc = X[:n_chunks * chunk_len].reshape(n_chunks, chunk_len, C)
    # RMS per chunk per channel
    P = np.sqrt(np.mean(Xc**2, axis=1))
    return P, n_chunks

def best_shift(a, b, max_shift):
    # a, b: (C,)
    # returns integer shift s where shifting a by s best aligns to b
    C = a.shape[0]
    best_s, best_score = 0, -np.inf
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            aa = a[-s:]          # drop first -s
            bb = b[:C+s]         # drop last -s
        elif s > 0:
            aa = a[:C-s]
            bb = b[s:]
        else:
            aa, bb = a, b

        # normalized dot (cosine similarity-ish)
        denom = (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8)
        score = float(np.dot(aa, bb) / denom)
        if score > best_score:
            best_score = score
            best_s = s
    return best_s

def smooth_shifts(shifts, k=5):
    # simple median smoothing
    shifts = np.asarray(shifts)
    out = shifts.copy()
    r = k // 2
    for i in range(len(shifts)):
        lo = max(0, i - r)
        hi = min(len(shifts), i + r + 1)
        out[i] = int(np.median(shifts[lo:hi]))
    return out

def apply_channel_shift_chunkwise(X, shifts, chunk_len):
    # X: (T, C), shifts: (n_chunks,) integer channel shifts
    T, C = X.shape
    n_chunks = len(shifts)
    X_aligned = X.copy()

    for k in range(n_chunks):
        s = shifts[k]
        t0 = k * chunk_len
        t1 = t0 + chunk_len
        chunk = X[t0:t1]  # (chunk_len, C)

        out = np.zeros_like(chunk)
        if s == 0:
            out = chunk
        elif s > 0:
            # shift "down" channels: channel i goes to i+s
            out[:, :-s] = chunk[:, s:]
        else:
            s2 = -s
            out[:, s2:] = chunk[:, :-s2]

        X_aligned[t0:t1] = out

    return X_aligned

def drift_align_session(X, fs=50, chunk_seconds=5, max_shift=12, smooth_k=5, ref_mode="early"):
    print("max_shift in channels:", max_shift)
    
    chunk_len = int(fs * chunk_seconds)

    P, n_chunks = chunk_rms_profile(X, chunk_len)
    print("P chunk0 vs chunk10 L2:", float(np.linalg.norm(P[0] - P[min(10, len(P)-1)])))
    print("P std across chunks:", float(P.std()))
    d = np.linalg.norm(P[0] - P[min(10, len(P)-1)])
    r = d / (np.linalg.norm(P[0]) + 1e-8)
    print("relative L2:", float(r))

    # per-channel variability over time, then average across channels
    print("mean per-channel std over time:", float(P.std(axis=0).mean()))

    # per-chunk variability across channels, then average across chunks
    print("mean per-chunk std across channels:", float(P.std(axis=1).mean()))

    if ref_mode == "early":
        ref = np.median(P[:min(10, n_chunks)], axis=0)
    else:
        ref = np.median(P, axis=0)

    shifts = np.array([best_shift(P[k], ref, max_shift) for k in range(n_chunks)], dtype=int)
    print("raw shifts:  mean", shifts.mean(), "std", shifts.std(), "min", shifts.min(), "max", shifts.max())

    shifts_smooth = smooth_shifts(shifts, k=smooth_k)
    print("smooth shifts: mean", shifts_smooth.mean(), "std", shifts_smooth.std(), "min", shifts_smooth.min(), "max", shifts_smooth.max())

    X_aligned = apply_channel_shift_chunkwise(X, shifts_smooth, chunk_len)
    return X_aligned, shifts, shifts_smooth, P

def load_session_data(session_id, train):
    if train:
        file_path = f"kaggle_data/train/{session_id}_sbp.npy"
    else:
        file_path = f"kaggle_data/test/{session_id}_sbp_masked.npy"
    return np.load(file_path)

def plot_heatmaps(X, X_aligned, shifts, shifts_smooth):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.imshow(X.T, aspect='auto', origin='lower')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Aligned Data")
    plt.imshow(X_aligned.T, aspect='auto', origin='lower')
    plt.colorbar()
    
    plt.figure(figsize=(10, 4))
    plt.plot(shifts, label='Raw Shifts')
    plt.plot(shifts_smooth, label='Smoothed Shifts', linewidth=2)
    plt.title("Channel Shifts Over Time")
    plt.xlabel("Chunk Index")
    plt.ylabel("Channel Shift")
    plt.legend()
    plt.grid()
    plt.show()

def robust_session_norm(X, eps=1e-6):
    # log + robust z-score per channel (within session)
    X_log = np.log(X + eps)
    med = np.median(X_log, axis=0, keepdims=True)
    mad = np.median(np.abs(X_log - med), axis=0, keepdims=True)
    Xn = (X_log - med) / (mad + eps)
    return Xn

def channel_fingerprint(X):
    # one scalar per channel
    return np.median(np.abs(X), axis=0)

def cosine_sim(a, b, eps=1e-9):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))

def sliding_channel_corr(X, window=2000):
    T, C = X.shape
    corrs = []

    for t in range(0, T-window, window):
        w = X[t:t+window]
        corr = np.mean(np.corrcoef(w.T))
        corrs.append(corr)

    plt.plot(corrs)
    plt.title("Average Channel Correlation Over Time")
    plt.xlabel("Window")
    plt.ylabel("Mean Correlation")
    plt.show()

def similarity_curve(fingerprints):

    S = len(fingerprints)
    dist = []
    sim = []

    for i in range(S):
        for j in range(i+1, S):
            dist.append(abs(i-j))
            sim.append(np.corrcoef(fingerprints[i], fingerprints[j])[0,1])

    dist = np.array(dist)
    sim = np.array(sim)

    max_d = dist.max()
    mean_sim = []

    for d in range(1, max_d+1):
        mean_sim.append(sim[dist == d].mean())

    return mean_sim

if __name__ == "__main__":
        # Example usage

    directory = "kaggle_data/train"
    file_names = os.listdir(directory)

    session_ids = sorted({fn.split("_")[0] for fn in file_names if fn.endswith(".npy")})

    fingerprints_raw = []
    fingerprints_norm = []

    for sid in session_ids:
        X = load_session_data(sid, train=True)  # shape (T, 96)

        fp_raw = channel_fingerprint(X)

        Xn = robust_session_norm(X)
        fp_norm = channel_fingerprint(Xn)

        fingerprints_raw.append(fp_raw)
        fingerprints_norm.append(fp_norm)

    fingerprints_raw = np.asarray(fingerprints_raw)   # (S, C)
    fingerprints_norm = np.asarray(fingerprints_norm) # (S, C)

    print("Raw:", fingerprints_raw.shape, "Norm:", fingerprints_norm.shape)

    # --- correlation matrices (session-session) ---
    corr_raw = np.corrcoef(fingerprints_raw)
    corr_norm = np.corrcoef(fingerprints_norm)


   #----plot fingerprints side by side for all sessions---
   #first flip x and y for better visualization (channels on y-axis)
    

    # --- similarity vs distance (cosine) ---
    dist = []
    sim_raw = []
    sim_norm = []   

    S = len(session_ids)
    for i in range(S):
        for j in range(i + 1, S):
            d = j - i
            dist.append(d)
            sim_raw.append(cosine_sim(fingerprints_raw[i], fingerprints_raw[j]))
            sim_norm.append(cosine_sim(fingerprints_norm[i], fingerprints_norm[j]))

    dist = np.asarray(dist)
    sim_raw = np.asarray(sim_raw)
    sim_norm = np.asarray(sim_norm)

    # --- plot similarity vs distance ---
    raw_curve = similarity_curve(fingerprints_raw)
    norm_curve = similarity_curve(fingerprints_norm)
    """
    plt.figure(figsize=(8,5))
    plt.plot(raw_curve, label="RAW")
    plt.plot(norm_curve, label="NORMALIZED")
    plt.xlabel("Session distance")
    plt.ylabel("Mean similarity")
    plt.title("Session similarity vs time")
    plt.legend()
    plt.show()"""

    fingerprints_raw = fingerprints_raw.T
    fingerprints_norm = fingerprints_norm.T
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Fingerprints (RAW)")
    plt.imshow(fingerprints_raw, aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Fingerprints (NORMALIZED)")
    plt.imshow(fingerprints_norm, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()


    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    raw_2d = pca.fit_transform(fingerprints_raw)
    norm_2d = pca.fit_transform(fingerprints_norm)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("PCA of RAW fingerprints")
    plt.scatter(raw_2d[:, 0], raw_2d[:, 1])
    plt.subplot(1, 2, 2)
    plt.title("PCA of NORMALIZED fingerprints")
    plt.scatter(norm_2d[:, 0], norm_2d[:, 1])
    plt.show()



