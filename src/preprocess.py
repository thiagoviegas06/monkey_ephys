from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_train_session_stats(sbp: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = sbp.mean(axis=0)
    std = sbp.std(axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_test_session_stats(
    sbp_masked: np.ndarray,
    mask: np.ndarray,
    fallback_mean: Optional[np.ndarray] = None,
    fallback_std: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    min_observed_per_channel: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sbp_masked.shape != mask.shape:
        raise ValueError(f"Shape mismatch: sbp={sbp_masked.shape}, mask={mask.shape}")

    n_channels = sbp_masked.shape[1]
    mean = np.zeros((n_channels,), dtype=np.float64)
    std = np.ones((n_channels,), dtype=np.float64)
    observed_counts = np.zeros((n_channels,), dtype=np.int64)

    observed_all = sbp_masked[~mask]
    global_mean = float(observed_all.mean()) if observed_all.size > 0 else 0.0
    global_std = float(observed_all.std()) if observed_all.size > 0 else 1.0
    if global_std < eps:
        global_std = 1.0

    for ch in range(n_channels):
        observed = sbp_masked[~mask[:, ch], ch]
        observed_counts[ch] = observed.size
        if observed.size >= min_observed_per_channel:
            ch_mean = float(observed.mean())
            ch_std = float(observed.std())
        else:
            if fallback_mean is not None:
                ch_mean = float(fallback_mean[ch])
            else:
                ch_mean = global_mean

            if fallback_std is not None:
                ch_std = float(fallback_std[ch])
            else:
                ch_std = global_std

        if not np.isfinite(ch_mean):
            ch_mean = global_mean
        if (not np.isfinite(ch_std)) or (ch_std < eps):
            if fallback_std is not None and np.isfinite(fallback_std[ch]) and fallback_std[ch] >= eps:
                ch_std = float(fallback_std[ch])
            else:
                ch_std = global_std

        mean[ch] = ch_mean
        std[ch] = max(ch_std, eps)

    return mean.astype(np.float32), std.astype(np.float32), observed_counts


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def apply_denorm(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x_norm * std + mean).astype(np.float32)
