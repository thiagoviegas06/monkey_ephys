from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.io import SessionData
from src.preprocess import apply_norm, compute_train_session_stats


@dataclass
class _PreparedSession:
    session_id: str
    sbp_norm: np.ndarray
    kinematics: np.ndarray
    trial_ids: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def _valid_centers_for_trial_ids(trial_ids: np.ndarray, half_window: int) -> np.ndarray:
    n_bins = len(trial_ids)
    if n_bins == 0:
        return np.empty((0,), dtype=np.int64)

    boundaries = np.where(np.diff(trial_ids) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n_bins]])

    centers: List[np.ndarray] = []
    for start, end in zip(starts, ends):
        trial_id = trial_ids[start]
        if trial_id < 0:
            continue

        lo = start + half_window
        hi = end - half_window
        if lo < hi:
            centers.append(np.arange(lo, hi, dtype=np.int64))

    if not centers:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(centers)


class SBPWindowDataset(Dataset):
    def __init__(
        self,
        sessions: Sequence[SessionData],
        window_size: int = 201,
        split: str = "train",
        seed: int = 42,
        mask_channels: int = 30,
        mask_channels_min: int | None = None,
        mask_channels_max: int | None = None,
        deterministic_masks: bool = False,
        max_centers_per_session: int | None = None,
    ) -> None:
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")

        self.window_size = window_size
        self.half_window = window_size // 2
        self.split = split
        self.seed = seed
        self.deterministic_masks = deterministic_masks
        self.mask_channels = int(mask_channels)
        self.mask_channels_min = int(mask_channels if mask_channels_min is None else mask_channels_min)
        self.mask_channels_max = int(mask_channels if mask_channels_max is None else mask_channels_max)

        if not (1 <= self.mask_channels_min <= 96):
            raise ValueError(f"mask_channels_min must be in [1,96], got {self.mask_channels_min}")
        if not (1 <= self.mask_channels_max <= 96):
            raise ValueError(f"mask_channels_max must be in [1,96], got {self.mask_channels_max}")
        if self.mask_channels_min > self.mask_channels_max:
            raise ValueError(
                f"mask_channels_min must be <= mask_channels_max, got {self.mask_channels_min} > {self.mask_channels_max}"
            )
        if not (1 <= self.mask_channels <= 96):
            raise ValueError(f"mask_channels must be in [1,96], got {self.mask_channels}")

        rng = np.random.default_rng(seed)

        self.sessions: List[_PreparedSession] = []
        self.index_map: List[Tuple[int, int]] = []

        for session in sessions:
            mean, std = compute_train_session_stats(session.sbp)
            sbp_norm = apply_norm(session.sbp, mean, std)

            prepared = _PreparedSession(
                session_id=session.session_id,
                sbp_norm=sbp_norm,
                kinematics=session.kinematics.astype(np.float32),
                trial_ids=session.trial_ids.astype(np.int64),
                mean=mean,
                std=std,
            )
            self.sessions.append(prepared)

            centers = _valid_centers_for_trial_ids(prepared.trial_ids, self.half_window)
            if max_centers_per_session is not None and len(centers) > max_centers_per_session:
                centers = np.sort(rng.choice(centers, size=max_centers_per_session, replace=False))

            session_index = len(self.sessions) - 1
            self.index_map.extend((session_index, int(c)) for c in centers)

        if not self.index_map:
            raise ValueError("No valid window centers found. Check trial boundaries and window size.")

    def __len__(self) -> int:
        return len(self.index_map)

    def _sample_channel_mask(self, idx: int) -> np.ndarray:
        if self.deterministic_masks or self.split != "train":
            local_rng = np.random.default_rng(self.seed + idx)
            k = self.mask_channels
        else:
            local_rng = np.random.default_rng()
            k = int(local_rng.integers(self.mask_channels_min, self.mask_channels_max + 1))

        channels = local_rng.choice(96, size=k, replace=False)
        mask = np.zeros((96,), dtype=np.float32)
        mask[channels] = 1.0
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        session_idx, center = self.index_map[idx]
        session = self.sessions[session_idx]

        start = center - self.half_window
        end = center + self.half_window + 1

        x_kin = session.kinematics[start:end].astype(np.float32, copy=True)
        x_sbp = session.sbp_norm[start:end].astype(np.float32, copy=True)
        y_seq = session.sbp_norm[start:end].astype(np.float32, copy=True)

        mask = self._sample_channel_mask(idx)                 # (96,) 1=masked
        mask_bool = mask.astype(bool)

        # zero-out masked channels in the input
        x_sbp[:, mask_bool] = 0.0

        # observed-indicator feature: (T,96) where 1=observed, 0=masked
        obs_mask = (1.0 - mask)[None, :].repeat(x_sbp.shape[0], axis=0).astype(np.float32)

        return {
            "x_sbp": torch.from_numpy(x_sbp),
            "x_kin": torch.from_numpy(x_kin),
            "obs_mask": torch.from_numpy(obs_mask),
            "y_seq": torch.from_numpy(y_seq),
            "mask": torch.from_numpy(mask),
            "session_id": session.session_id,
        }
