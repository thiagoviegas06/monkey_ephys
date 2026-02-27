from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SessionData:
    session_id: str
    split: str
    sbp: np.ndarray
    kinematics: np.ndarray
    trial_ids: np.ndarray
    mask: Optional[np.ndarray] = None


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)
    required = {"session_id", "day"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"metadata.csv must include {sorted(required)}. Found: {list(df.columns)}"
        )
    return df


def list_session_ids(data_dir: str | Path, split: str) -> List[str]:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    suffix = "_sbp.npy" if split == "train" else "_sbp_masked.npy"
    session_ids = sorted(p.name.replace(suffix, "") for p in split_dir.glob(f"*{suffix}"))
    if not session_ids:
        raise FileNotFoundError(
            f"No sessions found in {split_dir} with suffix {suffix}. "
            "TODO: adapt file discovery if names differ."
        )
    return session_ids


def _load_trial_ids_from_file(path: Path, n_bins: int) -> np.ndarray:
    if path.suffix == ".npz":
        trial_info = np.load(path)
        keys = set(trial_info.files)
        if {"start_bins", "end_bins"}.issubset(keys):
            starts = trial_info["start_bins"].astype(np.int64)
            ends = trial_info["end_bins"].astype(np.int64)
            trial_ids = np.full((n_bins,), -1, dtype=np.int64)
            for idx, (start, end) in enumerate(zip(starts, ends)):
                if start < 0 or end > n_bins or end <= start:
                    raise ValueError(
                        f"Invalid trial boundaries in {path}: start={start}, end={end}, n_bins={n_bins}"
                    )
                trial_ids[start:end] = idx
            return trial_ids

        if "trial_ids" in keys:
            trial_ids = trial_info["trial_ids"].astype(np.int64)
            if len(trial_ids) != n_bins:
                raise ValueError(f"trial_ids length mismatch in {path}: {len(trial_ids)} vs {n_bins}")
            return trial_ids

        raise ValueError(
            f"Unsupported .npz trial format in {path}. Keys={sorted(keys)}. "
            "TODO: add parser for this trial format."
        )

    if path.suffix == ".npy":
        trial_ids = np.load(path).astype(np.int64)
        if trial_ids.shape[0] != n_bins:
            raise ValueError(f"trial_ids length mismatch in {path}: {len(trial_ids)} vs {n_bins}")
        return trial_ids

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "trial_id" in df.columns:
            trial_ids = df["trial_id"].to_numpy(dtype=np.int64)
            if len(trial_ids) != n_bins:
                raise ValueError(f"trial_id length mismatch in {path}: {len(trial_ids)} vs {n_bins}")
            return trial_ids

        if {"start_bin", "end_bin"}.issubset(df.columns):
            trial_ids = np.full((n_bins,), -1, dtype=np.int64)
            for idx, row in df.reset_index(drop=True).iterrows():
                start = int(row["start_bin"])
                end = int(row["end_bin"])
                if start < 0 or end > n_bins or end <= start:
                    raise ValueError(
                        f"Invalid trial boundaries in {path}: start={start}, end={end}, n_bins={n_bins}"
                    )
                trial_ids[start:end] = idx
            return trial_ids

        raise ValueError(
            f"Unsupported CSV trial format in {path}. Need column 'trial_id' or ['start_bin','end_bin']. "
            "TODO: add parser for your trial CSV format."
        )

    raise ValueError(
        f"Unsupported trial file type: {path}. "
        "TODO: add parser for this trial boundary format."
    )


def _load_trial_ids(split_dir: Path, session_id: str, n_bins: int) -> np.ndarray:
    candidates = [
        split_dir / f"{session_id}_trial_info.npz",
        split_dir / f"{session_id}_trial_ids.npy",
        split_dir / f"{session_id}_trials.csv",
        split_dir / session_id / "trial_info.npz",
        split_dir / session_id / "trial_ids.npy",
        split_dir / session_id / "trials.csv",
    ]

    found = [p for p in candidates if p.exists()]
    if not found:
        raise FileNotFoundError(
            f"No trial boundary file found for {session_id} in {split_dir}. "
            "Tried *_trial_info.npz, *_trial_ids.npy, *_trials.csv and nested session-dir variants. "
            "TODO: update _load_trial_ids() with your actual filename pattern."
        )

    return _load_trial_ids_from_file(found[0], n_bins=n_bins)


def _load_sbp_and_kin(split_dir: Path, session_id: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    if split == "train":
        sbp_candidates = [
            split_dir / f"{session_id}_sbp.npy",
            split_dir / session_id / "sbp.npy",
        ]
    else:
        sbp_candidates = [
            split_dir / f"{session_id}_sbp_masked.npy",
            split_dir / session_id / "sbp_masked.npy",
            split_dir / f"{session_id}_sbp.npy",
            split_dir / session_id / "sbp.npy",
        ]

    kin_candidates = [
        split_dir / f"{session_id}_kinematics.npy",
        split_dir / session_id / "kinematics.npy",
    ]

    sbp_path = next((p for p in sbp_candidates if p.exists()), None)
    kin_path = next((p for p in kin_candidates if p.exists()), None)

    if sbp_path is None:
        raise FileNotFoundError(
            f"Missing SBP file for {session_id} in {split_dir}. "
            "TODO: update file pattern in _load_sbp_and_kin()."
        )
    if kin_path is None:
        raise FileNotFoundError(
            f"Missing kinematics file for {session_id} in {split_dir}. "
            "TODO: update file pattern in _load_sbp_and_kin()."
        )

    sbp = np.load(sbp_path).astype(np.float32)
    kin = np.load(kin_path).astype(np.float32)

    if sbp.ndim != 2 or sbp.shape[1] != 96:
        raise ValueError(f"Expected SBP shape (N,96), got {sbp.shape} for {session_id}")
    if kin.ndim != 2 or kin.shape[1] != 4:
        raise ValueError(f"Expected kinematics shape (N,4), got {kin.shape} for {session_id}")
    if sbp.shape[0] != kin.shape[0]:
        raise ValueError(
            f"Length mismatch for {session_id}: sbp has {sbp.shape[0]} bins, kinematics has {kin.shape[0]} bins"
        )

    return sbp, kin


def _load_test_mask(
    split_dir: Path,
    session_id: str,
    sbp_like: np.ndarray,
    infer_mask_from_zero: bool,
) -> np.ndarray:
    candidates = [
        split_dir / f"{session_id}_mask.npy",
        split_dir / session_id / "mask.npy",
    ]
    mask_path = next((p for p in candidates if p.exists()), None)

    if mask_path is not None:
        mask = np.load(mask_path)
        if mask.shape != sbp_like.shape:
            raise ValueError(
                f"Mask shape mismatch for {session_id}: mask={mask.shape}, sbp={sbp_like.shape}"
            )
        return mask.astype(bool)

    if infer_mask_from_zero:
        return sbp_like == 0

    raise FileNotFoundError(
        f"Mask file missing for test session {session_id}. "
        "Set infer_mask_from_zero=True only if zeros are guaranteed to indicate masked entries."
    )


def load_session(
    data_dir: str | Path,
    session_id: str,
    split: str,
    infer_mask_from_zero: bool = False,
) -> SessionData:
    split_dir = Path(data_dir) / split
    sbp, kin = _load_sbp_and_kin(split_dir, session_id, split)
    trial_ids = _load_trial_ids(split_dir, session_id, n_bins=sbp.shape[0])

    mask: Optional[np.ndarray] = None
    if split == "test":
        mask = _load_test_mask(
            split_dir=split_dir,
            session_id=session_id,
            sbp_like=sbp,
            infer_mask_from_zero=infer_mask_from_zero,
        )

    return SessionData(session_id=session_id, split=split, sbp=sbp, kinematics=kin, trial_ids=trial_ids, mask=mask)


def load_sessions(
    data_dir: str | Path,
    session_ids: Iterable[str],
    split: str,
    infer_mask_from_zero: bool = False,
) -> List[SessionData]:
    return [
        load_session(
            data_dir=data_dir,
            session_id=session_id,
            split=split,
            infer_mask_from_zero=infer_mask_from_zero,
        )
        for session_id in session_ids
    ]


def session_ids_from_metadata(metadata: pd.DataFrame, split: str) -> List[str]:
    if "split" in metadata.columns:
        out = metadata.loc[metadata["split"] == split, "session_id"].astype(str).tolist()
        if out:
            return sorted(out)
    raise ValueError(
        f"Could not extract session IDs for split='{split}' from metadata. "
        "Need a 'split' column or use list_session_ids()."
    )
