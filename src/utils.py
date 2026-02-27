from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val_sessions(
    metadata: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    del seed  # deterministic chronological split

    if "split" in metadata.columns:
        base_df = metadata.loc[metadata["split"] == "train"].copy()
    else:
        base_df = metadata.copy()

    if base_df.empty:
        raise ValueError("No training sessions found in metadata.")

    # Chronological holdout by day is more realistic under session drift.
    base_df = base_df.sort_values(["day", "session_id"]).reset_index(drop=True)
    n_val = max(1, int(round(len(base_df) * val_fraction)))

    val_ids = base_df.iloc[-n_val:]["session_id"].astype(str).tolist()
    train_ids = base_df.iloc[:-n_val]["session_id"].astype(str).tolist()

    if not train_ids:
        raise ValueError("Validation fraction too large; no train sessions left.")

    return train_ids, val_ids


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
