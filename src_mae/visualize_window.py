from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src_mae.preprocessing import preprocess


def build_window_payload(
    data_path: str,
    session_id: str | None,
    window_size: int,
    k_windows: int,
    seed: int,
    p_mask_trial: float,
    sample_idx: int,
):
    samples_by_session = preprocess(
        data_path=data_path,
        window_size=window_size,
        K=k_windows,
        seed=seed,
        p_mask_trial=p_mask_trial,
        return_samples=True,
        target_session_id=session_id,
    )

    if not samples_by_session:
        raise RuntimeError("No preprocessed samples were generated")

    if session_id is None:
        selected_session_id = sorted(samples_by_session.keys())[0]
    else:
        if session_id not in samples_by_session:
            raise ValueError(f"No samples generated for session {session_id}")
        selected_session_id = session_id

    samples = samples_by_session[selected_session_id]
    if not samples:
        raise RuntimeError(f"Session {selected_session_id} returned 0 samples")

    if sample_idx < 0 or sample_idx >= len(samples):
        raise ValueError(f"sample_idx out of range [0, {len(samples) - 1}]")

    sample = samples[sample_idx]
    a = sample["a"]
    b = sample["b"]

    return {
        "session_id": selected_session_id,
        "trial_idx": int(sample["trial_idx"]),
        "w0": int(sample["w0"]),
        "trial_span": (int(a), int(b)),
        "sbp_true": sample["window_sbp"],
        "sbp_masked": sample["x"],
        "kin": sample["window_kin"],
        "mask_vec": sample["mask_vec"],
        "sample_idx": int(sample_idx),
        "num_samples": int(len(samples)),
    }


def plot_window(payload: dict, save_path: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc

    a, b = payload["trial_span"]
    y = payload["sbp_true"]
    x = payload["sbp_masked"]
    kin = payload["kin"]
    mask_vec = payload["mask_vec"]

    masked_channels = np.flatnonzero(mask_vec > 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(y.T, aspect="auto", interpolation="nearest", origin="lower")
    axes[0, 0].axvspan(a, b, color="white", alpha=0.18)
    axes[0, 0].set_title("Ground-truth SBP window")
    axes[0, 0].set_xlabel("Window bin")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(x.T, aspect="auto", interpolation="nearest", origin="lower")
    axes[0, 1].axvspan(a, b, color="white", alpha=0.18)
    axes[0, 1].set_title("Masked input SBP window")
    axes[0, 1].set_xlabel("Window bin")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    mean_all = y.mean(axis=1)
    axes[1, 0].plot(mean_all, label="mean SBP (all channels)", linewidth=2)
    if masked_channels.size > 0:
        mean_masked_true = y[:, masked_channels].mean(axis=1)
        mean_masked_in = x[:, masked_channels].mean(axis=1)
        axes[1, 0].plot(mean_masked_true, label="mean masked channels (true)", linewidth=1.7)
        axes[1, 0].plot(mean_masked_in, label="mean masked channels (input)", linewidth=1.7)
    axes[1, 0].axvspan(a, b, color="gray", alpha=0.2)
    axes[1, 0].set_title("SBP time profile in window")
    axes[1, 0].set_xlabel("Window bin")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(kin)
    axes[1, 1].axvspan(a, b, color="gray", alpha=0.2)
    axes[1, 1].set_title("Kinematics (4 channels)")
    axes[1, 1].set_xlabel("Window bin")
    axes[1, 1].set_ylabel("Value")

    mask_count = int(mask_vec.sum())
    fig.suptitle(
        (
            f"session={payload['session_id']}  trial={payload['trial_idx']}  "
            f"sample={payload['sample_idx']}/{payload['num_samples'] - 1}  "
            f"w0={payload['w0']}  trial_span=[{a},{b})  masked_channels={mask_count}"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one sample from preprocess(...) output")
    parser.add_argument("--data-path", type=str, default="kaggle_data", help="Path containing metadata.csv, train/, test/")
    parser.add_argument("--session-id", type=str, default=None, help="Train session id (e.g. S008). If omitted, first generated session is used.")
    parser.add_argument("--window-size", type=int, default=128, help="Window size")
    parser.add_argument("--k-windows", type=int, default=500, help="K argument passed to preprocess(...) per session")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--p-mask-trial", type=float, default=0.80, help="p_mask_trial passed to preprocess(...)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to visualize after preprocessing")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to save figure instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_window_payload(
        data_path=args.data_path,
        session_id=args.session_id,
        window_size=args.window_size,
        k_windows=args.k_windows,
        seed=args.seed,
        p_mask_trial=args.p_mask_trial,
        sample_idx=args.sample_idx,
    )
    plot_window(payload, save_path=args.save_path)


if __name__ == "__main__":
    main()
