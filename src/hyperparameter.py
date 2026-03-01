import random
import csv
import os
import time
from types import SimpleNamespace

from train import train

def sample_params(rng: random.Random, param_space: dict) -> dict:
    params = {k: rng.choice(v) for k, v in param_space.items()}

    if params["mask_min"] > params["mask_max"]:
        params["mask_min"], params["mask_max"] = params["mask_max"], params["mask_min"]

    if params["kernel_size"] % 2 == 0:
        params["kernel_size"] += 1

    return params


def random_search(num_trials=50, seed=42, results_file="hyperparam_results.csv"):
    rng = random.Random(seed)

    param_space = {
        "num_layers": [4, 8, 12, 16],
        "hidden_dim": [64, 128, 192, 256, 384],
        "kernel_size": [3, 5, 7, 9, 13],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "dilation_base": [2, 3, 4],
        "dilation_cap": [32, 64, 128],
        "gn_groups": [4, 8, 16],
        "batch_size": [32, 64, 128, 256, 512],
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
        "mask_min": [5, 10, 20],
        "mask_max": [40, 60, 80],
        "val_fraction": [0.1, 0.2, 0.3, 0.4],
        "epochs": [20, 40, 60],
    }

    fieldnames = (
        ["trial", "seed", "status", "error", "wall_sec"]
        + list(param_space.keys())
        + ["val_nmse", "checkpoint_name"]
    )

    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for trial in range(1, num_trials + 1):
            params = sample_params(rng, param_space)
            checkpoint_name = f"trial{trial:03d}.pt"  # FIXED

            args = SimpleNamespace(
                model="tcn",
                num_layers=params["num_layers"],
                hidden_dim=params["hidden_dim"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
                dilation_base=params["dilation_base"],
                dilation_cap=params["dilation_cap"],
                gn_groups=params["gn_groups"],
                batch_size=params["batch_size"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                mask_min=params["mask_min"],
                mask_max=params["mask_max"],
                mask_channels=30,  # RECOMMENDED: fixed val masking for fair comparison
                val_fraction=params["val_fraction"],
                epochs=params["epochs"],
                checkpoint_name=checkpoint_name,
                seed=seed,
                device="auto",
                window_size=201,
                data_dir="kaggle_data",
                metadata_file="metadata.csv",
                checkpoint_dir="checkpoints_randomsearch",
                num_workers=0,
                use_cosine_scheduler=True,
            )

            print(f"[trial {trial}/{num_trials}] {checkpoint_name}")
            t0 = time.time()

            row = {
                "trial": trial,
                "seed": seed,
                **params,
                "checkpoint_name": checkpoint_name,
                "val_nmse": None,
                "status": "ok",
                "error": "",
                "wall_sec": None,
            }

            try:
                result = train(args)  # FIXED: actually run training

                if isinstance(result, dict):
                    row["val_nmse"] = result.get("best_val_nmse", result.get("val_nmse"))
                elif isinstance(result, (float, int)):
                    row["val_nmse"] = float(result)

            except Exception as e:
                row["status"] = "fail"
                row["error"] = repr(e)

            row["wall_sec"] = round(time.time() - t0, 2)
            writer.writerow(row)
            csvfile.flush()

if __name__ == "__main__":
    random_search(num_trials=50, seed=42)