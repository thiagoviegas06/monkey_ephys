
import random
import csv
from train import train

def random_search(num_trials=50, seed=42):
    random.seed(seed)
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
    }

    results_file = "hyperparam_results.csv"
    with open(results_file, "w", newline="") as csvfile:
        fieldnames = list(param_space.keys()) + ["val_nmse", "checkpoint_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for trial in range(num_trials):
            params = {k: random.choice(v) for k, v in param_space.items()}
            checkpoint_name = (
                f"tcn_layers{params['num_layers']}_hidden{params['hidden_dim']}_kernel{params['kernel_size']}"
                f"_dropout{params['dropout']}_dilationbase{params['dilation_base']}_dilationcap{params['dilation_cap']}"
                f"_gngroups{params['gn_groups']}_batch{params['batch_size']}_lr{params['lr']}_wd{params['weight_decay']}"
                f"_maskmin{params['mask_min']}_maskmax{params['mask_max']}_valfrac{params['val_fraction']}.pt"
            )
            args = [
                "--model", "tcn",
                "--num-layers", str(params["num_layers"]),
                "--hidden-dim", str(params["hidden_dim"]),
                "--kernel-size", str(params["kernel_size"]),
                "--dropout", str(params["dropout"]),
                "--dilation-base", str(params["dilation_base"]),
                "--dilation-cap", str(params["dilation_cap"]),
                "--gn-groups", str(params["gn_groups"]),
                "--batch-size", str(params["batch_size"]),
                "--lr", str(params["lr"]),
                "--weight-decay", str(params["weight_decay"]),
                "--mask-min", str(params["mask_min"]),
                "--mask-max", str(params["mask_max"]),
                "--val-fraction", str(params["val_fraction"]),
                "--checkpoint-name", checkpoint_name,
            ]
            print(f"Trial {trial+1}/{num_trials}: Training with args: {' '.join(args)}")
            result = train(train.parse_args(args))
            # Assume train() returns val_nmse, update if needed
            val_nmse = None
            if isinstance(result, dict) and "best_val_nmse" in result:
                val_nmse = result["best_val_nmse"]
            writer.writerow({**params, "val_nmse": val_nmse, "checkpoint_name": checkpoint_name})

if __name__ == "__main__":
    random_search(num_trials=50)



