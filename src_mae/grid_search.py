import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BEST_LINE_RE = re.compile(r"Best model: epoch\s+(\d+)\s+with val_loss=([0-9]*\.?[0-9]+)")


def parse_list(raw, cast_fn):
    if raw is None or raw.strip() == "":
        return []
    return [cast_fn(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search runner for src_mae/train.py")

    parser.add_argument("--output-dir", type=str, default="grid_search_runs", help="Directory for run artifacts")
    parser.add_argument("--max-runs", type=int, default=5, help="Optional cap on number of runs (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only")

    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--learning-rates", type=str, default="1e-4,7e-5", help="Comma-separated learning rates")
    parser.add_argument("--weight-decays", type=str, default="1e-5", help="Comma-separated weight decay values")
    parser.add_argument("--base-channels", type=str, default="64,96", help="Comma-separated base channel widths")
    parser.add_argument("--batch-sizes", type=str, default="16", help="Comma-separated batch sizes")
    parser.add_argument("--num-epochs", type=str, default="20", help="Comma-separated epoch counts")

    parser.add_argument("--split-modes", type=str, default="test_proximity_session", help="Comma-separated split modes")
    parser.add_argument("--val-ratios", type=str, default="0.2", help="Comma-separated val ratios")
    parser.add_argument("--temps", type=str, default="40,80", help="Comma-separated proximity temperatures (days)")
    parser.add_argument("--randomness", type=str, default="0.2,0.35", help="Comma-separated proximity randomness")

    parser.add_argument("--early-patience", type=str, default="6", help="Comma-separated early stopping patience")
    parser.add_argument("--early-min-delta", type=str, default="1e-4", help="Comma-separated early stopping min delta")

    parser.add_argument("--model-name", type=str, default="unet", help="Model name (unet/simple_cnn/resnet)")
    parser.add_argument("--windows-dir", type=str, default="kaggle_data/masked_windows", help="Windows directory")
    parser.add_argument("--data-path", type=str, default="kaggle_data", help="Data root path")
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint cadence")

    return parser.parse_args()


def build_grid(args):
    seeds = parse_list(args.seeds, int)
    learning_rates = parse_list(args.learning_rates, float)
    weight_decays = parse_list(args.weight_decays, float)
    base_channels = parse_list(args.base_channels, int)
    batch_sizes = parse_list(args.batch_sizes, int)
    num_epochs = parse_list(args.num_epochs, int)

    split_modes = [x.strip() for x in args.split_modes.split(",") if x.strip()]
    val_ratios = parse_list(args.val_ratios, float)
    temps = parse_list(args.temps, float)
    randomness = parse_list(args.randomness, float)

    early_patience = parse_list(args.early_patience, int)
    early_min_delta = parse_list(args.early_min_delta, float)

    if not all([seeds, learning_rates, weight_decays, base_channels, batch_sizes, num_epochs, split_modes, val_ratios, temps, randomness, early_patience, early_min_delta]):
        raise ValueError("One of the search dimensions is empty. Check CLI arguments.")

    combos = list(
        itertools.product(
            seeds,
            learning_rates,
            weight_decays,
            base_channels,
            batch_sizes,
            num_epochs,
            split_modes,
            val_ratios,
            temps,
            randomness,
            early_patience,
            early_min_delta,
        )
    )

    if args.max_runs and args.max_runs > 0:
        combos = combos[: args.max_runs]

    return combos


def make_run_config(args, combo, run_dir):
    (
        seed,
        learning_rate,
        weight_decay,
        base_channels,
        batch_size,
        num_epochs,
        split_mode,
        val_ratio,
        temp,
        rand,
        early_patience,
        early_min_delta,
    ) = combo

    cfg = {
        "preprocess": False,
        "data_path": args.data_path,
        "windows_dir": args.windows_dir,
        "model_name": args.model_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "base_channels": base_channels,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "split_mode": split_mode,
        "val_ratio": val_ratio,
        "proximity_temperature_days": temp,
        "proximity_randomness": rand,
        "early_stopping_patience": early_patience,
        "early_stopping_min_delta": early_min_delta,
        "save_every": args.save_every,
        "checkpoint_dir": str(run_dir / "checkpoints"),
    }
    return cfg


def launch_training(cfg, run_dir):
    code = (
        "import json, os; "
        "import train as t; "
        "cfg=json.loads(os.environ['GRID_CFG']); "
        "[setattr(t.Config, k, v) for k, v in cfg.items()]; "
        "t.main()"
    )

    env = os.environ.copy()
    env["GRID_CFG"] = json.dumps(cfg)

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
        env=env,
    )

    log_path = run_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\n[STDERR]\n")
            f.write(proc.stderr)

    best_epoch = None
    best_val = None
    match = BEST_LINE_RE.search(proc.stdout)
    if match:
        best_epoch = int(match.group(1))
        best_val = float(match.group(2))

    return {
        "return_code": proc.returncode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "log_path": str(log_path),
    }


def save_results_csv(results, out_csv):
    fields = [
        "run_name",
        "return_code",
        "best_epoch",
        "best_val_loss",
        "seed",
        "learning_rate",
        "weight_decay",
        "base_channels",
        "batch_size",
        "num_epochs",
        "split_mode",
        "val_ratio",
        "proximity_temperature_days",
        "proximity_randomness",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "checkpoint_dir",
        "log_path",
    ]

    sorted_results = sorted(
        results,
        key=lambda r: (float("inf") if r["best_val_loss"] is None else r["best_val_loss"]),
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted_results:
            writer.writerow(row)


def main():
    args = parse_args()
    combos = build_grid(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = Path(args.output_dir) / f"run_{timestamp}"
    root_out.mkdir(parents=True, exist_ok=True)

    print(f"Planned runs: {len(combos)}")
    print(f"Output directory: {root_out}")

    if args.dry_run:
        for i, combo in enumerate(combos, 1):
            print(f"[{i:03d}] {combo}")
        return

    results = []

    for run_idx, combo in enumerate(combos, 1):
        run_name = f"run_{run_idx:03d}"
        run_dir = root_out / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = make_run_config(args, combo, run_dir)

        print("=" * 80)
        print(f"[{run_idx}/{len(combos)}] Starting {run_name}")
        print(
            f"seed={cfg['seed']} lr={cfg['learning_rate']} wd={cfg['weight_decay']} "
            f"ch={cfg['base_channels']} bs={cfg['batch_size']} ep={cfg['num_epochs']} "
            f"split={cfg['split_mode']} vr={cfg['val_ratio']} temp={cfg['proximity_temperature_days']} rand={cfg['proximity_randomness']}"
        )

        outcome = launch_training(cfg, run_dir)

        merged = {
            "run_name": run_name,
            **outcome,
            **cfg,
        }
        results.append(merged)

        if outcome["return_code"] != 0:
            print(f"{run_name} FAILED (code={outcome['return_code']}) -> {outcome['log_path']}")
        else:
            print(
                f"{run_name} done: best_epoch={outcome['best_epoch']} "
                f"best_val_loss={outcome['best_val_loss']}"
            )

    results_csv = root_out / "results.csv"
    save_results_csv(results, results_csv)

    print("=" * 80)
    print(f"Grid search complete. Results: {results_csv}")

    best_ok = [r for r in results if r["return_code"] == 0 and r["best_val_loss"] is not None]
    if best_ok:
        best = min(best_ok, key=lambda r: r["best_val_loss"])
        print(
            "Best run: "
            f"{best['run_name']} | best_val_loss={best['best_val_loss']:.6f} | "
            f"seed={best['seed']} lr={best['learning_rate']} temp={best['proximity_temperature_days']} rand={best['proximity_randomness']}"
        )


if __name__ == "__main__":
    main()
