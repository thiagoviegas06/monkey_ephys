# Neuroinformatics Project 2 (Phase 1) PyTorch Baseline

This repository contains a baseline for masked SBP reconstruction using a temporal 1D CNN.

## Baseline Summary

- Input per time bin: `SBP(96) + kinematics(4) = 100` features.
- Model: temporal Conv1D over windows of length `T` (default `201`), predicts SBP `(96,)` at the center bin.
- Loss: masked MSE on masked channels only (simulated during train/val).
- Normalization strategy (Option B): per-session SBP channel-wise normalization.
  - Train sessions: mean/std from full session SBP.
  - Test sessions: mean/std from observed entries only (`mask == False`), with safe fallback to train global stats.

## Project Structure

- `src/config.py`
- `src/io.py`
- `src/preprocess.py`
- `src/dataset.py`
- `src/model.py`
- `src/losses.py`
- `src/train.py`
- `src/eval.py`
- `src/predict.py`
- `src/utils.py`

## Data Assumptions

Default paths and names match the provided Kaggle bundle layout:

- `kaggle_data/metadata.csv`
- `kaggle_data/train/{session_id}_sbp.npy`
- `kaggle_data/train/{session_id}_kinematics.npy`
- `kaggle_data/train/{session_id}_trial_info.npz`
- `kaggle_data/test/{session_id}_sbp_masked.npy`
- `kaggle_data/test/{session_id}_kinematics.npy`
- `kaggle_data/test/{session_id}_mask.npy`
- `kaggle_data/test/{session_id}_trial_info.npz`
- `kaggle_data/sample_submission.csv`

`trial_info.npz` is expected to contain `start_bins` and `end_bins`.

## TODO Adaptation Points

- If trial boundary filenames/formats differ, update `_load_trial_ids()` / `_load_trial_ids_from_file()` in `src/io.py`.
- If test mask files are absent and zeros are guaranteed to represent masked entries, run predict with `--infer-mask-from-zero`.
- If submission schema differs from `sample_submission.csv`, adapt `build_submission()` in `src/predict.py`.

## Train

```bash
python -m src.train \
  --data-dir kaggle_data \
  --epochs 20 \
  --batch-size 64
```

Outputs best checkpoint at:

- `checkpoints/best.pt`

## Evaluate (Validation)

```bash
python -m src.eval \
  --data-dir kaggle_data \
  --checkpoint-path checkpoints/best.pt
```

## Predict + Submission

```bash
python -m src.predict \
  --data-dir kaggle_data \
  --checkpoint-path checkpoints/best.pt \
  --output-file submission.csv
```

This writes `submission.csv`, using `sample_submission.csv` row ordering when available.
