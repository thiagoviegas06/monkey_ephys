"""
Kaggle custom metric for Phase 2 — Position Decoding competition.

Computes R² (coefficient of determination) between predicted and true
position values, averaged across all (session, position_channel) groups.
Higher is better. R² = 0.0 means predicting the channel mean.

The solution DataFrame has columns:
  sample_id, session_id, time_bin, true_index_pos, true_mrp_pos, Usage

The submission DataFrame has columns:
  sample_id, index_pos, mrp_pos
"""

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    R² (coefficient of determination) for position decoding.

    For each (session, position_channel) group, computes:
        R² = 1 - SS_res / SS_tot
    where SS_res = sum((pred - true)²) and SS_tot = sum((true - mean(true))²).

    Final score is the mean R² across all groups (24 sessions × 2 channels = 48).

    R² = 1.0: perfect prediction
    R² = 0.0: equivalent to predicting the channel mean per session
    R² < 0.0: worse than predicting the mean

    Parameters
    ----------
    solution : pd.DataFrame
        Ground truth with columns [sample_id, session_id, time_bin,
        true_index_pos, true_mrp_pos, Usage].
    submission : pd.DataFrame
        Participant predictions with columns [sample_id,
        index_pos, mrp_pos].
    row_id_column_name : str
        Name of the row ID column used for alignment (typically 'sample_id').

    Returns
    -------
    float
        The mean R² score (higher is better).

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "sample_id"
    >>> sol = pd.DataFrame({
    ...     "sample_id": [0, 1, 2, 3],
    ...     "session_id": ["D1", "D1", "D1", "D1"],
    ...     "time_bin": [0, 1, 2, 3],
    ...     "true_index_pos": [0.1, 0.2, 0.3, 0.4],
    ...     "true_mrp_pos": [0.5, 0.6, 0.7, 0.8],
    ...     "Usage": ["Public", "Public", "Public", "Public"]
    ... })
    >>> sub = pd.DataFrame({
    ...     "sample_id": [0, 1, 2, 3],
    ...     "index_pos": [0.1, 0.2, 0.3, 0.4],
    ...     "mrp_pos": [0.5, 0.6, 0.7, 0.8],
    ... })
    >>> score(sol.copy(), sub.copy(), row_id_column_name)
    1.0
    """
    kin_pred_cols = ['index_pos', 'mrp_pos']
    kin_true_cols = ['true_index_pos', 'true_mrp_pos']

    # Validate columns
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission missing column: {row_id_column_name}")

    for col in kin_pred_cols:
        if col not in submission.columns:
            raise ParticipantVisibleError(
                f"Submission must have a '{col}' column. "
                f"Found columns: {list(submission.columns)}"
            )
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(
                f"Column '{col}' must contain numeric values."
            )
        if submission[col].isna().any():
            raise ParticipantVisibleError(
                f"Column '{col}' contains NaN values. All entries must be finite numbers."
            )
        if submission[col].abs().eq(float("inf")).any():
            raise ParticipantVisibleError(
                f"Column '{col}' contains infinite values. All entries must be finite numbers."
            )

    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Submission has {len(submission)} rows but expected {len(solution)} rows."
        )

    # Merge on row ID to align
    merge_cols = [row_id_column_name] + kin_pred_cols
    merged = solution.merge(submission[merge_cols], on=row_id_column_name, how="left")

    for col in kin_pred_cols:
        if merged[col].isna().any():
            n_missing = int(merged[col].isna().sum())
            raise ParticipantVisibleError(
                f"Submission is missing predictions for {n_missing} sample IDs "
                f"in column '{col}'."
            )

    # Compute R² per (session, position channel)
    r2_values = []
    min_ss_tot = 1e-10  # guard against zero-variance

    for session_id in merged['session_id'].unique():
        sess_mask = merged['session_id'] == session_id
        sess_data = merged[sess_mask]

        for true_col, pred_col in zip(kin_true_cols, kin_pred_cols):
            true_vals = sess_data[true_col].values.astype(np.float64)
            pred_vals = sess_data[pred_col].values.astype(np.float64)

            ss_res = np.sum((pred_vals - true_vals) ** 2)
            ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
            ss_tot = max(ss_tot, min_ss_tot)

            r2 = 1.0 - ss_res / ss_tot
            r2_values.append(r2)

    mean_r2 = float(np.mean(r2_values))
    return mean_r2
