# evaluation/metrics.py

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalized by the range (max - min) of the true values in the
    evaluated set -- makes the error comparable across reaches/periods with
    very different CIR/volume magnitudes."""
    value_range = y_true.max() - y_true.min()
    if value_range < 1e-9:
        return float("nan")
    return rmse(y_true, y_pred) / value_range


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error. CIR crosses zero constantly (it's a
    signed rate), so plain MAPE is unstable wherever the true value is near
    zero (the denominator blows up). The denominator is floored at 1% of the
    series' own standard deviation to avoid divide-by-near-zero explosions --
    but MAPE is still a fundamentally awkward fit for a zero-crossing
    quantity like CIR, and should be read as a rough guide rather than a
    precise one."""
    eps = 0.01 * np.std(y_true)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def back_calculated_volume_errors(reach_ids, window_start_dates, predicted_cir_windows, volume_by_reach):
    """Integrate each predicted CIR window forward from its real volume
    anchor (the actual recorded volume the day before the window starts) and
    compare to the real recorded volume over the same window. Windows may
    have different lengths (e.g. a truncated recursive forecast), so results
    are concatenated into flat 1D arrays rather than stacked into a 2D array.
    Skips any window missing its anchor or any real volume value.
    """
    import pandas as pd

    real_all, pred_all = [], []
    for reach_id, start, pred_cir in zip(reach_ids, window_start_dates, predicted_cir_windows):
        anchor_date = pd.Timestamp(start) - pd.Timedelta(days=1)
        anchor = volume_by_reach[reach_id].get(anchor_date, np.nan)
        if np.isnan(anchor):
            continue
        pred_cir = np.asarray(pred_cir)
        dates = pd.date_range(start, periods=len(pred_cir), freq="D")
        real = volume_by_reach[reach_id].reindex(dates).to_numpy()
        if np.isnan(real).any():
            continue
        pred = anchor + np.cumsum(pred_cir)
        real_all.append(real)
        pred_all.append(pred)
    if not real_all:
        return np.array([]), np.array([])
    return np.concatenate(real_all), np.concatenate(pred_all)
