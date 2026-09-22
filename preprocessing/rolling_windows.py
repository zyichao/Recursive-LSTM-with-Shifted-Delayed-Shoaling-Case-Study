# preprocessing/rolling_windows.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def create_rolling_windows(
    input_df: pd.DataFrame,
    output_series: pd.Series,
    input_window_size: int,
    output_window_size: int,
):
    """Slide a window over aligned input/output series. Input and output
    window lengths are independent -- they don't need to match.

    For position i:
        X = input_df.iloc[i : i+input_window_size]
            (past input_window_size days: CIR_prev + exo features)
        y = output_series.iloc[i+input_window_size : i+input_window_size+output_window_size]
            (next output_window_size days of CIR -- the forecast target)

    input_df and output_series must share the same index. A window (spanning
    both the input and output halves) is only kept if every consecutive pair
    of dates in it is exactly 1 day apart -- this skips any window straddling
    an actual missing calendar date, and also guarantees the output window
    starts immediately (the very next day) after the input window ends.
    """
    X, y, t_input_start, t_output_start = [], [], [], []
    idx = input_df.index
    n = len(idx)
    total = input_window_size + output_window_size
    for i in range(n - total + 1):
        window_idx = idx[i: i + total]
        date_diff = window_idx[1:] - window_idx[:-1]
        if not (date_diff == pd.Timedelta(days=1)).all():
            continue
        X.append(input_df.iloc[i: i + input_window_size].to_numpy())
        y.append(output_series.iloc[i + input_window_size: i + total].to_numpy())
        t_input_start.append(window_idx[0])
        t_output_start.append(window_idx[input_window_size])
    return X, y, t_input_start, t_output_start


def build_windowed_dataset(
    reach_ids,
    cir_by_reach: dict,
    exo_shifted: pd.DataFrame,
    input_window_size: int,
    output_window_size: int,
):
    """Build rolling-window samples per reach ("previous CIR" + shifted
    exogenous discharge, joined by date) and concatenate across reaches.

    Returns
    -------
    X_all : np.ndarray, shape (n_samples, input_window_size, n_features)
    y_all : np.ndarray, shape (n_samples, output_window_size)
    reach_id_all : np.ndarray, shape (n_samples,)
    t_input_all, t_output_all : pd.DatetimeIndex, shape (n_samples,)
    feature_columns : list[str]
    """
    feature_columns = ["CIR_prev"] + list(exo_shifted.columns)

    X_list, y_list, reach_id_list, t_input_list, t_output_list = [], [], [], [], []

    for reach_id in reach_ids:
        cir_series = cir_by_reach[reach_id]

        # "previous CIR" + shifted exogenous discharge, joined by date; only
        # dates present in both (and with a fully-populated exo row) are usable.
        reach_input = pd.DataFrame({"CIR_prev": cir_series}).join(exo_shifted, how="inner").dropna()
        reach_output = cir_series.reindex(reach_input.index)

        X, y, t_in, t_out = create_rolling_windows(
            reach_input, reach_output, input_window_size, output_window_size
        )
        print(f"{reach_id}: {len(reach_input)} aligned days -> {len(X)} window samples")

        X_list.extend(X)
        y_list.extend(y)
        reach_id_list.extend([reach_id] * len(X))
        t_input_list.extend(t_in)
        t_output_list.extend(t_out)

    X_all = np.stack(X_list)
    y_all = np.stack(y_list)
    reach_id_all = np.array(reach_id_list)
    t_input_all = pd.DatetimeIndex(t_input_list)
    t_output_all = pd.DatetimeIndex(t_output_list)

    return X_all, y_all, reach_id_all, t_input_all, t_output_all, feature_columns


def compute_zero_crossing_flags(y_all: np.ndarray) -> np.ndarray:
    """Per-sample zero-crossing flag (Zeng et al. 2026, Eq. 36): 1.0 if the
    target CIR window contains both a positive and a negative value -- i.e.
    the sedimentation trend reverses (a local peak or valley) somewhere
    within the forecast window -- else 0.0. Used to weight these
    operationally important transition windows more heavily during training
    (see training/losses.py's WeightedZeroCrossingMSELoss).
    """
    return ((y_all.min(axis=1) < 0) & (y_all.max(axis=1) > 0)).astype(np.float32)


def chronological_train_test_split(X_all, y_all, reach_id_all, t_input_all, t_output_all, test_start_date):
    """A sample is a TEST sample iff its OUTPUT window (the forecast target)
    starts on or after `test_start_date`. Its INPUT window is allowed to
    reach back before that date -- forecasting into the test period needs
    the preceding history, which is standard practice for time-series splits.
    """
    test_start_date = pd.Timestamp(test_start_date)
    is_test = t_output_all >= test_start_date
    zc_all = compute_zero_crossing_flags(y_all)

    return {
        "X_train": X_all[~is_test], "y_train": y_all[~is_test],
        "X_test": X_all[is_test], "y_test": y_all[is_test],
        "reach_train": reach_id_all[~is_test], "reach_test": reach_id_all[is_test],
        "t_out_train": t_output_all[~is_test], "t_out_test": t_output_all[is_test],
        "zc_train": zc_all[~is_test], "zc_test": zc_all[is_test],
        "is_test": is_test,
    }


def scale_and_tensorize(X_train, y_train, X_test, y_test):
    """Fit MinMaxScaler on TRAIN only (to avoid leaking test-period statistics
    into normalization), apply to both splits, and return PyTorch tensors.
    """
    n_features = X_train.shape[-1]

    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # MinMaxScaler expects 2D (n_samples, n_features); flatten the window
    # dimension into the sample dimension for fitting/transforming, then
    # reshape back.
    input_scaler.fit(X_train.reshape(-1, n_features))
    output_scaler.fit(y_train.reshape(-1, 1))

    def _scale_X(X):
        shape = X.shape
        return input_scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)

    def _scale_y(y):
        shape = y.shape
        return output_scaler.transform(y.reshape(-1, 1)).reshape(shape)

    X_train_scaled, X_test_scaled = _scale_X(X_train), _scale_X(X_test)
    y_train_scaled, y_test_scaled = _scale_y(y_train), _scale_y(y_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_scaler, output_scaler


def make_loader(*tensors, batch_size, shuffle):
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def chronological_train_val_split(X_train_tensor, y_train_tensor, zc_train, t_out_train, val_fraction):
    """Carve a validation set out of the TRAIN split (not the test set) using
    the most recent `val_fraction` of training dates across all reaches
    (sorted chronologically, not just the tail of the concatenation order,
    since samples are grouped by reach then by date).
    """
    sort_idx = np.argsort(t_out_train.values)
    n_val = int(len(sort_idx) * val_fraction)
    val_idx, tr_idx = sort_idx[-n_val:], sort_idx[:-n_val]

    X_tr, y_tr = X_train_tensor[tr_idx], y_train_tensor[tr_idx]
    X_val, y_val = X_train_tensor[val_idx], y_train_tensor[val_idx]
    zc_tr_tensor = torch.tensor(zc_train[tr_idx], dtype=torch.float32)
    zc_val_tensor = torch.tensor(zc_train[val_idx], dtype=torch.float32)

    return X_tr, y_tr, zc_tr_tensor, X_val, y_val, zc_val_tensor
