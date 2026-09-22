# inference/recursive_prediction.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def recursive_predict(
    model: nn.Module,
    reach_id: str,
    forecast_start_date,
    cir_by_reach: dict,
    exo_shifted: pd.DataFrame,
    input_scaler,
    output_scaler,
    n_iterations: int,
    input_window_size: int,
    output_window_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Forecast `n_iterations * output_window_size` days of CIR forward from
    `forecast_start_date`.

    Each iteration's model INPUT is the most recent `input_window_size` days
    of "previous CIR" (real history, increasingly backfilled with the
    model's own prior predictions as the forecast extends further out) plus
    the REAL exogenous features for that same date range; the model then
    produces the NEXT `output_window_size` days of predicted CIR. A rolling
    buffer holds the last `input_window_size` days of CIR and is advanced by
    `output_window_size` days per iteration -- the general version of "feed
    the whole output back in as the next input", which only works when
    input and output window lengths are equal.

    Stops early (with a message) if the real exogenous data needed for the
    next window isn't available.

    Returns a DataFrame indexed by date with columns CIR_predicted,
    CIR_actual (the latter for comparison, NaN if beyond the real CIR record).
    """
    cir_series = cir_by_reach[reach_id]
    forecast_start_date = pd.Timestamp(forecast_start_date)

    seed_dates = pd.date_range(end=forecast_start_date - pd.Timedelta(days=1), periods=input_window_size, freq="D")
    if seed_dates.difference(cir_series.index).size or seed_dates.difference(exo_shifted.index).size:
        raise ValueError(f"Missing real history for the seed window ending {forecast_start_date.date()}.")

    # Rolling buffer of the last `input_window_size` days of CIR
    # (date-indexed); starts as real history and gets each iteration's
    # predictions appended (oldest days dropped) as it advances.
    cir_buffer = cir_series.reindex(seed_dates)

    predicted_blocks = []
    for it in range(n_iterations):
        future_dates = pd.date_range(
            start=forecast_start_date + pd.Timedelta(days=it * output_window_size),
            periods=output_window_size, freq="D",
        )
        input_dates = pd.date_range(end=future_dates.min() - pd.Timedelta(days=1), periods=input_window_size, freq="D")

        if input_dates.difference(exo_shifted.index).size or future_dates.difference(exo_shifted.index).size:
            print(f"  [{reach_id}] stopping after {it} iteration(s): no exogenous data for "
                  f"{input_dates.min().date()}..{future_dates.max().date()}.")
            break

        exo_block = exo_shifted.reindex(input_dates).to_numpy()
        if np.isnan(exo_block).any():
            print(f"  [{reach_id}] stopping after {it} iteration(s): NaN in exogenous data for this window.")
            break

        cir_prev_values = cir_buffer.reindex(input_dates).to_numpy()
        window_features = np.column_stack([cir_prev_values, exo_block])  # (input_window_size, n_features)
        window_scaled = input_scaler.transform(window_features)
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            y_hat_scaled = model(x).cpu().numpy()
        y_hat = output_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).flatten()

        predicted_series = pd.Series(y_hat, index=future_dates)
        predicted_blocks.append(predicted_series)

        # Advance the buffer: append the new predictions, keep only the
        # most recent input_window_size days.
        cir_buffer = pd.concat([cir_buffer, predicted_series]).iloc[-input_window_size:]

    if not predicted_blocks:
        return pd.DataFrame(columns=["CIR_predicted", "CIR_actual"])

    predicted = pd.concat(predicted_blocks)
    actual = cir_series.reindex(predicted.index)
    return pd.DataFrame({"CIR_predicted": predicted, "CIR_actual": actual})


def sample_forecast_starts(t_out_test, reach_test, reach_id: str, stride_days: int) -> pd.DatetimeIndex:
    """Evenly-spaced forecast start dates spanning reach_id's test region."""
    reach_test_dates = t_out_test[reach_test == reach_id]
    return pd.date_range(reach_test_dates.min(), reach_test_dates.max(), freq=f"{stride_days}D")
