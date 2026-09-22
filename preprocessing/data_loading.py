# preprocessing/data_loading.py

import numpy as np
import pandas as pd


def _mad_sigma(x: np.ndarray) -> float:
    """Robust standard-deviation estimate (median absolute deviation, scaled)."""
    x = x[~np.isnan(x)]
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def find_leading_cutoff_idx(
    values: np.ndarray,
    mad_multiplier: float = 10.0,
    consecutive_required: int = 3,
) -> int:
    """Index of the first position that starts a run of `consecutive_required`
    values that are neither NaN nor an extreme leading outlier (within
    `mad_multiplier` robust MADs of the series median).

    Every reach's raw record starts with a day of NaN (before the first valid
    interval midpoint) immediately followed by a huge one-off jump -- the
    record is effectively zero-padded before real monitoring began, and the
    first real reading(s) look like an enormous single-day spike relative to
    everything that follows. Requiring several CONSECUTIVE in-range values
    (rather than stopping at the very first one) avoids being fooled by a
    small, deceptively "normal-looking" value sandwiched between two large
    spikes (e.g. a raw sequence like [NaN, 815, 1059886, 1061317, 4494, ...]:
    stopping at the lone 815 would leave the two ~1e6 spikes right after it in
    the trimmed series). This is a data artifact, not a real event, so it is
    trimmed using a robust-scale (MAD-based) approach rather than a
    hard-coded day count.
    """
    sigma = _mad_sigma(values)
    med = np.nanmedian(values)
    n = len(values)
    for i in range(n):
        window = values[i:i + consecutive_required]
        if len(window) < consecutive_required:
            break
        if np.any(np.isnan(window)):
            continue
        if np.all(np.abs(window - med) <= mad_multiplier * sigma):
            return i
    return 0


def load_cir_and_volume(
    cir_daily_path,
    cir_target_column: str,
    volume_target_column: str,
    mad_multiplier: float = 10.0,
    consecutive_required: int = 3,
):
    """Load the per-reach daily CIR/volume table, trim each reach's leading
    NaN/initialization-artifact rows, and return per-reach CIR and volume
    series plus the sorted list of reach ids.

    Parameters
    ----------
    cir_daily_path : str or Path
        CSV with columns [date, reach_id, cir_target_column, volume_target_column, ...].
    cir_target_column, volume_target_column : str
        Column names for the CIR forecast target/autoregressive input and the
        (Student's-t Kalman-smoothed) sediment volume used for back-calculated
        volume metrics/plots.
    mad_multiplier, consecutive_required : see `find_leading_cutoff_idx`.

    Returns
    -------
    cir_by_reach : dict[str, pd.Series]
        Date-indexed, trimmed, NaN-dropped CIR series per reach. Plays two
        roles downstream: the "previous CIR" autoregressive input feature,
        and (independently) the forecast target.
    volume_by_reach : dict[str, pd.Series]
        Date-indexed smoothed volume series per reach (untrimmed), used only
        for back-calculated-sediment-volume metrics/plots.
    reach_ids : list[str]
        Sorted reach identifiers.
    """
    cir_daily = pd.read_csv(cir_daily_path, parse_dates=["date"])

    reach_ids = sorted(cir_daily["reach_id"].unique())

    volume_by_reach = {
        reach_id: g.set_index("date")[volume_target_column].sort_index()
        for reach_id, g in cir_daily.groupby("reach_id")
    }

    leading_cutoff_date = {}
    for reach_id, g in cir_daily.groupby("reach_id"):
        g = g.sort_values("date")
        idx = find_leading_cutoff_idx(
            g[cir_target_column].to_numpy(dtype=float),
            mad_multiplier=mad_multiplier,
            consecutive_required=consecutive_required,
        )
        leading_cutoff_date[reach_id] = g["date"].iloc[idx]

    cutoff_per_row = cir_daily["reach_id"].map(leading_cutoff_date)
    cir_daily_trimmed = cir_daily[cir_daily["date"] >= cutoff_per_row].reset_index(drop=True)

    cir_by_reach = {
        reach_id: g.set_index("date")[cir_target_column].sort_index().dropna()
        for reach_id, g in cir_daily_trimmed.groupby("reach_id")
    }

    return cir_by_reach, volume_by_reach, reach_ids
