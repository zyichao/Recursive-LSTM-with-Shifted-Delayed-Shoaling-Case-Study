# preprocessing/exogenous.py

import pandas as pd


def _normalize_feature_name(name: str, exo_columns) -> str:
    """Strip a trailing '.N' pandas dedup suffix (e.g. 'Discharge_X.1' ->
    'Discharge_X'), which shows up when the source table had a genuinely
    duplicated column name."""
    base = name.rsplit(".", 1)[0]
    return base if base in exo_columns else name


def select_and_shift_exogenous_features(exo_raw: pd.DataFrame, gsa_top: pd.DataFrame, top_n: int):
    """Pick the top-N exogenous discharge features by GSA sensitivity ranking
    and shift each forward by its own optimal lag.

    `gsa_top` has two rows -- `max_value` (peak sensitivity/importance) and
    `row_number` (the lag, in days, at which that peak occurs) -- one column
    per discharge station, already sorted by descending `max_value`, so the
    "top N" features are simply the first N columns.

    Selected features are shifted **forward** by their `row_number` days, so
    a station whose influence on sedimentation peaks ~20 days later has its
    value moved 20 days ahead -- e.g. the value recorded on 2013-01-01 lines
    up with 2013-01-21.

    Returns
    -------
    exo_shifted : pd.DataFrame
        Date-indexed, lag-shifted feature matrix (one column per selected
        feature, using its normalized name). Leading rows are NaN during the
        shift warm-up period.
    top_feature_map : dict[str, int]
        Normalized feature name -> forward shift (days).
    """
    assert gsa_top.loc["max_value"].is_monotonic_decreasing, \
        "gsa_top table is expected to be pre-sorted by max_value"

    # shift(n) on a *contiguous daily* index is exactly a date-shift: the
    # value at date D ends up at row/date D + n. Verify contiguity first so
    # that equivalence holds.
    assert (exo_raw.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), \
        "Exogenous discharge series must be a contiguous daily series for positional shift() to equal a date shift."

    top_feature_cols_raw = gsa_top.columns[:top_n].tolist()
    top_feature_map: dict[str, int] = {}
    for col in top_feature_cols_raw:
        norm = _normalize_feature_name(col, exo_raw.columns)
        shift_days = int(round(gsa_top.loc["row_number", col]))
        if norm in top_feature_map and top_feature_map[norm] != shift_days:
            raise ValueError(
                f"Conflicting shift_days for duplicated feature {norm!r}: "
                f"{top_feature_map[norm]} vs {shift_days}"
            )
        top_feature_map[norm] = shift_days

    exo_shifted = pd.DataFrame(index=exo_raw.index)
    for feat, shift_days in top_feature_map.items():
        exo_shifted[feat] = exo_raw[feat].shift(shift_days)

    return exo_shifted, top_feature_map
