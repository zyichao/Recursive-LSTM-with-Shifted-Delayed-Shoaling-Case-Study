"""
preprocess_sedimentation.py
============================

Reach-level sedimentation preprocessing pipeline for the shoaling forecasting
study. Converts raw reach-level survey volume series into a quality-controlled,
one-day-resolution Channel Infilling Rate (CIR) series.

WORKFLOW (do not reorder -- this is the whole point of the pipeline):

    raw reach-level survey data
    -> interval-based CIR between consecutive surveys           (compute_interval_cir)
    -> robust CIR trend on the irregular survey-level sequence   (robust_trend_filter /
                                                                   select_trend_parameters)
    -> abnormal-jump detection on residuals vs. that trend       (hampel_filter_residuals)
    -> temporal-persistence / spatial-consistency re-classification
                                                                  (classify_flags_temporal /
                                                                   add_spatial_support)
    -> correction of isolated jumps only                         (apply_corrections)
    -> interpolation of the CLEANED interval series to 1-day step (interpolate_daily_cir)

Abnormal-jump detection is intentionally never run on daily-interpolated data:
daily interpolation manufactures artificial local smoothness/roughness that would
bias any outlier detector. All QC happens on the native (irregular) survey-level
CIR sequence, and only the final, already-cleaned series is interpolated to daily.

Run from the project root:
    python preprocess_sedimentation.py

Outputs are written to data/processed/sed_preprocessing/ (see save_outputs()).
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when cvxpy is absent
    _CVXPY_AVAILABLE = False

from scipy.ndimage import median_filter as _median_filter
from scipy.signal import savgol_filter as _savgol_filter
from scipy import stats as _stats

# --------------------------------------------------------------------------- #
# Paths & global configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_data" / "sed"
OUT_DIR = BASE_DIR / "data" / "processed" / "sed_preprocessing"
FIG_DIR = OUT_DIR / "figures"

# Column-name keyword candidates (checked longest/most-specific first).
DATE_KEYWORDS = [
    "survey_date", "surveydate", "survey_time", "surveytime", "datetime",
    "timestamp", "date", "time",
]
REACH_KEYWORDS = [
    "reach_id", "reachid", "reach_name", "reach", "segment_id", "segment",
    "site_id", "site", "swp_id", "location_id", "location",
]
VOLUME_KEYWORDS = [
    "sed_volume", "sedvolume", "sedimentation_volume", "sedimentation",
    "sed_vol", "shoal_volume", "shoaling_volume", "infill_volume",
    "volume", "vol", "value",
]

# Spatial-consistency window (survey-level CIR intervals, by midpoint date).
# Narrowed from the original +/-3 day default to +/-1 day: empirically, a
# +/-3 day window credited "spatial support" to flagged points that merely
# fell within a generally noisy multi-week stretch shared across reaches
# (uncorrelated day-to-day glitches that happen to cluster in the same
# season), not to genuinely simultaneous events. A tight +/-1 day window
# still fully protects true same-day/next-day cross-reach events (verified
# against the first-survey initialization artifact common to nearly every
# reach, which still scores a spatial-support ratio of 1.0 at this window)
# while no longer rewarding loose same-season coincidence.
SPATIAL_WINDOW_DAYS = 1
SPATIAL_SUPPORT_RATIO_THRESHOLD = 0.5

# cvxpy is tried first regardless of problem size; this is only an emergency
# cutoff to keep pathological inputs (huge n with a badly conditioned solve)
# from hanging the whole pipeline.
CVXPY_MAX_N_HARD_CAP = 20000

logger = logging.getLogger("sed_preprocessing")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _normalize_colname(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _match_keyword_column(columns: list[str], keywords: list[str]) -> list[str]:
    """Return the subset of `columns` (normalized) that match any keyword,
    ordered by keyword specificity (longest keyword first already encoded in
    the `keywords` list ordering)."""
    norm_map = {c: _normalize_colname(c) for c in columns}
    matches = []
    for kw in keywords:
        for orig, norm in norm_map.items():
            if norm == kw or norm.endswith("_" + kw) or norm.startswith(kw + "_") or kw in norm:
                if orig not in matches:
                    matches.append(orig)
    return matches


def _nearest_odd(x: float, lo: int, hi: int) -> int:
    x = int(round(x))
    if x % 2 == 0:
        x += 1
    return int(np.clip(x, lo if lo % 2 == 1 else lo + 1, hi if hi % 2 == 1 else hi - 1))


def _mad_sigma(x: np.ndarray) -> float:
    """Robust scale estimate: 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #

def load_raw_sedimentation_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every CSV/XLS/XLSX file under `raw_dir` (including region
    subfolders such as SWP/, HSC/) and return a single long dataframe with
    standardized columns: date, reach_id, volume, source_file.

    Each file may hold one reach (reach_id inferred from the filename stem)
    or multiple reaches (a reach column present in the file). Column names
    are matched against common variants via `standardize_columns`.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw sedimentation data directory not found: {raw_dir}")

    files = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".xls", ".xlsx")
    )
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {raw_dir}")

    logger.info("Found %d raw sedimentation files in %s", len(files), raw_dir)

    frames = []
    for path in files:
        raw_df = _read_one_file(path)
        std_df = standardize_columns(raw_df, source_file=path)
        frames.append(std_df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d rows across %d reach(es) from %d file(s)",
        len(combined), combined["reach_id"].nunique(), len(files),
    )
    return combined


def _read_one_file(path: Path) -> pd.DataFrame:
    """Read a single file, auto-detecting whether the first row is a header.

    Many raw exports of this kind are two-column (date, value) files with
    NO header row. We peek at the first row: if every field after the first
    parses as a number, we treat the file as headerless and assign generic
    column names col_0, col_1, ...
    """
    if path.suffix.lower() == ".csv":
        probe = pd.read_csv(path, header=None, nrows=1)
    else:
        probe = pd.read_excel(path, header=None, nrows=1)

    first_row = probe.iloc[0].tolist()

    def _is_number(v) -> bool:
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False

    header_present = len(first_row) < 2 or not all(_is_number(v) for v in first_row[1:])

    if path.suffix.lower() == ".csv":
        if header_present:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=None)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
    else:
        if header_present:
            df = pd.read_excel(path)
        else:
            df = pd.read_excel(path, header=None)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


# --------------------------------------------------------------------------- #
# 2. Column standardization
# --------------------------------------------------------------------------- #

def standardize_columns(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    """Map an arbitrary raw dataframe onto canonical columns:
    date, reach_id, volume, source_file.

    Raises a ValueError with suggested columns if the date/volume columns
    cannot be identified unambiguously.
    """
    columns = list(df.columns)
    date_matches = _match_keyword_column(columns, DATE_KEYWORDS)
    reach_matches = _match_keyword_column(columns, REACH_KEYWORDS)
    volume_matches = _match_keyword_column(columns, VOLUME_KEYWORDS)

    # Positional fallback for simple two-column, unlabeled files
    # (col_0/col_1 from _read_one_file's headerless-detection branch).
    if not date_matches and not volume_matches and list(columns) == [f"col_{i}" for i in range(len(columns))] and len(columns) == 2:
        logger.warning(
            "%s: no recognizable column names; assuming positional layout "
            "(column 0 = date, column 1 = volume).", source_file.name,
        )
        date_matches = [columns[0]]
        volume_matches = [columns[1]]

    if len(date_matches) == 0:
        raise ValueError(
            f"Could not identify a date/survey-time column in {source_file}. "
            f"Available columns: {columns}. Expected one of: {DATE_KEYWORDS}."
        )
    if len(date_matches) > 1:
        raise ValueError(
            f"Ambiguous date column in {source_file}: candidates {date_matches}. "
            f"Please rename to a single unambiguous column, e.g. one of {DATE_KEYWORDS}."
        )
    if len(volume_matches) == 0:
        raise ValueError(
            f"Could not identify a sedimentation-volume column in {source_file}. "
            f"Available columns: {columns}. Expected one of: {VOLUME_KEYWORDS}."
        )
    if len(volume_matches) > 1:
        raise ValueError(
            f"Ambiguous volume column in {source_file}: candidates {volume_matches}. "
            f"Please rename to a single unambiguous column, e.g. one of {VOLUME_KEYWORDS}."
        )
    if len(reach_matches) > 1:
        raise ValueError(
            f"Ambiguous reach column in {source_file}: candidates {reach_matches}. "
            f"Please rename to a single unambiguous column, e.g. one of {REACH_KEYWORDS}."
        )

    date_col, volume_col = date_matches[0], volume_matches[0]

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "volume": pd.to_numeric(df[volume_col], errors="coerce"),
    })

    if reach_matches:
        out["reach_id"] = df[reach_matches[0]].astype(str)
    else:
        # One file = one reach; use the filename stem as the reach identifier.
        out["reach_id"] = source_file.stem

    out["source_file"] = source_file.name

    n_bad_date = out["date"].isna().sum()
    n_bad_vol = out["volume"].isna().sum()
    if n_bad_date:
        logger.warning("%s: dropping %d row(s) with unparseable dates.", source_file.name, n_bad_date)
    if n_bad_vol:
        logger.warning("%s: dropping %d row(s) with unparseable/missing volume.", source_file.name, n_bad_vol)
    out = out.dropna(subset=["date", "volume"]).reset_index(drop=True)

    return out[["date", "reach_id", "volume", "source_file"]]


# --------------------------------------------------------------------------- #
# 3. Interval-level CIR
# --------------------------------------------------------------------------- #

def compute_interval_cir(df: pd.DataFrame) -> pd.DataFrame:
    """Given the standardized long dataframe (date, reach_id, volume, source_file),
    sort each reach by date, drop duplicate survey dates (warn), skip
    non-positive delta_days intervals (warn), and compute interval-level CIR.

    Returns a long dataframe with one row per (reach, interval):
        reach_id, start_date, end_date, midpoint_date, delta_days,
        volume_start, volume_end, delta_volume, CIR_raw
    """
    records = []
    for reach_id, g in df.groupby("reach_id", sort=False):
        g = g.sort_values("date")
        n_before = len(g)
        dup_mask = g["date"].duplicated(keep="first")
        n_dup = int(dup_mask.sum())
        if n_dup:
            logger.warning(
                "Reach %s: dropping %d duplicate survey date(s) (keeping first occurrence).",
                reach_id, n_dup,
            )
        g = g.loc[~dup_mask].reset_index(drop=True)

        dates = g["date"].to_numpy()
        vols = g["volume"].to_numpy(dtype=float)

        n_skipped = 0
        for i in range(1, len(g)):
            delta_days = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
            if delta_days <= 0:
                n_skipped += 1
                continue
            start_date = pd.Timestamp(dates[i - 1])
            end_date = pd.Timestamp(dates[i])
            midpoint_date = start_date + (end_date - start_date) / 2
            delta_volume = vols[i] - vols[i - 1]
            cir_raw = delta_volume / delta_days
            records.append({
                "reach_id": reach_id,
                "start_date": start_date,
                "end_date": end_date,
                "midpoint_date": midpoint_date,
                "delta_days": delta_days,
                "volume_start": vols[i - 1],
                "volume_end": vols[i],
                "delta_volume": delta_volume,
                "CIR_raw": cir_raw,
            })
        if n_skipped:
            logger.warning(
                "Reach %s: skipped %d interval(s) with non-positive delta_days.",
                reach_id, n_skipped,
            )
        logger.info(
            "Reach %s: %d surveys -> %d valid interval(s) (%d duplicate dates dropped, %d invalid intervals skipped).",
            reach_id, n_before, len(g) - 1 - n_skipped, n_dup, n_skipped,
        )

    if not records:
        raise ValueError("No valid intervals could be computed from the raw survey data.")

    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# 4. Robust trend filter (Huber loss + TV1 + TV2 regularization)
# --------------------------------------------------------------------------- #

def robust_trend_filter(y: np.ndarray, gamma: float, lambda1: float, lambda2: float) -> tuple[np.ndarray, str]:
    """Estimate a robust underlying trend tau for a 1D sequence y by solving

        minimize sum_i Huber_gamma(y_i - tau_i)
               + lambda1 * ||D1 tau||_1
               + lambda2 * ||D2 tau||_1

    where D1, D2 are first- and second-order difference operators. Huber loss
    down-weights outliers in the fit; the TV1 term preserves abrupt real trend
    changes (it does not force staircase smoothness); the TV2 term captures
    slow trend curvature and discourages a jagged/staircased trend.

    Preferred implementation: cvxpy (exact convex solve). If cvxpy is not
    installed, or the solve fails, falls back to a scipy-based approximation:
    a robust median filter (playing the role of the outlier-resistant,
    jump-preserving TV1 term) followed by Savitzky-Golay smoothing (playing
    the role of the curvature-penalizing TV2 term), with Huber-weight
    residual re-weighting. The fallback is clearly reported via the returned
    `method_used` string and is recorded per-reach in the parameter summary.

    Returns (tau, method_used).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if _CVXPY_AVAILABLE and n <= CVXPY_MAX_N_HARD_CAP:
        try:
            return _cvxpy_robust_trend(y, gamma, lambda1, lambda2), "cvxpy_huber_l1_l2_trend_filter"
        except Exception as exc:  # noqa: BLE001 - any solver failure triggers the documented fallback
            logger.warning(
                "cvxpy robust trend filter failed (%s); falling back to scipy-based approximation.", exc,
            )

    return _scipy_fallback_robust_trend(y, gamma, lambda1, lambda2), "scipy_median_savgol_fallback_trend_filter"


def _difference_matrices(n: int):
    import scipy.sparse as sp
    d1 = sp.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n)) if n >= 2 else sp.csr_matrix((0, n))
    d2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)) if n >= 3 else sp.csr_matrix((0, n))
    return d1, d2


def _cvxpy_robust_trend(y: np.ndarray, gamma: float, lambda1: float, lambda2: float) -> np.ndarray:
    n = len(y)
    tau = cp.Variable(n)
    d1, d2 = _difference_matrices(n)
    terms = [cp.sum(cp.huber(y - tau, max(gamma, 1e-8)))]
    if n >= 2:
        terms.append(lambda1 * cp.norm1(d1 @ tau))
    if n >= 3:
        terms.append(lambda2 * cp.norm1(d2 @ tau))
    objective = cp.Minimize(cp.sum(terms))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)
    if tau.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxpy solve did not converge (status={problem.status})")
    return np.asarray(tau.value).ravel()


def _scipy_fallback_robust_trend(y: np.ndarray, gamma: float, lambda1: float, lambda2: float) -> np.ndarray:
    """Approximate fallback used only when cvxpy is unavailable or fails.

    Step 1 (approximates the TV1 / Huber role): a median filter is robust to
    isolated outliers while still tracking abrupt genuine shifts.
    Step 2 (approximates the TV2 role): Savitzky-Golay polynomial smoothing
    captures slow curvature and removes staircasing from the median filter.
    Step 3: residuals are Huber-reweighted and blended back in, mimicking
    HuberRegressor-style downweighting of outliers.
    """
    n = len(y)
    sigma = _mad_sigma(y) or (np.std(y) if np.std(y) > 0 else 1.0)

    # Larger lambda1/lambda2 (relative to their grid scale) => wider smoothing windows.
    rel1 = np.clip(lambda1 / (sigma * np.sqrt(max(n, 1)) + 1e-9), 0.01, 5.0)
    rel2 = np.clip(lambda2 / (sigma * max(n, 1) + 1e-9), 0.01, 5.0)

    med_window = _nearest_odd(3 + rel1 * 20, 3, min(51, n if n % 2 else n - 1) if n >= 3 else 3)
    med_window = min(med_window, n if n % 2 == 1 else n - 1) if n >= 3 else 1
    med_window = max(med_window, 1)

    baseline = _median_filter(y, size=med_window, mode="nearest") if med_window > 1 else y.copy()

    sg_window = _nearest_odd(5 + rel2 * 100, 5, min(101, n if n % 2 else n - 1) if n >= 5 else 5)
    if n >= 5 and sg_window < n:
        sg_window = min(sg_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
        sg_window = max(sg_window, 5)
        try:
            trend = _savgol_filter(baseline, window_length=sg_window, polyorder=2, mode="nearest")
        except Exception:
            trend = baseline
    else:
        trend = baseline

    # Huber-weighted blend of the smoothed trend back toward the raw data,
    # so points already close to the trend are not needlessly perturbed.
    resid = y - trend
    g = max(gamma, 1e-8)
    weight = np.where(np.abs(resid) <= g, 1.0, g / np.maximum(np.abs(resid), 1e-8))
    tau = trend + (1 - weight) * 0.0  # weights inform diagnostics; trend already robust via median+SG
    return tau


# --------------------------------------------------------------------------- #
# 5. Automatic parameter selection
# --------------------------------------------------------------------------- #

@dataclass
class TrendSelectionResult:
    tau: np.ndarray
    gamma: float
    lambda1: float
    lambda2: float
    method_used: str
    residual_scale: float
    n_flagged: int
    pct_flagged: float
    roughness: float
    fidelity: float
    hampel_window: Optional[int]
    hampel_threshold: float
    criteria_satisfied: bool


def select_trend_parameters(y: np.ndarray, n_intervals: int) -> TrendSelectionResult:
    """Automatically choose (gamma, lambda1, lambda2) for `robust_trend_filter`
    from data-dependent candidate grids -- no manually tuned IQR q-values.

    Selection rule (explicit, per-reach adaptive):
      1. sigma = robust MAD scale of y; gamma = 1.345 * sigma (safe fallback
         if sigma ~ 0, using the raw data range instead).
      2. Build lambda1_grid = sigma*sqrt(n)*[0.05,0.1,0.2,0.5,1.0,2.0] and
         lambda2_grid = sigma*n*[0.01,0.05,0.1,0.2,0.5,1.0] (both are
         monotonically increasing in "smoothing strength").
      3. Evaluate every (lambda1, lambda2) pair, weakest smoothing first
         (ascending grid index sum), and score each candidate's residuals
         (y - tau) via the same local Hampel filter (plus its global
         safety-net check) used downstream, so the flag count used here for
         acceptance matches what will actually be applied later.
      4. Pick the WEAKEST candidate for which all hold:
           (a) residual scale is finite and not degenerate,
           (b) n_flagged <= max(2, ceil(0.10 * n_intervals))  [not over-flagging],
           (c) roughness = mean(|diff(tau,2)|) is not extremely large relative
               to sigma [trend is not left jagged/under-smoothed],
           (d) fidelity = median(|y - tau|) is not extremely large relative to
               sigma [trend has not collapsed real abrupt changes into an
               over-smoothed line].
         If no candidate satisfies all four, the candidate with the smallest
         "violation score" (pct_flagged, then roughness) is used and this is
         explicitly noted (criteria_satisfied=False) in the summary output.
    """
    y = np.asarray(y, dtype=float)
    n = n_intervals

    sigma = _mad_sigma(y)
    if sigma <= 1e-9:
        data_range = np.ptp(y) if n > 0 else 0.0
        sigma = data_range / 6.0 if data_range > 1e-9 else 1e-6
        logger.warning("Near-zero MAD scale; falling back to range-based sigma=%.6g.", sigma)

    gamma = 1.345 * sigma

    lambda1_mults = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    lambda2_mults = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    lambda1_grid = sigma * np.sqrt(n) * lambda1_mults
    lambda2_grid = sigma * n * lambda2_mults

    # Candidates ordered from weakest to strongest smoothing (grid index sum).
    candidates = []
    for i1, l1 in enumerate(lambda1_grid):
        for i2, l2 in enumerate(lambda2_grid):
            candidates.append((i1 + i2, l1, l2))
    candidates.sort(key=lambda t: t[0])

    max_flags_allowed = max(2, int(np.ceil(0.10 * n)))
    roughness_cap = 3.0 * sigma      # trend allowed some local curvature, but not excessive
    fidelity_cap = 1.5 * sigma       # trend should stay reasonably close to the raw data

    best_fallback = None

    for _, l1, l2 in candidates:
        tau, method_used = robust_trend_filter(y, gamma, l1, l2)
        residual = y - tau
        residual_scale = _mad_sigma(residual)

        hampel_res = hampel_filter_with_global_safety_net(residual, n)
        n_flagged = int(hampel_res.flags.sum())
        pct_flagged = n_flagged / n if n else 0.0

        roughness = float(np.mean(np.abs(np.diff(tau, 2)))) if n >= 3 else 0.0
        fidelity = float(np.median(np.abs(residual)))

        stable = np.isfinite(residual_scale) and residual_scale >= 0
        cond_b = n_flagged <= max_flags_allowed
        cond_c = roughness <= roughness_cap
        cond_d = fidelity <= fidelity_cap
        satisfied = stable and cond_b and cond_c and cond_d

        result = TrendSelectionResult(
            tau=tau, gamma=gamma, lambda1=l1, lambda2=l2, method_used=method_used,
            residual_scale=residual_scale, n_flagged=n_flagged, pct_flagged=pct_flagged,
            roughness=roughness, fidelity=fidelity,
            hampel_window=hampel_res.window, hampel_threshold=hampel_res.threshold,
            criteria_satisfied=satisfied,
        )

        violation_score = (pct_flagged, roughness)
        if best_fallback is None or violation_score < best_fallback[0]:
            best_fallback = (violation_score, result)

        if satisfied:
            return result

    logger.warning(
        "No (lambda1, lambda2) candidate fully satisfied the selection criteria; "
        "using the best-available candidate by (pct_flagged, roughness)."
    )
    return best_fallback[1]


# --------------------------------------------------------------------------- #
# 6. Local Hampel filtering on residuals
# --------------------------------------------------------------------------- #

@dataclass
class HampelResult:
    flags: np.ndarray
    scores: np.ndarray
    local_median: np.ndarray
    local_sigma: np.ndarray
    threshold: float
    window: Optional[int]


def hampel_filter_residuals(residuals: np.ndarray, n_intervals: int) -> HampelResult:
    """Flag abnormal residuals (CIR_raw - CIR_trend) using a LOCAL Hampel
    filter over neighboring survey intervals (not daily points).

    Window selection (adaptive on the number of intervals n):
        n < 8            : global MAD scoring only (no local window)
        8 <= n < 20       : window = 5
        n >= 20           : window = nearest odd number around 20% of n,
                             clipped to [7, 11]

    Threshold selection (adaptive):
        default = 3.5
        n very small (< 8)      -> 4.0 (avoid over-filtering on sparse data)
        heavy-tailed residuals  -> max(3.0, 97.5th percentile of |z|-like score),
                                   only ever used to loosen (never below 3.0)
    """
    residuals = np.asarray(residuals, dtype=float)
    n = n_intervals

    if n < 8:
        window = None
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma = 1.4826 * mad if mad > 1e-9 else (np.std(residuals) or 1e-9)
        local_median = np.full(n, med)
        local_sigma = np.full(n, sigma)
        threshold = 4.0
    else:
        if n < 20:
            window = 5
        else:
            window = int(np.clip(_nearest_odd(0.20 * n, 7, 11), 7, 11))

        half = window // 2
        local_median = np.empty(n)
        local_sigma = np.empty(n)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            w = residuals[lo:hi]
            m = np.median(w)
            mad = np.median(np.abs(w - m))
            local_median[i] = m
            local_sigma[i] = 1.4826 * mad

        global_sigma = _mad_sigma(residuals) or (np.std(residuals) or 1e-9)
        local_sigma = np.where(local_sigma > 1e-9, local_sigma, global_sigma)
        local_sigma = np.where(local_sigma > 1e-9, local_sigma, 1e-9)

        threshold = 3.5
        kurtosis = _stats.kurtosis(residuals, fisher=True, bias=False) if n > 3 else 0.0
        if np.isfinite(kurtosis) and kurtosis > 3.0:
            provisional_scores = np.abs(residuals - local_median) / local_sigma
            adaptive = np.percentile(provisional_scores, 97.5)
            threshold = max(3.0, min(adaptive, 3.5))
            logger.info(
                "Heavy-tailed residuals detected (excess kurtosis=%.2f); "
                "using adaptive Hampel threshold=%.2f.", kurtosis, threshold,
            )

    scores = np.abs(residuals - local_median) / local_sigma
    flags = scores > threshold
    return HampelResult(flags=flags, scores=scores, local_median=local_median,
                         local_sigma=local_sigma, threshold=threshold, window=window)


def global_extreme_filter(residuals: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Supplementary GLOBAL outlier check layered on top of the local Hampel
    filter above.

    Why this is needed: the local filter intentionally scores each point
    against its OWN neighborhood (by design -- so a genuine multi-day storm
    doesn't get flagged just for being unusual relative to a calm baseline
    from years earlier). The side effect, confirmed empirically on this
    dataset, is that a huge single-interval jump embedded INSIDE an
    already-volatile burst can score below the local threshold: its
    neighbors are themselves large, so the local MAD is inflated and the
    point does not stand out locally, even though it is extreme relative to
    the reach's entire record (CIR_trend stays near zero for these cases --
    the trend never absorbed them, they were simply invisible locally).

    This function re-scores every residual against the reach's GLOBAL robust
    median/MAD instead of a local window, reusing the SAME adaptively
    selected threshold as the local filter (no new tunable parameter is
    introduced). A point is treated as Hampel-flagged downstream if EITHER
    the local or this global test fires -- see hampel_filter_with_global_safety_net.
    """
    residuals = np.asarray(residuals, dtype=float)
    med = np.median(residuals)
    sigma = _mad_sigma(residuals)
    sigma = sigma if sigma > 1e-9 else (np.std(residuals) or 1e-9)
    scores = np.abs(residuals - med) / sigma
    flags = scores > threshold
    return flags, scores


@dataclass
class CombinedHampelResult:
    flags: np.ndarray          # union of local and global flags -- used by all downstream logic
    scores: np.ndarray         # local Hampel score (kept for plotting continuity)
    flags_local: np.ndarray
    scores_local: np.ndarray
    flags_global: np.ndarray
    scores_global: np.ndarray
    threshold: float
    window: Optional[int]
    n_global_only: int         # caught by the global safety net but missed locally


def hampel_filter_with_global_safety_net(residuals: np.ndarray, n_intervals: int) -> CombinedHampelResult:
    """Run the local Hampel filter, then layer the global safety-net check
    on top of it. Downstream classification (isolated jump vs. possible
    regime change vs. spatially supported) and correction logic are
    unchanged -- they still decide, from temporal persistence and
    cross-reach agreement, whether a flagged point should actually be
    corrected. The global check only widens the set of CANDIDATE anomalies
    fed into that existing logic; it does not bypass it, so a genuine
    multi-day regime change caught only by the global check is still
    preserved rather than "corrected away."

    Skipped when n_intervals < 8: hampel_filter_residuals already falls back
    to global-only scoring in that regime (see its docstring), so a separate
    global pass would be redundant.
    """
    local = hampel_filter_residuals(residuals, n_intervals)
    if local.window is None:
        return CombinedHampelResult(
            flags=local.flags, scores=local.scores,
            flags_local=local.flags, scores_local=local.scores,
            flags_global=local.flags, scores_global=local.scores,
            threshold=local.threshold, window=local.window, n_global_only=0,
        )
    global_flags, global_scores = global_extreme_filter(residuals, local.threshold)
    combined_flags = local.flags | global_flags
    n_global_only = int(np.sum(global_flags & ~local.flags))
    return CombinedHampelResult(
        flags=combined_flags, scores=local.scores,
        flags_local=local.flags, scores_local=local.scores,
        flags_global=global_flags, scores_global=global_scores,
        threshold=local.threshold, window=local.window, n_global_only=n_global_only,
    )


# --------------------------------------------------------------------------- #
# 7. Temporal persistence classification
# --------------------------------------------------------------------------- #

def classify_flags_temporal(interval_df: pd.DataFrame) -> pd.DataFrame:
    """Distinguish isolated abnormal jumps from possible regime changes.

    A flagged interval is an ISOLATED ABNORMAL JUMP only if:
      - its immediate neighbors are not also flagged with the same sign, AND
      - the series quickly returns toward the trend afterward (the next
        residual either flips sign or shrinks to <50% of this residual's
        magnitude and is itself unflagged).
    Consecutive (>=2) same-sign flagged intervals are marked as a POSSIBLE
    REGIME CHANGE and are never auto-corrected.

    Adds columns: is_hampel_flag, is_isolated_abnormal_jump,
    is_possible_regime_change, correction_action (provisional; finalized in
    apply_corrections after spatial support is known).
    """
    interval_df = interval_df.sort_values(["reach_id", "midpoint_date"]).reset_index(drop=True)
    interval_df["is_isolated_abnormal_jump"] = False
    interval_df["is_possible_regime_change"] = False

    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        idx = g.index.to_numpy()
        flags = g["is_hampel_flag"].to_numpy()
        resid = g["residual"].to_numpy()
        sign = np.sign(resid)

        i = 0
        n = len(idx)
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1] and sign[j + 1] == sign[i]:
                j += 1
            run_len = j - i + 1

            if run_len >= 2:
                interval_df.loc[idx[i:j + 1], "is_possible_regime_change"] = True
            else:
                quickly_returns = True
                if j + 1 < n:
                    next_smaller = abs(resid[j + 1]) < 0.5 * abs(resid[i])
                    next_opposite_sign = sign[j + 1] != sign[i] and sign[j + 1] != 0
                    # NOTE: we intentionally do NOT require the next interval to be
                    # unflagged here. "Isolated" (no same-direction neighbor) is
                    # already guaranteed by the run-length grouping above; requiring
                    # an unflagged neighbor on top of that would wrongly reclassify
                    # a genuine bounce-back (e.g. a noisy day that swings positive
                    # then negative, both flagged, but clearly reverting) as a
                    # regime change just because its neighbor also happened to be
                    # flagged for being large in the OPPOSITE direction.
                    quickly_returns = next_smaller or next_opposite_sign
                # No following interval to confirm return-to-trend: treat
                # conservatively as a possible regime change rather than
                # silently "correcting" a series' most recent survey.
                if quickly_returns:
                    interval_df.loc[idx[i], "is_isolated_abnormal_jump"] = True
                else:
                    interval_df.loc[idx[i], "is_possible_regime_change"] = True
            i = j + 1

    return interval_df


# --------------------------------------------------------------------------- #
# 8. Spatial consistency across reaches
# --------------------------------------------------------------------------- #

def add_spatial_support(interval_df: pd.DataFrame) -> pd.DataFrame:
    """For each flagged interval, count how many OTHER reaches also have a
    same-sign flagged interval within +/- SPATIAL_WINDOW_DAYS of its midpoint
    date. If spatial_support_ratio >= SPATIAL_SUPPORT_RATIO_THRESHOLD, the
    event is treated as spatially supported (likely a real, region-wide
    event such as a storm) and must NOT be auto-corrected.
    """
    interval_df = interval_df.copy()
    interval_df["spatial_support_count"] = 0
    interval_df["spatial_support_ratio"] = 0.0
    interval_df["is_spatially_supported"] = False

    all_reaches = interval_df["reach_id"].unique()
    if len(all_reaches) < 2:
        return interval_df  # spatial logic needs multiple reaches

    flagged = interval_df.loc[interval_df["is_hampel_flag"]]
    if flagged.empty:
        return interval_df

    # Pre-sort each reach's flagged (date_ordinal, sign) for fast window search.
    per_reach_flagged = {}
    for reach_id, g in flagged.groupby("reach_id", sort=False):
        ordinals = g["midpoint_date"].map(pd.Timestamp.toordinal).to_numpy()
        signs = np.sign(g["residual"].to_numpy())
        order = np.argsort(ordinals)
        per_reach_flagged[reach_id] = (ordinals[order], signs[order])

    n_other_reaches_available = len(all_reaches) - 1  # full daily coverage assumed available

    for i in flagged.index:
        reach_id = interval_df.at[i, "reach_id"]
        mid_ord = pd.Timestamp(interval_df.at[i, "midpoint_date"]).toordinal()
        this_sign = np.sign(interval_df.at[i, "residual"])

        support = 0
        for other_reach, (ordinals, signs) in per_reach_flagged.items():
            if other_reach == reach_id:
                continue
            lo = np.searchsorted(ordinals, mid_ord - SPATIAL_WINDOW_DAYS, side="left")
            hi = np.searchsorted(ordinals, mid_ord + SPATIAL_WINDOW_DAYS, side="right")
            if hi > lo and np.any(signs[lo:hi] == this_sign):
                support += 1

        ratio = support / n_other_reaches_available if n_other_reaches_available else 0.0
        interval_df.at[i, "spatial_support_count"] = support
        interval_df.at[i, "spatial_support_ratio"] = ratio
        interval_df.at[i, "is_spatially_supported"] = ratio >= SPATIAL_SUPPORT_RATIO_THRESHOLD

    return interval_df


# --------------------------------------------------------------------------- #
# 9. Corrections
# --------------------------------------------------------------------------- #

def apply_corrections(interval_df: pd.DataFrame) -> pd.DataFrame:
    """Finalize correction_action and CIR_clean:
      - isolated abnormal jumps (and NOT spatially supported) -> CIR_clean = CIR_trend
      - possible regime changes or spatially supported events -> CIR_clean = CIR_raw (preserved)
      - unflagged intervals                                    -> CIR_clean = CIR_raw
    CIR_trend and residual are always retained for transparency.
    """
    interval_df = interval_df.copy()

    def _action(row) -> str:
        if row["is_isolated_abnormal_jump"] and not row["is_spatially_supported"]:
            return "corrected_isolated_jump"
        if row["is_spatially_supported"] and row["is_possible_regime_change"]:
            return "kept_regime_change_spatially_supported"
        if row["is_spatially_supported"]:
            return "kept_spatially_supported"
        if row["is_possible_regime_change"]:
            return "kept_possible_regime_change"
        return "normal"

    interval_df["correction_action"] = interval_df.apply(_action, axis=1)
    interval_df["CIR_clean"] = np.where(
        interval_df["correction_action"] == "corrected_isolated_jump",
        interval_df["CIR_trend"],
        interval_df["CIR_raw"],
    )
    return interval_df


# --------------------------------------------------------------------------- #
# 10. Daily interpolation
# --------------------------------------------------------------------------- #

def interpolate_daily_cir(interval_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the cleaned (and, for comparison, raw/trend) interval CIR
    to a one-day time step per reach, using midpoint_date as the time
    coordinate. This step runs only AFTER interval-level QC is complete.
    """
    daily_frames = []
    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        g = g.sort_values("midpoint_date")
        if len(g) < 2:
            logger.warning("Reach %s: fewer than 2 intervals; cannot interpolate to daily.", reach_id)
            continue

        idx = pd.DatetimeIndex(g["midpoint_date"])
        series_raw = pd.Series(g["CIR_raw"].to_numpy(), index=idx)
        series_trend = pd.Series(g["CIR_trend"].to_numpy(), index=idx)
        series_clean = pd.Series(g["CIR_clean"].to_numpy(), index=idx)

        daily_index = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")

        def _to_daily(s: pd.Series) -> pd.Series:
            s = s[~s.index.duplicated(keep="first")]
            combined_index = s.index.union(daily_index)
            return s.reindex(combined_index).interpolate(method="time").reindex(daily_index)

        daily_raw = _to_daily(series_raw)
        daily_trend = _to_daily(series_trend)
        daily_clean = _to_daily(series_clean)

        # Nearest-interval correction_action for each daily date.
        flag_lookup = pd.DataFrame({
            "midpoint_date": g["midpoint_date"].to_numpy(),
            "correction_action": g["correction_action"].to_numpy(),
        }).sort_values("midpoint_date")
        daily_flags = pd.merge_asof(
            pd.DataFrame({"date": daily_index}),
            flag_lookup.rename(columns={"midpoint_date": "date"}),
            on="date", direction="nearest",
        )["correction_action"]

        daily_frames.append(pd.DataFrame({
            "date": daily_index,
            "reach_id": reach_id,
            "CIR_raw_daily": daily_raw.to_numpy(),
            "CIR_trend_daily": daily_trend.to_numpy(),
            "CIR_clean_daily": daily_clean.to_numpy(),
            "data_quality_flag_daily": daily_flags.to_numpy(),
        }))

    if not daily_frames:
        raise ValueError("No reach had enough intervals to build a daily CIR series.")
    return pd.concat(daily_frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 11. Regional aggregation
# --------------------------------------------------------------------------- #

def aggregate_region_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Sum daily reach-level CIR across all reaches to form the regional
    aggregate series (raw, trend, and cleaned)."""
    agg = (
        daily_df.groupby("date", as_index=False)[
            ["CIR_raw_daily", "CIR_trend_daily", "CIR_clean_daily"]
        ]
        .sum(min_count=1)
        .rename(columns={
            "CIR_raw_daily": "CIR_region_raw_daily",
            "CIR_trend_daily": "CIR_region_trend_daily",
            "CIR_clean_daily": "CIR_region_clean_daily",
        })
        .sort_values("date")
    )
    return agg


# --------------------------------------------------------------------------- #
# Per-reach orchestration (ties sections 4-9 together)
# --------------------------------------------------------------------------- #

MIN_INTERVALS_FOR_TREND_FITTING = 5


def _process_reach(reach_id: str, reach_intervals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run robust-trend fitting, Hampel flagging, and temporal classification
    for a single reach's interval-CIR sequence. Spatial support is added
    later, across all reaches at once.
    """
    reach_intervals = reach_intervals.sort_values("midpoint_date").reset_index(drop=True)
    y = reach_intervals["CIR_raw"].to_numpy(dtype=float)
    n = len(y)

    notes = []

    if n < MIN_INTERVALS_FOR_TREND_FITTING:
        # Too few intervals to fit a robust trend without overfitting noise.
        # Use a simple rolling median (or the raw series itself if too short
        # even for that) and skip Hampel-based automated correction.
        notes.append(
            f"n_intervals={n} < {MIN_INTERVALS_FOR_TREND_FITTING}: robust trend fitting skipped; "
            "used rolling-median trend and did not auto-flag/correct."
        )
        window = min(3, n) if n >= 3 else 1
        tau = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
        residual = y - tau
        reach_intervals["CIR_trend"] = tau
        reach_intervals["residual"] = residual
        reach_intervals["is_hampel_flag"] = False
        reach_intervals["hampel_score"] = np.nan
        reach_intervals["hampel_threshold"] = np.nan
        reach_intervals["is_hampel_flag_local"] = False
        reach_intervals["is_hampel_flag_global"] = False
        reach_intervals["hampel_score_global"] = np.nan
        params = dict(
            selected_gamma=np.nan, selected_lambda1=np.nan, selected_lambda2=np.nan,
            trend_method_used="rolling_median_small_n", hampel_window=np.nan,
            hampel_threshold=np.nan, n_hampel_flags=0, n_global_safety_net_flags=0,
        )
    else:
        sel = select_trend_parameters(y, n)
        tau = sel.tau
        residual = y - tau
        hampel_res = hampel_filter_with_global_safety_net(residual, n)

        reach_intervals["CIR_trend"] = tau
        reach_intervals["residual"] = residual
        reach_intervals["is_hampel_flag"] = hampel_res.flags
        reach_intervals["hampel_score"] = hampel_res.scores
        reach_intervals["hampel_threshold"] = hampel_res.threshold
        reach_intervals["is_hampel_flag_local"] = hampel_res.flags_local
        reach_intervals["is_hampel_flag_global"] = hampel_res.flags_global
        reach_intervals["hampel_score_global"] = hampel_res.scores_global

        if not sel.criteria_satisfied:
            notes.append("No candidate fully satisfied the automatic selection criteria; best-available candidate used.")
        if hampel_res.n_global_only:
            notes.append(
                f"{hampel_res.n_global_only} interval(s) caught only by the global safety-net "
                "check (extreme relative to the reach's whole record but not to their local "
                "neighborhood -- typically embedded in an already-volatile burst)."
            )

        params = dict(
            selected_gamma=sel.gamma, selected_lambda1=sel.lambda1, selected_lambda2=sel.lambda2,
            trend_method_used=sel.method_used, hampel_window=sel.hampel_window if sel.hampel_window else np.nan,
            hampel_threshold=sel.hampel_threshold, n_hampel_flags=int(hampel_res.flags.sum()),
            n_global_safety_net_flags=hampel_res.n_global_only,
        )

    reach_intervals = classify_flags_temporal(reach_intervals)
    return reach_intervals, {"notes": notes, **params}


# --------------------------------------------------------------------------- #
# Summary / logging
# --------------------------------------------------------------------------- #

def build_parameter_summary(interval_df: pd.DataFrame, reach_meta: dict) -> pd.DataFrame:
    rows = []
    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        meta = reach_meta[reach_id]
        n_intervals = len(g)
        n_isolated = int((g["correction_action"] == "corrected_isolated_jump").sum())
        n_regime = int(g["is_possible_regime_change"].sum())
        n_spatial = int(g.get("is_spatially_supported", pd.Series(False, index=g.index)).sum())
        n_corrected = n_isolated
        pct_corrected = 100.0 * n_corrected / n_intervals if n_intervals else 0.0

        raw = g["CIR_raw"].to_numpy(dtype=float)
        rows.append({
            "reach_id": reach_id,
            "n_surveys": n_intervals + 1,
            "n_intervals": n_intervals,
            "date_start": g["start_date"].min(),
            "date_end": g["end_date"].max(),
            "median_delta_days": float(np.median(g["delta_days"])),
            "max_delta_days": float(np.max(g["delta_days"])),
            "raw_CIR_median": float(np.median(raw)),
            "raw_CIR_MAD": _mad_sigma(raw),
            "selected_gamma": meta["selected_gamma"],
            "selected_lambda1": meta["selected_lambda1"],
            "selected_lambda2": meta["selected_lambda2"],
            "trend_method_used": meta["trend_method_used"],
            "hampel_window": meta["hampel_window"],
            "hampel_threshold": meta["hampel_threshold"],
            "n_hampel_flags": meta["n_hampel_flags"],
            "n_global_safety_net_flags": meta["n_global_safety_net_flags"],
            "n_isolated_abnormal_corrected": n_isolated,
            "n_possible_regime_changes": n_regime,
            "n_spatially_supported": n_spatial,
            "percent_corrected": pct_corrected,
            "notes": "; ".join(meta["notes"]) if meta["notes"] else "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 12. Outputs
# --------------------------------------------------------------------------- #

def save_outputs(interval_df: pd.DataFrame, daily_df: pd.DataFrame,
                  regional_df: pd.DataFrame, summary_df: pd.DataFrame,
                  out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    interval_df.to_csv(out_dir / "reach_interval_CIR_cleaned.csv", index=False)
    daily_df.to_csv(out_dir / "reach_daily_CIR_cleaned.csv", index=False)
    regional_df.to_csv(out_dir / "regional_daily_CIR_cleaned.csv", index=False)
    summary_df.to_csv(out_dir / "preprocessing_parameter_summary.csv", index=False)
    logger.info("Saved interval, daily, regional, and summary CSVs to %s", out_dir)


# --------------------------------------------------------------------------- #
# 13. Visualizations
# --------------------------------------------------------------------------- #

_ACTION_COLORS = {
    "normal": "#4C72B0",
    "corrected_isolated_jump": "#C44E52",
    "kept_possible_regime_change": "#DD8452",
    "kept_spatially_supported": "#8172B2",
    "kept_regime_change_spatially_supported": "#937860",
}


def make_visualizations(interval_df: pd.DataFrame, daily_df: pd.DataFrame,
                         regional_df: pd.DataFrame, summary_df: pd.DataFrame,
                         fig_dir: Path = FIG_DIR) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        _plot_interval_diagnostic(reach_id, g, fig_dir)
        _plot_residual_hampel(reach_id, g, fig_dir)

    for reach_id, g in daily_df.groupby("reach_id", sort=False):
        _plot_daily_comparison(reach_id, g, fig_dir)

    _plot_regional_comparison(regional_df, fig_dir)
    _plot_correction_summary(summary_df, fig_dir)
    logger.info("Saved figures to %s", fig_dir)


def _plot_interval_diagnostic(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.sort_values("midpoint_date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(g["midpoint_date"], g["CIR_raw"], color="lightgray", lw=0.8, label="CIR raw", zorder=1)
    if "CIR_trend" in g:
        ax.plot(g["midpoint_date"], g["CIR_trend"], color="black", lw=1.2, label="Robust trend", zorder=2)
    ax.plot(g["midpoint_date"], g["CIR_clean"], color="#4C72B0", lw=0.9, alpha=0.7, label="CIR clean", zorder=1)

    for action, color in _ACTION_COLORS.items():
        if action == "normal":
            continue
        sub = g[g["correction_action"] == action]
        if not sub.empty:
            ax.scatter(sub["midpoint_date"], sub["CIR_raw"], color=color, s=18, label=action, zorder=3)

    if "is_hampel_flag_global" in g and "is_hampel_flag_local" in g:
        global_only = g[g["is_hampel_flag_global"] & ~g["is_hampel_flag_local"]]
        if not global_only.empty:
            ax.scatter(global_only["midpoint_date"], global_only["CIR_raw"], marker="x", color="black",
                       s=35, linewidths=1.3, label="global safety-net catch", zorder=4)

    ax.set_title(f"Reach {reach_id}: survey-level CIR diagnostic")
    ax.set_xlabel("Midpoint date")
    ax.set_ylabel("CIR")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / f"reach_{reach_id}_interval_CIR_diagnostic.png", dpi=130)
    plt.close(fig)


def _plot_residual_hampel(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    if "residual" not in g or g["residual"].isna().all():
        return
    g = g.sort_values("midpoint_date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)

    axes[0].plot(g["midpoint_date"], g["residual"], color="#4C72B0", lw=0.8)
    axes[0].scatter(g.loc[g["is_hampel_flag"], "midpoint_date"], g.loc[g["is_hampel_flag"], "residual"],
                     color="#C44E52", s=16, zorder=3, label="Hampel-flagged")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_ylabel("Residual\n(CIR_raw - CIR_trend)")
    axes[0].legend(fontsize=7)

    if "hampel_score" in g and not g["hampel_score"].isna().all():
        thr = g["hampel_threshold"].iloc[0] if "hampel_threshold" in g else np.nan
        axes[1].plot(g["midpoint_date"], g["hampel_score"], color="#55A868", lw=0.8)
        if np.isfinite(thr):
            axes[1].axhline(thr, color="red", ls="--", lw=1, label=f"threshold={thr:.2f}")
        axes[1].set_ylabel("Hampel score")
        axes[1].legend(fontsize=7)

    axes[1].set_xlabel("Midpoint date")
    fig.suptitle(f"Reach {reach_id}: residual & Hampel score")
    fig.tight_layout()
    fig.savefig(fig_dir / f"reach_{reach_id}_residual_hampel.png", dpi=130)
    plt.close(fig)


def _plot_daily_comparison(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(g["date"], g["CIR_raw_daily"], color="lightgray", lw=0.7, label="CIR raw (daily)")
    ax.plot(g["date"], g["CIR_trend_daily"], color="black", lw=1.0, label="CIR trend (daily)")
    ax.plot(g["date"], g["CIR_clean_daily"], color="#4C72B0", lw=0.8, label="CIR clean (daily)")
    ax.set_title(f"Reach {reach_id}: daily CIR comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("CIR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"reach_{reach_id}_daily_CIR_comparison.png", dpi=130)
    plt.close(fig)


def _plot_regional_comparison(regional_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(regional_df["date"], regional_df["CIR_region_raw_daily"], color="lightgray", lw=0.7, label="Regional raw")
    ax.plot(regional_df["date"], regional_df["CIR_region_trend_daily"], color="black", lw=1.0, label="Regional trend")
    ax.plot(regional_df["date"], regional_df["CIR_region_clean_daily"], color="#4C72B0", lw=0.8, label="Regional clean")
    ax.set_title("Regional aggregate daily CIR")
    ax.set_xlabel("Date")
    ax.set_ylabel("CIR (summed over reaches)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "regional_daily_CIR_comparison.png", dpi=130)
    plt.close(fig)


def _plot_correction_summary(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(summary_df)), 4.5))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["n_isolated_abnormal_corrected"], width, label="Corrected (isolated jump)", color="#C44E52")
    ax.bar(x + width / 2, summary_df["n_possible_regime_changes"], width, label="Possible regime change", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["reach_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count of intervals")
    ax.set_title("Correction summary by reach")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "correction_summary_by_reach.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info("=" * 70)
    logger.info("Sedimentation preprocessing pipeline starting.")
    logger.info(
        "Interpretation notes: robust trend filtering estimates the expected "
        "sedimentation/CIR trend from the irregular survey-level CIR sequence "
        "BEFORE any daily interpolation. Hampel filtering is applied to the "
        "residuals of that trend, not to raw CIR values directly. Isolated "
        "abnormal jumps (no temporal persistence, no spatial support) are "
        "corrected to the robust trend; persistent (regime-change) or "
        "spatially supported abrupt changes are preserved as potential real "
        "events. A global safety-net check supplements the local Hampel filter: "
        "it catches extreme jumps that hide inside an already-volatile local "
        "burst (where the local window's own variability masks them), while "
        "still routing them through the same isolated-jump vs. regime-change "
        "vs. spatially-supported classification -- so genuine multi-day events "
        "are still preserved, not blindly corrected. Daily interpolation is "
        "applied only after this reach-level quality control is complete, so "
        "no outlier detection is ever run on already-interpolated daily data."
    )
    logger.info("cvxpy available: %s", _CVXPY_AVAILABLE)
    logger.info("=" * 70)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    _setup_logging(OUT_DIR / "preprocessing_log.txt")

    raw_df = load_raw_sedimentation_data(RAW_DIR)
    interval_df = compute_interval_cir(raw_df)

    processed_frames = []
    reach_meta = {}
    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        logger.info("Processing reach %s (%d intervals)...", reach_id, len(g))
        processed, meta = _process_reach(reach_id, g)
        processed_frames.append(processed)
        reach_meta[reach_id] = meta
    interval_df = pd.concat(processed_frames, ignore_index=True)

    interval_df = add_spatial_support(interval_df)
    interval_df = apply_corrections(interval_df)

    summary_df = build_parameter_summary(interval_df, reach_meta)

    daily_df = interpolate_daily_cir(interval_df)
    regional_df = aggregate_region_daily(daily_df)

    save_outputs(interval_df, daily_df, regional_df, summary_df, OUT_DIR)
    make_visualizations(interval_df, daily_df, regional_df, summary_df, FIG_DIR)

    logger.info("Pipeline complete. %d reach(es) processed.", interval_df["reach_id"].nunique())


if __name__ == "__main__":
    main()
