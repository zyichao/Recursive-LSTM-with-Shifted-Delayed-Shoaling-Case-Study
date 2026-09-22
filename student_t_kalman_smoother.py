"""
student_t_kalman_smoother.py
=============================

Robust and trend-following CIR smoothing using a Student's t Kalman
smoother, following Aravkin, Burke, and Pillonetto (2014), "Robust and
Trend-Following Student's t Kalman Smoothers," SIAM J. Control Optim.
52(5), 2891-2916.

This is an ADDITIONAL, alternative smoothing method for the reach-level
survey-interval CIR sequence, benchmarked against the existing robust trend
filter in preprocess_sedimentation.py (Huber loss + L1 first/second-
difference penalties, solved via cvxpy). It does NOT modify
preprocess_sedimentation.py, its outputs, or any other existing file --
it only READS the already-generated reach_interval_CIR_cleaned.csv and
writes its own new outputs.

METHOD
------
State-space model -- a "local level" model (an integrated random walk,
the simplest member of the state-space family used in the paper's
Section 6.1 spline-reconstruction and Section 6.4-6.5 trend-tracking
experiments), applied to each reach's irregular survey-interval sequence:

    x_k = x_{k-1} + w_k      process:  the smooth CIR trend evolves as a
                              random walk, scaled by delta_days_k (the
                              actual number of days between survey k-1 and
                              k), since survey spacing is irregular
    z_k = x_k + v_k           measurement: the observed interval CIR_raw

Two of the paper's three named smoothers are combined into a "double-T"
(all-Student's-t) smoother (Section 4, equations 4.5-4.6), because our
data has both of the problems the paper's two special cases each target
individually:

  - T-ROBUST (measurement residual v_k modeled as Student's t): downweights
    an isolated bad/outlier CIR_raw reading -- playing the same role as the
    Huber loss in the existing pipeline's robust trend filter.
  - T-TREND (process residual w_k modeled as Student's t): lets the smooth
    state JUMP when the data supports a real, sudden shift, instead of
    forcing artificial smoothness through it -- playing the same role as
    the L1 first-difference (TV1) penalty in the existing pipeline.

IMPLEMENTATION NOTE: the paper's general framework handles nonlinear g_k,
h_k via an iterative Gauss-Newton scheme with a specialized block-
tridiagonal Hessian approximation (their Algorithm 5.1). Our state-space
model is LINEAR and SCALAR (g_k, h_k are both the identity), so that
general scheme reduces exactly to the classical IRLS (iteratively
reweighted least squares) simplification: repeatedly (a) run a standard
linear-Gaussian RTS Kalman smoother with the CURRENT per-step noise
variances, (b) recompute residuals from the new smoothed estimate, (c) turn
each residual into a new Student's t reweighting of that step's noise
variance (the paper's equations 4.2/4.4/4.6, specialized to a scalar
state), and repeat to convergence. This is the same MAP estimate as the
paper's general algorithm for this special linear case, just without
needing the general nonlinear machinery.

SECOND SMOOTHING TARGET -- VOLUME instead of CIR: each reach's raw SURVEY
VOLUME sequence (reconstructed from the interval table's
volume_start/volume_end) is ALSO smoothed directly, and CIR is then
obtained as a byproduct by finite-differencing the smoothed volume
(CIR_from_smoothed_volume = diff(V_smooth)/delta_days) -- no further
smoothing is applied to that derived CIR series. Rationale: CIR_raw is
itself already a first difference of volume, and differencing amplifies
noise (roughly doubles the noise variance and divides by dt^2); smoothing
the raw MEASURED quantity (volume) first, then differencing once, avoids
smoothing an already noise-amplified signal. It also lets the Student's t
measurement-weight correctly attribute a single bad volume reading to that
one survey point, rather than "smearing" it across the two adjacent CIR
intervals a bad volume reading corrupts.

This volume smoothing does NOT reuse the scalar local-level model: volume
has a genuine, often large, ongoing rate of change, not just noise around a
constant, so a local-level model (which assumes the state stays roughly
constant) under-smooths it -- an early version of this script tried exactly
that and produced a differenced CIR nearly as noisy as CIR_raw. Instead,
rts_smoother_local_linear_trend / student_t_kalman_smoother_volume implement
a 2-state "local linear trend" (level + rate) model, matching the paper's
own Section 6.1 spline-reconstruction example (position/velocity):
    V_k = V_{k-1} + R_{k-1}*dt_k + w1_k,  R_k = R_{k-1} + w2_k,  z_k = V_k + v_k
Both the original CIR-direct smoothing and this volume-based approach are
computed and saved side by side for comparison.

Run from the project root:
    python student_t_kalman_smoother.py

Outputs go to data/processed/student_t_smoothing/ (see save_outputs()).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
INTERVAL_CSV_PATH = BASE_DIR / "data" / "processed" / "sed_preprocessing" / "reach_interval_CIR_cleaned.csv"

OUT_DIR = BASE_DIR / "data" / "processed" / "student_t_smoothing"
FIG_DIR = OUT_DIR / "figures"

# Which residuals get Student's t treatment. "double_t" (both) is the
# default and combines the two problems ("T-robust" alone doesn't track
# real jumps well; "T-trend" alone doesn't downweight isolated bad readings).
SMOOTHER_MODE = "double_t"  # "double_t" | "t_robust" | "t_trend" | "gaussian"

# Degrees of freedom for the Student's t distributions. The source paper
# treats these as fixed, known constants in all of its experiments (dof=4
# throughout) and explicitly leaves automatic dof estimation to future work
# (Section 7); we follow that same convention here.
PROCESS_DOF = 4.0
MEASUREMENT_DOF = 4.0

# Automatic, per-reach noise-scale selection (no manually tuned per-reach
# constants): both Q0 (nominal process variance per day) and R0 (measurement
# variance) are derived from each reach's own MAD-based robust scale of
# CIR_raw, matching the auto-parameter philosophy already used throughout
# preprocess_sedimentation.py.
Q_TO_R_RATIO = 0.03  # nominal process std-dev, as a fraction of the measurement std-dev

MAX_IRLS_ITER = 25
IRLS_TOL = 1e-4

EXAMPLE_REACHES = ["CEMVN_SW_01_SWP_01", "CEMVN_SW_05_SWP_01", "CEMVN_SW_10_SWP_01"]
N_ZOOM_EVENTS_PER_REACH = 3
ZOOM_HALF_WINDOW_DAYS = 45

logger = logging.getLogger("student_t_kalman_smoother")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def select_noise_scales(cir_raw: np.ndarray, q_to_r_ratio: float = Q_TO_R_RATIO) -> tuple[float, float]:
    """Automatic per-reach (Q0, R0): R0 from the reach's own robust CIR_raw
    scale, Q0 as a smaller fraction of it (the smooth trend should drift
    less per day than the full raw measurement noise)."""
    sigma = _mad_sigma(cir_raw)
    if sigma < 1e-9:
        sigma = np.std(cir_raw) or 1.0
    R0 = sigma ** 2
    Q0 = (q_to_r_ratio * sigma) ** 2
    return Q0, R0


def select_volume_measurement_noise(cir_raw: np.ndarray, delta_days: np.ndarray) -> float:
    """Automatic measurement-noise variance (R0) for smoothing VOLUME
    directly -- derived from the reach's already-computed CIR_raw noise
    scale rather than picked independently.

    CIR_raw_k = (V_k - V_{k-1}) / dt_k. If V has i.i.d. measurement noise of
    variance sigma_v^2, the noise CONTRIBUTION to CIR_raw has variance
    2*sigma_v^2/dt_k^2 (differencing doubles the variance, then divides by
    dt^2). Inverting that relationship with the reach's typical (median)
    survey spacing gives sigma_v^2 = 0.5 * sigma_CIR^2 * median(dt)^2.

    Note this deliberately does NOT also derive a new Q0 for the volume
    model's rate component: that role is filled by the SAME Q0 already used
    to smooth CIR directly (see process_reach) -- both represent the same
    physical quantity, how fast the true infilling rate can drift per day,
    so reusing it keeps a single auto-tuned process-noise scale for the
    whole module rather than inventing a second, redundant one."""
    sigma_cir = _mad_sigma(cir_raw)
    if sigma_cir < 1e-9:
        sigma_cir = np.std(cir_raw) or 1.0
    positive_dt = delta_days[delta_days > 0]
    median_dt = np.median(positive_dt) if positive_dt.size else 1.0
    return 0.5 * sigma_cir ** 2 * median_dt ** 2


def build_survey_series(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the survey-level (date, volume) sequence from the
    interval table: n intervals -> n+1 surveys (the first interval's start,
    then every interval's end). delta_days_survey[0] is unused, mirroring
    the convention already used for Q_nominal[0] in
    student_t_kalman_smoother."""
    # Build via pd.concat (not np.concatenate) so the single leading Timestamp
    # scalar and the datetime64[ns] array merge cleanly -- np.concatenate can
    # silently promote one side to object/int64 depending on numpy version,
    # which then breaks the pd.to_datetime() call below ("mixed datetimes
    # and integers").
    dates = pd.concat(
        [pd.Series([g["start_date"].iloc[0]]), g["end_date"]], ignore_index=True
    ).to_numpy(dtype="datetime64[ns]")
    volumes = np.concatenate([[g["volume_start"].iloc[0]], g["volume_end"].to_numpy()]).astype(float)
    delta_days_survey = np.empty(len(volumes))
    delta_days_survey[0] = 0.0
    delta_days_survey[1:] = g["delta_days"].to_numpy(dtype=float)
    return pd.to_datetime(dates), volumes, delta_days_survey


# --------------------------------------------------------------------------- #
# Linear-Gaussian RTS Kalman smoother (the inner solver reused by IRLS)
# --------------------------------------------------------------------------- #

def rts_smoother_scalar(z: np.ndarray, Q: np.ndarray, R: np.ndarray, x0: float, P0: float) -> tuple[np.ndarray, np.ndarray]:
    """Standard Rauch-Tung-Striebel smoother for the scalar local-level
    model x_k = x_{k-1} + w_k, z_k = x_k + v_k, w_k ~ N(0, Q[k]),
    v_k ~ N(0, R[k]). Q[0] is unused (there is no k=0 -> k=-1 transition).
    """
    n = len(z)
    x_pred = np.empty(n)
    P_pred = np.empty(n)
    x_filt = np.empty(n)
    P_filt = np.empty(n)

    x_pred[0] = x0
    P_pred[0] = P0
    K0 = P_pred[0] / (P_pred[0] + R[0])
    x_filt[0] = x_pred[0] + K0 * (z[0] - x_pred[0])
    P_filt[0] = (1 - K0) * P_pred[0]

    for k in range(1, n):
        x_pred[k] = x_filt[k - 1]
        P_pred[k] = P_filt[k - 1] + Q[k]
        K = P_pred[k] / (P_pred[k] + R[k])
        x_filt[k] = x_pred[k] + K * (z[k] - x_pred[k])
        P_filt[k] = (1 - K) * P_pred[k]

    x_smooth = np.empty(n)
    P_smooth = np.empty(n)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for k in range(n - 2, -1, -1):
        C = P_filt[k] / P_pred[k + 1]
        x_smooth[k] = x_filt[k] + C * (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C ** 2 * (P_smooth[k + 1] - P_pred[k + 1])

    return x_smooth, P_smooth


# --------------------------------------------------------------------------- #
# Student's t Kalman smoother (IRLS wrapper around the RTS smoother)
# --------------------------------------------------------------------------- #

def student_t_kalman_smoother(z: np.ndarray, delta_days: np.ndarray, Q0: float, R0: float,
                                process_dof: float | None = PROCESS_DOF,
                                measurement_dof: float | None = MEASUREMENT_DOF,
                                max_iter: int = MAX_IRLS_ITER, tol: float = IRLS_TOL) -> dict:
    """MAP smoothing under the state-space model above, with either or both
    of the process/measurement residuals modeled as Student's t (passing
    `process_dof=None` or `measurement_dof=None` reduces that residual to
    plain Gaussian, recovering the "T-robust", "T-trend", or fully-Gaussian
    special cases as needed).

    Implements the IRLS reduction described in the module docstring: this
    is the same MAP estimate the paper's general Gauss-Newton algorithm
    would produce for this linear, scalar state-space model.

    Returns dict with x_smooth (the trend estimate), alpha (final process
    Student's t weights, 1.0 where Gaussian/undefined), beta (final
    measurement Student's t weights), and n_iter (IRLS iterations used).
    """
    n = len(z)
    Q_nominal = np.empty(n)
    Q_nominal[0] = Q0  # unused by the filter but kept for a well-defined array
    Q_nominal[1:] = Q0 * delta_days[1:]
    R_nominal = np.full(n, R0)

    Q = Q_nominal.copy()
    R = R_nominal.copy()
    x0 = z[0]
    P0 = 1e6 * R0  # diffuse prior: let the first measurement dominate the initial estimate

    alpha = np.ones(n)  # process weights (index k = weight on transition INTO k)
    beta = np.ones(n)   # measurement weights

    x_smooth_prev = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        x_smooth, _ = rts_smoother_scalar(z, Q, R, x0, P0)

        if measurement_dof is not None:
            v = z - x_smooth
            beta = measurement_dof / (measurement_dof + v ** 2 / R0)
            R = R_nominal / beta

        if process_dof is not None:
            w = np.empty(n)
            w[0] = 0.0
            w[1:] = x_smooth[1:] - x_smooth[:-1]
            alpha = np.ones(n)
            alpha[1:] = process_dof / (process_dof + w[1:] ** 2 / Q_nominal[1:])
            Q = Q_nominal / alpha

        if x_smooth_prev is not None:
            change = np.max(np.abs(x_smooth - x_smooth_prev)) / (np.max(np.abs(x_smooth_prev)) + 1e-9)
            if change < tol:
                x_smooth_prev = x_smooth
                break
        x_smooth_prev = x_smooth

    return dict(x_smooth=x_smooth_prev, alpha=alpha, beta=beta, n_iter=n_iter)


# --------------------------------------------------------------------------- #
# Local-linear-trend (2-state: level + rate) RTS smoother, for smoothing
# VOLUME directly and deriving CIR from it by differencing only.
# --------------------------------------------------------------------------- #
#
# A first attempt at "smooth volume directly" reused the scalar local-LEVEL
# model above with volume as the state (x_k = x_{k-1} + w_k). That model
# assumes the true state stays roughly CONSTANT except for small noise-scale
# drift -- a fine assumption for CIR (which hovers around a typical value),
# but wrong for volume, which has a genuine, often large, ongoing rate of
# change. Empirically, that first attempt produced a differenced CIR nearly
# as noisy as CIR_raw itself: with Q0 calibrated to a small MEASUREMENT-noise
# scale, the filter could only track the real trend by injecting noise-sized
# jumps into the level every step, which then show up as noise once
# differenced. The fix is a genuine LOCAL LINEAR TREND ("integrated random
# walk") model with the rate as its own explicit second state component --
# exactly the structure of the paper's own Section 6.1 spline-reconstruction
# example (position/velocity), applied here to volume/CIR-rate:
#
#     V_k = V_{k-1} + R_{k-1} * dt_k + w1_k     (volume integrates the rate)
#     R_k = R_{k-1} + w2_k                       (the rate itself drifts)
#     z_k = V_k + v_k                            (measurement: observed volume)
#
# with w_k = [w1_k, w2_k] ~ N(0, Q_k), the standard integrated-Wiener-process
# covariance Q_k = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]] for a scalar
# intensity q, and v_k ~ N(0, R_k).

def _llt_transition(dt: float) -> np.ndarray:
    return np.array([[1.0, dt], [0.0, 1.0]])


def _llt_process_cov(dt: float, q: float) -> np.ndarray:
    return q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])


def rts_smoother_local_linear_trend(z: np.ndarray, delta_days: np.ndarray, Q: np.ndarray, R: np.ndarray,
                                      x0: np.ndarray, P0: np.ndarray) -> np.ndarray:
    """RTS smoother for the 2-state local linear trend model described
    above. Q[k] (shape (2,2)) and R[k] (scalar) are the PER-STEP noise
    covariances (already reweighted by the caller's IRLS loop, if
    applicable); Q[0] is unused. Returns x_smooth, shape (n, 2), columns
    [level, rate]."""
    n = len(z)
    H = np.array([1.0, 0.0])

    x_pred = np.empty((n, 2))
    P_pred = np.empty((n, 2, 2))
    x_filt = np.empty((n, 2))
    P_filt = np.empty((n, 2, 2))

    x_pred[0] = x0
    P_pred[0] = P0
    S = P_pred[0][0, 0] + R[0]
    K = P_pred[0] @ H / S
    x_filt[0] = x_pred[0] + K * (z[0] - x_pred[0][0])
    P_filt[0] = P_pred[0] - np.outer(K, H @ P_pred[0])

    for k in range(1, n):
        Gk = _llt_transition(delta_days[k])
        x_pred[k] = Gk @ x_filt[k - 1]
        P_pred[k] = Gk @ P_filt[k - 1] @ Gk.T + Q[k]
        S = P_pred[k][0, 0] + R[k]
        K = P_pred[k] @ H / S
        x_filt[k] = x_pred[k] + K * (z[k] - x_pred[k][0])
        P_filt[k] = P_pred[k] - np.outer(K, H @ P_pred[k])

    x_smooth = np.empty((n, 2))
    P_smooth = np.empty((n, 2, 2))
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for k in range(n - 2, -1, -1):
        Gk1 = _llt_transition(delta_days[k + 1])
        C = P_filt[k] @ Gk1.T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

    return x_smooth


def student_t_kalman_smoother_volume(z: np.ndarray, delta_days: np.ndarray, q0: float, r0: float,
                                       process_dof: float | None = PROCESS_DOF,
                                       measurement_dof: float | None = MEASUREMENT_DOF,
                                       max_iter: int = MAX_IRLS_ITER, tol: float = IRLS_TOL,
                                       rate0: float | None = None) -> dict:
    """IRLS Student's t wrapper around rts_smoother_local_linear_trend, the
    volume-smoothing analogue of student_t_kalman_smoother. q0 is the same
    process-noise intensity already used for CIR-direct smoothing (see
    process_reach); r0 is the volume measurement-noise variance from
    select_volume_measurement_noise.

    The measurement residual reweighting (beta) is identical to the scalar
    case. The process residual is now a 2-vector w_k = x_smooth_k -
    G_k @ x_smooth_{k-1}; it is reweighted via its Mahalanobis distance under
    the nominal Q_k (d_k^2 = w_k^T Q_k^-1 w_k), using the same
    process_dof / (process_dof + d^2) form as the scalar case -- a single
    shared scalar weight applied to the whole Q_k matrix, which is exactly
    how a multivariate Student's t scale-mixture reweights a shared scale
    parameter, so this is the natural multivariate generalization of the
    scalar formula rather than an ad hoc simplification.

    rate0 (the initial rate/CIR guess) should be supplied by the caller as a
    ROBUST estimate (e.g. a median over the first few raw CIR values), not
    derived here from a single first difference: a single difference is
    exactly as vulnerable to an isolated bad first reading as the
    quantity this whole module exists to be robust to, and with a diffuse
    P0 on the rate a single bad rate0 leaks into the first several smoothed
    points before being corrected. Defaults to the single-difference
    fallback only if the caller doesn't supply one.
    """
    n = len(z)
    R_nominal = np.full(n, r0)
    Q_nominal = np.empty((n, 2, 2))
    Q_nominal[0] = np.eye(2)  # unused by the filter, kept well-defined
    for k in range(1, n):
        Q_nominal[k] = _llt_process_cov(delta_days[k], q0)

    Q = Q_nominal.copy()
    R = R_nominal.copy()

    if rate0 is None:
        rate0 = (z[1] - z[0]) / delta_days[1] if n > 1 and delta_days[1] > 0 else 0.0
    x0 = np.array([z[0], rate0])
    P0 = np.diag([1e6 * r0, 1e6 * q0])

    alpha = np.ones(n)
    beta = np.ones(n)
    x_smooth_prev = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        x_smooth = rts_smoother_local_linear_trend(z, delta_days, Q, R, x0, P0)

        if measurement_dof is not None:
            v = z - x_smooth[:, 0]
            beta = measurement_dof / (measurement_dof + v ** 2 / r0)
            R = R_nominal / beta

        if process_dof is not None:
            d2 = np.zeros(n)
            for k in range(1, n):
                Gk = _llt_transition(delta_days[k])
                w = x_smooth[k] - Gk @ x_smooth[k - 1]
                d2[k] = w @ np.linalg.solve(Q_nominal[k], w)
            alpha = np.ones(n)
            alpha[1:] = process_dof / (process_dof + d2[1:])
            for k in range(1, n):
                Q[k] = Q_nominal[k] / alpha[k]

        if x_smooth_prev is not None:
            change = (np.max(np.abs(x_smooth[:, 0] - x_smooth_prev[:, 0]))
                      / (np.max(np.abs(x_smooth_prev[:, 0])) + 1e-9))
            if change < tol:
                x_smooth_prev = x_smooth
                break
        x_smooth_prev = x_smooth

    return dict(x_smooth=x_smooth_prev, alpha=alpha, beta=beta, n_iter=n_iter)


# --------------------------------------------------------------------------- #
# Per-reach processing
# --------------------------------------------------------------------------- #

def process_reach(reach_id: str, g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("midpoint_date").reset_index(drop=True)
    z = g["CIR_raw"].to_numpy(dtype=float)
    delta_days = g["delta_days"].to_numpy(dtype=float)

    process_dof = PROCESS_DOF if SMOOTHER_MODE in ("double_t", "t_trend") else None
    measurement_dof = MEASUREMENT_DOF if SMOOTHER_MODE in ("double_t", "t_robust") else None

    # ---- (A) Original approach: Student's t smoothing applied directly to
    # CIR_raw. Kept for comparison against the volume-based approach below. ----
    Q0, R0 = select_noise_scales(z)
    result_cir = student_t_kalman_smoother(z, delta_days, Q0, R0, process_dof, measurement_dof)
    g["CIR_trend_studentt"] = result_cir["x_smooth"]
    g["studentt_process_weight"] = result_cir["alpha"]
    g["studentt_measurement_weight"] = result_cir["beta"]

    # ---- (B) New approach: smooth VOLUME with a local-linear-trend (level +
    # rate) Kalman smoother, derive CIR by differencing only. q0 reuses the
    # SAME Q0 as the CIR-direct smoothing above (see student_t_kalman_smoother_volume
    # docstring); only the measurement-noise scale r0 is newly derived. ----
    survey_dates, survey_volumes, survey_dt = build_survey_series(g)
    R0_vol = select_volume_measurement_noise(z, delta_days)
    rate0 = float(np.median(z[: min(5, len(z))]))
    result_vol = student_t_kalman_smoother_volume(
        survey_volumes, survey_dt, Q0, R0_vol, process_dof, measurement_dof, rate0=rate0,
    )
    V_smooth = result_vol["x_smooth"][:, 0]

    g["V_smooth_start"] = V_smooth[:-1]
    g["V_smooth_end"] = V_smooth[1:]
    g["CIR_from_smoothed_volume"] = (V_smooth[1:] - V_smooth[:-1]) / delta_days
    g["volume_studentt_process_weight"] = result_vol["alpha"][1:]
    g["volume_studentt_measurement_weight"] = result_vol["beta"][1:]

    # ---- Volume trend implied by the EXISTING Huber CIR_trend, integrated
    # forward from a single real anchor (the first survey). This is a FREE-
    # RUNNING (compounding) integration, not the re-anchored-at-corrections
    # approach used in reconstruct_sedimentation_volume.py -- it is included
    # here deliberately AS a free-running reference, to make any long-run
    # compounding drift visible in the volume comparison plot rather than
    # hidden. See plot_volume_comparison's legend/label. ----
    anchor = float(g["volume_start"].iloc[0])
    V_huber = np.empty(len(g) + 1)
    V_huber[0] = anchor
    V_huber[1:] = anchor + np.cumsum(g["CIR_trend"].to_numpy(dtype=float) * delta_days)
    g["V_huber_trend_start"] = V_huber[:-1]
    g["V_huber_trend_end"] = V_huber[1:]

    logger.info(
        "Reach %s: [CIR-direct] Q0=%.3g R0=%.3g iters=%d meanW=(proc=%.3f,meas=%.3f) | "
        "[volume] Q0=%.3g R0_vol=%.3g iters=%d meanW=(proc=%.3f,meas=%.3f)",
        reach_id, Q0, R0, result_cir["n_iter"], result_cir["alpha"].mean(), result_cir["beta"].mean(),
        Q0, R0_vol, result_vol["n_iter"], result_vol["alpha"][1:].mean(), result_vol["beta"][1:].mean(),
    )
    return g


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #

RAW_COLOR = "#B0B0B0"
HUBER_COLOR = "#4C72B0"
STUDENTT_COLOR = "#C44E52"
VOLUME_DERIVED_COLOR = "#55A868"


def plot_full_comparison(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(g["midpoint_date"], g["CIR_raw"], color=RAW_COLOR, lw=0.5, label="CIR raw", zorder=1)
    ax.plot(g["midpoint_date"], g["CIR_trend"], color=HUBER_COLOR, lw=1.0, label="CIR trend (Huber+TV1+TV2)", zorder=2)
    ax.plot(g["midpoint_date"], g["CIR_trend_studentt"], color=STUDENTT_COLOR, lw=1.0, label="CIR trend (Student's t Kalman, direct on CIR)", zorder=3)
    ax.plot(g["midpoint_date"], g["CIR_from_smoothed_volume"], color=VOLUME_DERIVED_COLOR, lw=1.0, label="CIR (from Student's t smoothed volume)", zorder=4)
    ax.set_title(f"{reach_id}: trend comparison across the full record", fontsize=10)
    ax.set_xlabel("Midpoint date")
    ax.set_ylabel("CIR")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"full_comparison_{reach_id}.png", dpi=200)
    plt.close(fig)


def _survey_arrays_from_group(g: pd.DataFrame) -> dict:
    """Rebuild the survey-indexed (n+1 point) volume series -- raw, Huber-
    trend-integrated, and Student's t smoothed -- for the volume plots."""
    # See the matching note in build_survey_series: pd.concat (not
    # np.concatenate) avoids a numpy-version-dependent dtype promotion that
    # breaks pd.to_datetime() below.
    dates = pd.concat(
        [pd.Series([g["start_date"].iloc[0]]), g["end_date"]], ignore_index=True
    ).to_numpy(dtype="datetime64[ns]")
    raw = np.concatenate([[g["volume_start"].iloc[0]], g["volume_end"].to_numpy()]).astype(float)
    smooth = np.concatenate([[g["V_smooth_start"].iloc[0]], g["V_smooth_end"].to_numpy()]).astype(float)
    huber = np.concatenate([[g["V_huber_trend_start"].iloc[0]], g["V_huber_trend_end"].to_numpy()]).astype(float)
    return dict(dates=pd.to_datetime(dates), raw=raw, smooth=smooth, huber=huber)


def plot_volume_comparison(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    s = _survey_arrays_from_group(g)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(s["dates"], s["raw"], color=RAW_COLOR, lw=0.6, marker=".", ms=3, label="Volume (raw survey)", zorder=1)
    ax.plot(s["dates"], s["huber"], color=HUBER_COLOR, lw=1.2, label="Volume (Huber CIR-trend, free-running integration)", zorder=2)
    ax.plot(s["dates"], s["smooth"], color=VOLUME_DERIVED_COLOR, lw=1.2, label="Volume (Student's t Kalman, smoothed directly)", zorder=3)
    ax.set_title(f"{reach_id}: sediment volume trend comparison across the full record", fontsize=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"volume_comparison_{reach_id}.png", dpi=200)
    plt.close(fig)


def plot_volume_zoomed_comparison(reach_id: str, g: pd.DataFrame, center: pd.Timestamp, fig_dir: Path) -> None:
    s = _survey_arrays_from_group(g)
    lo, hi = center - pd.Timedelta(days=ZOOM_HALF_WINDOW_DAYS), center + pd.Timedelta(days=ZOOM_HALF_WINDOW_DAYS)
    mask = (s["dates"] >= lo) & (s["dates"] <= hi)
    if not mask.any():
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s["dates"][mask], s["raw"][mask], color=RAW_COLOR, lw=1.0, marker="o", ms=4, label="Volume (raw survey)", zorder=1)
    ax.plot(s["dates"][mask], s["huber"][mask], color=HUBER_COLOR, lw=1.6, label="Volume (Huber CIR-trend, free-running integration)", zorder=2)
    ax.plot(s["dates"][mask], s["smooth"][mask], color=VOLUME_DERIVED_COLOR, lw=1.6, label="Volume (Student's t Kalman, smoothed directly)", zorder=3)
    ax.axvline(center, color="gray", lw=0.7, ls=":")
    ax.set_title(f"{reach_id}: zoomed volume comparison around {center.date()}", fontsize=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"volume_zoom_{reach_id}_{center.date()}.png", dpi=200)
    plt.close(fig)


def find_zoom_events(g: pd.DataFrame, n_events: int = N_ZOOM_EVENTS_PER_REACH, min_gap_days: int = 90) -> list[pd.Timestamp]:
    """Pick the dates of the largest |CIR_raw| spikes, spaced apart so the
    same event isn't picked more than once."""
    remaining = g[["midpoint_date", "CIR_raw"]].copy()
    remaining["abs_cir"] = remaining["CIR_raw"].abs()
    centers = []
    for _ in range(n_events):
        if remaining.empty:
            break
        row = remaining.loc[remaining["abs_cir"].idxmax()]
        center = row["midpoint_date"]
        centers.append(center)
        lo, hi = center - pd.Timedelta(days=min_gap_days), center + pd.Timedelta(days=min_gap_days)
        remaining = remaining[(remaining["midpoint_date"] < lo) | (remaining["midpoint_date"] > hi)]
    return centers


def plot_zoomed_comparison(reach_id: str, g: pd.DataFrame, center: pd.Timestamp, fig_dir: Path) -> None:
    lo = center - pd.Timedelta(days=ZOOM_HALF_WINDOW_DAYS)
    hi = center + pd.Timedelta(days=ZOOM_HALF_WINDOW_DAYS)
    window = g[(g["midpoint_date"] >= lo) & (g["midpoint_date"] <= hi)]
    if window.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(window["midpoint_date"], window["CIR_raw"], color=RAW_COLOR, lw=1.0, label="CIR raw", zorder=1)
    ax.plot(window["midpoint_date"], window["CIR_trend"], color=HUBER_COLOR, lw=1.6, label="CIR trend (Huber+TV1+TV2)", zorder=2)
    ax.plot(window["midpoint_date"], window["CIR_trend_studentt"], color=STUDENTT_COLOR, lw=1.6, label="CIR trend (Student's t Kalman, direct on CIR)", zorder=3)
    ax.plot(window["midpoint_date"], window["CIR_from_smoothed_volume"], color=VOLUME_DERIVED_COLOR, lw=1.6, label="CIR (from Student's t smoothed volume)", zorder=4)
    ax.axvline(center, color="gray", lw=0.7, ls=":")
    ax.set_title(f"{reach_id}: zoomed comparison around {center.date()}", fontsize=10)
    ax.set_xlabel("Midpoint date")
    ax.set_ylabel("CIR")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"zoom_{reach_id}_{center.date()}.png", dpi=200)
    plt.close(fig)


def plot_weights_diagnostic(reach_id: str, g: pd.DataFrame, fig_dir: Path) -> None:
    """Shows which points the Student's t smoother is downweighting (low
    measurement weight = treated as an outlier) or allowing a jump at
    (low process weight = a real, sudden shift was accommodated)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    axes[0].plot(g["midpoint_date"], g["studentt_measurement_weight"], color=STUDENTT_COLOR, lw=0.8)
    axes[0].set_ylabel("measurement weight\n(low = outlier)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(g["midpoint_date"], g["studentt_process_weight"], color=HUBER_COLOR, lw=0.8)
    axes[1].set_ylabel("process weight\n(low = real jump)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Midpoint date")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{reach_id}: Student's t IRLS weights", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"weights_{reach_id}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Daily interpolation (for downstream use, e.g. the ML pipeline), matching
# preprocess_sedimentation.py's interpolate_daily_cir() convention: a daily
# grid spanning each reach's own record, time-interpolated from the
# irregular interval/survey dates.
# --------------------------------------------------------------------------- #

def build_daily_studentt_series(result_df: pd.DataFrame) -> pd.DataFrame:
    """Per-reach daily table with the volume-derived CIR (interval-indexed by
    midpoint_date) and the directly-smoothed volume (survey-indexed, i.e. one
    point per actual survey date) both time-interpolated onto the same daily
    grid, so this is a drop-in daily replacement for
    reach_daily_CIR_cleaned.csv's CIR_clean_daily / reach_daily_volume_
    reconstructed.csv's volume_original. CIR_raw is carried along too (as
    CIR_raw_daily, matching reach_daily_CIR_cleaned.csv's column of the same
    name) purely for reference/diagnostic plotting against the new series."""
    daily_frames = []
    for reach_id, g in result_df.groupby("reach_id", sort=False):
        g = g.sort_values("midpoint_date")
        if len(g) < 2:
            continue

        cir_idx = pd.DatetimeIndex(g["midpoint_date"])
        cir_series = pd.Series(g["CIR_from_smoothed_volume"].to_numpy(), index=cir_idx)
        cir_series = cir_series[~cir_series.index.duplicated(keep="first")]
        cir_raw_series = pd.Series(g["CIR_raw"].to_numpy(), index=cir_idx)
        cir_raw_series = cir_raw_series[~cir_raw_series.index.duplicated(keep="first")]

        survey_dates, survey_volumes, _ = build_survey_series(g)
        vol_series = pd.Series(survey_volumes, index=pd.DatetimeIndex(survey_dates))
        vol_series = vol_series[~vol_series.index.duplicated(keep="first")]

        daily_index = pd.date_range(cir_idx.min().normalize(), cir_idx.max().normalize(), freq="D")

        def _to_daily(s: pd.Series) -> pd.Series:
            combined_index = s.index.union(daily_index)
            return s.reindex(combined_index).interpolate(method="time").reindex(daily_index)

        daily_frames.append(pd.DataFrame({
            "date": daily_index,
            "reach_id": reach_id,
            "CIR_studentt_daily": _to_daily(cir_series).to_numpy(),
            "CIR_raw_daily": _to_daily(cir_raw_series).to_numpy(),
            "volume_studentt_daily": _to_daily(vol_series).to_numpy(),
        }))

    return pd.concat(daily_frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 70)
    logger.info("Student's t Kalman smoother (SMOOTHER_MODE=%s) starting.", SMOOTHER_MODE)
    logger.info(
        "This benchmarks an alternative to preprocess_sedimentation.py's Huber+TV1+TV2 "
        "robust trend filter. Both aim to be robust to outliers AND track genuine sudden "
        "shifts, via different mathematical frameworks (convex penalized least squares vs. "
        "Student's t MAP estimation / IRLS Kalman smoothing)."
    )
    logger.info("=" * 70)


def main() -> None:
    _setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INTERVAL_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{INTERVAL_CSV_PATH} not found. Run preprocess_sedimentation.py first."
        )
    interval_df = pd.read_csv(INTERVAL_CSV_PATH, parse_dates=["start_date", "end_date", "midpoint_date"])

    processed = []
    for reach_id, g in interval_df.groupby("reach_id", sort=False):
        processed.append(process_reach(reach_id, g))
    result_df = pd.concat(processed, ignore_index=True)

    keep_cols = [
        "reach_id", "start_date", "end_date", "midpoint_date", "delta_days",
        "volume_start", "volume_end",
        "CIR_raw", "CIR_trend", "CIR_trend_studentt", "CIR_from_smoothed_volume",
        "studentt_process_weight", "studentt_measurement_weight",
        "V_smooth_start", "V_smooth_end", "V_huber_trend_start", "V_huber_trend_end",
        "volume_studentt_process_weight", "volume_studentt_measurement_weight",
    ]
    result_df[keep_cols].to_csv(OUT_DIR / "reach_interval_CIR_studentt.csv", index=False)
    logger.info("Saved %s", OUT_DIR / "reach_interval_CIR_studentt.csv")

    daily_df = build_daily_studentt_series(result_df)
    daily_df.to_csv(OUT_DIR / "reach_daily_studentt.csv", index=False)
    logger.info("Saved %s", OUT_DIR / "reach_daily_studentt.csv")

    for reach_id in EXAMPLE_REACHES:
        g = result_df[result_df["reach_id"] == reach_id]
        if g.empty:
            logger.warning("Example reach %s not found in data; skipping its figures.", reach_id)
            continue
        plot_full_comparison(reach_id, g, FIG_DIR)
        plot_weights_diagnostic(reach_id, g, FIG_DIR)
        plot_volume_comparison(reach_id, g, FIG_DIR)
        for center in find_zoom_events(g):
            plot_zoomed_comparison(reach_id, g, center, FIG_DIR)
            plot_volume_zoomed_comparison(reach_id, g, center, FIG_DIR)

    logger.info("Saved figures to %s", FIG_DIR)
    logger.info("Done. %d reach(es) processed.", result_df["reach_id"].nunique())


if __name__ == "__main__":
    main()
