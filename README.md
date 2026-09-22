# CIR Reach Forecasting

**Multi-reach Channel Infilling Rate (CIR) forecasting with a baseline LSTM and recursive multi-step inference**

---

## Reference

This repository is the code accompanying:

> Zeng, Y., Ammar, D., Chadha, M., Wang, D., Asborno, M., Miele, S., McKnight,
> C.J., Memarsadeghi, N.P., Hartman, M.A., Mitchell, K.N., Gugaratshan, G.,
> Todd, M.D., & Hu, Z. (2026). **A recursive shoaling forecasting framework
> with shifted delays and its application to the Southwest Pass in the
> United States.** *Machine Learning with Applications*, 25, 100968.
> https://doi.org/10.1016/j.mlwa.2026.100968

If you use this code, please cite the paper above. This repository
reproduces the core recursive-LSTM / shifted-delay / Student's-t-smoothing
pipeline as runnable, modular code; a few of the paper's other components
(live GSA computation, ARIMA/RF/MLP baselines, full hyperparameter search,
SHAP explainability, and the HSC verification study) are not included here
— see §5 for what each implemented stage does and how it maps to the
paper's equations.

---

## 1. Overview

This project forecasts reach-level **Channel Infilling Rate (CIR)** — the
daily rate of change of sediment volume — for the 13 Southwest Pass (SWP)
reaches, using each reach's own CIR history plus a handful of GSA-selected,
lag-shifted exogenous discharge features. Sediment volume is recovered by
cumulative-summing predicted CIR from a real volume anchor.

It reproduces, as runnable modular code, the workflow from raw survey data
through model training, evaluation, and recursive inference, at a glance:

```
raw survey volume  →  QC (quality control: flag/correct bad readings) + Student's-t Kalman smoothing  →  daily CIR/volume
        →  GSA-shifted exogenous features  →  windowed LSTM  →  recursive forecast
```

**Student's-t Kalman smoothing is the only smoothing technique used to
produce the final CIR/volume target series** — see §5 for how each stage
works.

---

## 2. Project structure

```
cir_reach_forecasting/
├── preprocess_sedimentation.py     # Upstream: raw survey volume -> QC'd interval CIR
├── student_t_kalman_smoother.py    # Upstream: Student's-t Kalman smoothing -> daily CIR/volume
├── config/config.yaml              # All experiment settings
│
├── data/                           # Raw + processed data -- see §3
├── preprocessing/                  # Data loading, GSA shifting, windowing, regional aggregation
├── models/lstm.py                  # BaselineLSTM
├── training/train_loop.py          # Training loop
├── inference/recursive_prediction.py
├── evaluation/                     # Metrics + results-saving
│
├── saved_models/                   # Trained weights (generated)
├── results/                        # Predictions, metrics, figures (generated) -- see §7
│
├── run_main.py                     # End-to-end entry point
├── requirements.txt
└── README.md
```

---

## 3. Data

| File | Meaning | Index |
|------|---------|-------|
| `data/raw_data/sed/SWP/<reach>.csv` | Raw per-reach survey volume (date, cumulative volume), irregular survey spacing | date |
| `data/processed/student_t_smoothing/reach_daily_studentt.csv` | Per-reach daily CIR (Student's-t Kalman-smoothed target `CIR_studentt_daily`, raw `CIR_raw_daily`) and smoothed volume (`volume_studentt_daily`) | `date`, `reach_id` |
| `data/processed/exo_input/preprocessed_input_0620_discharge.csv` | Candidate exogenous discharge/gage-height/salinity features, daily, contiguous | date |
| `data/processed/exo_input/gsa_top_0620.csv` | Two rows per candidate feature: `max_value` (peak GSA sensitivity, pre-sorted descending) and `row_number` (lag in days at which it peaks) | feature name (columns) |

Only the raw survey CSVs plus the two `exo_input` files are strictly
required; everything under `data/processed/sed_preprocessing/`,
`data/processed/student_t_smoothing/`, `data/processed/ml_input/`, and
`saved_models/` is regenerable by running the pipeline (a
`reach_daily_studentt.csv` is shipped pre-generated so `run_main.py` can
skip straight to the ML stage — see §4/§6).

**Note:** the exogenous discharge record covers a narrower span than the CIR
record, so it — not the CIR series — is the effective upper bound on usable
data range and on the test set's end date.

---

## 4. Configuration (`config/config.yaml`)

- **`paths`** — input CSVs and output directories (models, windowed tensors/scalers).
- **`upstream_preprocessing`** — `run_if_missing` (default `true`): regenerate
  `cir_daily` from raw survey data via `preprocess_sedimentation.py` +
  `student_t_kalman_smoother.py` only if it doesn't already exist;
  `force_rerun` (default `false`): always regenerate, even if it exists.
- **`columns`** — `cir_target` (`CIR_studentt_daily`) and `volume_target` (`volume_studentt_daily`).
- **`region`** — `enabled` (default `false`): when `true`, sum `reach_ids`
  (empty = all reaches) into one regional-total CIR/volume series
  (`region_id`, default `"SWP_ALL"`) and forecast that single cumulated
  series instead of each reach individually — see §5's "Regional
  aggregation" note.
- **`windows`** — `input_window_size` (30 days), `output_window_size` (10 days,
  independent of input length), `batch_size`.
- **`features.top_n_exo_features`** — how many discharge stations to keep (5), ranked by `gsa_top`'s `max_value`.
- **`leading_trim`** — MAD multiplier / consecutive-day threshold for trimming each reach's leading data artifact.
- **`split.test_start_date`** — a sample is a **test** sample iff its OUTPUT window starts on/after this date.
- **`model.lstm`** — `hidden_size` (64), `num_layers` (3), `dropout` (0.2).
- **`training`** — learning rate, weight decay, epochs, batch size, validation
  fraction (most recent 10% of training dates), early-stopping patience, seed.
- **`recursive_inference`** — `n_iterations` (3) → total forecast horizon = `n_iterations × output_window_size` days.

---

## 5. Method details

### Upstream QC (`preprocess_sedimentation.py`)
Raw survey volume readings are irregular in time and contain occasional bad
readings and abrupt dredging-driven jumps. This stage: computes interval
CIR between consecutive surveys → fits a robust trend on that irregular
sequence (Huber loss + L1 first/second-difference penalties, via `cvxpy` if
installed, else an equivalent `scipy`-based fallback) purely to obtain
residuals → flags abnormal jumps in those residuals with an adaptive Hampel
filter (temporal-persistence and spatial-consistency checks reclassify
"kept" vs. "corrected" flags) → corrects only isolated jumps → interpolates
the cleaned, irregular interval series to a 1-day grid. **This trend filter
is QC/outlier-correction machinery only** — it does not produce any of the
series used as the ML target; that's the next stage's job.

### Student's-t Kalman smoothing (`student_t_kalman_smoother.py`) — the only smoothing technique
Reads the QC'd interval table and reconstructs each reach's raw survey
volume sequence, then smooths it with a Student's-t ("double-T") Kalman/RTS
smoother (Aravkin, Burke & Pillonetto 2014) — a 2-state local-linear-trend
model (level + rate) whose measurement AND process residuals are both
modeled as Student's-t, solved via IRLS-reweighted linear-Gaussian RTS
smoothing to convergence. This is the paper's **Stage 1** (Eq. 3–4 in
Zeng et al. 2026): the raw sediment-volume series is replaced by the
smoothed latent volume trajectory, and CIR is then obtained by
finite-differencing that smoothed volume (Eq. 5) rather than re-smoothing
an already noise-amplified derivative.

**Simplification vs. the paper:** the paper's **Stage 2** (Eq. 6–10) runs a
*second*, separate Student's-t Kalman smoother directly on the differenced
CIR series and *selectively* replaces only the ~1–3% of points whose
standardized residual against that second smoother exceeds a MAD-based
threshold `τ_R` — leaving the other ~97–99% of days as the raw
Stage-1-derived CIR. This repository's `CIR_studentt_daily` is the Stage-1
output only (the full finite-differenced smoothed-volume series, with no
selective Stage-2 correction pass applied). In practice this still removes
essentially all large fluctuations (Stage 1 already down-weights isolated
volume outliers), but it is not a byte-for-byte reproduction of the paper's
two-stage selective procedure.

The result — `CIR_studentt_daily` / `volume_studentt_daily` in
`reach_daily_studentt.csv` — is the **only** smoothed series this project
uses downstream; no other smoothing method (Savitzky-Golay, the QC stage's
Huber+L1 trend, etc.) feeds the ML pipeline.

`preprocessing/upstream.py`'s `ensure_studentt_cir_daily()` runs both
scripts (in order, as subprocesses from the project root) only if
`reach_daily_studentt.csv` doesn't already exist, unless `force_rerun` is
set — see §4.

### Regional aggregation (`preprocessing/regional.py`)
Set `region.enabled: true` in `config.yaml` to forecast the SWP region's
**cumulated** (summed-across-reaches) sediment volume directly, rather than
each of the 13 reaches individually. `aggregate_region()` sums the listed
reaches' `CIR_studentt_daily` and `volume_studentt_daily` on dates common to
all of them (summing rates ≡ differencing the summed volume) into one
`region_id`-labeled series, which then flows through windowing, training,
evaluation, and recursive inference completely unchanged — the rest of the
pipeline just sees one "reach" instead of 13. Note the aggregated CIR series
is far noisier and more zero-crossing than any single reach's (13 independent
signals summed), so expect worse raw CIR MAPE/nRMSE than the per-reach runs;
the more informative comparison is the back-calculated regional sediment
volume RMSE.

### Leading-artifact trimming (`preprocessing/data_loading.py`)
Every reach's raw record starts with a NaN day immediately followed by a
huge one-off jump (zero-padding before real monitoring began). A robust,
MAD-based scan finds the first run of `consecutive_required` in-range,
non-NaN values and trims everything before it — avoiding both the initial
spike and being fooled by a small "normal-looking" value sandwiched between
two large spikes.

### Exogenous feature selection (`preprocessing/exogenous.py`)
`gsa_top_0620.csv` columns are pre-sorted by descending `max_value`, so the
top-N features are simply its first N columns. Each selected feature is
shifted **forward** by its own `row_number` (lag, in days) — a station whose
influence on sedimentation peaks 20 days later has its value moved 20 days
ahead so peak influence lines up with the current date. This mirrors the
paper's Global Sensitivity Analysis (GSA) procedure (§3.4.2–3.4.3, Eq.
16–33): a modularized first-order Sobol-index estimator (Li & Mahadevan,
2016) is evaluated per candidate discharge station across a scanned range
of uniform shift days (0–183/0–365 in the paper), aggregated over all
input/output window positions, and each station's optimal shift is the
`arg max` over that scan (Eq. 33).

**Simplification vs. the paper:** this repository does **not** include the
GSA computation itself — `gsa_top_0620.csv` (`max_value`, `row_number` per
station) is a precomputed ranking table consumed as-is by
`select_and_shift_exogenous_features()`. The paper's live GSA scan (Fig. 11,
Table 3) is not reproduced as runnable code here.

### Windowing (`preprocessing/rolling_windows.py`)
Per reach: `CIR_prev` (autoregressive) + shifted exogenous features, inner-joined
by date and NaN-dropped. Rolling `input_window_size`-day windows map to the
immediately-following `output_window_size`-day window (no gap); a window is
kept only if every date inside it is exactly one calendar day after the
previous one. Windows are built independently per reach, then concatenated.
Train/test split is chronological by the sample's **output**-window start
date; `MinMaxScaler` for both input and target is fit on the training split
only.

### Model (`models/lstm.py`)
`BaselineLSTM`: stacked LSTM → ReLU over the full output sequence → Dropout
→ take the last time step → Linear → all `output_window_size` future CIR
values at once (direct multi-output, no per-step teacher forcing). This is
the paper's Multi-Input Multi-Output (MIMO) LSTM-NARX architecture (§3.4.1,
Fig. 4–5, Eq. 12–15): 3 stacked LSTM layers with dropout between them,
feeding a dense output layer that produces the whole `Wpred`-length forecast
in one pass. The default `config.yaml` hyperparameters (`hidden_size: 64`,
`num_layers: 3`, `dropout: 0.2`, `learning_rate: 0.0001`, `n_epochs: 300`,
Adam optimizer) match the paper's grid-search optimum (§4.2.2) for those
settings; `training.batch_size` (64 here) differs from the paper's optimum
of 16, and `windows.input_window_size` (30 here) differs from the paper's
optimum of 60 (`Wpast`) — both are configurable in `config.yaml` if you want
to match the published setup exactly.

### Training (`training/train_loop.py`, `training/losses.py`)
Adam, chronological train/validation split (validation = most recent
`val_fraction` of training dates), early stopping on validation loss with
best weights restored at the end.

**Loss function:** by default, training uses the paper's custom
zero-crossing-weighted MSE (§3.5, Eq. 36–37): each sample's target window
is flagged as a zero crossing if it contains both a positive and a negative
CIR value (a sedimentation trend reversal — a local peak or valley), and
weighted `w_high` instead of `w_low` in the loss. `config.yaml`'s `loss`
block ships with the paper's grid-search optimum (`w_high: 10`, `w_low: 1`
— Table 9); set `loss.weighted_zero_crossing: false` to fall back to plain,
unweighted MSE. The zero-crossing flags are computed once from the raw
(unscaled) target windows in `preprocessing/rolling_windows.py`'s
`compute_zero_crossing_flags()` and threaded through training and
validation batches alongside `X`/`y`.

### Evaluation (`evaluation/metrics.py`)
- **RMSE** in native units.
- **nRMSE** — RMSE normalized by the true-value range, for cross-reach/period comparability.
- **MAPE** — denominator floored at 1% of the true values' std, since CIR
  crosses zero constantly and plain MAPE is unstable near zero.
- **`back_calculated_volume_errors`** — integrates predicted CIR forward from
  the real volume the day before the window starts, and compares to the real
  volume over the same window (the more physically meaningful metric).

### Recursive inference (`inference/recursive_prediction.py`)
`recursive_predict` extends the forecast beyond `output_window_size` days by
keeping a rolling `input_window_size`-day buffer of CIR — real history at
first, increasingly backfilled with the model's own predictions — advanced by
`output_window_size` days each iteration. The exogenous features fed each
iteration are always the **real** (already-shifted) values, since discharge
is an external driver the model doesn't need to forecast. Stops early if the
real exogenous record runs out before the requested horizon. This
implements the paper's recursive prediction scheme (§3.6.2, Fig. 8, Eq.
39–44); the default `recursive_inference.n_iterations: 3` with
`windows.output_window_size: 10` reproduces the paper's optimal **3 × 10-day**
configuration (30-day total horizon, §4.2.2).

---

## 6. Running an experiment

### Step 1 — Environment

```bash
python -m venv cir_env
source cir_env/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

Runs on Apple-Silicon **MPS** or **CUDA** if available, otherwise CPU. On an
M-series Mac, confirm MPS is actually active before a long training run --
some older/conda-distributed `torch` builds report CPU-only even when the
hardware supports it:
```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```
If either prints `False`, `pip install --upgrade torch` (see `requirements.txt`).
The difference is dramatic: in testing, MPS trained the full 300-epoch
baseline in ~25 minutes versus an estimated ~15 hours on a CPU-only `torch`
build on the same machine.

### Step 2 — Configure

Edit `config/config.yaml` if you want different window sizes, feature counts,
model hyperparameters, or the test-set cutoff date.

### Step 3 — Run

```bash
python run_main.py
```

A pre-generated `reach_daily_studentt.csv` ships with the repo, so by
default this skips straight to the ML stage (fast). To regenerate it from
the raw survey data instead — e.g. after editing the raw CSVs, or to pick up
a `cvxpy` install for exact QC fidelity — either delete
`data/processed/student_t_smoothing/reach_daily_studentt.csv` first, or set
`upstream_preprocessing.force_rerun: true` in `config.yaml`. Regenerating
runs `preprocess_sedimentation.py` then `student_t_kalman_smoother.py`
end-to-end (a few minutes for all 13 reaches) and also produces their own
diagnostic CSVs/figures under `data/processed/sed_preprocessing/` and
`data/processed/student_t_smoothing/`.

### Outputs
- `data/processed/sed_preprocessing/`, `data/processed/student_t_smoothing/` — upstream QC/smoothing
  artifacts (only regenerated if triggered — see Step 3)
- `data/processed/ml_input/cir_windowed_tensors.pt`, `cir_scalers.joblib`, `cir_sample_metadata.csv`
- `saved_models/baseline_lstm.pt` — weights + config + metadata (input size, feature order, window sizes)
- `results/` — predictions, metrics, and figures; see §7 below
- console: selected exogenous features + lags, per-reach window counts, train/test split sizes,
  per-epoch training/validation loss, test-set CIR and back-calculated sediment-volume metrics
  (RMSE / nRMSE / MAPE), and recursive-inference metrics for a few example reach/start-date pairs

---

## 7. Results (`evaluation/results.py`, `results/`)

Every run of `run_main.py` writes its predictions, metrics, and diagnostic
figures to `results/`, mirroring the layout of the reference project's
`lstm_gsa_modified_cir` / `region_level_lstm` output folders. `<scope>` below
is a reach_id in per-reach mode, or the single `region_id` in region mode
(§4/§5).

```
results/
├── metrics_summary.csv                       # one row per (scope, target) pair -- see below
├── predicted_CIR_<scope>.csv                 # wide: rows=forecast start date, cols=day offset 0..N-1
├── predicted_sediment_<scope>.csv            # same, back-calculated volume (anchor + cumsum(predicted CIR))
└── figures/
    ├── training_curve.png                    # train/val loss per epoch
    ├── <scope>_CIR_overview.png              # real CIR (whole test period) + several sampled recursive forecasts
    └── <scope>_volume_overview.png           # same, back-calculated sediment volume
```

- **`predicted_CIR_<scope>.csv` / `predicted_sediment_<scope>.csv`** — one row
  per **direct**, single-window test-set prediction (i.e. the model's native
  `output_window_size`-day forecast, not the extended recursive one), for
  every reach with test samples (or the region, in region mode).
- **`metrics_summary.csv`** — for every scope **and** an aggregate `"ALL"`
  row, both CIR-level and back-calculated-sediment-volume metrics
  (MAE/RMSE/nRMSE/MAPE + sample count) on the direct test-window predictions.
- **`training_curve.png`** — one plot for the run's model (all reaches share
  one pooled model, so there's a single curve, not one per scope).
- **`<scope>_CIR_overview.png` / `<scope>_volume_overview.png`** — generated
  for up to 3 example scopes (`REACH_IDS[:3]`, or the one region), each
  showing the real CIR/volume trend across the whole test period with
  several **recursive** (`recursive_inference.n_iterations`-length) forecasts
  sampled at a regular stride (`recursive_inference.overview_stride_multiplier
  × output_window_size` days) overlaid — mirrors the reference project's
  whole-test-region overview plots, not just cherry-picked examples.

> This is a **baseline**: modest hidden size and default hyperparameters, not
> a tuned final model. Natural next steps: hyperparameter search over
> `config.yaml`'s `model`/`training` blocks, a CNN-LSTM architecture, more
> exogenous features, or per-reach vs. pooled training.
