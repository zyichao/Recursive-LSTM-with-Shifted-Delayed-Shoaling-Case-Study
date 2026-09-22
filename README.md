# CIR Reach Forecasting

**Multi-reach Channel Infilling Rate (CIR) forecasting with a baseline LSTM and recursive multi-step inference**

---

## Reference

Code accompanying:

> Zeng, Y., Ammar, D., Chadha, M., Wang, D., Asborno, M., Miele, S., McKnight,
> C.J., Memarsadeghi, N.P., Hartman, M.A., Mitchell, K.N., Gugaratshan, G.,
> Todd, M.D., & Hu, Z. (2026). **A recursive shoaling forecasting framework
> with shifted delays and its application to the Southwest Pass in the
> United States.** *Machine Learning with Applications*, 25, 100968.
> https://doi.org/10.1016/j.mlwa.2026.100968

If you use this code, please cite the paper above. This repo reproduces the
core recursive-LSTM / shifted-delay / Student's-t-smoothing pipeline; a few
of the paper's other components (live GSA computation, ARIMA/RF/MLP
baselines, full hyperparameter search, SHAP explainability, the HSC study)
aren't included.

---

## 1. Overview

Forecasts reach-level **Channel Infilling Rate (CIR)** — the daily rate of
change of sediment volume — for the 13 Southwest Pass (SWP) reaches, using
each reach's own CIR history plus GSA-selected, lag-shifted exogenous
discharge features. Sediment volume is recovered by cumulative-summing
predicted CIR from a real volume anchor.

```
raw survey volume  →  QC + Student's-t Kalman smoothing  →  daily CIR/volume
        →  GSA-shifted exogenous features  →  windowed LSTM  →  recursive forecast
```

Student's-t Kalman smoothing is the only smoothing technique used to
produce the final CIR/volume target series.

---

## 2. Project structure

```
cir_reach_forecasting/
├── preprocess_sedimentation.py     # Raw survey volume -> QC'd interval CIR
├── student_t_kalman_smoother.py    # Student's-t Kalman smoothing -> daily CIR/volume
├── config/config.yaml              # All experiment settings
│
├── data/                           # Raw + processed data -- see §3
├── preprocessing/                  # Data loading, GSA shifting, windowing, regional aggregation
├── models/lstm.py                  # BaselineLSTM
├── training/                       # Training loop + custom loss function
├── inference/recursive_prediction.py
├── evaluation/                     # Metrics + results-saving
│
├── saved_models/                   # Trained weights (generated)
├── results/                        # Predictions, metrics, figures (generated) -- see §6
│
├── run_main.py                     # End-to-end entry point
├── requirements.txt
└── README.md
```

---

## 3. Data

| File | Meaning |
|------|---------|
| `data/raw_data/sed/SWP/<reach>.csv` | Raw per-reach survey volume |
| `data/processed/student_t_smoothing/reach_daily_studentt.csv` | Per-reach daily CIR + smoothed volume (the ML target) |
| `data/processed/exo_input/preprocessed_input_0620_discharge.csv` | Candidate exogenous discharge/gage/salinity features |
| `data/processed/exo_input/gsa_top_0620.csv` | Precomputed GSA ranking: `max_value` + optimal lag (`row_number`) per feature |

Only the raw survey CSVs and the two `exo_input` files are strictly
required — everything else is regenerable by running the pipeline (a
`reach_daily_studentt.csv` ships pre-generated so `run_main.py` can skip
straight to the ML stage).

---

## 4. Configuration (`config/config.yaml`)

- **`upstream_preprocessing`** — regenerate the daily CIR/volume table from raw data if missing (`run_if_missing`) or always (`force_rerun`).
- **`region`** — set `enabled: true` to sum reaches into one regional-total series instead of forecasting each reach individually.
- **`windows`** — `input_window_size` (30d), `output_window_size` (10d), `batch_size`.
- **`features.top_n_exo_features`** — how many discharge stations to keep, ranked by GSA.
- **`split.test_start_date`** — a sample is test iff its output window starts on/after this date.
- **`model.lstm`** — `hidden_size`, `num_layers`, `dropout`.
- **`training`** — learning rate, epochs, batch size, validation fraction, early-stopping patience.
- **`loss`** — `weighted_zero_crossing` (the paper's custom loss, on by default), `w_low`, `w_high`.
- **`recursive_inference`** — `n_iterations` × `output_window_size` = total forecast horizon.

---

## 5. Running an experiment

```bash
python -m venv cir_env
source cir_env/bin/activate
pip install -r requirements.txt
python run_main.py
```

Runs on Apple-Silicon MPS or CUDA if available, otherwise CPU — on an
M-series Mac, verify MPS is actually active before a long run (some
`torch` builds silently fall back to CPU):
```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```
If either prints `False`, `pip install --upgrade torch`.

**Outputs:** `saved_models/baseline_lstm.pt` (trained weights), `results/`
(predictions, metrics, figures — see §6), and console logs of the training
progress and test-set metrics.

---

## 6. Results (`results/`)

Every run writes to `results/`, mirroring the reference project's output
layout. `<scope>` is a reach_id (per-reach mode) or the region_id (region
mode).

```
results/
├── metrics_summary.csv               # MAE/RMSE/nRMSE/MAPE per scope + an "ALL" aggregate row
├── predicted_CIR_<scope>.csv         # direct test-window CIR predictions
├── predicted_sediment_<scope>.csv    # back-calculated sediment volume
└── figures/
    ├── training_curve.png
    ├── <scope>_CIR_overview.png      # real CIR + sampled recursive forecasts, whole test period
    └── <scope>_volume_overview.png   # same, for sediment volume
```

> This is a **baseline**: modest hidden size and default hyperparameters,
> not a tuned final model. Natural next steps: hyperparameter search, a
> CNN-LSTM architecture, more exogenous features, or per-reach vs. pooled
> training.
