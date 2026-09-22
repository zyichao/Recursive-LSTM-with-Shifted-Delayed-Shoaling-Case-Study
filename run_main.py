# ======================================================
# Main runner: multi-reach CIR forecasting
# Baseline LSTM (direct multi-output) + recursive multi-step inference
#
# Duplicates the workflow of ml_pipeline_data_ingestion.ipynb (0703_2026)
# as a runnable script. Run from this project's root directory.
# ======================================================

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from preprocessing.upstream import ensure_studentt_cir_daily
from preprocessing.data_loading import load_cir_and_volume
from preprocessing.regional import aggregate_region
from preprocessing.exogenous import select_and_shift_exogenous_features
from preprocessing.rolling_windows import (
    build_windowed_dataset,
    chronological_train_test_split,
    chronological_train_val_split,
    scale_and_tensorize,
    make_loader,
)
from models.lstm import BaselineLSTM
from training.train_loop import train_baseline_lstm
from training.losses import build_loss_fn
from evaluation.metrics import rmse, nrmse, mape, back_calculated_volume_errors
from evaluation.results import (
    save_predicted_cir_csv,
    save_predicted_sediment_csv,
    build_metrics_rows,
    save_metrics_summary_csv,
    plot_training_curve,
    plot_overview_figures,
)
from inference.recursive_prediction import recursive_predict, sample_forecast_starts

# ======================================================
# 1. Load configuration
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parent

with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CIR_TARGET_COLUMN = cfg["columns"]["cir_target"]
VOLUME_TARGET_COLUMN = cfg["columns"]["volume_target"]
INPUT_WINDOW_SIZE = cfg["windows"]["input_window_size"]
OUTPUT_WINDOW_SIZE = cfg["windows"]["output_window_size"]
TOP_N_EXO_FEATURES = cfg["features"]["top_n_exo_features"]
TEST_START_DATE = pd.Timestamp(cfg["split"]["test_start_date"])

ML_OUTPUT_DIR = PROJECT_ROOT / cfg["paths"]["outputs"]["ml_input"]
ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / cfg["paths"]["outputs"]["models"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["outputs"]["results"]
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. Reproducibility & device
# ======================================================
seed = cfg["training"]["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Using device:", device)

# ======================================================
# 2.5. Upstream preprocessing: raw survey volume -> QC'd interval CIR
#      (preprocess_sedimentation.py) -> Student's-t Kalman-smoothed daily
#      CIR/volume (student_t_kalman_smoother.py). Regenerated only if the
#      target CSV is missing, unless force_rerun is set.
# ======================================================
CIR_DAILY_PATH = PROJECT_ROOT / cfg["paths"]["data"]["cir_daily"]
ensure_studentt_cir_daily(
    PROJECT_ROOT, CIR_DAILY_PATH,
    run_if_missing=cfg["upstream_preprocessing"]["run_if_missing"],
    force_rerun=cfg["upstream_preprocessing"]["force_rerun"],
)

# ======================================================
# 3. Load CIR / volume (per reach, leading-artifact trimmed)
# ======================================================
cir_by_reach, volume_by_reach, REACH_IDS = load_cir_and_volume(
    CIR_DAILY_PATH,
    cir_target_column=CIR_TARGET_COLUMN,
    volume_target_column=VOLUME_TARGET_COLUMN,
    mad_multiplier=cfg["leading_trim"]["mad_multiplier"],
    consecutive_required=cfg["leading_trim"]["consecutive_required"],
)
print(f"{len(REACH_IDS)} reaches: {REACH_IDS}")

# ======================================================
# 3.5. Optional regional aggregation: collapse the listed reaches into ONE
#      regional-total CIR/volume series (summed across reaches on their
#      common dates) so the rest of the pipeline forecasts the SWP region's
#      cumulated volume directly, instead of each reach individually.
# ======================================================
if cfg["region"]["enabled"]:
    region_reach_ids = cfg["region"]["reach_ids"] or REACH_IDS
    cir_by_reach, volume_by_reach, REACH_IDS = aggregate_region(
        cir_by_reach, volume_by_reach, region_reach_ids,
        region_id=cfg["region"]["region_id"],
    )

# ======================================================
# 4. Load exogenous discharge input + GSA feature ranking, shift by lag
# ======================================================
exo_raw = pd.read_csv(PROJECT_ROOT / cfg["paths"]["data"]["exo_input"], index_col=0, parse_dates=True).sort_index()
gsa_top = pd.read_csv(PROJECT_ROOT / cfg["paths"]["data"]["gsa_top"], index_col=0)

exo_shifted, top_feature_map = select_and_shift_exogenous_features(exo_raw, gsa_top, TOP_N_EXO_FEATURES)
print(f"Selected top {len(top_feature_map)} exogenous feature(s) and their forward shift (days):")
for feat, shift in top_feature_map.items():
    print(f"  {feat}: +{shift} days")

# ======================================================
# 5. Build rolling-window dataset (per reach, concatenated)
# ======================================================
X_all, y_all, reach_id_all, t_input_all, t_output_all, FEATURE_COLUMNS = build_windowed_dataset(
    REACH_IDS, cir_by_reach, exo_shifted, INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE
)
print("\nConcatenated across all reaches:")
print("X_all:", X_all.shape, " y_all:", y_all.shape)
print("Feature order:", FEATURE_COLUMNS)

# ======================================================
# 6. Train / test split (chronological, by OUTPUT window start)
# ======================================================
split = chronological_train_test_split(X_all, y_all, reach_id_all, t_input_all, t_output_all, TEST_START_DATE)
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
reach_train, reach_test = split["reach_train"], split["reach_test"]
t_out_train, t_out_test = split["t_out_train"], split["t_out_test"]
zc_train = split["zc_train"]

print(f"Train: {X_train.shape[0]:6d} samples  ({t_out_train.min().date()} to {t_out_train.max().date()})")
print(f"Test:  {X_test.shape[0]:6d} samples  ({t_out_test.min().date()} to {t_out_test.max().date()})")

# ======================================================
# 7. Scale (fit on TRAIN only) + tensorize
# ======================================================
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_scaler, output_scaler = scale_and_tensorize(
    X_train, y_train, X_test, y_test
)
n_features = X_train.shape[-1]

# ======================================================
# 8. Save ingestion artifacts (tensors, scalers, sample metadata)
# ======================================================
torch.save({
    "X_train": X_train_tensor, "y_train": y_train_tensor,
    "X_test": X_test_tensor, "y_test": y_test_tensor,
    "feature_columns": FEATURE_COLUMNS,
    "input_window_size": INPUT_WINDOW_SIZE,
    "output_window_size": OUTPUT_WINDOW_SIZE,
}, ML_OUTPUT_DIR / "cir_windowed_tensors.pt")

joblib.dump(
    {"input_scaler": input_scaler, "output_scaler": output_scaler},
    ML_OUTPUT_DIR / "cir_scalers.joblib",
)

pd.DataFrame({
    "reach_id": reach_id_all,
    "t_input_start": t_input_all,
    "t_output_start": t_output_all,
    "is_test": split["is_test"],
}).to_csv(ML_OUTPUT_DIR / "cir_sample_metadata.csv", index=False)

print("Saved tensors, scalers, and sample metadata to", ML_OUTPUT_DIR)

# ======================================================
# 9. Train / validation split (most recent val_fraction of TRAIN dates)
# ======================================================
X_tr, y_tr, zc_tr, X_val, y_val, zc_val = chronological_train_val_split(
    X_train_tensor, y_train_tensor, zc_train, t_out_train, cfg["training"]["val_fraction"]
)
print(f"Zero-crossing windows: {zc_tr.mean() * 100:.1f}% of train, {zc_val.mean() * 100:.1f}% of val")

batch_size = cfg["training"]["batch_size"]
train_loader = make_loader(X_tr, y_tr, zc_tr, batch_size=batch_size, shuffle=True)
val_loader = make_loader(X_val, y_val, zc_val, batch_size=batch_size, shuffle=False)

# ======================================================
# 10. Train the baseline LSTM
# ======================================================
model = BaselineLSTM(
    input_size=n_features,
    hidden_size=cfg["model"]["lstm"]["hidden_size"],
    output_size=OUTPUT_WINDOW_SIZE,
    num_layers=cfg["model"]["lstm"]["num_layers"],
    dropout=cfg["model"]["lstm"]["dropout"],
).to(device)

loss_fn = build_loss_fn(cfg["loss"])
print(f"Loss function: {type(loss_fn).__name__}")

model, history, best_val_loss = train_baseline_lstm(
    model, train_loader, val_loader, device,
    learning_rate=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
    n_epochs=cfg["training"]["n_epochs"],
    early_stopping_patience=cfg["training"]["early_stopping_patience"],
    loss_fn=loss_fn,
)

# ======================================================
# 11. Save the trained model
# ======================================================
MODEL_PATH = MODEL_DIR / "baseline_lstm.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": cfg,
    "input_size": n_features,
    "output_size": OUTPUT_WINDOW_SIZE,
    "feature_columns": FEATURE_COLUMNS,
    "input_window_size": INPUT_WINDOW_SIZE,
    "output_window_size": OUTPUT_WINDOW_SIZE,
    "best_val_loss": best_val_loss,
}, MODEL_PATH)
print("Saved trained model to", MODEL_PATH)

# ======================================================
# 12. Test-set metrics: back-calculated sediment volume + CIR (direct window)
# ======================================================
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor.to(device)).cpu().numpy()
y_test_scaled = y_test_tensor.numpy()

y_pred_test = output_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).reshape(y_pred_test_scaled.shape)
y_true_test = output_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

real_vol_test, pred_vol_test = back_calculated_volume_errors(reach_test, t_out_test, y_pred_test, volume_by_reach)
print(f"\nTest set -- back-calculated SEDIMENT VOLUME vs. real volume "
      f"(single {OUTPUT_WINDOW_SIZE}-day window, n={len(real_vol_test)} samples):")
print(f"  RMSE:  {rmse(real_vol_test, pred_vol_test):,.1f}")
print(f"  nRMSE: {nrmse(real_vol_test, pred_vol_test):.4f}")
print(f"  MAPE:  {mape(real_vol_test, pred_vol_test):.1f}%  (see CIR zero-crossing caveat)")

overall_rmse = rmse(y_true_test.flatten(), y_pred_test.flatten())
overall_nrmse = nrmse(y_true_test.flatten(), y_pred_test.flatten())
overall_mape = mape(y_true_test.flatten(), y_pred_test.flatten())
print(f"\n(For reference) Test set -- CIR directly (single {OUTPUT_WINDOW_SIZE}-day window, n={len(y_true_test)} samples):")
print(f"  RMSE:  {overall_rmse:,.1f}")
print(f"  nRMSE: {overall_nrmse:.4f}")
print(f"  MAPE:  {overall_mape:.1f}%")

# ======================================================
# 13. Recursive multi-step inference (a few example reach/start-date pairs)
# ======================================================
N_INFERENCE_ITERATIONS = cfg["recursive_inference"]["n_iterations"]
recursive_examples = [(reach_id, TEST_START_DATE.date().isoformat()) for reach_id in REACH_IDS[:3]]

print(f"\nRecursive inference ({N_INFERENCE_ITERATIONS} iterations, "
      f"{N_INFERENCE_ITERATIONS * OUTPUT_WINDOW_SIZE}-day horizon):")
for reach_id, start_date in recursive_examples:
    result = recursive_predict(
        model, reach_id, start_date,
        cir_by_reach, exo_shifted, input_scaler, output_scaler,
        n_iterations=N_INFERENCE_ITERATIONS,
        input_window_size=INPUT_WINDOW_SIZE, output_window_size=OUTPUT_WINDOW_SIZE,
        device=device,
    )
    valid = result.dropna()
    if not len(valid):
        continue
    real_vol, pred_vol = back_calculated_volume_errors(
        [reach_id], [start_date], [valid["CIR_predicted"].to_numpy()], volume_by_reach
    )
    print(f"{reach_id} from {start_date} -- {len(result)}-day forecast:")
    if len(real_vol):
        print(f"  SEDIMENT VOLUME: RMSE={rmse(real_vol, pred_vol):,.1f}  "
              f"nRMSE={nrmse(real_vol, pred_vol):.4f}  MAPE={mape(real_vol, pred_vol):.1f}%")
    print(f"  CIR (reference): RMSE={rmse(valid['CIR_actual'].to_numpy(), valid['CIR_predicted'].to_numpy()):,.1f}  "
          f"nRMSE={nrmse(valid['CIR_actual'].to_numpy(), valid['CIR_predicted'].to_numpy()):.4f}")

# ======================================================
# 14. Persist results: predicted CIR / sediment volume CSVs, metrics
#      summary, training curve, and per-scope CIR/volume overview figures.
# ======================================================
print(f"\nSaving results to {RESULTS_DIR} ...")

save_predicted_cir_csv(t_out_test, y_pred_test, reach_test, RESULTS_DIR)
save_predicted_sediment_csv(t_out_test, y_pred_test, reach_test, volume_by_reach, RESULTS_DIR)

metrics_rows = build_metrics_rows(t_out_test, y_true_test, y_pred_test, reach_test, volume_by_reach)
save_metrics_summary_csv(metrics_rows, RESULTS_DIR / "metrics_summary.csv")

plot_training_curve(history, FIGURES_DIR / "training_curve.png")

overview_stride_days = cfg["recursive_inference"]["overview_stride_multiplier"] * OUTPUT_WINDOW_SIZE
for scope in REACH_IDS[:3]:
    plot_overview_figures(
        scope, model, cir_by_reach, volume_by_reach, exo_shifted,
        input_scaler, output_scaler, t_out_test, reach_test,
        INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE, N_INFERENCE_ITERATIONS, overview_stride_days,
        device, FIGURES_DIR,
    )

print("Saved:")
print(f"  {RESULTS_DIR}/metrics_summary.csv")
print(f"  {RESULTS_DIR}/predicted_CIR_<scope>.csv, predicted_sediment_<scope>.csv")
print(f"  {FIGURES_DIR}/training_curve.png, <scope>_CIR_overview.png, <scope>_volume_overview.png")

print("\nDone.")
