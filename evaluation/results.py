# evaluation/results.py
#
# Persists run_main.py's predictions, metrics, and diagnostic figures to a
# results/ folder, mirroring the layout of the reference project's
# lstm_gsa_modified_cir / region_level_lstm output folders:
#   results/
#     metrics_summary.csv
#     predicted_CIR_<scope>.csv        (wide: rows=forecast start date, cols=day offset)
#     predicted_sediment_<scope>.csv   (same, back-calculated volume)
#     figures/
#       training_curve.png
#       <scope>_CIR_overview.png
#       <scope>_volume_overview.png
# where <scope> is a reach_id (per-reach mode) or the region_id (region mode).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import rmse, nrmse, mape
from inference.recursive_prediction import recursive_predict, sample_forecast_starts


def save_predicted_cir_csv(t_out_test, y_pred_test, scope_test, output_dir):
    """Per-scope (reach_id or region_id) wide CSV of direct test-window CIR
    predictions: rows = forecast start date, columns = day offset (0..N-1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for scope in sorted(set(scope_test)):
        mask = scope_test == scope
        df = pd.DataFrame(y_pred_test[mask], index=pd.DatetimeIndex(t_out_test[mask]))
        df.sort_index().to_csv(output_dir / f"predicted_CIR_{scope}.csv")


def save_predicted_sediment_csv(t_out_test, y_pred_test, scope_test, volume_by_reach, output_dir):
    """Per-scope wide CSV of back-calculated predicted sediment volume for
    each direct test window: anchor (real volume the day before the window
    starts) + cumsum(predicted CIR). Same row/column convention as
    save_predicted_cir_csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for scope in sorted(set(scope_test)):
        mask = scope_test == scope
        starts = pd.DatetimeIndex(t_out_test[mask])
        preds = y_pred_test[mask]

        rows, kept_index = [], []
        for start, pred_cir in zip(starts, preds):
            anchor_date = start - pd.Timedelta(days=1)
            anchor = volume_by_reach[scope].get(anchor_date, np.nan)
            if np.isnan(anchor):
                continue
            rows.append(anchor + np.cumsum(pred_cir))
            kept_index.append(start)

        if rows:
            pd.DataFrame(rows, index=kept_index).sort_index().to_csv(output_dir / f"predicted_sediment_{scope}.csv")


def save_metrics_summary_csv(rows, path):
    """rows: list of dicts, e.g.
    {"scope": reach_id, "target": "CIR"|"volume", "MAE":..., "RMSE":..., "nRMSE":..., "MAPE":..., "n_samples":...}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_metrics_rows(t_out_test, y_true_test, y_pred_test, scope_test, volume_by_reach):
    """Compute per-scope + overall CIR and back-calculated-volume metrics,
    as rows ready for save_metrics_summary_csv."""
    from evaluation.metrics import back_calculated_volume_errors

    rows = []
    scopes = sorted(set(scope_test))
    for scope in scopes + ["ALL"]:
        mask = np.ones(len(scope_test), dtype=bool) if scope == "ALL" else (scope_test == scope)

        yt, yp = y_true_test[mask].flatten(), y_pred_test[mask].flatten()
        rows.append({
            "scope": scope, "target": "CIR",
            "MAE": float(np.mean(np.abs(yt - yp))), "RMSE": rmse(yt, yp),
            "nRMSE": nrmse(yt, yp), "MAPE": mape(yt, yp),
            "n_samples": int(mask.sum()),
        })

        real_vol, pred_vol = back_calculated_volume_errors(
            scope_test[mask], t_out_test[mask], y_pred_test[mask], volume_by_reach
        )
        if len(real_vol):
            rows.append({
                "scope": scope, "target": "sediment_volume",
                "MAE": float(np.mean(np.abs(real_vol - pred_vol))), "RMSE": rmse(real_vol, pred_vol),
                "nRMSE": nrmse(real_vol, pred_vol), "MAPE": mape(real_vol, pred_vol),
                "n_samples": len(real_vol),
            })
    return rows


def plot_training_curve(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], color="#4C72B0", label="train loss")
    ax.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], color="#D55E00", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled units)")
    ax.set_title("Baseline LSTM training curves")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview_figures(
    scope, model, cir_by_reach, volume_by_reach, exo_shifted,
    input_scaler, output_scaler, t_out_test, reach_test,
    input_window_size, output_window_size, n_iterations, stride_days,
    device, fig_dir,
):
    """CIR and sediment-volume overview figures for one scope (reach_id or
    region_id): the real trend across the whole test period, with several
    recursive forecasts sampled at a regular stride overlaid on top --
    mirrors the reference project's <reach>_CIR_overview.png /
    <reach>_volume_overview.png.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    starts = sample_forecast_starts(t_out_test, reach_test, scope, stride_days)

    results = []
    for start in starts:
        try:
            r = recursive_predict(
                model, scope, start, cir_by_reach, exo_shifted, input_scaler, output_scaler,
                n_iterations=n_iterations, input_window_size=input_window_size,
                output_window_size=output_window_size, device=device,
            )
        except ValueError:
            continue
        if not r.empty:
            results.append(r)

    if not results:
        print(f"  [{scope}] no recursive forecasts available for overview figures -- skipping.")
        return

    reach_test_dates = t_out_test[reach_test == scope]
    full_range = pd.date_range(reach_test_dates.min(), reach_test_dates.max(), freq="D")

    # --- CIR overview ---
    real_cir = cir_by_reach[scope].reindex(full_range)
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(real_cir.index, real_cir, color="#B0B0B0", lw=0.8, label="real CIR (test region)", zorder=1)
    for i, r in enumerate(results):
        ax.plot(r.index, r["CIR_predicted"], color="#4C72B0", lw=1.2, alpha=0.9, zorder=2,
                 label="sampled recursive prediction" if i == 0 else None)
    ax.set_title(f"{scope}: real vs. sampled recursive CIR forecasts (test period)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scope}_CIR_overview.png", dpi=150)
    plt.close(fig)

    # --- Volume overview (back-calculated from each sampled forecast's own anchor) ---
    real_volume = volume_by_reach[scope].reindex(full_range)
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.plot(real_volume.index, real_volume, color="#B0B0B0", lw=0.8, label="real volume (test region)", zorder=1)
    for i, r in enumerate(results):
        anchor_date = r.index.min() - pd.Timedelta(days=1)
        anchor = volume_by_reach[scope].get(anchor_date, np.nan)
        if np.isnan(anchor):
            continue
        predicted_volume = anchor + r["CIR_predicted"].cumsum()
        ax.plot(predicted_volume.index, predicted_volume, color="#4C72B0", lw=1.2, alpha=0.9, zorder=2,
                 label="sampled back-calculated predicted volume" if i == 0 else None)
    ax.set_title(f"{scope}: real vs. sampled back-calculated predicted volume (test period)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scope}_volume_overview.png", dpi=150)
    plt.close(fig)
