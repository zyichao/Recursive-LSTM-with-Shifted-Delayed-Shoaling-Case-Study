# Recursive LSTM with Shifted Delays for Shoaling Forecasting

This repository contains the code associated with the manuscript:

**A Recursive Shoaling Forecasting Framework with Shifted Delays and Its Application to the Southwest Pass in the United States**

The study develops a machine-learning framework for forecasting channel infilling rate (CIR) and reconstructing sediment volume for navigation-channel shoaling prediction. The framework integrates:

- Two-stage Student-t Kalman preprocessing for sediment-volume and CIR target construction
- Global Sensitivity Analysis (GSA)-based feature selection
- Station-specific shifted-delay input construction for discharge variables
- Recursive multi-step LSTM forecasting
- A custom weighted loss function for CIR transition periods
- Baseline model comparison and visualization scripts

## Repository Status

This repository is currently being prepared for public release. The work was supported by the United States Army Corps of Engineers through the U.S. Army Engineer Research and Development Center Research Cooperative Agreement **W912HZ24C0044**. Public release of the complete code package is pending the required review and approval process.

At this stage, the repository may contain placeholder files, documentation, or partial code examples. The complete implementation will be uploaded after approval is received.

## Project Overview

Shoaling in navigation channels can reduce available depth and disrupt maritime transportation. Accurate short-term forecasting of sediment accumulation is important for dredging planning, fleet scheduling, and sediment-management decisions.

This project proposes a recursive LSTM-based forecasting framework that predicts CIR over a 30-day horizon. Instead of directly forecasting cumulative sediment volume, the model forecasts the rate of sediment accumulation or erosion, which is then used to reconstruct future sediment volume. The framework further accounts for delayed hydrologic effects by applying GSA-selected shifted delays to upstream discharge variables.

## Main Workflow

The overall workflow includes the following steps:

1. **Target preprocessing**
   - Raw sediment-volume records are smoothed using a Student-t Kalman smoother.
   - CIR is computed from the smoothed sediment-volume series.
   - Isolated abnormal CIR spikes are selectively corrected using a second Student-t Kalman smoother.

2. **Input preprocessing**
   - Hydrologic and environmental variables are collected from USGS and NOAA sources.
   - Input variables are cleaned, resampled, and aligned with the CIR target.
   - Rate-of-change forms of exogenous variables are considered for consistency with the CIR target.

3. **GSA-based feature selection**
   - Global Sensitivity Analysis is used to identify important exogenous variables.
   - For discharge variables, candidate shifted days are evaluated to capture delayed hydrologic effects.
   - Non-discharge variables are not shifted in the current implementation.

4. **Recursive LSTM forecasting**
   - The model predicts CIR using a recursive multi-step strategy.
   - A 30-day prediction horizon is divided into shorter prediction blocks.
   - Predicted CIR values are used to reconstruct future sediment volume.

5. **Model evaluation**
   - Performance is evaluated using RMSE, nRMSE, and MAPE.
   - Baseline models include ARIMA, Random Forest, MLP, and baseline LSTM models.

## Planned Repository Structure

The final released repository is expected to follow the structure below:

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── preprocessing/
│   │   ├── sediment_volume_processing.py
│   │   ├── cir_processing.py
│   │   └── student_t_kalman_smoother.py
│   ├── feature_selection/
│   │   └── gsa_shifted_delay_selection.py
│   ├── models/
│   │   ├── lstm_recursive.py
│   │   ├── custom_loss.py
│   │   └── baselines.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── visualization/
│       └── plot_results.py
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
└── docs/
