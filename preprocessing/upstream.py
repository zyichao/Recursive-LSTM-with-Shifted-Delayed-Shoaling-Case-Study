# preprocessing/upstream.py

import subprocess
import sys
from pathlib import Path


def ensure_studentt_cir_daily(
    project_root: Path,
    cir_daily_path: Path,
    run_if_missing: bool = True,
    force_rerun: bool = False,
) -> None:
    """Make sure the Student's-t-smoothed daily CIR/volume table exists,
    regenerating it from raw survey data if needed.

    Runs, in order, from `project_root`:
        1. preprocess_sedimentation.py  -- raw survey volume -> QC'd,
           corrected, irregular-interval CIR (reach_interval_CIR_cleaned.csv).
           Robust-trend-filter + Hampel jump detection are used ONLY to flag
           and correct isolated bad survey readings here -- this stage does
           NOT produce the smoothed series used downstream.
        2. student_t_kalman_smoother.py -- reads that interval CIR and
           applies a Student's-t (double-T) Kalman/RTS smoother -- the ONLY
           smoothing technique used to produce the final daily CIR/volume
           target series (reach_daily_studentt.csv).

    Both scripts are self-contained (their own `data/raw_data/...` and
    `data/processed/...` paths, relative to `project_root`) and idempotent:
    re-running them regenerates their outputs from scratch.
    """
    if cir_daily_path.exists() and not force_rerun:
        if not run_if_missing:
            return
        print(f"Found existing {cir_daily_path.relative_to(project_root)} -- skipping upstream preprocessing.")
        return

    print("Running upstream preprocessing (raw survey data -> QC -> Student's-t smoothing)...")
    for script in ("preprocess_sedimentation.py", "student_t_kalman_smoother.py"):
        print(f"  -> {script}")
        subprocess.run([sys.executable, script], cwd=project_root, check=True)

    if not cir_daily_path.exists():
        raise RuntimeError(
            f"Upstream preprocessing finished but expected output was not found: {cir_daily_path}"
        )
