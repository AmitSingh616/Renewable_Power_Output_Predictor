"""
Solar 24-Hour-Ahead Direct Forecasting Demonstration
====================================================
Objective:
1. Direct multi-step demonstration: X_t -> Y_(t+24).
2. Uses the target_t24 = Active_Power.shift(-24).
3. Features: Best feature subset determined from feature selection (or full features).
4. Model: Locked tuned LightGBM model from Phase 4.
5. Strict chronological 80/20 train/test split.
6. Evaluate on final untouched test set.
7. Save metrics table and actual vs predicted plot.

Constraints:
- Dataset cap: tail(35000)
- Single direct forecast demonstration. No full re-tuning.
"""

import sys, os, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "solar" / "solar_features_1h.csv"
TABLE_DIR = ROOT / "results" / "tables"
SOLAR_TABLE_DIR = TABLE_DIR / "solar"
FIG_DIR = ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOLAR_TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_ROWS = 35000

# Tuned parameters for LightGBM from Phase 4
LGBM_PARAMS = {
    "n_estimators": 60,
    "num_leaves": 39,
    "max_depth": 8,
    "learning_rate": 0.10200369308281163,
    "subsample": 0.6,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

DIVIDER = "=" * 75

def sec(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def main():
    sec("1. LOAD SOLAR DATASET & CONSTRUCT 24H TARGET")
    assert DATA_PATH.exists(), f"Features file not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    print(f"  Raw feature rows: {len(df):,}")
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)
        print(f"  [!] Applied dataset cap: most recent {len(df):,} rows (~4 years)")

    print(f"  Active time range: {df.index.min()} -> {df.index.max()}")

    # Construct direct 24-hour ahead target: Active_Power at t+24
    # Note: 'Active_Power' represents power at t in solar_features_1h.csv
    assert "Active_Power" in df.columns, "Active_Power column missing from dataset"
    df["target_t24"] = df["Active_Power"].shift(-24)

    # Drop NaNs created by shifting 24 steps
    valid_mask = ~df["target_t24"].isna()
    df_clean = df[valid_mask].copy()
    print(f"  Rows after 24h shift drop: {len(df_clean):,} (dropped {len(df) - len(df_clean)} tail rows)")

    # Check for feature subset definition file; fallback to 16 features if not found
    subsets_file = SOLAR_TABLE_DIR / "solar_feature_subsets.csv"
    if not subsets_file.exists():
        subsets_file = TABLE_DIR / "solar_feature_subsets.csv"

    features_to_use = [
        "Global_Horizontal_Radiation",
        "Diffuse_Horizontal_Radiation",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Wind_Speed",
        "Wind_Direction_sin",
        "Wind_Direction_cos",
        "Hour_sin",
        "Hour_cos",
        "Day_sin",
        "Day_cos",
        "Month_sin",
        "Month_cos",
        "Power_lag_1h",
        "Power_lag_2h",
        "Power_lag_24h",
    ]

    selected_subset_name = "Full (16 features)"
    if subsets_file.exists():
        df_sub = pd.read_csv(subsets_file)
        # Select the 8 or 12 feature subset, or the one with optimal trade-off
        # Let's inspect if an 8-feature subset exists
        row_8 = df_sub[df_sub["subset_name"].str.contains("8")]
        if not row_8.empty:
            features_to_use = [f.strip() for f in row_8.iloc[0]["features"].split(";")]
            selected_subset_name = row_8.iloc[0]["subset_name"]
            print(f"  Selected feature subset: {selected_subset_name} ({len(features_to_use)} features)")
        else:
            print("  Using full feature set.")
    else:
        print("  Subsets file not yet generated, defaulting to all 16 features.")

    X = df_clean[features_to_use]
    y_24 = df_clean["target_t24"]

    # ---------------------------------------------------------------------------
    # Chronological Split (80% dev, 20% test)
    # ---------------------------------------------------------------------------
    sec("2. CHRONOLOGICAL SPLIT (80% DEV / 20% TEST)")
    n_total = len(df_clean)
    n_dev = int(n_total * 0.80)
    n_test = n_total - n_dev

    X_dev, y_dev = X.iloc[:n_dev], y_24.iloc[:n_dev]
    X_test, y_test = X.iloc[n_dev:], y_24.iloc[n_dev:]

    print(f"  Development rows (80%): {n_dev:,} ({df_clean.index[0]} to {df_clean.index[n_dev-1]})")
    print(f"  Final test rows  (20%): {n_test:,} ({df_clean.index[n_dev]} to {df_clean.index[-1]})")

    # ---------------------------------------------------------------------------
    # Model Training and Prediction
    # ---------------------------------------------------------------------------
    sec("3. TRAIN LIGHTGBM ON 24-HOUR HORIZON")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", LGBMRegressor(**LGBM_PARAMS))
    ])

    t_fit_start = time.perf_counter()
    pipe.fit(X_dev, y_dev)
    t_fit = time.perf_counter() - t_fit_start

    t_inf_start = time.perf_counter()
    y_pred = pipe.predict(X_test)
    t_inf = time.perf_counter() - t_inf_start

    rmse, mae, r2 = eval_metrics(y_test, y_pred)
    latency_ms = (t_inf / len(X_test)) * 1000.0

    # Also compute persistence baseline for 24h: Y_(t+24) = Y_t (or lag 24)
    # If daytime filter causes jumps, persistence can use Active_Power directly as t forecast
    baseline_pred = df_clean.iloc[n_dev:]["Active_Power"].values
    base_rmse, base_mae, base_r2 = eval_metrics(y_test, baseline_pred)

    print("\n  =======================================================")
    print(f"  LightGBM 24h Forecast Metrics:")
    print(f"    - Test RMSE: {rmse:.4f} kW")
    print(f"    - Test MAE : {mae:.4f} kW")
    print(f"    - Test R²  : {r2:.4f}")
    print(f"    - Latency  : {latency_ms:.4f} ms/sample")
    print(f"  Persistence Baseline (Y_t -> Y_t+24):")
    print(f"    - Baseline RMSE: {base_rmse:.4f} kW")
    print(f"    - Baseline MAE : {base_mae:.4f} kW")
    print(f"    - Baseline R²  : {base_r2:.4f}")
    print("  =======================================================\n")

    # Save metrics table
    res_rows = [
        {
            "Horizon": "24-Hour-Ahead (Direct)",
            "Model": "LightGBM (Tuned)",
            "Feature Subset": selected_subset_name,
            "Feature Count": len(features_to_use),
            "Test RMSE (kW)": round(rmse, 4),
            "Test MAE (kW)": round(mae, 4),
            "Test R²": round(r2, 4),
            "Inference Latency (ms/sample)": round(latency_ms, 4)
        },
        {
            "Horizon": "24-Hour-Ahead (Direct)",
            "Model": "Persistence Baseline (Y_t)",
            "Feature Subset": "Active_Power only",
            "Feature Count": 1,
            "Test RMSE (kW)": round(base_rmse, 4),
            "Test MAE (kW)": round(base_mae, 4),
            "Test R²": round(base_r2, 4),
            "Inference Latency (ms/sample)": 0.0
        }
    ]
    df_metrics = pd.DataFrame(res_rows)
    metrics_path1 = TABLE_DIR / "solar_24h_results.csv"
    metrics_path2 = SOLAR_TABLE_DIR / "solar_24h_results.csv"
    df_metrics.to_csv(metrics_path1, index=False)
    df_metrics.to_csv(metrics_path2, index=False)
    print(f"  Saved 24h metrics -> {metrics_path1}")

    # ---------------------------------------------------------------------------
    # Plot Actual vs Predicted (Sample Window)
    # ---------------------------------------------------------------------------
    sec("4. GENERATING 24-HOUR FORECAST PLOT")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Plot 1: 14-day sample trace
    sample_window = 14 * 12 # ~14 daytime periods (assuming ~12 daytime hours/day)
    if len(y_test) > sample_window:
        y_test_sub = y_test.iloc[:sample_window]
        y_pred_sub = y_pred[:sample_window]
    else:
        y_test_sub = y_test
        y_pred_sub = y_pred

    ax = axes[0]
    ax.plot(y_test_sub.index, y_test_sub.values, label="Actual Active Power (t+24)", color="#1f77b4", linewidth=1.5)
    ax.plot(y_test_sub.index, y_pred_sub, label="Predicted (LightGBM 24h direct)", color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Direct 24-Hour Ahead Forecast vs Actual Power (14-Day Daytime Sample)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Active Power (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Scatter parity plot
    ax = axes[1]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#2ca02c", label=f"R² = {r2:.4f}")
    min_val = min(y_test.min(), np.min(y_pred))
    max_val = max(y_test.max(), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="1:1 Perfect Forecast")
    ax.set_title("Actual vs Predicted Scatter Parity (Full Untouched Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Active Power (kW)")
    ax.set_ylabel("Predicted Active Power (kW)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = FIG_DIR / "solar_24h_actual_vs_predicted.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure -> {fig_path}")

    sec("24-HOUR DEMONSTRATION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
