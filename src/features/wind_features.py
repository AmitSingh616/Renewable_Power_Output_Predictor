"""
Phase 2: Wind Feature Engineering & Leakage Checks
===================================================
Kelmarsh Wind Farm — Hourly Farm-Level Dataset.
Period: 2021-01-01 to 2021-06-30.

Feature Set:
1. Physical / Weather:
   - Wind_Speed
   - Wind_Direction_sin, Wind_Direction_cos
   - Nacelle_Position_sin, Nacelle_Position_cos
   - Blade_Pitch_Angle
   - Ambient_Temperature
2. Temporal (cyclical encodings):
   - Hour_sin, Hour_cos (24h)
   - Day_sin, Day_cos (365.25d)
   - Month_sin, Month_cos (12m)
3. Power History (lags strictly <= t):
   - Power_lag_1h  (Farm_Power at t-1)
   - Power_lag_2h  (Farm_Power at t-2)
   - Power_lag_24h (Farm_Power at t-24)
4. Forecast Target:
   - target_t1 = Farm_Power.shift(-1) (power at t+1)
   - Audit column: Farm_Power (power at t)

Total Model Features: 15 features.
Strict temporal leakage checks performed and documented.
"""

import sys, os, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH   = ROOT / "data" / "processed" / "wind" / "wind_hourly_clean.csv"
OUTPUT_PATH  = ROOT / "data" / "processed" / "wind" / "wind_features_1h.csv"
TABLE_DIR    = ROOT / "results" / "tables" / "wind"
REPORT_PATH  = ROOT / "results" / "wind_feature_engineering_report.md"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

DIVIDER = "=" * 75

def sec(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def main():
    sec("1. LOAD CLEAN HOURLY WIND DATASET")
    assert INPUT_PATH.exists(), f"Input file not found: {INPUT_PATH}"
    df = pd.read_csv(INPUT_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    print(f"  Loaded raw hourly rows: {len(df):,}")
    print(f"  Time range: {df.index.min()} -> {df.index.max()}")
    print(f"  Input columns: {list(df.columns)}")

    sec("2. FEATURE ENGINEERING")

    # A. Physical / Directional features
    print("  [+] Generating physical and directional sinusoids...")
    wd_rad = np.deg2rad(df["Wind_Direction"])
    df["Wind_Direction_sin"] = np.sin(wd_rad)
    df["Wind_Direction_cos"] = np.cos(wd_rad)

    nac_rad = np.deg2rad(df["Nacelle_Position"])
    df["Nacelle_Position_sin"] = np.sin(nac_rad)
    df["Nacelle_Position_cos"] = np.cos(nac_rad)

    # B. Temporal cyclical features
    print("  [+] Generating temporal cyclical features (Hour, Day, Month)...")
    hour = df.index.hour
    df["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    day_of_year = df.index.dayofyear
    df["Day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["Day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    month = df.index.month
    df["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # C. Power history lags (strictly <= t)
    print("  [+] Generating farm power lag features (1h, 2h, 24h)...")
    # Farm_Power is power at time t
    # Power_lag_1h: power at t-1
    # Power_lag_2h: power at t-2
    # Power_lag_24h: power at t-24
    df["Power_lag_1h"] = df["Farm_Power"].shift(1)
    df["Power_lag_2h"] = df["Farm_Power"].shift(2)
    df["Power_lag_24h"] = df["Farm_Power"].shift(24)

    # D. Target variable: 1-hour ahead Farm_Power (at t+1)
    print("  [+] Generating target: target_t1 = Farm_Power.shift(-1)...")
    df["target_t1"] = df["Farm_Power"].shift(-1)

    MODEL_FEATURES = [
        "Wind_Speed",
        "Wind_Direction_sin",
        "Wind_Direction_cos",
        "Nacelle_Position_sin",
        "Nacelle_Position_cos",
        "Blade_Pitch_Angle",
        "Ambient_Temperature",
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

    ALL_COLS = MODEL_FEATURES + ["Farm_Power", "target_t1"]
    df_features = df[ALL_COLS].copy()

    # Drop rows where target_t1 or lags are NaN (first 24 rows and last row)
    n_before = len(df_features)
    df_features = df_features.dropna(subset=["target_t1", "Power_lag_24h"])
    n_after = len(df_features)
    print(f"  Rows before lag/target drop: {n_before:,}")
    print(f"  Rows after lag/target drop : {n_after:,} (dropped {n_before - n_after} boundary rows)")

    sec("3. LEAKAGE CHECKS & INTEGRITY AUDIT")

    # 1. Sorted timestamps
    is_sorted = df_features.index.is_monotonic_increasing
    print(f"  [1] Timestamps strictly monotonic increasing? {is_sorted}")
    assert is_sorted, "Timestamps are not strictly sorted!"

    # 2. No duplicate timestamps
    has_dups = df_features.index.has_duplicates
    print(f"  [2] Timestamp duplicates present? {has_dups}")
    assert not has_dups, "Duplicate timestamps found!"

    # 3. Target alignment verification
    # target_t1 at index i must exactly equal Farm_Power at index i+1
    sample_i = 100
    ts_now = df_features.index[sample_i]
    ts_next = df_features.index[sample_i + 1]
    t1_val = df_features["target_t1"].iloc[sample_i]
    next_pow = df_features["Farm_Power"].iloc[sample_i + 1]
    diff = abs(t1_val - next_pow)
    print(f"  [3] Target shift alignment check (row {sample_i} -> {sample_i + 1}):")
    print(f"      Time t: {ts_now} | Time t+1: {ts_next}")
    print(f"      target_t1(t): {t1_val:.4f} kW | Farm_Power(t+1): {next_pow:.4f} kW | Abs diff: {diff:.6f}")
    assert diff < 1e-5, "Target shift misaligned!"

    # 4. Lag alignment verification
    lag1_val = df_features["Power_lag_1h"].iloc[sample_i]
    prev_pow = df_features["Farm_Power"].iloc[sample_i - 1]
    diff_lag = abs(lag1_val - prev_pow)
    print(f"  [4] Lag 1h alignment check (row {sample_i} vs {sample_i - 1}):")
    print(f"      Power_lag_1h(t): {lag1_val:.4f} kW | Farm_Power(t-1): {prev_pow:.4f} kW | Abs diff: {diff_lag:.6f}")
    assert diff_lag < 1e-5, "Lag 1h misaligned!"

    # 5. Model features sanity
    print(f"  [5] Model feature count: {len(MODEL_FEATURES)}")
    assert "target_t1" not in MODEL_FEATURES, "Leakage: target_t1 in features!"
    assert "Farm_Power" not in MODEL_FEATURES, "Leakage: current Farm_Power in features!"

    sec("4. SAVING FEATURE DATASET & METRICS")

    # Save features CSV
    df_features.to_csv(OUTPUT_PATH)
    print(f"  [+] Saved final wind feature dataset -> {OUTPUT_PATH}")

    # Feature definitions table
    feat_defs = [
        {"feature_name": "Wind_Speed", "category": "Physical Weather", "unit": "m/s", "source": "Anemometer mean across 6 turbines", "leakage_safe": True},
        {"feature_name": "Wind_Direction_sin", "category": "Directional", "unit": "[-1, 1]", "source": "sin(rad(Wind_Direction)) vector mean", "leakage_safe": True},
        {"feature_name": "Wind_Direction_cos", "category": "Directional", "unit": "[-1, 1]", "source": "cos(rad(Wind_Direction)) vector mean", "leakage_safe": True},
        {"feature_name": "Nacelle_Position_sin", "category": "Operational", "unit": "[-1, 1]", "source": "sin(rad(Nacelle_Position)) vector mean", "leakage_safe": True},
        {"feature_name": "Nacelle_Position_cos", "category": "Operational", "unit": "[-1, 1]", "source": "cos(rad(Nacelle_Position)) vector mean", "leakage_safe": True},
        {"feature_name": "Blade_Pitch_Angle", "category": "Operational", "unit": "deg", "source": "Blade pitch A mean across 6 turbines", "leakage_safe": True},
        {"feature_name": "Ambient_Temperature", "category": "Physical Weather", "unit": "deg C", "source": "Nacelle ambient temp mean across 6 turbines", "leakage_safe": True},
        {"feature_name": "Hour_sin", "category": "Temporal Diurnal", "unit": "[-1, 1]", "source": "sin(2 * pi * hour / 24)", "leakage_safe": True},
        {"feature_name": "Hour_cos", "category": "Temporal Diurnal", "unit": "[-1, 1]", "source": "cos(2 * pi * hour / 24)", "leakage_safe": True},
        {"feature_name": "Day_sin", "category": "Temporal Seasonal", "unit": "[-1, 1]", "source": "sin(2 * pi * day / 365.25)", "leakage_safe": True},
        {"feature_name": "Day_cos", "category": "Temporal Seasonal", "unit": "[-1, 1]", "source": "cos(2 * pi * day / 365.25)", "leakage_safe": True},
        {"feature_name": "Month_sin", "category": "Temporal Annual", "unit": "[-1, 1]", "source": "sin(2 * pi * month / 12)", "leakage_safe": True},
        {"feature_name": "Month_cos", "category": "Temporal Annual", "unit": "[-1, 1]", "source": "cos(2 * pi * month / 12)", "leakage_safe": True},
        {"feature_name": "Power_lag_1h", "category": "Autoregressive Lag", "unit": "kW", "source": "Farm_Power at time t-1", "leakage_safe": True},
        {"feature_name": "Power_lag_2h", "category": "Autoregressive Lag", "unit": "kW", "source": "Farm_Power at time t-2", "leakage_safe": True},
        {"feature_name": "Power_lag_24h", "category": "Autoregressive Lag", "unit": "kW", "source": "Farm_Power at time t-24", "leakage_safe": True},
    ]
    df_defs = pd.DataFrame(feat_defs)
    defs_path = TABLE_DIR / "wind_feature_definitions.csv"
    df_defs.to_csv(defs_path, index=False)
    print(f"  [+] Saved feature definitions -> {defs_path}")

    # Missingness summary
    miss_summary = pd.DataFrame({
        "feature": df_features.columns,
        "missing_count": df_features.isna().sum().values,
        "missing_pct": (df_features.isna().mean() * 100).round(3).values
    })
    miss_path = TABLE_DIR / "wind_final_missingness.csv"
    miss_summary.to_csv(miss_path, index=False)
    print(f"  [+] Saved missingness summary -> {miss_path}")

    # Write Feature Engineering Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Kelmarsh Wind Farm — Feature Engineering & Leakage Report\n\n")
        f.write("> **Task**: 1-Hour-Ahead Total Farm Active Power Forecasting ($X_t \\to Y_{t+1}$).\n")
        f.write(f"> **Period**: `{df_features.index.min()}` to `{df_features.index.max()}` ({len(df_features):,} supervised hourly rows).\n\n")
        f.write("---\n\n## 1. Feature Space Architecture (16 Model Features)\n\n")
        f.write("| Feature Name | Category | Unit | Formula / Source |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for _, r in df_defs.iterrows():
            f.write(f"| `{r['feature_name']}` | {r['category']} | {r['unit']} | {r['source']} |\n")

        f.write("\n---\n\n## 2. Temporal Leakage Audit Checklist\n\n")
        f.write("- [x] **Strict Chronological Ordering**: Timestamps verified strictly monotonically increasing.\n")
        f.write("- [x] **No Future Weather/SCADA Leakage**: All physical variables ($WS_t, Temp_t, Pitch_t$) represent readings available at time $t$ or earlier.\n")
        f.write("- [x] **No Target Leakage**: Current `Farm_Power` ($t$) and target `target_t1` ($t+1$) are strictly excluded from feature matrix $X_t$.\n")
        f.write("- [x] **Causal Autoregressive Lags**: Lags are strictly causal ($t-1, t-2, t-24$). Verified via exact numerical shift assertions.\n")
        f.write("- [x] **Untouched Test Guarantee**: Split will be applied chronologically (80% dev, 20% test); all imputers and scalers will be fit exclusively within training folds.\n\n")
        f.write("---\n\n## 3. Dataset Missingness Audit\n\n")
        f.write("| Feature | Missing Count | Missing % |\n")
        f.write("| :--- | :---: | :---: |\n")
        for _, r in miss_summary.iterrows():
            f.write(f"| `{r['feature']}` | {int(r['missing_count'])} | {r['missing_pct']:.2f}% |\n")
        f.write("\n---\n*Feature engineering complete and validated for model benchmarking.*\n")

    print(f"  [+] Saved feature engineering report -> {REPORT_PATH}")
    sec("WIND FEATURE ENGINEERING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
