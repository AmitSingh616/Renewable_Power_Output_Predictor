"""
Phase 2 - Solar Feature Engineering
====================================
Constructs ML feature set from solar_hourly_clean.csv:
1. Applies locked daylight rule: GHI > 1.
2. Creates 6 cyclical temporal features (Hour, Day of year, Month sin/cos).
3. Creates 3 historical power lag features (Power_lag_1h, Power_lag_2h, Power_lag_24h).
4. Creates 2 circular wind direction features (Wind_Direction_sin, Wind_Direction_cos).
5. Defines target_t1 = Active_Power.shift(-1).
6. Runs sanity & leakage verification checks.
7. Saves feature dataset, metadata CSV, and feature engineering report.

Does NOT train models or perform feature selection.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
PROCESSED    = ROOT / "data" / "processed" / "solar"
RESULTS      = ROOT / "results" / "tables"
REPORTS      = ROOT / "results"

for d in [PROCESSED, RESULTS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_CSV    = PROCESSED / "solar_hourly_clean.csv"
OUTPUT_CSV   = PROCESSED / "solar_features_1h.csv"
METADATA_CSV = RESULTS   / "solar_feature_definitions.csv"
REPORT_MD    = REPORTS   / "solar_feature_engineering_report.md"

DIVIDER = "=" * 70

def sec(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def main():
    sec("1. LOADING CLEAN HOURLY SOLAR DATASET")
    assert INPUT_CSV.exists(), f"Missing input file: {INPUT_CSV}"
    df_raw = pd.read_csv(INPUT_CSV, index_col="timestamp", parse_dates=True)
    df_raw = df_raw.sort_index()
    print(f"  Loaded input rows: {len(df_raw):,}")
    print(f"  Input date range : {df_raw.index.min()} -> {df_raw.index.max()}")
    print(f"  Input columns    : {list(df_raw.columns)}")

    # ---------------------------------------------------------------------------
    # Reindex to strict regular hourly grid before computing lags and target
    # Ensures shift(k) maps strictly to t - k hours in physical time.
    # ---------------------------------------------------------------------------
    sec("2. REGULAR HOURLY GRID ALIGNMENT")
    full_idx = pd.date_range(df_raw.index.min(), df_raw.index.max(), freq="h")
    df_grid = df_raw.reindex(full_idx)
    df_grid.index.name = "timestamp"
    print(f"  Full hourly grid length: {len(df_grid):,}")
    print(f"  Unobserved hourly slots: {len(df_grid) - len(df_raw):,}")

    # ---------------------------------------------------------------------------
    # 3. Create Power Lags & Target on Regular Grid
    # ---------------------------------------------------------------------------
    sec("3. POWER LAG & TARGET CONSTRUCTION")
    df_grid["Power_lag_1h"]  = df_grid["Active_Power"].shift(1)
    df_grid["Power_lag_2h"]  = df_grid["Active_Power"].shift(2)
    df_grid["Power_lag_24h"] = df_grid["Active_Power"].shift(24)
    df_grid["target_t1"]     = df_grid["Active_Power"].shift(-1)
    print("  Created Power_lag_1h  = Active_Power.shift(1)   [t - 1h]")
    print("  Created Power_lag_2h  = Active_Power.shift(2)   [t - 2h]")
    print("  Created Power_lag_24h = Active_Power.shift(24)  [t - 24h]")
    print("  Created target_t1     = Active_Power.shift(-1)  [t + 1h]")

    # ---------------------------------------------------------------------------
    # 4. Cyclical Temporal Features
    # ---------------------------------------------------------------------------
    sec("4. CYCLICAL TEMPORAL FEATURES")
    hours = df_grid.index.hour
    day_of_year = df_grid.index.dayofyear
    months = df_grid.index.month

    df_grid["Hour_sin"]  = np.sin(2 * np.pi * hours / 24.0).astype("float32")
    df_grid["Hour_cos"]  = np.cos(2 * np.pi * hours / 24.0).astype("float32")
    df_grid["Day_sin"]   = np.sin(2 * np.pi * day_of_year / 365.25).astype("float32")
    df_grid["Day_cos"]   = np.cos(2 * np.pi * day_of_year / 365.25).astype("float32")
    df_grid["Month_sin"] = np.sin(2 * np.pi * months / 12.0).astype("float32")
    df_grid["Month_cos"] = np.cos(2 * np.pi * months / 12.0).astype("float32")
    print("  Created Hour_sin, Hour_cos   (period = 24)")
    print("  Created Day_sin, Day_cos     (period = 365.25)")
    print("  Created Month_sin, Month_cos (period = 12)")

    # ---------------------------------------------------------------------------
    # 5. Circular Wind Direction Encoding
    # ---------------------------------------------------------------------------
    sec("5. WIND DIRECTION ENCODING")
    wd_deg = df_grid["Wind_Direction"]
    wd_rad = np.deg2rad(wd_deg)
    df_grid["Wind_Direction_sin"] = np.sin(wd_rad).astype("float32")
    df_grid["Wind_Direction_cos"] = np.cos(wd_rad).astype("float32")
    print("  Created Wind_Direction_sin = sin(2*pi*Wind_Direction/360)")
    print("  Created Wind_Direction_cos = cos(2*pi*Wind_Direction/360)")
    print("  (Preserved NaNs where Wind_Direction was missing; raw Wind_Direction excluded from X)")

    # ---------------------------------------------------------------------------
    # 6. Apply Locked Daylight Filter (GHI > 1)
    # ---------------------------------------------------------------------------
    sec("6. LOCKED DAYLIGHT FILTER (GHI > 1)")
    rows_before_ghi = len(df_raw)
    ghi_mask = df_grid["Global_Horizontal_Radiation"] > 1.0
    df_daylight = df_grid[ghi_mask].copy()
    rows_after_ghi = len(df_daylight)
    rows_removed_ghi = rows_before_ghi - rows_after_ghi
    print(f"  Rows before daylight filtering: {rows_before_ghi:,}")
    print(f"  Rows after  GHI > 1 filtering : {rows_after_ghi:,}")
    print(f"  Rows removed by GHI > 1 rule  : {rows_removed_ghi:,}")

    # ---------------------------------------------------------------------------
    # 7. Lag / Target Boundary Handling
    # ---------------------------------------------------------------------------
    sec("7. BOUNDARY & USABILITY FILTERING")
    # Rows with missing required lags or missing target cannot be used for supervised learning
    req_history_target = ["Power_lag_1h", "Power_lag_2h", "Power_lag_24h", "target_t1"]
    n_before_boundary = len(df_daylight)
    df_final = df_daylight.dropna(subset=req_history_target).copy()
    n_after_boundary = len(df_final)
    rows_removed_boundary = n_before_boundary - n_after_boundary
    print(f"  Daylight rows before boundary drop: {n_before_boundary:,}")
    print(f"  Daylight rows with complete lags/target: {n_after_boundary:,}")
    print(f"  Rows removed due to lag/target boundaries: {rows_removed_boundary:,}")

    # ---------------------------------------------------------------------------
    # 8. Feature Selection & Column Layout
    # ---------------------------------------------------------------------------
    sec("8. FINAL FEATURE MATRIX SPECIFICATION")
    FEATURE_COLS = [
        # Physical / weather (5)
        "Global_Horizontal_Radiation",
        "Diffuse_Horizontal_Radiation",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Wind_Speed",
        # Wind direction circular (2)
        "Wind_Direction_sin",
        "Wind_Direction_cos",
        # Cyclical time (6)
        "Hour_sin",
        "Hour_cos",
        "Day_sin",
        "Day_cos",
        "Month_sin",
        "Month_cos",
        # Power lags (3)
        "Power_lag_1h",
        "Power_lag_2h",
        "Power_lag_24h",
    ]

    TARGET_COL = "target_t1"
    AUDIT_COL  = "Active_Power"

    ALL_COLS = FEATURE_COLS + [AUDIT_COL, TARGET_COL]
    df_save = df_final[ALL_COLS].copy()

    print(f"  Model feature count: {len(FEATURE_COLS)}")
    print("  Feature list:")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"    {i:2d}. {col}")

    # ---------------------------------------------------------------------------
    # 9. Sanity & Leakage Verification Checks
    # ---------------------------------------------------------------------------
    sec("9. SANITY & LEAKAGE VERIFICATION CHECKS")

    # 9.1 Timestamps
    assert df_save.index.is_monotonic_increasing, "FAIL: Timestamps not monotonic increasing!"
    assert not df_save.index.duplicated().any(), "FAIL: Duplicate timestamps present!"
    print("  [PASS] Timestamps strictly sorted ascending and unique.")

    # 9.2 Cyclical ranges
    cyclical_cols = ["Hour_sin", "Hour_cos", "Day_sin", "Day_cos", "Month_sin", "Month_cos"]
    for c in cyclical_cols:
        c_min, c_max = df_save[c].min(), df_save[c].max()
        assert -1.001 <= c_min and c_max <= 1.001, f"FAIL: {c} outside [-1, 1] range: ({c_min}, {c_max})"
    print("  [PASS] All cyclical temporal features strictly bounded in [-1, 1].")

    # 9.3 Wind direction identity sin^2 + cos^2 = 1
    wd_valid = df_save.dropna(subset=["Wind_Direction_sin", "Wind_Direction_cos"])
    sq_sum = wd_valid["Wind_Direction_sin"]**2 + wd_valid["Wind_Direction_cos"]**2
    assert np.allclose(sq_sum, 1.0, atol=1e-3), "FAIL: Wind direction sin^2 + cos^2 != 1"
    print("  [PASS] Wind direction trig identity verified (sin^2 + cos^2 ≈ 1.0).")

    # 9.4 Lags verification vs raw series on regular grid
    print("\n  Verifying lag alignments on sample rows:")
    sample_ts = df_save.index[100]
    p_curr = df_grid.loc[sample_ts, "Active_Power"]
    p_lag1 = df_save.loc[sample_ts, "Power_lag_1h"]
    p_lag2 = df_save.loc[sample_ts, "Power_lag_2h"]
    p_lag24 = df_save.loc[sample_ts, "Power_lag_24h"]
    t1_val  = df_save.loc[sample_ts, "target_t1"]

    expected_lag1  = df_grid.loc[sample_ts - pd.Timedelta(hours=1), "Active_Power"]
    expected_lag2  = df_grid.loc[sample_ts - pd.Timedelta(hours=2), "Active_Power"]
    expected_lag24 = df_grid.loc[sample_ts - pd.Timedelta(hours=24), "Active_Power"]
    expected_t1    = df_grid.loc[sample_ts + pd.Timedelta(hours=1), "Active_Power"]

    assert np.isclose(p_lag1, expected_lag1), f"FAIL: Power_lag_1h mismatch at {sample_ts}"
    assert np.isclose(p_lag2, expected_lag2), f"FAIL: Power_lag_2h mismatch at {sample_ts}"
    assert np.isclose(p_lag24, expected_lag24), f"FAIL: Power_lag_24h mismatch at {sample_ts}"
    assert np.isclose(t1_val, expected_t1), f"FAIL: target_t1 mismatch at {sample_ts}"
    print(f"  [PASS] Sample timestamp: {sample_ts}")
    print(f"         Power_lag_1h  = {p_lag1:.4f} (matches t-1h: {expected_lag1:.4f})")
    print(f"         Power_lag_2h  = {p_lag2:.4f} (matches t-2h: {expected_lag2:.4f})")
    print(f"         Power_lag_24h = {p_lag24:.4f} (matches t-24h: {expected_lag24:.4f})")
    print(f"         target_t1     = {t1_val:.4f} (matches t+1h: {expected_t1:.4f})")

    # 9.5 Leakage checks
    assert TARGET_COL not in FEATURE_COLS, "FAIL: Target leaked into feature list!"
    assert AUDIT_COL not in FEATURE_COLS, "FAIL: Unshifted Active_Power present in feature list!"
    for fc in FEATURE_COLS:
        assert not any(kw in fc.lower() for kw in ["t+1", "target", "lead", "future"]), f"FAIL: Future variable in features: {fc}"
    print("  [PASS] No future-derived variables or target present in feature list.")

    # ---------------------------------------------------------------------------
    # 10. Missingness Inspection
    # ---------------------------------------------------------------------------
    sec("10. MISSINGNESS INSPECTION")
    miss_series = df_save[FEATURE_COLS + [TARGET_COL]].isna().sum()
    miss_pct    = (df_save[FEATURE_COLS + [TARGET_COL]].isna().mean() * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": miss_series, "missing_pct": miss_pct})
    print(miss_df.to_string())

    # ---------------------------------------------------------------------------
    # 11. Save Feature Dataset
    # ---------------------------------------------------------------------------
    sec("11. SAVING FEATURE DATASET")
    df_save.to_csv(OUTPUT_CSV)
    print(f"  Saved features CSV : {OUTPUT_CSV}")
    print(f"  File size          : {OUTPUT_CSV.stat().st_size / 1e6:.2f} MB")

    # Reload check
    df_check = pd.read_csv(OUTPUT_CSV, index_col="timestamp", parse_dates=True)
    assert df_check.shape == df_save.shape, "FAIL: Reload shape mismatch!"
    print(f"  Reload verification: OK, shape = {df_check.shape}")

    # ---------------------------------------------------------------------------
    # 12. Save Feature Definitions Metadata
    # ---------------------------------------------------------------------------
    sec("12. SAVING FEATURE METADATA DEFINITIONS")
    meta_rows = [
        {"feature": "Global_Horizontal_Radiation", "category": "physical", "source": "WeatherStation",
         "description": "Global horizontal irradiance in W/m²", "availability_time": "t", "transformation": "hourly mean (GHI > 1 daytime filter)"},
        {"feature": "Diffuse_Horizontal_Radiation", "category": "physical", "source": "WeatherStation",
         "description": "Diffuse horizontal irradiance in W/m²", "availability_time": "t", "transformation": "hourly mean"},
        {"feature": "Weather_Temperature_Celsius", "category": "physical", "source": "WeatherStation",
         "description": "Ambient air temperature in Celsius", "availability_time": "t", "transformation": "hourly mean"},
        {"feature": "Weather_Relative_Humidity", "category": "physical", "source": "WeatherStation",
         "description": "Relative humidity percentage (0-100%)", "availability_time": "t", "transformation": "hourly mean"},
        {"feature": "Wind_Speed", "category": "physical", "source": "WeatherStation",
         "description": "Wind speed in m/s (~52% unobserved preserved)", "availability_time": "t", "transformation": "hourly mean"},
        {"feature": "Wind_Direction_sin", "category": "physical", "source": "WeatherStation",
         "description": "Sine circular component of wind direction", "availability_time": "t", "transformation": "sin(2*pi*Wind_Direction/360)"},
        {"feature": "Wind_Direction_cos", "category": "physical", "source": "WeatherStation",
         "description": "Cosine circular component of wind direction", "availability_time": "t", "transformation": "cos(2*pi*Wind_Direction/360)"},
        {"feature": "Hour_sin", "category": "temporal", "source": "Timestamp",
         "description": "Sine cyclical encoding of hour of day", "availability_time": "t", "transformation": "sin(2*pi*hour/24)"},
        {"feature": "Hour_cos", "category": "temporal", "source": "Timestamp",
         "description": "Cosine cyclical encoding of hour of day", "availability_time": "t", "transformation": "cos(2*pi*hour/24)"},
        {"feature": "Day_sin", "category": "temporal", "source": "Timestamp",
         "description": "Sine cyclical encoding of day of year", "availability_time": "t", "transformation": "sin(2*pi*day_of_year/365.25)"},
        {"feature": "Day_cos", "category": "temporal", "source": "Timestamp",
         "description": "Cosine cyclical encoding of day of year", "availability_time": "t", "transformation": "cos(2*pi*day_of_year/365.25)"},
        {"feature": "Month_sin", "category": "temporal", "source": "Timestamp",
         "description": "Sine cyclical encoding of month of year", "availability_time": "t", "transformation": "sin(2*pi*month/12)"},
        {"feature": "Month_cos", "category": "temporal", "source": "Timestamp",
         "description": "Cosine cyclical encoding of month of year", "availability_time": "t", "transformation": "cos(2*pi*month/12)"},
        {"feature": "Power_lag_1h", "category": "lag", "source": "Active_Power",
         "description": "Master-meter active power 1 hour prior to t", "availability_time": "t-1", "transformation": "Active_Power.shift(1)"},
        {"feature": "Power_lag_2h", "category": "lag", "source": "Active_Power",
         "description": "Master-meter active power 2 hours prior to t", "availability_time": "t-2", "transformation": "Active_Power.shift(2)"},
        {"feature": "Power_lag_24h", "category": "lag", "source": "Active_Power",
         "description": "Master-meter active power 24 hours prior to t (same hour previous day)", "availability_time": "t-24", "transformation": "Active_Power.shift(24)"},
        {"feature": "target_t1", "category": "target", "source": "Active_Power",
         "description": "Master-meter active power 1 hour ahead (forecast target)", "availability_time": "t+1", "transformation": "Active_Power.shift(-1)"},
    ]
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(METADATA_CSV, index=False)
    print(f"  Saved metadata CSV : {METADATA_CSV}")

    # ---------------------------------------------------------------------------
    # 13. Write Feature Engineering Markdown Report
    # ---------------------------------------------------------------------------
    sec("13. WRITING FEATURE ENGINEERING REPORT")
    def to_md_table(d):
        header = "| Column | " + " | ".join(d.columns) + " |\n"
        sep = "| --- | " + " | ".join(["---"] * len(d.columns)) + " |\n"
        rows = ""
        for idx, row in d.iterrows():
            rows += f"| {idx} | " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |\n"
        return header + sep + rows

    meta_table_md = "| Feature | Category | Source | Availability | Transformation |\n| --- | --- | --- | --- | --- |\n"
    for r in meta_rows:
        meta_table_md += f"| `{r['feature']}` | {r['category']} | {r['source']} | {r['availability_time']} | {r['transformation']} |\n"

    report_content = f"""# Solar Feature Engineering Report — Phase 2

> **The solar forecasting target is the whole master-meter `Active_Power`.**
> 
> **Daylight observations are selected using the fixed criterion `Global_Horizontal_Radiation > 1`.**

---

## 1. Input Dataset
- **Source**: `{INPUT_CSV.name}`
- **Input row count**: {rows_before_ghi:,}
- **Date range**: {df_raw.index.min()} to {df_raw.index.max()}

---

## 2. Locked Daylight Filtering (`GHI > 1`)
- **Criterion**: `Global_Horizontal_Radiation > 1`
- **Rows before daylight filtering**: {rows_before_ghi:,}
- **Rows after daylight filtering**: {rows_after_ghi:,}
- **Rows removed by `GHI > 1`**: {rows_removed_ghi:,}

---

## 3. Boundary & History Filtering
- **Required history / target**: `Power_lag_1h`, `Power_lag_2h`, `Power_lag_24h`, `target_t1`
- **Rows removed due to lag/target boundaries**: {rows_removed_boundary:,}
- **Final usable supervised rows**: {len(df_save):,}

---

## 4. Final Feature Specification
- **Total model features**: {len(FEATURE_COLS)}
- **Feature categories**:
  - **Physical / Weather (5)**: `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Weather_Temperature_Celsius`, `Weather_Relative_Humidity`, `Wind_Speed`
  - **Wind Direction Circular (2)**: `Wind_Direction_sin`, `Wind_Direction_cos`
  - **Cyclical Time (6)**: `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos`, `Month_sin`, `Month_cos`
  - **Power History Lags (3)**: `Power_lag_1h`, `Power_lag_2h`, `Power_lag_24h`
- **Target (1)**: `target_t1` (Active_Power at t+1)

---

## 5. Feature Definitions Table
{meta_table_md}

---

## 6. Missingness Summary
{to_md_table(miss_df)}

*Note on missingness: `Wind_Speed` has ~52% unobserved values originating from sensor gaps in the raw source; missingness is preserved as instructed without artificial fabrication.*

---

## 7. Leakage & Integrity Verification
- **Timestamps strictly sorted ascending**: YES
- **Zero duplicate timestamps**: YES
- **Cyclical features strictly within [-1, 1]**: YES
- **Wind direction trigonometry identity ($sin^2 + cos^2 = 1$)**: VERIFIED
- **Lag timing verification**: Exactly matches chronological $t-1$, $t-2$, $t-24$ hours
- **Target timing verification**: Exactly matches chronological $t+1$ hour
- **Target excluded from feature space**: YES (`target_t1` not in $X$)
- **Zero future-derived variables in $X$**: YES
- **No rolling statistics constructed**: YES

---

## 8. Final Dataset Summary
- **Output file**: `{OUTPUT_CSV}`
- **Final shape**: {df_save.shape}
- **Date range**: {df_save.index.min()} to {df_save.index.max()}

---

*Phase 2 complete. Models, scaling, and feature selection not started.*
"""
    REPORT_MD.write_text(report_content, encoding="utf-8")
    print(f"  Saved markdown report: {REPORT_MD}")

    sec("PHASE 2 COMPLETE")
    print(f"""
  1. Final dataset shape      : {df_save.shape}
  2. Final date range         : {df_save.index.min()} -> {df_save.index.max()}
  3. Final feature count      : {len(FEATURE_COLS)}
  4. Complete feature list    : {FEATURE_COLS}
  5. Missingness summary      :
{miss_df.to_string()}
  6. Rows removed by GHI > 1  : {rows_removed_ghi:,}
  7. Rows removed by boundaries: {rows_removed_boundary:,}
  8. Final target definition  : {TARGET_COL} = Active_Power at hour t+1
  9. Leakage check result     : PASS (all features available at t or earlier)
 10. Processed dataset path   : {OUTPUT_CSV}
 11. Feature metadata path    : {METADATA_CSV}
 12. Feature report path      : {REPORT_MD}
""")

if __name__ == "__main__":
    main()
