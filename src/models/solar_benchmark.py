"""
Phase 3-4: Chronological Evaluation and Model Benchmarking for Solar Power Forecasting
========================================================================================
(Optimized for faster training & high terminal verbosity)
"""

import sys, os, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Optuna set to INFO to print the exact score and hyperparameter guess after every single trial
optuna.logging.set_verbosity(optuna.logging.INFO)

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
DATA_PATH    = ROOT / "data" / "processed" / "solar" / "solar_features_1h.csv"
MODEL_DIR    = ROOT / "results" / "models" / "solar"
TABLE_DIR    = ROOT / "results" / "tables" / "solar"
REPORT_PATH  = ROOT / "results" / "solar_model_benchmark_report.md"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_SPLITS  = 3
N_OPTUNA_TRIALS = 15

FEATURE_COLS = [
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

TARGET_COL = "target_t1"

DIVIDER = "=" * 70

def sec(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

def cross_validate_pipeline(pipe_builder, X, y, cv):
    rmse_list, mae_list, r2_list = [], [], []
    t0 = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        print(f"      -> Running CV Fold {fold_idx}/3...")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipe = pipe_builder()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)

        rmse, mae, r2 = eval_metrics(y_val, preds)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
    elapsed = time.time() - t0
    return {
        "rmse_mean": np.mean(rmse_list),
        "rmse_std":  np.std(rmse_list),
        "mae_mean":  np.mean(mae_list),
        "mae_std":   np.std(mae_list),
        "r2_mean":   np.mean(r2_list),
        "r2_std":    np.std(r2_list),
        "fit_time":  elapsed
    }

def main():
    sec("1. LOAD SOLAR FEATURES DATASET & CHRONOLOGICAL SPLIT")
    assert DATA_PATH.exists(), f"Features dataset not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    print(f"  Original total rows: {len(df):,}")
    
    # Truncate dataset to the most recent ~4 years to massively speed up training
    MAX_ROWS = 35000
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)
        print(f"  [!] Truncated dataset to recent {len(df):,} rows for faster training.")

    print(f"  Active Date range  : {df.index.min()} -> {df.index.max()}")
    print(f"  Feature count      : {len(FEATURE_COLS)}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Chronological 80/20 split
    n_total = len(df)
    n_dev   = int(n_total * 0.80)
    n_test  = n_total - n_dev

    X_dev, y_dev = X.iloc[:n_dev], y.iloc[:n_dev]
    X_test, y_test = X.iloc[n_dev:], y.iloc[n_dev:]

    split_ts = df.index[n_dev]
    print(f"\n  Development set rows : {n_dev:,} (80.0%) [{df.index[0]} to {df.index[n_dev-1]}]")
    print(f"  Final test set rows  : {n_test:,} (20.0%) [{df.index[n_dev]} to {df.index[-1]}]")
    print(f"  Split cutoff timestamp: {split_ts}")

    # Save split summary
    split_summary = pd.DataFrame([{
        "total_rows": n_total,
        "dev_rows": n_dev,
        "test_rows": n_test,
        "dev_start": str(df.index[0]),
        "dev_end": str(df.index[n_dev-1]),
        "test_start": str(df.index[n_dev]),
        "test_end": str(df.index[-1]),
        "split_timestamp": str(split_ts),
        "cv_folds": N_CV_SPLITS,
    }])
    split_summary.to_csv(TABLE_DIR / "solar_split_summary.csv", index=False)

    # ---------------------------------------------------------------------------
    # TimeSeriesSplit on Development Set
    # ---------------------------------------------------------------------------
    sec("2. TIME SERIES SPLIT (Expanding Window, n_splits=3)")
    cv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_dev), 1):
        tr_start, tr_end = df.index[tr_idx[0]], df.index[tr_idx[-1]]
        va_start, va_end = df.index[val_idx[0]], df.index[val_idx[-1]]
        print(f"  Fold {fold}: Train={len(tr_idx):,} ({tr_start} -> {tr_end}) | Val={len(val_idx):,} ({va_start} -> {va_end})")

    # ---------------------------------------------------------------------------
    # Baseline: Ridge Regression
    # ---------------------------------------------------------------------------
    sec("3. BASELINE: RIDGE REGRESSION")
    def build_ridge():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])

    print("  Evaluating Ridge baseline (alpha=1.0) on 3-fold CV ...")
    res_ridge = cross_validate_pipeline(build_ridge, X_dev, y_dev, cv)
    print(f"    Ridge CV RMSE: {res_ridge['rmse_mean']:.4f} ± {res_ridge['rmse_std']:.4f}")
    print(f"    Ridge CV MAE : {res_ridge['mae_mean']:.4f} ± {res_ridge['mae_std']:.4f}")
    print(f"    Ridge CV R²  : {res_ridge['r2_mean']:.4f} ± {res_ridge['r2_std']:.4f}")
    print(f"    Training time: {res_ridge['fit_time']:.2f}s")

    # ---------------------------------------------------------------------------
    # Optuna Hyperparameter Tuning
    # ---------------------------------------------------------------------------
    sec(f"4. OPTUNA TUNING ({N_OPTUNA_TRIALS} trials per tree model on dev CV)")
    best_params = {}
    tuned_cv_results = {"Ridge": res_ridge}

    # --- 4.1 Random Forest ---
    print("\n  [1/4] Tuning Random Forest ...")
    def obj_rf(trial):
        n_estimators      = trial.suggest_int("n_estimators", 50, 100, step=25)
        max_depth         = trial.suggest_int("max_depth", 8, 14, step=2)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8, step=2)
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features      = trial.suggest_categorical("max_features", ["sqrt", 0.8, 1.0])

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=2 # Triggers sklearn progress prints in terminal
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_rf = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(obj_rf, n_trials=N_OPTUNA_TRIALS)
    best_params["Random Forest"] = study_rf.best_params
    print(f"    Best RF params: {study_rf.best_params}")

    def build_best_rf():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**best_params["Random Forest"], random_state=RANDOM_STATE, n_jobs=-1, verbose=2))
        ])
    print("\n    -> Running final CV for best Random Forest...")
    tuned_cv_results["Random Forest"] = cross_validate_pipeline(build_best_rf, X_dev, y_dev, cv)

    # --- 4.2 XGBoost ---
    print("\n  [2/4] Tuning XGBoost ...")
    def obj_xgb(trial):
        n_estimators     = trial.suggest_int("n_estimators", 60, 120, step=20)
        max_depth        = trial.suggest_int("max_depth", 4, 8)
        learning_rate    = trial.suggest_float("learning_rate", 0.04, 0.15, log=True)
        subsample        = trial.suggest_float("subsample", 0.6, 0.9, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 7)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=2 # Triggers XGBoost internal tree-building logs
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(obj_xgb, n_trials=N_OPTUNA_TRIALS)
    best_params["XGBoost"] = study_xgb.best_params
    print(f"    Best XGBoost params: {study_xgb.best_params}")

    def build_best_xgb():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(**best_params["XGBoost"], random_state=RANDOM_STATE, n_jobs=-1, verbosity=2))
        ])
    print("\n    -> Running final CV for best XGBoost...")
    tuned_cv_results["XGBoost"] = cross_validate_pipeline(build_best_xgb, X_dev, y_dev, cv)

    # --- 4.3 LightGBM ---
    print("\n  [3/4] Tuning LightGBM ...")
    def obj_lgb(trial):
        n_estimators     = trial.suggest_int("n_estimators", 60, 120, step=20)
        num_leaves       = trial.suggest_int("num_leaves", 15, 63, step=8)
        max_depth        = trial.suggest_int("max_depth", 4, 8)
        learning_rate    = trial.suggest_float("learning_rate", 0.04, 0.15, log=True)
        subsample        = trial.suggest_float("subsample", 0.6, 0.9, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=1 # Prints LightGBM info logs instead of staying silent
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(obj_lgb, n_trials=N_OPTUNA_TRIALS)
    best_params["LightGBM"] = study_lgb.best_params
    print(f"    Best LightGBM params: {study_lgb.best_params}")

    def build_best_lgb():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(**best_params["LightGBM"], random_state=RANDOM_STATE, n_jobs=-1, verbose=1))
        ])
    print("\n    -> Running final CV for best LightGBM...")
    tuned_cv_results["LightGBM"] = cross_validate_pipeline(build_best_lgb, X_dev, y_dev, cv)

    # --- 4.4 CatBoost ---
    print("\n  [4/4] Tuning CatBoost ...")
    def obj_cb(trial):
        iterations    = trial.suggest_int("iterations", 60, 120, step=20)
        depth         = trial.suggest_int("depth", 4, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.04, 0.15, log=True)
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 9.0, step=2.0)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    random_seed=RANDOM_STATE,
                    verbose=50, # Prints metrics every 50 trees
                    thread_count=-1
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_cb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_cb.optimize(obj_cb, n_trials=N_OPTUNA_TRIALS)
    best_params["CatBoost"] = study_cb.best_params
    print(f"    Best CatBoost params: {study_cb.best_params}")

    def build_best_cb():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(**best_params["CatBoost"], random_seed=RANDOM_STATE, verbose=50, thread_count=-1))
        ])
    print("\n    -> Running final CV for best CatBoost...")
    tuned_cv_results["CatBoost"] = cross_validate_pipeline(build_best_cb, X_dev, y_dev, cv)

    # Save best parameters table
    param_records = []
    for model_name, p in best_params.items():
        param_records.append({
            "model": model_name,
            "best_cv_rmse": tuned_cv_results[model_name]["rmse_mean"],
            "best_params": str(p),
            "n_trials": N_OPTUNA_TRIALS,
            "random_seed": RANDOM_STATE
        })
    df_params = pd.DataFrame(param_records)
    df_params.to_csv(TABLE_DIR / "solar_best_parameters.csv", index=False)

    # ---------------------------------------------------------------------------
    # 5. Model Comparison on Development Cross-Validation
    # ---------------------------------------------------------------------------
    sec("5. DEVELOPMENT CROSS-VALIDATION COMPARISON")
    cv_table_rows = []
    models_order = ["Ridge", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    for m in models_order:
        r = tuned_cv_results[m]
        cv_table_rows.append({
            "Model": m,
            "CV RMSE Mean": round(r["rmse_mean"], 4),
            "CV RMSE Std":  round(r["rmse_std"], 4),
            "CV MAE Mean":  round(r["mae_mean"], 4),
            "CV R² Mean":   round(r["r2_mean"], 4),
            "Training Time (s)": round(r["fit_time"], 2)
        })
    df_cv = pd.DataFrame(cv_table_rows)
    df_cv.to_csv(TABLE_DIR / "solar_cv_results.csv", index=False)
    print(df_cv.to_string(index=False))

    # Select best model using CV RMSE
    best_model_name = df_cv.sort_values("CV RMSE Mean").iloc[0]["Model"]
    print(f"\n  ⭐ Selected Best Model: {best_model_name} (Lowest CV RMSE = {df_cv.loc[df_cv['Model']==best_model_name, 'CV RMSE Mean'].values[0]:.4f})")

    # Map model name to pipeline builder
    pipe_builders = {
        "Ridge": build_ridge,
        "Random Forest": build_best_rf,
        "XGBoost": build_best_xgb,
        "LightGBM": build_best_lgb,
        "CatBoost": build_best_cb,
    }

    # ---------------------------------------------------------------------------
    # 6. Fit Selected Best Model on Entire Dev Set & Evaluate on Untouched Test Set
    # ---------------------------------------------------------------------------
    sec(f"6. FINAL MODEL FIT ({best_model_name}) ON DEV SET & TEST EVALUATION")
    final_pipe = pipe_builders[best_model_name]()
    print(f"  Fitting {best_model_name} on entire development set ({len(X_dev):,} rows) ...")
    t_fit_start = time.time()
    final_pipe.fit(X_dev, y_dev)
    t_fit_total = time.time() - t_fit_start
    print(f"  Fit complete in {t_fit_total:.2f}s.")

    # Measure test set inference latency
    print(f"  Evaluating on untouched test set ({len(X_test):,} rows) ...")
    t_pred_start = time.perf_counter()
    y_test_pred = final_pipe.predict(X_test)
    t_pred_total = time.perf_counter() - t_pred_start

    ms_per_sample = (t_pred_total / len(X_test)) * 1000.0
    test_rmse, test_mae, test_r2 = eval_metrics(y_test, y_test_pred)

    print(f"    Test RMSE        : {test_rmse:.4f} kW")
    print(f"    Test MAE         : {test_mae:.4f} kW")
    print(f"    Test R²          : {test_r2:.4f}")
    print(f"    Inference Time   : {t_pred_total:.4f} s (Total for {len(X_test):,} samples)")
    print(f"    Latency / sample : {ms_per_sample:.4f} ms")

    # Save final model
    saved_model_path = MODEL_DIR / f"solar_{best_model_name.lower().replace(' ', '_')}_final.joblib"
    joblib.dump(final_pipe, saved_model_path)
    print(f"  Saved final model to: {saved_model_path}")

    # Also evaluate all other models on test set for comprehensive benchmark table
    sec("7. ALL MODELS FINAL TEST BENCHMARK")
    test_benchmark_rows = []
    for m in models_order:
        p = pipe_builders[m]()
        p.fit(X_dev, y_dev)

        t_start = time.perf_counter()
        preds = p.predict(X_test)
        t_lat = time.perf_counter() - t_start

        rmse, mae, r2 = eval_metrics(y_test, preds)
        test_benchmark_rows.append({
            "Model": m,
            "Test RMSE": round(rmse, 4),
            "Test MAE":  round(mae, 4),
            "Test R²":   round(r2, 4),
            "Inference Time (s)": round(t_lat, 4),
            "Latency (ms/sample)": round((t_lat / len(X_test)) * 1000.0, 4)
        })
    df_test = pd.DataFrame(test_benchmark_rows)
    df_test.to_csv(TABLE_DIR / "solar_final_test_results.csv", index=False)
    print(df_test.to_string(index=False))

    # ---------------------------------------------------------------------------
    # 8. Write Markdown Benchmark Report
    # ---------------------------------------------------------------------------
    sec("8. WRITING BENCHMARK REPORT")
    def to_md_table(d):
        header = "| " + " | ".join(d.columns) + " |\n"
        sep = "| " + " | ".join(["---"] * len(d.columns)) + " |\n"
        rows = ""
        for idx, row in d.iterrows():
            rows += "| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |\n"
        return header + sep + rows

    report_content = f"""# Solar Model Benchmark Report — Phase 3 & 4

> **Solar forecasting target = whole master-meter `Active_Power` (1-hour ahead: `target_t1`).**
> 
> **Evaluation performed strictly chronologically with expanding-window TimeSeriesSplit (3 folds).**
> **Final 20% test set remained untouched until final evaluation.**

---

## 1. Dataset & Chronological Split
- **Total supervised rows**: {n_total:,}
- **Development set (80%)**: {n_dev:,} rows (`{df.index[0]}` to `{df.index[n_dev-1]}`)
- **Final test set (20%)**: {n_test:,} rows (`{df.index[n_dev]}` to `{df.index[-1]}`)
- **Exact split timestamp**: `{split_ts}`
- **Feature space (16)**: 5 physical weather, 2 wind direction circular, 6 cyclical temporal, 3 power lags.
- **Missing values**: `SimpleImputer(strategy='median')` fitted strictly inside each training fold.

---

## 2. Cross-Validation Configuration
- **Validation method**: Expanding-window `TimeSeriesSplit` with `n_splits = 3` on development set only.
- **Optimization**: Optuna ({N_OPTUNA_TRIALS} trials per tree model, TPE sampler, minimize mean CV RMSE).

---

## 3. Development Cross-Validation Benchmark Table
{to_md_table(df_cv)}

---

## 4. Best Hyperparameters (Optuna)
```python
{str(best_params)}
```

---

## 5. Model Selection
- **Selected best model**: **{best_model_name}**
- **Criterion**: Lowest mean CV RMSE on development set ({df_cv.loc[df_cv['Model']==best_model_name, 'CV RMSE Mean'].values[0]:.4f} kW).

---

## 6. Final Test Set Evaluation (Untouched 20%)
{to_md_table(df_test)}

### Selected Model ({best_model_name}) Performance:
- **Test RMSE**: {test_rmse:.4f} kW
- **Test MAE**: {test_mae:.4f} kW
- **Test R²**: {test_r2:.4f}
- **Total Inference Time**: {t_pred_total:.4f} s
- **Per-Sample Latency**: {ms_per_sample:.4f} ms

---

## 7. Artifacts Saved
- **Final trained model**: `{saved_model_path}`
- **CV comparison table**: `{TABLE_DIR / 'solar_cv_results.csv'}`
- **Final test results table**: `{TABLE_DIR / 'solar_final_test_results.csv'}`
- **Best parameters table**: `{TABLE_DIR / 'solar_best_parameters.csv'}`
- **Split summary**: `{TABLE_DIR / 'solar_split_summary.csv'}`

---

## 8. Important Limitations
1. Standby inverter power (~ -0.46 kW) remains represented in target and predictions as whole-meter net power.
2. `Wind_Speed` has ~51.6% sensor missingness, handled via fold-level median imputation.
3. Chronological evaluation reflects realistic non-stationary meteorological and seasonal conditions over 17 years.
"""
    REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"  Saved benchmark report: {REPORT_PATH}")

    sec("PHASE 3-4 BENCHMARK COMPLETE")
    print(f"""
  1. Development row count   : {n_dev:,}
  2. Final test row count    : {n_test:,}
  3. Exact split timestamp   : {split_ts}
  4. TimeSeriesSplit config  : Expanding window, 3 folds, no shuffling
  5. Missing-value strategy  : SimpleImputer(median) fitted inside each train fold
  6. Baseline Ridge CV RMSE  : {res_ridge['rmse_mean']:.4f} ± {res_ridge['rmse_std']:.4f} kW
  7. Selected Best Model     : {best_model_name}
  8. Final Test RMSE         : {test_rmse:.4f} kW
  9. Final Test MAE          : {test_mae:.4f} kW
 10. Final Test R²           : {test_r2:.4f}
 11. Inference Latency       : {ms_per_sample:.4f} ms / sample ({t_pred_total:.4f} s total)
 12. Saved Model Path        : {saved_model_path}
 13. Saved Results Tables    : {TABLE_DIR}
 14. Saved Benchmark Report  : {REPORT_PATH}
""")

if __name__ == "__main__":
    main()