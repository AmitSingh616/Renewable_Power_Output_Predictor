"""
Phase 3-4: Chronological Evaluation, Hyperparameter Tuning & Model Benchmarking
for Kelmarsh Wind Farm Power Forecasting
================================================================================
Models:
1. Ridge (StandardScaler + Ridge, baseline)
2. Random Forest (Optuna tuned)
3. XGBoost (Optuna tuned)
4. LightGBM (Optuna tuned)
5. CatBoost (Optuna tuned)

Methodology:
- Dataset: All 4,313 supervised hourly observations from wind_features_1h.csv
- Chronological Split: 80% Dev (3,450 rows), 20% untouched Test (863 rows)
- TimeSeriesSplit(n_splits=3) on Dev set for CV and Optuna tuning (15 trials each)
- Evaluation metrics: RMSE, MAE, R2, Inference Latency
- Save CV, best hyperparams, and final test benchmark tables
- Save trained final models (.joblib)
- Save comparison visualization and markdown benchmark report
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
import matplotlib.pyplot as plt

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

# Keep Optuna logging concise (WARNING to reduce terminal clutter)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT         = Path(__file__).resolve().parents[2]
DATA_PATH    = ROOT / "data" / "processed" / "wind" / "wind_features_1h.csv"
MODEL_DIR    = ROOT / "results" / "models" / "wind"
TABLE_DIR    = ROOT / "results" / "tables"
WIND_TAB_DIR = TABLE_DIR / "wind"
FIG_DIR      = ROOT / "results" / "figures"
REPORT_PATH  = ROOT / "results" / "wind_model_benchmark_report.md"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
WIND_TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_SPLITS  = 3
N_OPTUNA_TRIALS = 15

FEATURE_COLS = [
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

TARGET_COL = "target_t1"

DIVIDER = "=" * 75

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

def main(dry_run=False):
    trials_budget = 2 if dry_run else N_OPTUNA_TRIALS

    sec("1. LOAD WIND FEATURES DATASET & CHRONOLOGICAL SPLIT")
    assert DATA_PATH.exists(), f"Features dataset not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    print(f"  Total supervised rows: {len(df):,}")
    print(f"  Active Date range    : {df.index.min()} -> {df.index.max()}")
    print(f"  Feature count        : {len(FEATURE_COLS)}")

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

    # TimeSeriesSplit on Development Set
    cv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    sec("2. BASELINE: RIDGE REGRESSION")
    def build_ridge():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE))
        ])

    print("  -> Evaluating Ridge across 3 expanding folds...")
    ridge_cv = cross_validate_pipeline(build_ridge, X_dev, y_dev, cv)
    print(f"     Ridge CV RMSE: {ridge_cv['rmse_mean']:.4f} kW (+/- {ridge_cv['rmse_std']:.4f}) | R2: {ridge_cv['r2_mean']:.4f}")

    sec(f"3. HYPERPARAMETER TUNING ({trials_budget} trials per tree model)")
    best_params = {}
    tuned_cv_results = {"Ridge": ridge_cv}

    # 3.1 Random Forest
    print("\n  [1/4] Tuning Random Forest...")
    def obj_rf(trial):
        n_estimators     = trial.suggest_int("n_estimators", 40, 100, step=20)
        max_depth        = trial.suggest_int("max_depth", 6, 14)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 6)
        max_features     = trial.suggest_categorical("max_features", [0.7, 0.9, 1.0])

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_rf = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(obj_rf, n_trials=trials_budget)
    best_params["Random Forest"] = study_rf.best_params
    print(f"    Best RF params: {study_rf.best_params}")

    def build_best_rf():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestRegressor(**best_params["Random Forest"], random_state=RANDOM_STATE, n_jobs=-1))
        ])
    tuned_cv_results["Random Forest"] = cross_validate_pipeline(build_best_rf, X_dev, y_dev, cv)

    # 3.2 XGBoost
    print("\n  [2/4] Tuning XGBoost...")
    def obj_xgb(trial):
        n_estimators     = trial.suggest_int("n_estimators", 40, 100, step=20)
        max_depth        = trial.suggest_int("max_depth", 4, 8)
        learning_rate    = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        subsample        = trial.suggest_float("subsample", 0.6, 0.9, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 7)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(obj_xgb, n_trials=trials_budget)
    best_params["XGBoost"] = study_xgb.best_params
    print(f"    Best XGBoost params: {study_xgb.best_params}")

    def build_best_xgb():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", XGBRegressor(**best_params["XGBoost"], random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))
        ])
    tuned_cv_results["XGBoost"] = cross_validate_pipeline(build_best_xgb, X_dev, y_dev, cv)

    # 3.3 LightGBM
    print("\n  [3/4] Tuning LightGBM...")
    def obj_lgb(trial):
        n_estimators      = trial.suggest_int("n_estimators", 40, 100, step=20)
        num_leaves        = trial.suggest_int("num_leaves", 15, 63, step=8)
        max_depth         = trial.suggest_int("max_depth", 4, 8)
        learning_rate     = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 50, step=10)
        subsample         = trial.suggest_float("subsample", 0.6, 0.9, step=0.1)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", LGBMRegressor(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(obj_lgb, n_trials=trials_budget)
    best_params["LightGBM"] = study_lgb.best_params
    print(f"    Best LightGBM params: {study_lgb.best_params}")

    def build_best_lgb():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", LGBMRegressor(**best_params["LightGBM"], random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
        ])
    tuned_cv_results["LightGBM"] = cross_validate_pipeline(build_best_lgb, X_dev, y_dev, cv)

    # 3.4 CatBoost
    print("\n  [4/4] Tuning CatBoost...")
    def obj_cat(trial):
        iterations      = trial.suggest_int("iterations", 60, 150, step=30)
        depth           = trial.suggest_int("depth", 4, 8)
        learning_rate   = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        l2_leaf_reg     = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, step=1.0)
        random_strength = trial.suggest_float("random_strength", 0.1, 2.0, step=0.3)

        def make_pipe():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    random_strength=random_strength,
                    random_seed=RANDOM_STATE,
                    thread_count=-1,
                    verbose=0
                ))
            ])
        res = cross_validate_pipeline(make_pipe, X_dev, y_dev, cv)
        return res["rmse_mean"]

    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_cat.optimize(obj_cat, n_trials=trials_budget)
    best_params["CatBoost"] = study_cat.best_params
    print(f"    Best CatBoost params: {study_cat.best_params}")

    def build_best_cat():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", CatBoostRegressor(**best_params["CatBoost"], random_seed=RANDOM_STATE, thread_count=-1, verbose=0))
        ])
    tuned_cv_results["CatBoost"] = cross_validate_pipeline(build_best_cat, X_dev, y_dev, cv)

    # Save Best Parameters
    param_records = []
    for m_name, p_dict in best_params.items():
        param_records.append({
            "model": m_name,
            "best_cv_rmse": tuned_cv_results[m_name]["rmse_mean"],
            "best_params": str(p_dict),
            "n_trials": trials_budget,
            "random_seed": RANDOM_STATE
        })
    df_params = pd.DataFrame(param_records)
    df_params.to_csv(TABLE_DIR / "wind_best_hyperparameters.csv", index=False)
    df_params.to_csv(WIND_TAB_DIR / "wind_best_hyperparameters.csv", index=False)

    # Save CV Results
    cv_records = []
    for m_name, res in tuned_cv_results.items():
        cv_records.append({
            "Model": m_name,
            "CV RMSE (kW)": round(res["rmse_mean"], 4),
            "CV RMSE Std": round(res["rmse_std"], 4),
            "CV MAE (kW)": round(res["mae_mean"], 4),
            "CV MAE Std": round(res["mae_std"], 4),
            "CV R²": round(res["r2_mean"], 4),
            "CV R² Std": round(res["r2_std"], 4),
            "Fit Time (s)": round(res["fit_time"], 4)
        })
    df_cv = pd.DataFrame(cv_records).sort_values(by="CV RMSE (kW)").reset_index(drop=True)
    df_cv.to_csv(TABLE_DIR / "wind_model_benchmark_results.csv", index=False)
    df_cv.to_csv(WIND_TAB_DIR / "wind_model_benchmark_results.csv", index=False)

    print("\n  Summary Cross-Validation Results:")
    print(df_cv.to_string(index=False))

    sec("4. EVALUATION ON UNTOUCHED FINAL TEST SET (20%)")
    final_builders = {
        "Ridge": build_ridge,
        "Random Forest": build_best_rf,
        "XGBoost": build_best_xgb,
        "LightGBM": build_best_lgb,
        "CatBoost": build_best_cat,
    }

    test_benchmark_rows = []
    test_predictions = {}

    for m_name, b_fn in final_builders.items():
        print(f"  Refitting and evaluating {m_name}...")
        p = b_fn()
        t_fit_start = time.perf_counter()
        p.fit(X_dev, y_dev)
        t_fit = time.perf_counter() - t_fit_start

        # Measure inference latency on test set
        t_inf_start = time.perf_counter()
        preds = p.predict(X_test)
        t_inf = time.perf_counter() - t_inf_start

        test_predictions[m_name] = preds
        rmse, mae, r2 = eval_metrics(y_test, preds)
        latency_ms = (t_inf / len(X_test)) * 1000.0

        # Save model joblib
        model_filename = f"wind_{m_name.lower().replace(' ', '_')}_final.joblib"
        joblib.dump(p, MODEL_DIR / model_filename)

        test_benchmark_rows.append({
            "Model": m_name,
            "Test RMSE (kW)": round(rmse, 4),
            "Test MAE (kW)": round(mae, 4),
            "Test R²": round(r2, 4),
            "Fit Time (s)": round(t_fit, 4),
            "Inference Time (s)": round(t_inf, 4),
            "Latency (ms/sample)": round(latency_ms, 4)
        })

    df_test = pd.DataFrame(test_benchmark_rows).sort_values(by="Test RMSE (kW)").reset_index(drop=True)
    df_test.to_csv(TABLE_DIR / "wind_final_model_results.csv", index=False)
    df_test.to_csv(WIND_TAB_DIR / "wind_final_model_results.csv", index=False)

    print("\n  Summary Untouched Test Set Results:")
    print(df_test.to_string(index=False))

    best_model_name = df_test.iloc[0]["Model"]
    print(f"\n  >>> BEST OVERALL WIND MODEL: {best_model_name} (Test RMSE = {df_test.iloc[0]['Test RMSE (kW)']} kW)")

    sec("5. GENERATING COMPARISON FIGURE & REPORT")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    m_names = df_test["Model"].tolist()
    rmses   = df_test["Test RMSE (kW)"].tolist()
    maes    = df_test["Test MAE (kW)"].tolist()
    r2s     = df_test["Test R²"].tolist()

    # RMSE
    axes[0].bar(m_names, rmses, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Test RMSE (kW) — Lower is Better", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("RMSE (kW)")
    axes[0].tick_params(axis="x", rotation=25)

    # MAE
    axes[1].bar(m_names, maes, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Test MAE (kW) — Lower is Better", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("MAE (kW)")
    axes[1].tick_params(axis="x", rotation=25)

    # R2
    axes[2].bar(m_names, r2s, color="#2ca02c", alpha=0.85)
    axes[2].set_title("Test R² — Higher is Better", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis="x", rotation=25)

    plt.suptitle("Kelmarsh Wind Farm — Model Benchmarking Results", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = FIG_DIR / "wind_model_benchmark.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved benchmark plot -> {fig_path}")

    # Write Markdown Benchmark Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Kelmarsh Wind Farm — Model Benchmark Report (Phase 3 & 4)\n\n")
        f.write("> **Task**: 1-Hour-Ahead Farm Active Power Forecasting ($X_t \\to Y_{t+1}$).\n")
        f.write(f"> **Dataset**: Complete {n_total:,} hourly observations (no 35k truncation applied).\n")
        f.write(f"> **Chronological Split**: {n_dev:,} dev (80%) / {n_test:,} test (20%) at `{split_ts}`.\n\n")
        f.write("---\n\n## 1. Cross-Validation Results (Expanding-Window TSS=3 on Dev Set)\n\n")
        f.write("| Model | CV RMSE (kW) | CV RMSE Std | CV MAE (kW) | CV R² |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in df_cv.iterrows():
            f.write(f"| {r['Model']} | {r['CV RMSE (kW)']:.4f} | {r['CV RMSE Std']:.4f} | {r['CV MAE (kW)']:.4f} | {r['CV R²']:.4f} |\n")

        f.write("\n---\n\n## 2. Final Untouched Test Set Evaluation (20% Test)\n\n")
        f.write("| Model | Test RMSE (kW) | Test MAE (kW) | Test R² | Latency (ms/sample) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in df_test.iterrows():
            f.write(f"| **{r['Model']}** | {r['Test RMSE (kW)']:.4f} | {r['Test MAE (kW)']:.4f} | {r['Test R²']:.4f} | {r['Latency (ms/sample)']:.4f} |\n")

        f.write(f"\n**Best Model Selected**: **{best_model_name}** with Test RMSE of **{df_test.iloc[0]['Test RMSE (kW)']:.4f} kW** and $R^2$ of **{df_test.iloc[0]['Test R²']:.4f}**.\n")

    print(f"  [+] Saved benchmark report -> {REPORT_PATH}")
    sec("WIND MODEL BENCHMARKING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    is_dry = "--dry-run" in sys.argv
    main(dry_run=is_dry)
