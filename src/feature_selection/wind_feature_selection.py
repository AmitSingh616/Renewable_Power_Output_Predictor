"""
Wind Feature Selection & Sensor-Cost Reduction
===============================================
Kelmarsh Wind Farm (~4,313 hourly observations).
Model: Best tuned tree model from Phase 3-4 (Random Forest).

Objective:
1. Mutual Information ranking on development set (80%).
2. RFECV with TimeSeriesSplit(n_splits=3) on development set.
3. Define 4 subsets: Full (16), 12, 8, 5 features based on MI & RFECV rankings.
4. Evaluate tuned Random Forest across all 4 subsets on dev CV and untouched test set (20%).
5. Save metrics tables and comparison plot.

Constraints:
- Complete dataset used (no 35k row cap).
- Chronological 80/20 train/test split.
- Final test set remains untouched during feature selection.
- Median imputation fitted inside CV folds/dev split only.
- Fixed hyperparameter set from Phase 4 for Random Forest.
"""

import sys, os, time, warnings, ast
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "wind" / "wind_features_1h.csv"
TABLE_DIR = ROOT / "results" / "tables"
WIND_TAB_DIR = TABLE_DIR / "wind"
FIG_DIR = ROOT / "results" / "figures"
REPORT_PATH = ROOT / "results" / "wind_feature_selection_and_24h_report.md"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
WIND_TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_SPLITS = 3

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

# Tuned parameters for Random Forest from Phase 4
# {'n_estimators': 100, 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 6, 'max_features': 1.0}
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 14,
    "min_samples_split": 8,
    "min_samples_leaf": 6,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# If hyperparams table exists, dynamically load
HYPER_FILE = WIND_TAB_DIR / "wind_best_hyperparameters.csv"
if not HYPER_FILE.exists():
    HYPER_FILE = TABLE_DIR / "wind_best_hyperparameters.csv"

if HYPER_FILE.exists():
    try:
        df_hp = pd.read_csv(HYPER_FILE)
        rf_row = df_hp[df_hp["model"] == "Random Forest"]
        if not rf_row.empty:
            p_dict = ast.literal_eval(rf_row.iloc[0]["best_params"])
            RF_PARAMS.update(p_dict)
            print(f"Loaded RF hyperparameters from {HYPER_FILE.name}: {RF_PARAMS}")
    except Exception as e:
        print(f"Using default locked RF parameters: {e}")

DIVIDER = "=" * 75

def sec(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def cross_validate_subset(features, X_dev, y_dev, cv):
    rmse_list, mae_list, r2_list = [], [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_dev), 1):
        X_tr, y_tr = X_dev.iloc[tr_idx][features], y_dev.iloc[tr_idx]
        X_va, y_va = X_dev.iloc[val_idx][features], y_dev.iloc[val_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestRegressor(**RF_PARAMS))
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)

        rmse, mae, r2 = eval_metrics(y_va, preds)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    return {
        "cv_rmse_mean": np.mean(rmse_list),
        "cv_rmse_std": np.std(rmse_list),
        "cv_mae_mean": np.mean(mae_list),
        "cv_mae_std": np.std(mae_list),
        "cv_r2_mean": np.mean(r2_list),
        "cv_r2_std": np.std(r2_list),
    }

def main(dry_run=False):
    sec("1. LOAD WIND FEATURES & CHRONOLOGICAL SPLIT")
    assert DATA_PATH.exists(), f"Features file not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    print(f"  Raw feature rows : {len(df):,}")
    print(f"  Active time range: {df.index.min()} -> {df.index.max()}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    n_total = len(df)
    n_dev = int(n_total * 0.80)
    n_test = n_total - n_dev

    X_dev, y_dev = X.iloc[:n_dev], y.iloc[:n_dev]
    X_test, y_test = X.iloc[n_dev:], y.iloc[n_dev:]

    print(f"  Development rows (80%): {n_dev:,} ({df.index[0]} to {df.index[n_dev-1]})")
    print(f"  Final test rows  (20%): {n_test:,} ({df.index[n_dev]} to {df.index[-1]})")

    # ---------------------------------------------------------------------------
    # Step 1: Mutual Information on Development Data
    # ---------------------------------------------------------------------------
    sec("2. MUTUAL INFORMATION ANALYSIS (DEV SET ONLY)")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_dev_imputed = pd.DataFrame(
        imputer.fit_transform(X_dev),
        columns=FEATURE_COLS,
        index=X_dev.index
    )

    print("  Calculating Mutual Information against target_t1...")
    mi_scores = mutual_info_regression(
        X_dev_imputed,
        y_dev,
        random_state=RANDOM_STATE
    )

    df_mi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mutual_information": mi_scores
    }).sort_values(by="mutual_information", ascending=False).reset_index(drop=True)
    df_mi["mi_rank"] = range(1, len(df_mi) + 1)

    print("\n  Mutual Information Rankings:")
    print(df_mi.to_string(index=False))

    df_mi.to_csv(TABLE_DIR / "wind_mutual_information.csv", index=False)
    df_mi.to_csv(WIND_TAB_DIR / "wind_mutual_information.csv", index=False)
    print(f"\n  Saved MI rankings -> {TABLE_DIR / 'wind_mutual_information.csv'}")

    # ---------------------------------------------------------------------------
    # Step 2: RFECV with TimeSeriesSplit on Development Data
    # ---------------------------------------------------------------------------
    sec("3. RFECV FEATURE RANKING (DEV SET ONLY, TSS=3)")
    cv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    rf_step = 2 if dry_run else 1
    rfecv_estimator = RandomForestRegressor(**RF_PARAMS)

    rfecv = RFECV(
        estimator=rfecv_estimator,
        step=rf_step,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        min_features_to_select=1
    )

    print(f"  Running RFECV over expanding folds (step={rf_step})...")
    t0 = time.time()
    rfecv.fit(X_dev_imputed, y_dev)
    t_rfecv = time.time() - t0
    print(f"  RFECV complete in {t_rfecv:.2f}s")
    print(f"  Optimal feature count per RFECV: {rfecv.n_features_}")

    df_rfecv = pd.DataFrame({
        "feature": FEATURE_COLS,
        "rfecv_rank": rfecv.ranking_,
        "rfecv_selected": rfecv.support_
    }).sort_values(by="rfecv_rank", ascending=True).reset_index(drop=True)

    print("\n  RFECV Rankings:")
    print(df_rfecv.to_string(index=False))

    # Consolidated Ranking (MI + RFECV)
    df_rankings = pd.merge(df_mi, df_rfecv, on="feature")
    df_rankings["combined_rank_score"] = df_rankings["mi_rank"] + df_rankings["rfecv_rank"]
    df_rankings = df_rankings.sort_values(by=["combined_rank_score", "mi_rank"]).reset_index(drop=True)
    print("\n  Consolidated Ranking (MI + RFECV):")
    print(df_rankings.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Step 3: Define 4 Subsets (Full 16, 12, 8, 5)
    # ---------------------------------------------------------------------------
    sec("4. DEFINING FEATURE SUBSETS")
    sorted_features = df_rankings["feature"].tolist()

    subsets = {
        "Full (16 features)": sorted_features[:16],
        "12 features": sorted_features[:12],
        "8 features": sorted_features[:8],
        "5 features": sorted_features[:5],
    }

    subset_rows = []
    for name, f_list in subsets.items():
        subset_rows.append({
            "subset_name": name,
            "feature_count": len(f_list),
            "features": "; ".join(f_list)
        })
        print(f"  {name} ({len(f_list)} features):")
        for f in f_list:
            print(f"    - {f}")

    df_subsets = pd.DataFrame(subset_rows)
    df_subsets.to_csv(TABLE_DIR / "wind_feature_subsets.csv", index=False)
    df_subsets.to_csv(WIND_TAB_DIR / "wind_feature_subsets.csv", index=False)
    print(f"\n  Saved subsets -> {TABLE_DIR / 'wind_feature_subsets.csv'}")

    # ---------------------------------------------------------------------------
    # Step 4: Evaluate Subsets on Dev CV and Final Untouched Test Set
    # ---------------------------------------------------------------------------
    sec("5. EVALUATION OF FEATURE SUBSETS")
    eval_records = []

    for name, f_list in subsets.items():
        print(f"\n  >>> Evaluating Subset: {name} ({len(f_list)} features)...")
        # 1. Dev Cross-Validation
        cv_res = cross_validate_subset(f_list, X_dev, y_dev, cv)

        # 2. Fit on full Dev Set, evaluate on untouched Test Set
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestRegressor(**RF_PARAMS))
        ])

        t_fit_start = time.perf_counter()
        pipe.fit(X_dev[f_list], y_dev)
        t_fit = time.perf_counter() - t_fit_start

        t_inf_start = time.perf_counter()
        test_preds = pipe.predict(X_test[f_list])
        t_inf = time.perf_counter() - t_inf_start

        test_rmse, test_mae, test_r2 = eval_metrics(y_test, test_preds)
        latency_ms = (t_inf / len(X_test)) * 1000.0

        eval_records.append({
            "Subset": name,
            "Feature Count": len(f_list),
            "CV RMSE (kW)": round(cv_res["cv_rmse_mean"], 4),
            "CV MAE (kW)": round(cv_res["cv_mae_mean"], 4),
            "CV R²": round(cv_res["cv_r2_mean"], 4),
            "Test RMSE (kW)": round(test_rmse, 4),
            "Test MAE (kW)": round(test_mae, 4),
            "Test R²": round(test_r2, 4),
            "Fit Time (s)": round(t_fit, 4),
            "Inference Time (s)": round(t_inf, 4),
            "Latency (ms/sample)": round(latency_ms, 4)
        })

    df_results = pd.DataFrame(eval_records)
    print("\n  Summary Evaluation Results:")
    print(df_results.to_string(index=False))

    df_results.to_csv(TABLE_DIR / "wind_feature_reduction_results.csv", index=False)
    df_results.to_csv(WIND_TAB_DIR / "wind_feature_reduction_results.csv", index=False)
    print(f"\n  Saved reduction results -> {TABLE_DIR / 'wind_feature_reduction_results.csv'}")

    # ---------------------------------------------------------------------------
    # Step 5: Generate Comparison Visualization
    # ---------------------------------------------------------------------------
    sec("6. GENERATING FEATURE REDUCTION VISUALIZATION")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    counts = df_results["Feature Count"].tolist()

    # Plot 1: RMSE
    ax = axes[0]
    ax.plot(counts, df_results["CV RMSE (kW)"], marker="o", label="CV RMSE (kW)", color="#1f77b4", linewidth=2)
    ax.plot(counts, df_results["Test RMSE (kW)"], marker="s", label="Test RMSE (kW)", color="#d62728", linewidth=2)
    ax.set_title("RMSE vs Feature Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE (kW)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Test R²
    ax = axes[1]
    ax.plot(counts, df_results["Test R²"], marker="^", color="#2ca02c", linewidth=2)
    ax.set_title("Test R² vs Feature Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("R²")
    ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Latency
    ax = axes[2]
    ax.bar([str(c) for c in counts], df_results["Latency (ms/sample)"], color="#9467bd", alpha=0.85, width=0.5)
    ax.set_title("Inference Latency per Sample", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Latency (ms/sample)")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Kelmarsh Wind Farm — Feature Selection & Sensor Reduction (Random Forest)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = FIG_DIR / "wind_feature_reduction.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure -> {fig_path}")

    sec("WIND FEATURE SELECTION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    is_dry = "--dry-run" in sys.argv
    main(dry_run=is_dry)
