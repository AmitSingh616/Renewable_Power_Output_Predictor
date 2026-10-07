# Solar Feature Reduction & 24-Hour Forecast Demonstration Report

> **Project**: BTech ML Project — Short-Term Solar and Wind Power Forecasting  
> **Dataset**: DKASC Alice Springs (`96-Site_DKA-MasterMeter1.csv`, `101-Site_DKA-WeatherStation.csv`)  
> **Target**: Whole master-meter `Active_Power` signal  
> **Daylight Filter**: `Global_Horizontal_Radiation > 1`  
> **Dataset Subsetting**: Most recent 35,000 observations (~4 years, `2021-05-24 18:00:00` to `2025-08-23 03:00:00`)  
> **Model**: Tuned LightGBM Regressor (`n_estimators=60, num_leaves=39, max_depth=8, lr=0.102, subsample=0.6, colsample_bytree=0.9`)

---

## 1. Computational Limitation & Methodology Note

To maintain computational feasibility on standard student hardware without sacrificing scientific validity:
- Modeling uses the **most recent ~35,000 hourly observations (~4 years)** of data.
- **Split Strategy**: Strict chronological split:
  - **Development set (80%)**: 28,000 rows (`2021-05-24 18:00:00` to `2024-10-01 20:00:00`).
  - **Untouched test set (20%)**: 7,000 rows (`2024-10-01 21:00:00` to `2025-08-23 03:00:00`).
- The final 20% test set is strictly shielded from feature selection, ranking, imputation, and hyperparameter tuning.

---

## 2. Part A — Feature Selection & Sensor-Cost Reduction

### 2.1 Feature Importance & Ranking Pipeline
Features are evaluated exclusively on the development set using two complementary methodologies:
1. **Mutual Information (MI)**: Non-linear statistical dependency between each feature and $Y_{t+1}$.
2. **Recursive Feature Elimination with Cross-Validation (RFECV)**: LightGBM estimator evaluated across 3-fold expanding chronological window (`TimeSeriesSplit(n_splits=3)`).

### 2.2 Feature Subsets Evaluated
Four subsets are constructed to assess performance retention vs sensor requirements:
- **Full (16 features)**: Complete meteorological, temporal, and power lag feature set.
- **12 features**: Drops least informative components (e.g., secondary wind directional sinusoids).
- **8 features**: Retains core solar radiation, key temporal cycles, and primary power lag.
- **5 features**: Minimal operational sensor configuration (e.g., GHI, primary lag, diurnal time encodings).

### 2.3 Generated Artifacts
- `results/tables/solar_mutual_information.csv`: Full MI scores on development set.
- `results/tables/solar_feature_subsets.csv`: Feature lists for Full, 12, 8, and 5 features.
- `results/tables/solar_feature_reduction_results.csv`: Cross-validation and untouched test metrics (RMSE, MAE, $R^2$, latency).
- `results/figures/solar_feature_reduction.png`: Trade-off curves comparing performance across subset sizes.

---

## 3. Part B — 24-Hour-Ahead Direct Forecasting Demonstration

### 3.1 Task Formulation
- Direct multi-step forecasting: $X_t \to Y_{t+24}$
- Target: `target_t24 = Active_Power.shift(-24)`
- Uses tuned LightGBM architecture with the optimal reduced feature subset.
- Baseline comparison: Persistence benchmark ($Y_t \to Y_{t+24}$).

### 3.2 Generated Artifacts
- `results/tables/solar_24h_results.csv`: Test set performance metrics for 24-hour horizon.
- `results/figures/solar_24h_actual_vs_predicted.png`: Time series actual vs predicted trace and scatter parity plot.

---

## 4. Execution Instructions

The scripts are implemented with high terminal verbosity and ready to run sequentially:

```powershell
# 1. Run Feature Selection & Sensor Reduction
C:\Users\Khush\AppData\Local\Python\bin\python.exe src/feature_selection/solar_feature_selection.py

# 2. Run 24-Hour Direct Forecast Demonstration
C:\Users\Khush\AppData\Local\Python\bin\python.exe src/models/solar_24h_demo.py
```
