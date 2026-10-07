# Short-Term Solar and Wind Power Forecasting Pipeline

> **BTech Machine Learning Project**  
> **Pipelines**: Solar PV (DKASC Alice Springs) & Wind Farm (Kelmarsh Wind Farm SCADA)  
> **Primary Task**: Direct 1-Hour-Ahead Power Forecasting ($X_t \to Y_{t+1}$)  
> **Secondary Task**: Direct 24-Hour-Ahead Power Forecasting ($X_t \to Y_{t+24}$)  
> **Core Principle**: Strictly chronological, temporal leakage-free, reproducible, and computationally practical machine learning.

---

## 1. Project Overview & Objectives

This project builds two end-to-end machine learning forecasting pipelines for renewable energy generation:
1. **Solar PV Pipeline**: Predicts total master-meter active power for the Desert Knowledge Australia Solar Centre (DKASC) in Alice Springs.
2. **Wind Farm Pipeline**: Predicts total farm-level active power across 6 Senvion MM92 turbines (~12.3 MW capacity) for the Kelmarsh Wind Farm in the UK.

Both pipelines strictly enforce **chronological splits (80% development, 20% untouched final test)** and expanding-window cross-validation (`TimeSeriesSplit(n_splits=3)`) to prevent temporal data leakage. All feature transformations, imputations, and tunings are fitted strictly within training folds.

```
                    +------------------------------------------+
                    | Raw Sensors / SCADA (10-min / 5-min)     |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Phase 1: Preprocessing & 1h Aggregation  |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Phase 2: Feature Engineering & Lags (t)  |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Phase 3-4: Chronological Benchmark & CV  |
                    |   (Ridge, RF, XGBoost, LightGBM, CatBoost)|
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Phase 5: Feature Reduction (MI & RFECV)  |
                    |          & Direct 24-Hour Demo           |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Phase 6: TreeExplainer SHAP Interpret    |
                    +------------------------------------------+
```

---

## 2. Phase 1 — Data Preprocessing & Aggregation

### 2.1 Solar Preprocessing (DKASC Alice Springs)
- **Source Files**: `96-Site_DKA-MasterMeter1.csv` (Active Power target) and `101-Site_DKA-WeatherStation.csv` (Dedicated weather sensors).
- **Target Definition**: Whole master-meter `Active_Power` signal (bidirectional; negative standby/inverter consumption preserved; no artificial clipping to 5.6 kW).
- **Sampling & Alignment**: Resampled from 5-minute sampling to regular 1-hour resolution.
- **Physical Filters**: Sensor extreme plausibility checks; missing values preserved as NaN without forward-filling across long sensor dropouts.
- **Output**: [`data/processed/solar/solar_hourly_clean.csv`](file:///c:/Users/Khush/Projects/ML%20Project/renewable_forecasting/data/processed/solar/solar_hourly_clean.csv).

### 2.2 Wind Preprocessing (Kelmarsh Wind Farm)
- **Source Files**: 6 turbine SCADA files (`Turbine_Data_Kelmarsh_1..6`) covering 2021-01-01 to 2021-06-30 at 10-minute resolution.
- **Hygiene & Filtering**:
  - Clamped physical ranges: power $[-50, 2200]$ kW per turbine, wind speed $[0, 50]$ m/s, pitch $[-5, 95]^\circ$, ambient temp $[-40, 50]^\circ\text{C}$.
  - Resampled each turbine to 1 hour (mean).
- **Farm Aggregation**:
  - **Farm Active Power**: Direct sum across all 6 turbines ($\text{Farm\_Power} = \sum_{i=1}^6 P_i$).
  - **Atmospheric / Operational Signals**: Anemometer wind speed, pitch angle, ambient temperature (mean); wind direction and nacelle position (circular vector average using $\sin/\cos$).
- **Output**: [`data/processed/wind/wind_hourly_clean.csv`](file:///c:/Users/Khush/Projects/ML%20Project/renewable_forecasting/data/processed/wind/wind_hourly_clean.csv).

---

## 3. Phase 2 — Feature Engineering & Leakage Prevention

### 3.1 Solar Feature Engineering
- **Daylight Filter Rule**: Locked project decision `Global_Horizontal_Radiation > 1` applied consistently. Night rows discarded.
- **Computational Limitation**: Development capped at the **most recent 35,000 hourly observations (~4 years)** to ensure fast execution on standard laptops.
- **16 Model Features**:
  - Physical Weather (5): `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Weather_Temperature_Celsius`, `Weather_Relative_Humidity`, `Wind_Speed`.
  - Circular Wind (2): `Wind_Direction_sin`, `Wind_Direction_cos`.
  - Cyclical Temporal (6): `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos`, `Month_sin`, `Month_cos`.
  - Power Lags (3): `Power_lag_1h` ($t-1$), `Power_lag_2h` ($t-2$), `Power_lag_24h` ($t-24$).
- **Target**: `target_t1 = Active_Power.shift(-1)`.
- **Output**: [`data/processed/solar/solar_features_1h.csv`](file:///c:/Users/Khush/Projects/ML%20Project/renewable_forecasting/data/processed/solar/solar_features_1h.csv).

### 3.2 Wind Feature Engineering
- **Dataset Size**: Full 4,313 supervised hourly observations (no 35k truncation needed).
- **16 Model Features**:
  - Physical & Operational (7): `Wind_Speed`, `Wind_Direction_sin`, `Wind_Direction_cos`, `Nacelle_Position_sin`, `Nacelle_Position_cos`, `Blade_Pitch_Angle`, `Ambient_Temperature`.
  - Cyclical Temporal (6): `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos`, `Month_sin`, `Month_cos`.
  - Power Lags (3): `Power_lag_1h`, `Power_lag_2h`, `Power_lag_24h`.
- **Target**: `target_t1 = Farm_Power.shift(-1)`.
- **Leakage Audits**: Monotonic timestamps verified; assertion $|Y_{t+1} - P_{t+1}| = 0$ passed; zero future information in $X_t$.
- **Output**: [`data/processed/wind/wind_features_1h.csv`](file:///c:/Users/Khush/Projects/ML%20Project/renewable_forecasting/data/processed/wind/wind_features_1h.csv).

---

## 4. Phase 3 & 4 — Model Benchmarking & Hyperparameter Tuning

Five model families were evaluated across both pipelines using strict chronological splits:
- **Ridge Regression**: Baseline linear pipeline (`SimpleImputer(median) -> StandardScaler() -> Ridge`).
- **Tree Models**: `SimpleImputer(median) -> Model` (no scaling).
- **Tuning**: Optuna (15 trials per tree model) evaluated strictly using expanding-window `TimeSeriesSplit(n_splits=3)` on the 80% development set.

### 4.1 Solar Model Benchmark Results (1-Hour Ahead)
*Untouched test set: 7,000 observations (`2024-10-01` to `2025-08-23`)*

| Model | CV RMSE (kW) | Test RMSE (kW) | Test MAE (kW) | Test $R^2$ | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 10.97 | 9.39 | 3.97 | 0.9779 | 0.0459 |
| **LightGBM (Selected)** | **10.96** | **9.47** | **4.15** | **0.9776** | **0.0117** |
| **CatBoost** | 11.12 | 9.59 | 4.42 | 0.9770 | 0.0052 |
| **XGBoost** | 11.29 | 9.65 | 4.31 | 0.9767 | 0.0070 |
| **Ridge (Baseline)** | 15.34 | 15.18 | 11.16 | 0.9423 | 0.0035 |

> **Solar Winner**: **LightGBM** achieved the lowest CV RMSE (10.96 kW) and high test $R^2$ (0.9776) with $4\times$ faster inference than Random Forest.

### 4.2 Wind Model Benchmark Results (1-Hour Ahead)
*Untouched test set: 863 observations (`2021-05-26` to `2021-06-30`)*

| Model | CV RMSE (kW) | Test RMSE (kW) | Test MAE (kW) | Test $R^2$ | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)**| **1080.79** | **672.63** | **491.56** | **0.8567** | **0.0315** |
| **LightGBM** | 1108.12 | 680.91 | 518.74 | 0.8531 | 0.0047 |
| **CatBoost** | 1090.25 | 690.69 | 512.07 | 0.8489 | 0.0030 |
| **XGBoost** | 1110.78 | 709.59 | 552.99 | 0.8405 | 0.0058 |
| **Ridge (Baseline)** | 1462.53 | 858.73 | 640.79 | 0.7664 | 0.0028 |

> **Wind Winner**: **Random Forest** achieved the lowest Test RMSE (672.63 kW) and highest test $R^2$ (0.8567).

---

## 5. Phase 5 — Feature Selection & 24-Hour Forecasting Demonstrations

### 5.1 Sensor-Cost Reduction via Feature Subsets
Features were ranked using Mutual Information (MI) and RFECV on development data only. Four subsets (Full 16, 12, 8, 5) were evaluated on the untouched test sets:

- **Solar (LightGBM)**:
  - Full (16 features): Test RMSE = **9.50 kW**, $R^2 = \mathbf{0.9774}$
  - 12 features: Test RMSE = **9.47 kW**, $R^2 = \mathbf{0.9776}$
  - 8 features: Test RMSE = **9.51 kW**, $R^2 = \mathbf{0.9774}$
  - 5 features: Test RMSE = 10.72 kW, $R^2 = 0.9713$
  - *Outcome*: The **8-feature subset** cuts sensor overhead by $50\%$ while losing zero accuracy.

- **Wind (Random Forest)**:
  - Full (16 features): Test RMSE = **672.45 kW**, $R^2 = \mathbf{0.8567}$
  - 12 features: Test RMSE = **674.02 kW**, $R^2 = \mathbf{0.8561}$
  - 8 features: Test RMSE = **700.55 kW**, $R^2 = 0.8445$
  - 5 features: Test RMSE = 712.80 kW, $R^2 = 0.8390$
  - *Outcome*: The **12-feature subset** maintains $99.7\%$ of accuracy; the minimal **5-feature subset** maintains $R^2 = 0.8390$.

### 5.2 Direct 24-Hour-Ahead Forecasting ($X_t \to Y_{t+24}$)
Evaluated direct forecasting using only historical information at time $t$ against the persistence benchmark ($Y_t \to Y_{t+24}$):

- **Solar 24h Forecast**:
  - LightGBM: **Test RMSE = 28.10 kW**, Test $R^2 = \mathbf{0.8026}$
  - Persistence: Test RMSE = 32.46 kW, Test $R^2 = 0.7364$
  - *Improvement*: $13.5\%$ error reduction over persistence driven by daily solar diurnal periodicity.
- **Wind 24h Forecast**:
  - Random Forest: Test RMSE = 4462.47 kW, Test $R^2 = -5.28$
  - Persistence: Test RMSE = 2373.33 kW, Test $R^2 = -0.78$
  - *Physical Insight*: Without future Numerical Weather Prediction (NWP) inputs, wind power lacks a rigid diurnal cycle, and seasonal synoptic shifts between development (spring) and test (summer) lead to degradation in direct 24h autoregression.

---

## 6. Phase 6 — SHAP Explainability & Physical Validation

Model interpretability was computed using `shap.TreeExplainer` on deterministic samples of 800 test observations.

### 6.1 Solar Interpretation (LightGBM)
- **Top Attributions**: `Global_Horizontal_Radiation` ($62.7\%$), `Hour_sin` ($16.2\%$), `Hour_cos` ($8.8\%$), `Power_lag_1h` ($3.6\%$).
- **Physical Alignment**: Strictly monotonic relationship with solar irradiance. Diurnal time encodings effectively capture the sun's elevation and zenith angle trajectories.

### 6.2 Wind Interpretation (Random Forest)
- **Top Attributions**: `Wind_Speed` ($85.98\%$), `Blade_Pitch_Angle` ($3.44\%$), `Power_lag_1h` ($3.24\%$), `Day_cos` ($1.35\%$).
- **Physical Alignment**: Reconstructed the classic aerodynamic turbine power curve ($P \propto v^3$). Blade pitch angle acts as an aerodynamic brake: near-zero pitch drives positive generation, while high pitch ($>15^\circ$) sharply drives negative SHAP adjustments.

---

## 7. Project Structure

```
renewable_forecasting/
├── data/
│   ├── interim/
│   │   ├── solar/
│   │   └── wind/                    # Cleaned 10-minute turbine SCADA files
│   ├── processed/
│   │   ├── solar/
│   │   │   ├── solar_hourly_clean.csv
│   │   │   └── solar_features_1h.csv
│   │   └── wind/
│   │       ├── wind_hourly_clean.csv
│   │       └── wind_features_1h.csv
│   └── raw/
│       ├── solar/                   # DKASC MasterMeter & WeatherStation CSVs
│       └── wind/                    # Kelmarsh 6 Turbine SCADA CSVs
├── results/
│   ├── figures/
│   │   ├── shap/solar/              # Bar, beeswarm, dependence, waterfall plots
│   │   ├── shap/wind/               # Bar, beeswarm, dependence, waterfall plots
│   │   ├── solar_feature_reduction.png
│   │   ├── solar_24h_actual_vs_predicted.png
│   │   ├── wind_model_benchmark.png
│   │   ├── wind_feature_reduction.png
│   │   └── wind_24h_actual_vs_predicted.png
│   ├── models/
│   │   ├── solar/solar_lightgbm_final.joblib
│   │   └── wind/wind_random_forest_final.joblib
│   └── tables/                      # Metrics, feature subsets, MI, and split summaries
├── src/
│   ├── data/
│   │   ├── preprocess_solar.py
│   │   └── preprocess_wind.py
│   ├── features/
│   │   ├── solar_features.py
│   │   └── wind_features.py
│   ├── feature_selection/
│   │   ├── solar_feature_selection.py
│   │   └── wind_feature_selection.py
│   ├── models/
│   │   ├── solar_benchmark.py
│   │   ├── solar_24h_demo.py
│   │   ├── wind_benchmark.py
│   │   └── wind_24h_demo.py
│   └── explainability/
│       ├── solar_shap.py
│       └── wind_shap.py
└── README.md
```

---

## 8. Execution Guide

To reproduce the entire pipeline sequentially from the terminal:

```powershell
# Navigate to project directory
cd "renewable_forecasting"

# 1. Preprocessing & Clean Hourly Aggregation
python src/data/preprocess_solar.py
python src/data/preprocess_wind.py

# 2. Feature Engineering & Leakage Checks
python src/features/solar_features.py
python src/features/wind_features.py

# 3. Model Benchmarking & Tuning (Ridge, RF, XGB, LGBM, CatBoost)
python src/models/solar_benchmark.py
python src/models/wind_benchmark.py

# 4. Feature Selection & Sensor-Cost Reduction
python src/feature_selection/solar_feature_selection.py
python src/feature_selection/wind_feature_selection.py

# 5. Direct 24-Hour Forecasting Demonstrations
python src/models/solar_24h_demo.py
python src/models/wind_24h_demo.py

# 6. TreeExplainer SHAP Explainability & Physical Alignment
python src/explainability/solar_shap.py
python src/explainability/wind_shap.py
```
