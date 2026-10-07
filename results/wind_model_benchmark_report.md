# Kelmarsh Wind Farm — Model Benchmark Report (Phase 3 & 4)

> **Task**: 1-Hour-Ahead Farm Active Power Forecasting ($X_t \to Y_{t+1}$).
> **Dataset**: Complete 4,313 hourly observations (no 35k truncation applied).
> **Chronological Split**: 3,450 dev (80%) / 863 test (20%) at `2021-05-26 00:00:00`.

---

## 1. Cross-Validation Results (Expanding-Window TSS=3 on Dev Set)

| Model | CV RMSE (kW) | CV RMSE Std | CV MAE (kW) | CV R² |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | 1080.7875 | 41.5511 | 718.9989 | 0.9127 |
| CatBoost | 1090.2537 | 54.2644 | 768.0243 | 0.9110 |
| LightGBM | 1108.1220 | 54.0417 | 775.1635 | 0.9088 |
| XGBoost | 1110.7806 | 57.6858 | 784.3431 | 0.9081 |
| Ridge | 1462.5320 | 132.5469 | 1125.3411 | 0.8366 |

---

## 2. Final Untouched Test Set Evaluation (20% Test)

| Model | Test RMSE (kW) | Test MAE (kW) | Test R² | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 672.6327 | 491.5578 | 0.8567 | 0.0365 |
| **LightGBM** | 680.9114 | 518.7359 | 0.8531 | 0.0055 |
| **CatBoost** | 690.6949 | 512.0658 | 0.8489 | 0.0035 |
| **XGBoost** | 709.5924 | 552.9903 | 0.8405 | 0.0067 |
| **Ridge** | 858.7317 | 640.7931 | 0.7664 | 0.0032 |

**Best Model Selected**: **Random Forest** with Test RMSE of **672.6327 kW** and $R^2$ of **0.8567**.
