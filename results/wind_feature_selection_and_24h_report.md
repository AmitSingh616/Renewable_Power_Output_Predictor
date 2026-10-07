# Kelmarsh Wind Farm — Feature Reduction & 24-Hour Forecast Report

> **Task**: Feature selection, sensor reduction, and direct 24-hour forecasting ($X_t \to Y_{t+24}$).
> **Model**: Tuned Random Forest (`{'n_estimators': 100, 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 6, 'max_features': 1.0, 'random_state': 42, 'n_jobs': -1}`).

---

## 1. 24-Hour Direct Forecast Performance

| Model | Features | Test RMSE (kW) | Test MAE (kW) | Test R² | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Random Forest (Tuned) | 8 features | 4462.9443 | 3966.0815 | -5.2847 | 0.0345 |
| Persistence Baseline (Y_t) | Farm_Power only | 2373.3266 | 1612.3537 | -0.7773 | 0.0000 |

---
*Direct 24-hour demonstration completed cleanly without recursive prediction or future information leakage.*
