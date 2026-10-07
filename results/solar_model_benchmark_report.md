# Solar Model Benchmark Report — Phase 3 & 4

> **Solar forecasting target = whole master-meter `Active_Power` (1-hour ahead: `target_t1`).**
> 
> **Evaluation performed strictly chronologically with expanding-window TimeSeriesSplit (3 folds).**
> **Final 20% test set remained untouched until final evaluation.**

---

## 1. Dataset & Chronological Split
- **Total supervised rows**: 35,000
- **Development set (80%)**: 28,000 rows (`2021-05-24 18:00:00` to `2024-10-01 20:00:00`)
- **Final test set (20%)**: 7,000 rows (`2024-10-01 21:00:00` to `2025-08-23 03:00:00`)
- **Exact split timestamp**: `2024-10-01 21:00:00`
- **Feature space (16)**: 5 physical weather, 2 wind direction circular, 6 cyclical temporal, 3 power lags.
- **Missing values**: `SimpleImputer(strategy='median')` fitted strictly inside each training fold.

---

## 2. Cross-Validation Configuration
- **Validation method**: Expanding-window `TimeSeriesSplit` with `n_splits = 3` on development set only.
- **Optimization**: Optuna (15 trials per tree model, TPE sampler, minimize mean CV RMSE).

---

## 3. Development Cross-Validation Benchmark Table
| Model | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV R² Mean | Training Time (s) |
| --- | --- | --- | --- | --- | --- |
| Ridge | 17.3868 | 1.8701 | 12.6407 | 0.9233 | 0.9200 |
| Random Forest | 10.9731 | 1.3209 | 5.0402 | 0.9693 | 71.5900 |
| XGBoost | 11.2907 | 1.7857 | 5.5825 | 0.9672 | 0.8200 |
| LightGBM | 10.9619 | 1.3891 | 5.2316 | 0.9693 | 0.3600 |
| CatBoost | 11.1214 | 1.4668 | 5.6279 | 0.9684 | 1.4500 |


---

## 4. Best Hyperparameters (Optuna)
```python
{'Random Forest': {'n_estimators': 100, 'max_depth': 14, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 1.0}, 'XGBoost': {'n_estimators': 80, 'max_depth': 5, 'learning_rate': 0.08980131249183412, 'subsample': 0.6, 'colsample_bytree': 0.7, 'min_child_weight': 3}, 'LightGBM': {'n_estimators': 60, 'num_leaves': 39, 'max_depth': 8, 'learning_rate': 0.10200369308281163, 'subsample': 0.6, 'colsample_bytree': 0.9}, 'CatBoost': {'iterations': 120, 'depth': 6, 'learning_rate': 0.10372960067462543, 'l2_leaf_reg': 3.0}}
```

---

## 5. Model Selection
- **Selected best model**: **LightGBM**
- **Criterion**: Lowest mean CV RMSE on development set (10.9619 kW).

---

## 6. Final Test Set Evaluation (Untouched 20%)
| Model | Test RMSE | Test MAE | Test R² | Inference Time (s) | Latency (ms/sample) |
| --- | --- | --- | --- | --- | --- |
| Ridge | 15.1805 | 11.1562 | 0.9423 | 0.0035 | 0.0005 |
| Random Forest | 9.3896 | 3.9657 | 0.9779 | 0.0459 | 0.0066 |
| XGBoost | 9.6494 | 4.3136 | 0.9767 | 0.0070 | 0.0010 |
| LightGBM | 9.4697 | 4.1528 | 0.9776 | 0.0117 | 0.0017 |
| CatBoost | 9.5927 | 4.4218 | 0.9770 | 0.0052 | 0.0007 |


### Selected Model (LightGBM) Performance:
- **Test RMSE**: 9.4697 kW
- **Test MAE**: 4.1528 kW
- **Test R²**: 0.9776
- **Total Inference Time**: 0.0082 s
- **Per-Sample Latency**: 0.0012 ms

---

## 7. Artifacts Saved
- **Final trained model**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\results\models\solar\solar_lightgbm_final.joblib`
- **CV comparison table**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\results\tables\solar\solar_cv_results.csv`
- **Final test results table**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\results\tables\solar\solar_final_test_results.csv`
- **Best parameters table**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\results\tables\solar\solar_best_parameters.csv`
- **Split summary**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\results\tables\solar\solar_split_summary.csv`

---

## 8. Important Limitations
1. Standby inverter power (~ -0.46 kW) remains represented in target and predictions as whole-meter net power.
2. `Wind_Speed` has ~51.6% sensor missingness, handled via fold-level median imputation.
3. Chronological evaluation reflects realistic non-stationary meteorological and seasonal conditions over 17 years.
