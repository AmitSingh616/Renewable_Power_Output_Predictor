# Solar PV Power Forecasting — SHAP Explainability & Physical Alignment Report

> **Model Analyzed**: Final Tuned LightGBM Regressor (`solar_lightgbm_final.joblib`).
> **Evaluation Set**: Chronological test set sample (800 observations between `2024-10-02 14:00:00` and `2025-08-22 19:00:00`).
> **Disclaimer**: SHAP analysis identifies feature attributions and associational dependencies within the learned tree structure; it does not constitute causal proof.

---

## 1. Global Feature Importance Hierarchy

| Rank | Feature | Mean |SHAP| (kW) | Relative Share (%) | Physical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Global_Horizontal_Radiation` | 46.5589 | 62.71% | Direct solar irradiance driver |
| 2 | `Hour_sin` | 12.0069 | 16.17% | Diurnal solar zenith angle cycle |
| 3 | `Hour_cos` | 6.5369 | 8.80% | Diurnal solar zenith angle cycle |
| 4 | `Power_lag_1h` | 2.6558 | 3.58% | Autoregressive persistence / inertia |
| 5 | `Diffuse_Horizontal_Radiation` | 2.1975 | 2.96% | Direct solar irradiance driver |
| 6 | `Power_lag_24h` | 1.1504 | 1.55% | Autoregressive persistence / inertia |
| 7 | `Power_lag_2h` | 1.0254 | 1.38% | Autoregressive persistence / inertia |
| 8 | `Weather_Temperature_Celsius` | 0.9810 | 1.32% | Thermal PV cell efficiency & air mass |
| 9 | `Day_cos` | 0.3182 | 0.43% | Seasonal sun declination cycle |
| 10 | `Day_sin` | 0.2439 | 0.33% | Seasonal sun declination cycle |
| 11 | `Month_cos` | 0.1296 | 0.17% | Seasonal sun declination cycle |
| 12 | `Wind_Direction_sin` | 0.1218 | 0.16% | Convective PV module cooling |
| 13 | `Weather_Relative_Humidity` | 0.1198 | 0.16% | Atmospheric attenuation / cloud moisture |
| 14 | `Wind_Direction_cos` | 0.1038 | 0.14% | Convective PV module cooling |
| 15 | `Month_sin` | 0.0912 | 0.12% | Seasonal sun declination cycle |

---

## 2. Physical Consistency & Model Mechanics

1. **Dominant Drivers (`Global_Horizontal_Radiation`, `Hour_sin`)**:
   - The model relies most strongly on `Global_Horizontal_Radiation` and `Hour_sin`, which is physically expected for short-term 1-hour PV forecasting.
   - Higher irradiance values correspond consistently to positive SHAP contributions (pushing forecasted generation upward), while low or declining radiation yields negative adjustments.

2. **Autoregressive Power Lags**:
   - Short-term lag (`Power_lag_1h`) serves as the baseline anchor, capturing immediate cloud coverage states and inverter operational constraints.
   - The 24-hour lag (`Power_lag_24h`) captures diurnal periodic baseline expectation, reflecting conditions at the exact same hour on the preceding day.

3. **Secondary Atmospheric Variables**:
   - `Weather_Temperature_Celsius` displays moderate importance consistent with seasonal ambient conditions, while `Wind_Speed` has minor negative or positive cooling adjustments.

4. **Representative Local Explanation**:
   - For the sample observation at `2024-10-09 11:00:00`, the model forecast of **170.03 kW** aligned closely with actual generation of **169.72 kW**.

---
*SHAP analysis executed cleanly using TreeExplainer with zero data leakage.*
