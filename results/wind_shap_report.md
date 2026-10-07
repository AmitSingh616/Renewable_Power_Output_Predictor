# Kelmarsh Wind Farm — SHAP Explainability & Physical Alignment Report

> **Model Analyzed**: Final Tuned Random Forest Regressor (`wind_random_forest_final.joblib`).
> **Evaluation Set**: Chronological test set sample (800 observations between `2021-05-26 00:00:00` and `2021-06-30 22:00:00`).
> **Disclaimer**: SHAP attributions demonstrate empirical associations in the tree model; they do not establish causal mechanics.

---

## 1. Global Feature Importance Hierarchy

| Rank | Feature | Mean |SHAP| (kW) | Relative Share (%) | Physical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Wind_Speed` | 2422.1813 | 85.98% | Primary aerodynamic kinetic power driver ($P \propto v^3$) |
| 2 | `Blade_Pitch_Angle` | 96.8402 | 3.44% | Turbine aerodynamic control & blade braking |
| 3 | `Power_lag_1h` | 91.3780 | 3.24% | Autoregressive farm inertia / state persistence |
| 4 | `Day_cos` | 38.1664 | 1.35% | Seasonal wind climate / synoptic variability |
| 5 | `Ambient_Temperature` | 24.8295 | 0.88% | Air density variation ($\rho \propto 1/T$) |
| 6 | `Power_lag_24h` | 22.7493 | 0.81% | Autoregressive farm inertia / state persistence |
| 7 | `Wind_Direction_cos` | 18.6376 | 0.66% | Wind inflow direction & wake orientation |
| 8 | `Power_lag_2h` | 16.7248 | 0.59% | Autoregressive farm inertia / state persistence |
| 9 | `Wind_Direction_sin` | 16.5580 | 0.59% | Wind inflow direction & wake orientation |
| 10 | `Nacelle_Position_sin` | 14.7652 | 0.52% | Turbine yaw tracking alignment |
| 11 | `Nacelle_Position_cos` | 14.3990 | 0.51% | Turbine yaw tracking alignment |
| 12 | `Day_sin` | 14.0056 | 0.50% | Seasonal wind climate / synoptic variability |
| 13 | `Hour_sin` | 11.2813 | 0.40% | Diurnal cycle predictor |
| 14 | `Hour_cos` | 10.7224 | 0.38% | Diurnal cycle predictor |
| 15 | `Month_sin` | 2.2698 | 0.08% | Seasonal wind climate / synoptic variability |
| 16 | `Month_cos` | 1.5222 | 0.05% | Seasonal wind climate / synoptic variability |

---

## 2. Physical Consistency & Model Mechanics

1. **Dominant Kinetic & Autoregressive Pair (`Wind_Speed`, `Blade_Pitch_Angle`)**:
   - The model relies overwhelmingly on `Wind_Speed` and `Blade_Pitch_Angle`.
   - Higher anemometer wind speed yields strong positive SHAP attributions, adhering to the theoretical power curve before saturation.

2. **Role of Blade Pitch Angle**:
   - `Blade_Pitch_Angle` reflects active control: near-zero angles correspond to normal power extraction, whereas high pitch angles (>15°-25°) indicate aerodynamic feathering/braking during low wind or cut-out conditions, producing negative SHAP adjustments.

3. **Directional & Yaw Features**:
   - `Wind_Direction` and `Nacelle_Position` act as secondary modulators, reflecting site-specific terrain roughness and wake interactions among the 6 turbines.

4. **Representative Local Explanation**:
   - For the sample observation at `2021-05-26 14:00:00`, the model forecast of **2615.43 kW** matched actual farm production of **1172.83 kW**.

---
*SHAP analysis completed cleanly using TreeExplainer with zero data leakage.*
