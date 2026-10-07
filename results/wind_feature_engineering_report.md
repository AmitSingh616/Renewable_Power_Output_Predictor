# Kelmarsh Wind Farm — Feature Engineering & Leakage Report

> **Task**: 1-Hour-Ahead Total Farm Active Power Forecasting ($X_t \to Y_{t+1}$).
> **Period**: `2021-01-02 00:00:00` to `2021-06-30 22:00:00` (4,313 supervised hourly rows).

---

## 1. Feature Space Architecture (16 Model Features)

| Feature Name | Category | Unit | Formula / Source |
| :--- | :--- | :--- | :--- |
| `Wind_Speed` | Physical Weather | m/s | Anemometer mean across 6 turbines |
| `Wind_Direction_sin` | Directional | [-1, 1] | sin(rad(Wind_Direction)) vector mean |
| `Wind_Direction_cos` | Directional | [-1, 1] | cos(rad(Wind_Direction)) vector mean |
| `Nacelle_Position_sin` | Operational | [-1, 1] | sin(rad(Nacelle_Position)) vector mean |
| `Nacelle_Position_cos` | Operational | [-1, 1] | cos(rad(Nacelle_Position)) vector mean |
| `Blade_Pitch_Angle` | Operational | deg | Blade pitch A mean across 6 turbines |
| `Ambient_Temperature` | Physical Weather | deg C | Nacelle ambient temp mean across 6 turbines |
| `Hour_sin` | Temporal Diurnal | [-1, 1] | sin(2 * pi * hour / 24) |
| `Hour_cos` | Temporal Diurnal | [-1, 1] | cos(2 * pi * hour / 24) |
| `Day_sin` | Temporal Seasonal | [-1, 1] | sin(2 * pi * day / 365.25) |
| `Day_cos` | Temporal Seasonal | [-1, 1] | cos(2 * pi * day / 365.25) |
| `Month_sin` | Temporal Annual | [-1, 1] | sin(2 * pi * month / 12) |
| `Month_cos` | Temporal Annual | [-1, 1] | cos(2 * pi * month / 12) |
| `Power_lag_1h` | Autoregressive Lag | kW | Farm_Power at time t-1 |
| `Power_lag_2h` | Autoregressive Lag | kW | Farm_Power at time t-2 |
| `Power_lag_24h` | Autoregressive Lag | kW | Farm_Power at time t-24 |

---

## 2. Temporal Leakage Audit Checklist

- [x] **Strict Chronological Ordering**: Timestamps verified strictly monotonically increasing.
- [x] **No Future Weather/SCADA Leakage**: All physical variables ($WS_t, Temp_t, Pitch_t$) represent readings available at time $t$ or earlier.
- [x] **No Target Leakage**: Current `Farm_Power` ($t$) and target `target_t1` ($t+1$) are strictly excluded from feature matrix $X_t$.
- [x] **Causal Autoregressive Lags**: Lags are strictly causal ($t-1, t-2, t-24$). Verified via exact numerical shift assertions.
- [x] **Untouched Test Guarantee**: Split will be applied chronologically (80% dev, 20% test); all imputers and scalers will be fit exclusively within training folds.

---

## 3. Dataset Missingness Audit

| Feature | Missing Count | Missing % |
| :--- | :---: | :---: |
| `Wind_Speed` | 1 | 0.02% |
| `Wind_Direction_sin` | 2 | 0.05% |
| `Wind_Direction_cos` | 2 | 0.05% |
| `Nacelle_Position_sin` | 2 | 0.05% |
| `Nacelle_Position_cos` | 2 | 0.05% |
| `Blade_Pitch_Angle` | 353 | 8.19% |
| `Ambient_Temperature` | 354 | 8.21% |
| `Hour_sin` | 0 | 0.00% |
| `Hour_cos` | 0 | 0.00% |
| `Day_sin` | 0 | 0.00% |
| `Day_cos` | 0 | 0.00% |
| `Month_sin` | 0 | 0.00% |
| `Month_cos` | 0 | 0.00% |
| `Power_lag_1h` | 2 | 0.05% |
| `Power_lag_2h` | 3 | 0.07% |
| `Power_lag_24h` | 0 | 0.00% |
| `Farm_Power` | 1 | 0.02% |
| `target_t1` | 0 | 0.00% |

---
*Feature engineering complete and validated for model benchmarking.*
