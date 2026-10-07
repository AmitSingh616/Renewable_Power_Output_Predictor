# Kelmarsh Wind Farm — Preprocessing & Aggregation Report

> **Task**: Preprocessing 6 Senvion MM92 turbines (2,050 kW each, 12.3 MW total farm capacity).
> **Period**: `2021-01-01 00:00:00` to `2021-06-30 23:00:00` (6 months, 10-minute SCADA resampled to 1 hour).

---

## 1. SCADA Hygiene & Data Quality
- **Turbines Included**: 6/6 turbines (`Kelmarsh 1` through `Kelmarsh 6`).
- **10-minute samples per turbine**: 26,064 records (181 days).
- **Hourly resampled farm rows**: 4,344 hourly observations.
- **Power limits**: Allowed standby power consumption down to -50 kW; maximum clamped at 2,200 kW per turbine.
- **Farm Power Formulation**: Direct sum of all 6 operational turbines (`Farm_Power = sum(Power_t1 .. Power_t6)`).
- **Directional Averaging**: Circular vector averaging using $\sin$ and $\cos$ decomposition for wind direction and nacelle position.
- **Predictors**: Farm wind speed, ambient temperature, pitch angle, nacelle position, wind direction.

---

## 2. Farm Hourly Summary Statistics

| Variable | Mean | Std | Min | Median | Max | Missing % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Farm_Power` | 3347.72 | 3576.54 | -31.13 | 1885.09 | 12287.45 | 0.07% |
| `Wind_Speed` | 5.85 | 2.76 | 0.44 | 5.36 | 19.20 | 0.07% |
| `Wind_Direction` | 178.58 | 98.85 | 1.61 | 199.13 | 359.24 | 0.09% |
| `Nacelle_Position` | 178.96 | 101.36 | 0.54 | 200.18 | 359.95 | 0.09% |
| `Blade_Pitch_Angle` | 11.24 | 15.49 | 0.00 | 3.50 | 92.34 | 8.20% |
| `Ambient_Temperature` | 8.30 | 5.54 | -3.35 | 7.82 | 26.93 | 8.22% |

---
*Preprocessing completed cleanly without temporal leakage or artificial data manipulation.*
