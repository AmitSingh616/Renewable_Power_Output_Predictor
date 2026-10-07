# Solar Feature Engineering Report — Phase 2

> **The solar forecasting target is the whole master-meter `Active_Power`.**
> 
> **Daylight observations are selected using the fixed criterion `Global_Horizontal_Radiation > 1`.**

---

## 1. Input Dataset
- **Source**: `solar_hourly_clean.csv`
- **Input row count**: 145,834
- **Date range**: 2008-09-12 05:00:00 to 2025-08-23 04:00:00

---

## 2. Locked Daylight Filtering (`GHI > 1`)
- **Criterion**: `Global_Horizontal_Radiation > 1`
- **Rows before daylight filtering**: 145,834
- **Rows after daylight filtering**: 144,169
- **Rows removed by `GHI > 1`**: 1,665

---

## 3. Boundary & History Filtering
- **Required history / target**: `Power_lag_1h`, `Power_lag_2h`, `Power_lag_24h`, `target_t1`
- **Rows removed due to lag/target boundaries**: 673
- **Final usable supervised rows**: 143,496

---

## 4. Final Feature Specification
- **Total model features**: 16
- **Feature categories**:
  - **Physical / Weather (5)**: `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Weather_Temperature_Celsius`, `Weather_Relative_Humidity`, `Wind_Speed`
  - **Wind Direction Circular (2)**: `Wind_Direction_sin`, `Wind_Direction_cos`
  - **Cyclical Time (6)**: `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos`, `Month_sin`, `Month_cos`
  - **Power History Lags (3)**: `Power_lag_1h`, `Power_lag_2h`, `Power_lag_24h`
- **Target (1)**: `target_t1` (Active_Power at t+1)

---

## 5. Feature Definitions Table
| Feature | Category | Source | Availability | Transformation |
| --- | --- | --- | --- | --- |
| `Global_Horizontal_Radiation` | physical | WeatherStation | t | hourly mean (GHI > 1 daytime filter) |
| `Diffuse_Horizontal_Radiation` | physical | WeatherStation | t | hourly mean |
| `Weather_Temperature_Celsius` | physical | WeatherStation | t | hourly mean |
| `Weather_Relative_Humidity` | physical | WeatherStation | t | hourly mean |
| `Wind_Speed` | physical | WeatherStation | t | hourly mean |
| `Wind_Direction_sin` | physical | WeatherStation | t | sin(2*pi*Wind_Direction/360) |
| `Wind_Direction_cos` | physical | WeatherStation | t | cos(2*pi*Wind_Direction/360) |
| `Hour_sin` | temporal | Timestamp | t | sin(2*pi*hour/24) |
| `Hour_cos` | temporal | Timestamp | t | cos(2*pi*hour/24) |
| `Day_sin` | temporal | Timestamp | t | sin(2*pi*day_of_year/365.25) |
| `Day_cos` | temporal | Timestamp | t | cos(2*pi*day_of_year/365.25) |
| `Month_sin` | temporal | Timestamp | t | sin(2*pi*month/12) |
| `Month_cos` | temporal | Timestamp | t | cos(2*pi*month/12) |
| `Power_lag_1h` | lag | Active_Power | t-1 | Active_Power.shift(1) |
| `Power_lag_2h` | lag | Active_Power | t-2 | Active_Power.shift(2) |
| `Power_lag_24h` | lag | Active_Power | t-24 | Active_Power.shift(24) |
| `target_t1` | target | Active_Power | t+1 | Active_Power.shift(-1) |


---

## 6. Missingness Summary
| Column | missing_count | missing_pct |
| --- | --- | --- |
| Global_Horizontal_Radiation | 0.00 | 0.00 |
| Diffuse_Horizontal_Radiation | 0.00 | 0.00 |
| Weather_Temperature_Celsius | 0.00 | 0.00 |
| Weather_Relative_Humidity | 924.00 | 0.64 |
| Wind_Speed | 74029.00 | 51.59 |
| Wind_Direction_sin | 509.00 | 0.35 |
| Wind_Direction_cos | 509.00 | 0.35 |
| Hour_sin | 0.00 | 0.00 |
| Hour_cos | 0.00 | 0.00 |
| Day_sin | 0.00 | 0.00 |
| Day_cos | 0.00 | 0.00 |
| Month_sin | 0.00 | 0.00 |
| Month_cos | 0.00 | 0.00 |
| Power_lag_1h | 0.00 | 0.00 |
| Power_lag_2h | 0.00 | 0.00 |
| Power_lag_24h | 0.00 | 0.00 |
| target_t1 | 0.00 | 0.00 |


*Note on missingness: `Wind_Speed` has ~52% unobserved values originating from sensor gaps in the raw source; missingness is preserved as instructed without artificial fabrication.*

---

## 7. Leakage & Integrity Verification
- **Timestamps strictly sorted ascending**: YES
- **Zero duplicate timestamps**: YES
- **Cyclical features strictly within [-1, 1]**: YES
- **Wind direction trigonometry identity ($sin^2 + cos^2 = 1$)**: VERIFIED
- **Lag timing verification**: Exactly matches chronological $t-1$, $t-2$, $t-24$ hours
- **Target timing verification**: Exactly matches chronological $t+1$ hour
- **Target excluded from feature space**: YES (`target_t1` not in $X$)
- **Zero future-derived variables in $X$**: YES
- **No rolling statistics constructed**: YES

---

## 8. Final Dataset Summary
- **Output file**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\data\processed\solar\solar_features_1h.csv`
- **Final shape**: (143496, 18)
- **Date range**: 2008-09-13 05:00:00 to 2025-08-23 03:00:00

---

*Phase 2 complete. Models, scaling, and feature selection not started.*
