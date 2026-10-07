# Solar Preprocessing Report — Phase 1

> **Solar forecasting target = whole master-meter `Active_Power`
> from `96-Site_DKA-MasterMeter1.csv`**
> (Not a single 5.6 kW Site 1 array.)

---

## Raw Data

| Property | Power File | Weather File |
|----------|-----------|--------------|
| Filename | `96-Site_DKA-MasterMeter1.csv` | `101-Site_DKA-WeatherStation.csv` |
| Raw rows | 1,767,903 | 1,761,522 |
| Date min | 2008-09-12 05:55:00 | 2008-09-12 05:55:00 |
| Date max | 2025-08-23 05:00:00 | 2025-08-23 05:55:00 |
| Median step | 5.0 min | 5.0 min |
| Duplicate timestamps | 0 | 0 |

---

## Cleaning

### Active_Power

| Rule | Count |
|------|-------|
| Values below -10.0 kW set to NaN | 0 |
| Values above 300.0 kW set to NaN | 0 |
| NaN before cleaning | 27,126 |

Negative values in the -10 to 0 range are **preserved** — they represent
genuine inverter standby/reactive draw on the whole-facility master meter.

### Weather

| Cleaning Rule | Values Set to NaN |
|---------------|-------------------|
| Global_Horizontal_Radiation (GHI < 0) | 1 |
| Diffuse_Horizontal_Radiation (DHI < 0) | 0 |
| Weather_Relative_Humidity (RH < 0 or > 100) | 14,589 |
| Wind_Speed (Wind_Speed < 0) | 16 |
| Wind_Speed (Wind_Speed > 100) | 0 |
| Wind_Direction (Wind_Direction < 0 or > 360) | 10,114 |
| Weather_Temperature_Celsius (Temp < -60 or > 80) | 0 |

---

## Resampling

| Stage | Rows |
|-------|------|
| Power hourly rows | 148,537 |
| Weather hourly rows | 148,537 |
| After outer merge | 148,537 |

Wind direction: circular mean (unit-vector method).
All other variables: arithmetic mean over 5-min intervals within each hour.

---

## Daylight Filter

| Stage | Rows |
|-------|------|
| Before GHI > 0 filter | 148,537 |
| After GHI > 0 filter | 145,863 |
| Removed | 2,674 |
| Dropped (NaN target_t1) | 29 |
| **Final daytime rows** | **145,834** |

---

## Target Statistics (daytime rows only)

| Stat | Value |
|------|-------|
| Min | -1.7873 kW |
| Max | 209.1591 kW |
| Mean | 44.8871 kW |
| Median | 0.0969 kW |
| Std | 61.4019 kW |
| % Negative | 49.78% |
| % Zero | 0.02% |
| % Positive | 50.20% |

---

## Final Dataset

- **Path**: `C:\Users\Khush\Projects\ML Project\renewable_forecasting\data\processed\solar\solar_hourly_clean.csv`
- **Shape**: (145834, 8)
- **Date range**: 2008-09-12 05:00:00 to 2025-08-23 04:00:00
- **Columns**: ['Active_Power', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed', 'Wind_Direction', 'target_t1']

### Missingness

| Column | missing_count | missing_pct |
| --- | --- | --- |
| Active_Power | 0.00 | 0.00 |
| Global_Horizontal_Radiation | 0.00 | 0.00 |
| Diffuse_Horizontal_Radiation | 0.00 | 0.00 |
| Weather_Temperature_Celsius | 0.00 | 0.00 |
| Weather_Relative_Humidity | 942.00 | 0.65 |
| Wind_Speed | 76021.00 | 52.13 |
| Wind_Direction | 557.00 | 0.38 |
| target_t1 | 0.00 | 0.00 |


---

## QC Results

- Timestamps sorted: YES
- No duplicate timestamps: YES
- GHI negatives in final dataset: 0
- Wind speed negatives in final dataset: 0
- RH out-of-range in final dataset: 0
- Raw files untouched: YES
- Reload verification: OK

---

*Phase 1 complete. Phase 2 (feature engineering) not started.*
