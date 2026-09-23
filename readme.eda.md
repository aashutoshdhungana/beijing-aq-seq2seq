# Beijing Multi-Site Air-Quality EDA — Theory and Modeling Guide

## 1. Purpose

This pipeline performs a reproducible exploratory analysis of the UCI Beijing
Multi-Site Air-Quality Dataset and prepares the dataset for the next stage:
a multivariate sequence-to-sequence forecasting model with:

- input context: **24 hourly observations**
- forecast horizon: **t+1 through t+24**
- multiple monitoring stations
- pollutant targets such as PM2.5 and/or O3

The EDA is not merely a collection of plots. Its purpose is to answer five
modeling questions:

1. **Is the data trustworthy and temporally continuous?**
2. **What does missingness look like, and how should it be handled?**
3. **Which variables contain predictable temporal, meteorological, and spatial information?**
4. **What transformations or representations are appropriate for ML?**
5. **How should the eventual seq2seq train/validation/test split be designed?**

---

# 2. Dataset mental model

The data are fundamentally a panel time series:

$$
X_{s,t}
$$

where:

- $s$ = monitoring station
- $t$ = hourly timestamp
- $X$ = vector containing pollutants, meteorology, and engineered variables.

This is different from ordinary tabular ML. The order of observations matters.

For forecasting PM2.5:

$$
\hat{y}_{t+1:t+24}
=
f(X_{t-23:t})
$$

A 24-hour input contains the recent diurnal cycle. However, EDA should determine
whether 24 hours is sufficient or whether additional lag/context features may
be useful.

---

# 3. Step 1 — Data ingestion and validation

## What the code does

The pipeline supports:

- a directory containing station CSV files;
- a single already-concatenated CSV;
- automatic station identification from standard UCI filenames when necessary;
- datetime construction;
- duplicate-row removal;
- numeric type conversion;
- chronological sorting.

## Why this matters

Time-series models assume that timestamps are correctly ordered.

If rows are accidentally shuffled, duplicated, or assigned to the wrong station,
later operations such as interpolation, lag construction, rolling statistics,
and sequence generation can become invalid.

### Key validation questions

Check:

- number of stations;
- number of observations per station;
- minimum/maximum timestamp;
- duplicate timestamps;
- gaps larger than one hour;
- unexpected station names;
- unexpected column names.

A nominally "four-year hourly dataset" does not automatically imply that every
station has a perfectly continuous hourly sequence.

---

# 4. Step 2 — Missing-value analysis

Missingness is especially important in environmental sensor data.

Let:

$$
M_{t} =
\begin{cases}
1 & \text{if the measurement is missing}\\
0 & \text{otherwise}
\end{cases}
$$

The EDA examines both:

- **variable-level missingness**
- **station × variable missingness**
- **temporal missingness patterns**

## Why station-level analysis matters

A pollutant may be almost complete at one station but heavily missing at
another. A global missing percentage would hide this.

## Why temporal missingness matters

A missing value in isolation is different from a six-hour or six-day sensor outage.

For forecasting, a long missing block can destroy many possible training windows.

---

# 5. Step 3 — Smart imputation

The pipeline uses a two-stage EDA imputation strategy.

## Stage A: short-gap interpolation

For gaps up to three consecutive hours:

$$
x_t \leftarrow \text{linear interpolation}(x_{t-k},x_{t+k})
$$

Time interpolation is appropriate when a sensor temporarily misses a few
measurements but the physical process is expected to evolve smoothly.

This should **not** be blindly applied to long outages.

## Stage B: grouped median fallback

Remaining missing values are filled using:

1. station × month × hour median
2. station × hour median
3. station median
4. global median as a final fallback

The first level is particularly useful because pollution distributions differ
by location, season, and hour.

### Important modeling warning

The EDA script fits these statistics over the complete dataset because it is an
EDA artifact.

For the final forecasting model, **never calculate imputation statistics using
future validation/test observations**.

Correct procedure:

```text
chronological split
       ↓
fit imputation statistics on TRAIN
       ↓
transform TRAIN
transform VALIDATION
transform TEST
```

This prevents temporal leakage.

---

# 6. Step 4 — Wind-vector engineering

The original `wd` variable is categorical, e.g.:

```text
N, NE, E, SE, S, SW, W, NW
```

Categorical encoding treats N and NE as unrelated categories, even though their
directions are geometrically close.

The pipeline converts wind direction to an angle $\theta$ and calculates:

$$
u = -WSPM\sin(\theta)
$$

$$
v = -WSPM\cos(\theta)
$$

where:

- $u$ = east-west wind component
- $v$ = north-south wind component

The negative sign follows the meteorological convention that wind direction
describes where the wind is **coming from**.

### Important terminology

`u_wind` and `v_wind` are horizontal vector components.

They are **not** vertical wind velocity. A true vertical component would require
different measurements that are not present in this dataset.

## Why this helps ML

The transformation gives the model a continuous representation of direction:

```text
N  → approximately (0, -speed)
E  → approximately (-speed, 0)
S  → approximately (0, +speed)
W  → approximately (+speed, 0)
```

It also avoids the artificial discontinuity between 359° and 0°.

---

# 7. Step 5 — Cyclical time features

Hour is not an ordinary numeric variable.

Treating:

```text
23 → 0
```

as a large numerical jump is undesirable.

The pipeline creates:

$$
hour_{sin}=\sin(2\pi hour/24)
$$

$$
hour_{cos}=\cos(2\pi hour/24)
$$

The same idea is applied to month.

This creates a circular representation where 23:00 and 00:00 are close.

For a neural forecasting model, these features can be useful when the architecture
does not already learn temporal periodicity sufficiently well.

---

# 8. Step 6 — Descriptive statistics

The pipeline reports:

- mean
- standard deviation
- median
- skewness
- excess kurtosis
- missing percentage

## Mean vs median

Air-pollution concentrations commonly contain episodic high-concentration
events.

Therefore:

$$
mean \neq median
$$

can be substantial.

The median is often a better description of the "typical" observation when
extreme pollution episodes are present.

## Skewness

Positive skewness indicates a long right tail.

This is expected for many pollutant variables because concentration cannot be
negative and occasional pollution episodes can be very large.

## Kurtosis

High excess kurtosis indicates heavier tails and/or more extreme observations
than a normal distribution.

This matters for:

- visualization;
- normalization;
- loss-function behavior;
- outlier diagnostics.

---

# 9. Step 7 — Distribution analysis and log transformation

The pipeline plots raw and:

$$
\log(1+x)
$$

transformed distributions.

`log1p(x)` is preferred over `log(x)` because it is defined at zero.

For a strongly right-skewed pollutant:

```text
raw:
        many small values + a long right tail

log1p:
        tail compressed
        distribution often more symmetric
```

## Why this can help seq2seq models

Neural networks often optimize more easily when features do not span extremely
different scales.

However, do not automatically log-transform every variable.

A transformation should be chosen after inspecting:

- skewness;
- domain meaning;
- model residuals;
- validation performance.

For a transformed PM2.5 target:

$$
y'=\log(1+y)
$$

the forecast must be inverted:

$$
\hat{y}=\exp(\hat{y}')-1
$$

Evaluation should normally include metrics in the original physical units.

---

# 10. Step 8 — Diurnal dynamics

Hourly profiles reveal repeated 24-hour behavior.

The code specifically compares:

- NO2
- CO
- O3

## Why NO2 and CO often show rush-hour structure

Traffic-related emissions can produce enhanced concentrations around commuting
periods. Boundary-layer dynamics can also affect concentrations.

## Why O3 behaves differently

Ozone is strongly influenced by photochemical processes and sunlight.

Consequently, its daily cycle can differ substantially from primary pollutants.

The purpose of this plot is not to prove a causal mechanism. It is to identify
predictable temporal structure that a forecasting model can potentially exploit.

---

# 11. Step 9 — Monthly and seasonal distributions

The monthly and seasonal boxplots investigate whether the conditional
distribution changes over the year.

For example:

```text
winter → potentially elevated PM2.5
summer → different temperature, humidity, mixing, and photochemical regime
```

The important ML concept is **non-stationarity**.

If:

$$
P(X_t,Y_t)
$$

changes with season, then a model trained mostly on one regime may perform
differently in another.

This is one reason chronological validation is important.

---

# 12. Step 10 — Month × hour heatmap

The month-hour heatmap combines two time scales:

- annual/seasonal variation
- diurnal variation

It can reveal structures such as:

```text
winter + early morning → high concentration
summer + afternoon → different O3 behavior
```

This is often more informative than a single hourly average because the latter
can hide seasonal interactions.

---

# 13. Step 11 — STL decomposition

STL means:

**Seasonal-Trend decomposition using Loess.**

Conceptually:

$$
Y_t = T_t + S_t + R_t
$$

where:

- $T_t$ = trend
- $S_t$ = seasonal component
- $R_t$ = remainder/residual

## Daily STL

For hourly data:

```python
period = 24
```

captures a 24-hour seasonal pattern.

## Annual STL

For annual seasonality, the script additionally aggregates to monthly means
and uses:

```python
period = 12
```

This is computationally and statistically easier to interpret than attempting
a huge 8760-hour STL period directly.

## Why STL matters for forecasting

It tells you whether the target contains:

- persistent trend;
- repeatable seasonality;
- irregular shocks.

If the remainder remains highly structured, the model may need additional
features or longer historical context.

---

# 14. Step 12 — Pearson vs Spearman correlation

## Pearson

Measures linear association:

$$
r =
\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
$$

## Spearman

Ranks the variables first and measures monotonic association.

This makes Spearman less sensitive to the exact functional form and somewhat
more robust to extreme values.

### Interpretation

Example:

```text
Pearson = 0.25
Spearman = 0.55
```

can suggest that the relationship is monotonic but not strongly linear.

Correlation does **not** establish causation.

With approximately 420,000 observations, statistical significance can become
almost trivial. Therefore, prioritize effect size and physical plausibility.

---

# 15. Step 13 — Temperature/Dew Point vs PM2.5

The regression plots examine:

- TEMP → PM2.5
- DEWP → PM2.5

These relationships can be nonlinear because meteorological variables affect:

- atmospheric mixing;
- humidity;
- aerosol properties;
- chemical reactions;
- boundary-layer behavior.

Therefore, treat the straight regression line as a diagnostic summary, not a
complete physical model.

For the future ML model, nonlinear neural architectures can capture interactions
that a simple regression cannot.

---

# 16. Step 14 — Wind-direction polar analysis

The polar plot groups observations by wind direction and separates them into:

- lower wind speed;
- higher wind speed.

The question is:

> Does pollutant concentration vary systematically with wind direction and
> wind-speed regime?

This can reveal directional patterns that are invisible in a standard
correlation matrix.

## Important limitation

A wind-direction plot is **not source apportionment**.

A high concentration from one direction does not prove that a particular source
exists there. Atmospheric transport, terrain, station location, and other
meteorological factors also matter.

---

# 17. Step 15 — Cross-station comparison

The station boxplots and ridge KDE plots answer:

- Which stations have different PM2.5 distributions?
- Which stations have different NO2 distributions?
- Are station distributions similarly shaped?
- Are there strong site-specific regimes?

This matters because the dataset is a multi-site panel rather than a single
time series.

A future model can be designed as:

### Option A — Global model

One model receives all stations plus station identity.

```text
features + station embedding → shared model
```

### Option B — One model per station

Each station receives its own model.

This may capture site-specific behavior but uses less data per model.

### Option C — Global + station embedding

Often a useful experimental design:

```text
shared temporal encoder
        +
station embedding
        ↓
forecast
```

The EDA should inform which strategy is worth testing.

---

# 18. Step 16 — Temporal integrity for seq2seq

This is one of the most important diagnostics.

A 24→24 sequence requires:

```text
t-23 ... t-2 t-1 t
t+1 ... t+22 t+23 t+24
```

with hourly continuity.

If timestamps jump:

```text
10:00
11:00
15:00
```

you cannot safely treat those observations as consecutive time steps.

The pipeline reports:

- gaps > 1 hour;
- gaps > 24 hours;
- duplicate timestamps;
- expected vs actual observations.

---

# 19. Step 17 — Seq2seq window diagnostics

The pipeline estimates the percentage of possible windows that have:

- continuous hourly timestamps;
- valid future PM2.5 targets.

For an input length $L=24$ and horizon $H=24$:

$$
X =
[x_{t-23},...,x_t]
$$

and:

$$
Y =
[y_{t+1},...,y_{t+24}]
$$

The model should never use:

```text
y(t+1 ... t+24)
```

as an input feature when predicting those same values.

That would be target leakage.

---

# 20. Recommended train/validation/test design

For forecasting, prefer chronological splitting.

Example:

```text
2013–2015  → TRAIN
2016       → VALIDATION
2017       → TEST
```

The exact split should be chosen after examining the time range.

Do not use ordinary random `train_test_split` for the final forecasting
experiment.

Why?

Random splitting can put:

```text
January 2015
```

in training and:

```text
January 2015
```

in validation.

The model can then indirectly see almost identical temporal regimes on both
sides of the split.

---

# 21. Leakage checklist for the future model

Before training, verify all of the following.

### Scaling

Correct:

```text
fit scaler on train
transform train
transform validation
transform test
```

Incorrect:

```text
fit scaler on all data
```

### Imputation

Correct:

```text
fit imputation statistics on train
apply to future
```

Incorrect:

```text
calculate station-month-hour medians using the complete dataset
```

### Feature engineering

Features must be available at forecast time.

For example, if future weather variables are unavailable, do not accidentally
feed observed future weather into the model during a real forecasting scenario.

There are two different forecasting settings:

1. **Exogenous weather known in advance**
2. **Weather must itself be forecast**

The EDA should help determine which setting is intended.

---

# 22. Recommended feature groups for the 24→24 model

A useful initial feature set is:

### Pollutants

```text
PM2.5
PM10
SO2
NO2
CO
O3
```

### Meteorology

```text
TEMP
PRES
DEWP
RAIN
WSPM
u_wind
v_wind
```

### Temporal

```text
hour_sin
hour_cos
month_sin
month_cos
```

### Station

```text
station ID / embedding
```

Do not assume every variable improves forecasting. Compare ablations.

---

# 23. Suggested modeling experiments after EDA

Build progressively.

## Experiment 1 — Persistence baseline

Predict:

$$
\hat{y}_{t+h}=y_t
$$

for every future horizon.

This is essential. A neural model should be compared against a simple baseline.

## Experiment 2 — Historical mean / seasonal baseline

For example:

```text
predict average PM2.5 for this station and hour-of-day
```

## Experiment 3 — GRU/LSTM seq2seq

Input:

```text
24 × num_features
```

Output:

```text
24 × target_dimensions
```

## Experiment 4 — Attention-based encoder-decoder

Add temporal attention to determine which historical time steps are most useful.

## Experiment 5 — Transformer-style temporal model

Only after simpler baselines are established.

---

# 24. Evaluation metrics

Do not rely on one metric.

Useful metrics include:

## MAE

$$
MAE = \frac{1}{n}\sum |y-\hat y|
$$

Easy to interpret in pollutant units.

## RMSE

$$
RMSE =
\sqrt{\frac{1}{n}\sum(y-\hat y)^2}
$$

Penalizes large errors more strongly.

## MAPE

Be cautious with MAPE when concentrations approach zero.

A safer alternative can be:

- sMAPE;
- MAE;
- RMSE;
- MASE;
- normalized MAE.

## Horizon-wise evaluation

For a 24-hour forecast, calculate:

```text
MAE at t+1
MAE at t+2
...
MAE at t+24
```

This is especially important because forecasting error usually changes with
horizon.

---

# 25. What to look for in the EDA before modeling

Create a short findings table with:

| Question                    | Evidence              | Modeling implication     |
| --------------------------- | --------------------- | ------------------------ |
| Strong right skew?          | Skewness/log plots    | Consider log1p target    |
| Strong hourly cycle?        | Hourly profiles/STL   | 24h input is justified   |
| Strong annual cycle?        | Monthly STL           | Add seasonal features    |
| Station differences?        | Boxplots/ridges       | Use station embedding    |
| Wind relationship?          | Polar plots           | Include wind vectors     |
| Long missing blocks?        | Gap diagnostics       | Remove affected windows  |
| Strong weather interaction? | Correlations/scatters | Include meteorology      |
| Nonlinear relationships?    | Scatter/KDE           | Consider nonlinear model |
| Large temporal gaps?        | Integrity report      | Repair/remove windows    |

---

# 26. Recommended directory structure

After running:

```text
eda_outputs/
├── data_dictionary_diagnostic.csv
├── missingness_summary.csv
├── station_missingness.csv
├── descriptive_statistics.csv
├── station_statistics.csv
├── station_summary.csv
├── temporal_integrity_report.csv
├── domain_value_diagnostics.csv
├── seq2seq_window_diagnostics.csv
├── run_metadata.csv
├── eda_analysis_ready.csv
│
├── 01_missingness_by_variable.png
├── 02_missingness_station_variable.png
├── 03_missingness_temporal_pattern.png
├── distribution_PM2_5.png
├── distribution_PM10.png
├── distribution_CO.png
├── ...
├── hourly_pollutant_profiles.png
├── monthly_boxplot_PM2_5.png
├── seasonal_boxplot_PM2_5.png
├── ...
├── stl_PM2_5_daily_24h_all.png
├── stl_PM2_5_annual_all.png
├── stl_O3_daily_24h_all.png
├── stl_O3_annual_all.png
├── correlation_pearson.png
├── correlation_spearman.png
├── scatter_pm25_vs_temp.png
├── scatter_pm25_vs_dewp.png
├── wind_polar_PM2_5.png
├── wind_polar_NO2.png
├── station_boxplot_PM2_5.png
├── station_boxplot_NO2.png
├── ridge_PM2_5.png
└── ridge_NO2.png
```

---

# 27. Running the pipeline

Install dependencies:

```bash
pip install pandas numpy seaborn matplotlib statsmodels scipy
```

Run against station files:

```bash
python beijing_air_quality_eda.py \
    --data-dir ./PRSA_Data \
    --output-dir ./eda_outputs
```

Run against a unified CSV:

```bash
python beijing_air_quality_eda.py \
    --input-csv ./beijing_air_quality.csv \
    --output-dir ./eda_outputs
```

For a different short-gap threshold:

```bash
python beijing_air_quality_eda.py \
    --data-dir ./PRSA_Data \
    --short-gap-hours 3
```

---

# 28. Practical improvements before production modeling

The EDA script is intentionally conservative. Before the seq2seq experiment,
consider adding:

1. **Outlier policy**
   - sensor-range validation;
   - physically impossible values;
   - robust outlier flags instead of blindly deleting extremes.

2. **Humidity-related features**
   - relative humidity estimated from temperature/dew point;
   - vapor pressure deficit if scientifically justified.

3. **Lag features for baseline models**
   - lag 1, 3, 6, 12, 24;
   - rolling 3h, 6h, 24h statistics.

4. **Cross-station features**
   - neighboring-station pollution;
   - spatial distance;
   - regional average.

5. **Better wind-rose analysis**
   - pollutant percentile by sector;
   - concentration-weighted wind frequency;
   - station-specific directional plots.

6. **Model-specific preprocessing**
   - train-only scalers;
   - train-only imputers;
   - reproducible sequence generation.

7. **Forecast-origin analysis**
   - evaluate morning, afternoon, evening, and nighttime forecast origins
     separately.

---

# 29. Important conceptual distinction: EDA vs preprocessing

EDA asks:

> "What does the data look like?"

Model preprocessing asks:

> "What transformation can be applied without using future information?"

Therefore, some operations in this EDA pipeline intentionally use the full
dataset for descriptive purposes.

Do not copy those operations directly into the final training pipeline without
changing them to fit only on the training period.

---

# 30. Final recommended workflow

```text
RAW CSV FILES
     │
     ▼
INGEST + VALIDATE
     │
     ├── schema
     ├── timestamps
     ├── duplicates
     └── station coverage
     │
     ▼
MISSINGNESS EDA
     │
     ├── variable
     ├── station
     └── temporal gaps
     │
     ▼
EDA IMPUTATION
     │
     ├── short interpolation
     └── grouped median
     │
     ▼
FEATURE ENGINEERING
     │
     ├── wind vectors
     ├── cyclical time
     └── season
     │
     ▼
STATISTICAL EDA
     │
     ├── distributions
     ├── skew/kurtosis
     └── correlations
     │
     ▼
TEMPORAL EDA
     │
     ├── hourly
     ├── monthly
     ├── seasonal
     └── STL
     │
     ▼
METEOROLOGICAL EDA
     │
     ├── weather/pollution
     ├── wind direction
     └── wind speed
     │
     ▼
SPATIAL EDA
     │
     ├── station boxplots
     └── ridge distributions
     │
     ▼
SEQ2SEQ READINESS
     │
     ├── temporal continuity
     ├── usable windows
     └── leakage checks
     │
     ▼
TRAIN / VALIDATION / TEST
     │
     ▼
24h ENCODER → 24h DECODER
```

The most important principle is:

> **Use the EDA to discover structure, but build the final forecasting
> preprocessing pipeline so that no information from the future can influence
> the past.**
