"""
==============================================================================
EDA + Preprocessing Pipeline
UCI Beijing Multi-Site Air-Quality Dataset
==============================================================================

Goal
----
Turn the 12 raw per-station CSVs into a clean, model-ready dataset for
sequence-to-sequence forecasting (LSTM / GRU / LSTM Autoencoder), using:

    INPUT_LEN  = 24   (24 hours of history)
    OUTPUT_LEN = 24   (24 hours forecast horizon)

Directory layout expected
--------------------------
    /datasets/raw_data/station1.csv ... station12.csv   (or any *.csv there)

Output produced
----------------
    /datasets/processed/
        cleaned_long.parquet          -> cleaned, feature-engineered long table
        train_windows.npz             -> X_train, y_train (+ meta)
        val_windows.npz               -> X_val,   y_val
        test_windows.npz              -> X_test,  y_test
        feature_scaler.pkl            -> fitted StandardScaler (train-only)
        target_scaler.pkl             -> fitted StandardScaler for target(s)
        feature_columns.json          -> ordered list of feature names
        eda_report/                   -> saved EDA plots (png)

Run
---
    python beijing_eda_pipeline.py --raw_dir /datasets/raw_data \
                                    --out_dir /datasets/processed \
                                    --target PM2.5

==============================================================================
"""

import os
import glob
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_COL = "wd"
NUMERIC_COLS = POLLUTANTS + WEATHER  # columns that get imputed/scaled

INPUT_LEN = 24     # hours of history fed to the encoder
OUTPUT_LEN = 24    # hours forecast by the decoder

# chronological split (time-based, NOT random, to avoid leakage)
TRAIN_END = "2015-12-31 23:00:00"
VAL_END = "2016-12-31 23:00:00"
# everything after VAL_END -> test


# =========================================================================
# STEP 1 — LOAD RAW DATA
# =========================================================================
def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Read every station CSV in raw_dir and concatenate into one long df."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"[Step 1] Found {len(files)} station files.")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        dfs.append(d)
        print(f"   loaded {os.path.basename(f):20s} shape={d.shape}")

    df = pd.concat(dfs, ignore_index=True)

    # build a single datetime column and drop the raw parts + row index col
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])

    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    print(f"[Step 1] Combined shape: {df.shape}, "
          f"stations: {df['station'].nunique()}, "
          f"date range: {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


# =========================================================================
# STEP 2 — INITIAL STRUCTURAL CHECKS
# =========================================================================
def initial_checks(df: pd.DataFrame, out_dir: str):
    """Print/save basic structural diagnostics before any cleaning."""
    print("\n[Step 2] Initial structural checks")
    print(df.dtypes)
    print("\nDuplicate rows:", df.duplicated().duplicated().sum())

    # verify every station has a complete, gap-free hourly index
    print("\nPer-station timestamp coverage:")
    gap_rows = []
    for station, g in df.groupby("station"):
        full_range = pd.date_range(g["datetime"].min(), g["datetime"].max(), freq="h")
        missing_ts = full_range.difference(g["datetime"])
        gap_rows.append((station, len(g), len(full_range), len(missing_ts)))
        print(f"   {station:15s} rows={len(g):6d} expected={len(full_range):6d} "
              f"missing_timestamps={len(missing_ts)}")

    pd.DataFrame(gap_rows, columns=["station", "rows", "expected_rows",
                                     "missing_timestamps"]).to_csv(
        os.path.join(out_dir, "eda_report", "timestamp_coverage.csv"), index=False)


# =========================================================================
# STEP 3 — MISSING VALUE ANALYSIS (BEFORE CLEANING)
# =========================================================================
def missing_value_report(df: pd.DataFrame, out_dir: str):
    print("\n[Step 3] Missing value report (raw)")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    print(report[report["missing_count"] > 0])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=report.index, y=report["missing_pct"], ax=ax)
    ax.set_ylabel("% missing")
    ax.set_title("Missing values by column (raw data)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "missing_values_raw.png"))
    plt.close(fig)

    # missingness per station heatmap
    pivot = df.groupby("station")[NUMERIC_COLS].apply(
        lambda g: g.isna().mean() * 100
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax)
    ax.set_title("% missing per station / column")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "missing_by_station.png"))
    plt.close(fig)


# =========================================================================
# STEP 4 — DISTRIBUTIONS & OUTLIERS (BEFORE CLEANING)
# =========================================================================
def distribution_and_outlier_report(df: pd.DataFrame, out_dir: str):
    print("\n[Step 4] Distribution / outlier report")
    print(df[NUMERIC_COLS].describe().T)

    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
    for ax in axes.flat[len(NUMERIC_COLS):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "boxplots_raw.png"))
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=50)
        ax.set_title(col)
    for ax in axes.flat[len(NUMERIC_COLS):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "histograms_raw.png"))
    plt.close(fig)

    # physically impossible negative values (should not exist, but check)
    neg_counts = (df[POLLUTANTS] < 0).sum()
    print("\nNegative pollutant readings (should be 0):")
    print(neg_counts)


# =========================================================================
# STEP 5 — CORRELATION ANALYSIS
# =========================================================================
def correlation_report(df: pd.DataFrame, out_dir: str):
    print("\n[Step 5] Correlation analysis")
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "correlation_matrix.png"))
    plt.close(fig)


# =========================================================================
# STEP 6 — TEMPORAL PATTERNS (SEASONALITY CHECK)
# =========================================================================
def temporal_pattern_report(df: pd.DataFrame, out_dir: str, target: str):
    print("\n[Step 6] Temporal pattern report (target =", target, ")")
    tmp = df.copy()
    tmp["hour"] = tmp["datetime"].dt.hour
    tmp["month"] = tmp["datetime"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=tmp, x="hour", y=target, ax=axes[0], errorbar="sd")
    axes[0].set_title(f"{target} by hour of day")
    sns.lineplot(data=tmp, x="month", y=target, ax=axes[1], errorbar="sd")
    axes[1].set_title(f"{target} by month")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "temporal_patterns.png"))
    plt.close(fig)

    # example daily trend, one station, first 30 days
    example_station = df["station"].unique()[0]
    ex = df[df["station"] == example_station].head(24 * 30)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ex["datetime"], ex[target])
    ax.set_title(f"{target} time series — {example_station} (first 30 days)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_report", "example_series.png"))
    plt.close(fig)


# =========================================================================
# STEP 7 — CLEANING: MISSING VALUE IMPUTATION (TIME-AWARE, PER STATION)
# =========================================================================
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values per-station using time-ordered
    interpolation, which is appropriate for hourly sensor series:
      1. linear interpolation for short gaps (bounded, time-based)
      2. forward/backward fill for any remaining edge gaps
      3. categorical wind direction: forward-fill then mode fallback
    """
    print("\n[Step 7] Cleaning missing values")
    df = df.sort_values(["station", "datetime"]).copy()

    def _impute_station(g):
        station_name = g.name  # pandas >=2.2 excludes the group col from g itself
        g = g.set_index("datetime")
        g[NUMERIC_COLS] = g[NUMERIC_COLS].interpolate(
            method="time", limit=6, limit_direction="both"
        )
        g[NUMERIC_COLS] = g[NUMERIC_COLS].ffill().bfill()

        g[WIND_DIR_COL] = g[WIND_DIR_COL].ffill().bfill()
        if g[WIND_DIR_COL].isna().any():
            g[WIND_DIR_COL] = g[WIND_DIR_COL].fillna(g[WIND_DIR_COL].mode()[0])
        g = g.reset_index()
        g["station"] = station_name
        return g

    df = df.groupby("station", group_keys=False).apply(_impute_station, include_groups=False)

    remaining = df[NUMERIC_COLS + [WIND_DIR_COL]].isna().sum().sum()
    print(f"   remaining missing values after cleaning: {remaining}")
    assert remaining == 0, "Missing values remain after imputation"
    return df


# =========================================================================
# STEP 8 — OUTLIER / PHYSICAL-RANGE CLIPPING
# =========================================================================
def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip clearly non-physical values and extreme statistical outliers
    (winsorize at 0.1% / 99.9%) instead of dropping rows, to preserve
    the continuous hourly sequence needed for windowing.
    """
    print("\n[Step 8] Clipping outliers")
    df = df.copy()
    # pollutants and wind speed cannot be negative
    for col in POLLUTANTS + ["WSPM"]:
        df[col] = df[col].clip(lower=0)

    # winsorize extreme tails per column (global, not per-station)
    for col in NUMERIC_COLS:
        lo, hi = df[col].quantile([0.001, 0.999])
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# =========================================================================
# STEP 9 — FEATURE ENGINEERING
# =========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - cyclical encode hour / day-of-week / month (sin/cos) so the model
      sees continuity across midnight / year-end
    - cyclical/one-hot encode wind direction
    - one-hot encode station id (needed since one model may serve all
      12 stations)
    """
    print("\n[Step 9] Feature engineering")
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)

    wd_dummies = pd.get_dummies(df[WIND_DIR_COL], prefix="wd")
    station_dummies = pd.get_dummies(df["station"], prefix="station")

    df = pd.concat([df, wd_dummies, station_dummies], axis=1)
    return df


# =========================================================================
# STEP 10 — CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# =========================================================================
def split_by_time(df: pd.DataFrame):
    print("\n[Step 10] Chronological split")
    train = df[df["datetime"] <= TRAIN_END]
    val = df[(df["datetime"] > TRAIN_END) & (df["datetime"] <= VAL_END)]
    test = df[df["datetime"] > VAL_END]
    print(f"   train: {train['datetime'].min()} -> {train['datetime'].max()} "
          f"({len(train)} rows)")
    print(f"   val:   {val['datetime'].min()} -> {val['datetime'].max()} "
          f"({len(val)} rows)")
    print(f"   test:  {test['datetime'].min()} -> {test['datetime'].max()} "
          f"({len(test)} rows)")
    return train, val, test


# =========================================================================
# STEP 11 — SCALING (FIT ON TRAIN ONLY, AVOID LEAKAGE)
# =========================================================================
def fit_scalers(train_df: pd.DataFrame, feature_cols: list, target_cols: list):
    print("\n[Step 11] Fitting scalers on training data only")
    feat_scaler = StandardScaler().fit(train_df[feature_cols])
    target_scaler = StandardScaler().fit(train_df[target_cols])
    return feat_scaler, target_scaler


def apply_scaling(df: pd.DataFrame, feature_cols, target_cols,
                   feat_scaler, target_scaler) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = feat_scaler.transform(df[feature_cols])
    # target columns are also inputs (e.g. PM2.5 history is a feature),
    # so keep a separately-scaled copy for building y windows
    scaled_targets = target_scaler.transform(df[target_cols])
    for i, c in enumerate(target_cols):
        df[c + "_scaled_target"] = scaled_targets[:, i]
    return df


# =========================================================================
# STEP 12 — WINDOWING: BUILD (24h IN -> 24h OUT) SEQUENCES PER STATION
# =========================================================================
def build_windows(df: pd.DataFrame, feature_cols, target_cols,
                   input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    """
    For each station, slide a window across its (gap-free, hourly)
    timeline and emit:
        X: (input_len,  n_features)   encoder input
        y: (output_len, n_targets)    decoder target (scaled)
    Windows are built independently per station so history never
    crosses a station boundary.
    """
    print(f"\n[Step 12] Building sliding windows "
          f"(input={input_len}h, output={output_len}h)")
    target_scaled_cols = [c + "_scaled_target" for c in target_cols]

    X_all, y_all, meta_all = [], [], []
    for station, g in df.groupby("station"):
        g = g.sort_values("datetime").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targs = g[target_scaled_cols].to_numpy(dtype=np.float32)
        times = g["datetime"].to_numpy()

        n = len(g)
        total_len = input_len + output_len
        for start in range(0, n - total_len + 1):
            X_all.append(feats[start: start + input_len])
            y_all.append(targs[start + input_len: start + total_len])
            meta_all.append((station, times[start], times[start + input_len]))

        print(f"   {station:15s} -> {max(0, n - total_len + 1):6d} windows")

    X = np.stack(X_all).astype(np.float32)
    y = np.stack(y_all).astype(np.float32)
    meta = np.array(meta_all, dtype=object)
    print(f"   total windows: X={X.shape}, y={y.shape}")
    return X, y, meta


# =========================================================================
# MAIN PIPELINE
# =========================================================================
def main(raw_dir, out_dir, target):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "eda_report"), exist_ok=True)

    # 1-2: load + structural checks
    df = load_raw_data(raw_dir)
    initial_checks(df, out_dir)

    # 3-6: EDA on raw data (run before touching values, so plots reflect
    #      the true raw data quality)
    missing_value_report(df, out_dir)
    distribution_and_outlier_report(df, out_dir)
    correlation_report(df, out_dir)
    temporal_pattern_report(df, out_dir, target=target)

    # 7-9: cleaning + feature engineering
    df = clean_missing_values(df)
    df = clip_outliers(df)
    df = engineer_features(df)

    # persist the cleaned long-form table (station, datetime, all features)
    df.to_parquet(os.path.join(out_dir, "cleaned_long.parquet"), index=False)
    print(f"\n[Save] cleaned_long.parquet -> {df.shape}")

    # 10: split
    train_df, val_df, test_df = split_by_time(df)

    # define feature/target columns (exclude identifiers)
    exclude = {"datetime", "station", WIND_DIR_COL}
    feature_cols = [c for c in df.columns if c not in exclude]
    target_cols = [target]  # extend to multiple targets if forecasting >1 pollutant

    # 11: scale (fit on train only)
    feat_scaler, target_scaler = fit_scalers(train_df, feature_cols, target_cols)
    train_df = apply_scaling(train_df, feature_cols, target_cols, feat_scaler, target_scaler)
    val_df = apply_scaling(val_df, feature_cols, target_cols, feat_scaler, target_scaler)
    test_df = apply_scaling(test_df, feature_cols, target_cols, feat_scaler, target_scaler)

    # 12: build windows for each split independently (no leakage across splits)
    X_train, y_train, meta_train = build_windows(train_df, feature_cols, target_cols)
    X_val, y_val, meta_val = build_windows(val_df, feature_cols, target_cols)
    X_test, y_test, meta_test = build_windows(test_df, feature_cols, target_cols)

    np.savez_compressed(os.path.join(out_dir, "train_windows.npz"),
                         X=X_train, y=y_train, meta=meta_train)
    np.savez_compressed(os.path.join(out_dir, "val_windows.npz"),
                         X=X_val, y=y_val, meta=meta_val)
    np.savez_compressed(os.path.join(out_dir, "test_windows.npz"),
                         X=X_test, y=y_test, meta=meta_test)

    with open(os.path.join(out_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(os.path.join(out_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)
    with open(os.path.join(out_dir, "feature_columns.json"), "w") as f:
        json.dump({"feature_cols": feature_cols, "target_cols": target_cols}, f, indent=2)

    print("\n[Done] Processed dataset saved to:", out_dir)
    print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape}  y_val   {y_val.shape}")
    print(f"  X_test  {X_test.shape}  y_test  {y_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/datasets/raw_data/")
    parser.add_argument("--out_dir", type=str, default="/datasets/processed/")
    parser.add_argument("--target", type=str, default="PM2.5",
                         help="Pollutant column to forecast")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.target)
