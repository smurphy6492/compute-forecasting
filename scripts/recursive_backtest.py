"""Recursive forecast backtest against known Jan-Jun 2026 actuals.

Trains the hybrid model on data through Jun 2025, then runs the recursive
forecasting loop over the test period (Jan-Jun 2026) where we have actuals.
Compares recursive MAPE to single-step MAPE at various horizons.

This answers: "How much does accuracy degrade when we feed predictions back
as lag features instead of using actual values?"
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from compute_forecasting.features import (
    RecursiveFeatureBuilder,
    build_features,
    get_feature_columns,
    fit_series_trends,
    compute_residual_ratios,
    predict_with_trend,
)

# ---------------------------------------------------------------------------
# Load data and build features
# ---------------------------------------------------------------------------
usage = pd.read_csv(ROOT / "data" / "compute_usage.csv", parse_dates=["date"])
holidays = pd.read_csv(ROOT / "data" / "holiday_calendar.csv", parse_dates=["date"])

df = build_features(usage, holidays)
feature_cols = get_feature_columns(df)
TARGET = "compute_hours"
CAT_FEATURES = ["compute_type", "customer_segment", "series_id"]

TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-06-30")

train = df[df["date"] <= TRAIN_END].copy()
val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test = df[df["date"] > VAL_END].copy()

PARAMS = {
    "objective": "quantile", "alpha": 0.5, "metric": "quantile",
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "n_estimators": 2000, "random_state": 42, "verbose": -1,
}

# ---------------------------------------------------------------------------
# 1. Train hybrid model (same as notebook 02, Section 11)
# ---------------------------------------------------------------------------
print("=" * 70)
print("RECURSIVE FORECAST BACKTEST")
print("=" * 70)

print("\n1. Training hybrid model on train set (through Jun 2025)...")
trends = fit_series_trends(train)

train_ratios = compute_residual_ratios(train, trends)
val_ratios = compute_residual_ratios(val, trends)

y_train_log_resid = np.log(train_ratios)
y_val_log_resid = np.log(val_ratios)

X_train, X_val, X_test = train[feature_cols], val[feature_cols], test[feature_cols]
y_test = test[TARGET]

models = {}
for alpha, label in [(0.1, "P10"), (0.5, "P50"), (0.9, "P90")]:
    p = PARAMS.copy()
    p["alpha"] = alpha
    m = lgb.LGBMRegressor(**p)
    m.fit(
        X_train, y_train_log_resid,
        eval_set=[(X_val, y_val_log_resid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=0)],
        categorical_feature=CAT_FEATURES,
    )
    models[label] = m
    print(f"  {label}: best_iteration={m.best_iteration_}")

# ---------------------------------------------------------------------------
# 2. Single-step predictions (using actual lags) — baseline comparison
# ---------------------------------------------------------------------------
print("\n2. Single-step predictions (actual lags)...")
ss_p50 = predict_with_trend(test, models["P50"].predict(X_test), trends)
ss_mape = mean_absolute_percentage_error(y_test, ss_p50) * 100
print(f"  Single-step test MAPE: {ss_mape:.2f}%")

# ---------------------------------------------------------------------------
# 3. Recursive forecast over test period
# ---------------------------------------------------------------------------
print("\n3. Running recursive forecast over test period (Jan-Jun 2026)...")

COMPUTE_TYPES = sorted(usage["compute_type"].unique())
SEGMENTS = sorted(usage["customer_segment"].unique())

# Feature builder with actuals through the forecast origin (train + val,
# i.e. everything known on Dec 31 2025). Test-period lag lookups fall through
# to the prediction cache, so the Jan-Jun 2026 forecast is truly recursive.
# History must extend to the origin: cutting it at TRAIN_END would leave a
# Jul-Dec 2025 blackout where lags are neither actuals nor cached predictions.
builder = RecursiveFeatureBuilder(usage[usage["date"] <= VAL_END], holidays)


test_dates = pd.date_range(TEST_START, TEST_END, freq="D")
all_recursive_rows = []

for i, dt in enumerate(test_dates):
    day_rows = []
    for ct in COMPUTE_TYPES:
        for seg in SEGMENTS:
            feat = builder.build_row(dt, ct, seg)
            day_rows.append(feat)

    day_df = pd.DataFrame(day_rows)
    day_df["compute_type"] = day_df["compute_type"].astype("category")
    day_df["customer_segment"] = day_df["customer_segment"].astype("category")
    day_df["series_id"] = (
        day_df["compute_type"].astype(str) + " | " + day_df["customer_segment"].astype(str)
    ).astype("category")

    # Trend values for reconstruction
    trend_vals = np.array([
        trends[(row["compute_type"], row["customer_segment"])].predict(pd.Series([dt]))[0]
        for _, row in day_df.iterrows()
    ])

    # Predict log(residual_ratio) and reconstruct
    log_resid_preds = models["P50"].predict(day_df[feature_cols])
    day_df["recursive_P50"] = np.maximum(trend_vals * np.exp(log_resid_preds), 0)

    # Cache P50 for next day's lag features
    for idx, row in day_df.iterrows():
        builder.cache_prediction(dt, row["compute_type"], row["customer_segment"], row["recursive_P50"])

    day_df["date"] = dt
    all_recursive_rows.append(day_df[["date", "compute_type", "customer_segment", "recursive_P50"]])

    if (i + 1) % 30 == 0:
        print(f"  Forecasted {i + 1} / {len(test_dates)} days")

rc = pd.concat(all_recursive_rows, ignore_index=True)
print(f"  Done: {len(rc):,} rows")

# ---------------------------------------------------------------------------
# 4. Merge recursive predictions with actuals and compare
# ---------------------------------------------------------------------------
print("\n4. Comparing single-step vs recursive accuracy...")

# Merge actuals
actuals = usage[usage["date"] >= TEST_START][["date", "compute_type", "customer_segment", "compute_hours"]]
rc = rc.merge(actuals, on=["date", "compute_type", "customer_segment"], how="left")

# Overall MAPE
rc_mape = mean_absolute_percentage_error(rc["compute_hours"], rc["recursive_P50"]) * 100
print(f"\n  {'Metric':<35} {'Single-step':>12} {'Recursive':>12}")
print(f"  {'-'*60}")
print(f"  {'Overall MAPE':<35} {ss_mape:>11.2f}% {rc_mape:>11.2f}%")

# By compute type
print(f"\n  MAPE by Compute Type:")
for ct in COMPUTE_TYPES:
    mask_ss = (test["compute_type"] == ct).values
    mask_rc = rc["compute_type"] == ct
    ss_ct = mean_absolute_percentage_error(y_test.values[mask_ss], ss_p50[mask_ss]) * 100
    rc_ct = mean_absolute_percentage_error(rc.loc[mask_rc, "compute_hours"], rc.loc[mask_rc, "recursive_P50"]) * 100
    print(f"    {ct:<25} {ss_ct:>11.2f}% {rc_ct:>11.2f}%")

# By segment
print(f"\n  MAPE by Customer Segment:")
for seg in SEGMENTS:
    mask_ss = (test["customer_segment"] == seg).values
    mask_rc = rc["customer_segment"] == seg
    ss_seg = mean_absolute_percentage_error(y_test.values[mask_ss], ss_p50[mask_ss]) * 100
    rc_seg = mean_absolute_percentage_error(rc.loc[mask_rc, "compute_hours"], rc.loc[mask_rc, "recursive_P50"]) * 100
    print(f"    {seg:<25} {ss_seg:>11.2f}% {rc_seg:>11.2f}%")

# ---------------------------------------------------------------------------
# 5. MAPE by forecast horizon (how fast does accuracy degrade?)
# ---------------------------------------------------------------------------
print("\n5. MAPE by forecast horizon (days into the future):")

rc["days_out"] = (rc["date"] - TEST_START).dt.days

horizon_bins = [
    ("Days 1-7", 0, 7),
    ("Days 8-14", 7, 14),
    ("Days 15-30", 14, 30),
    ("Days 31-60", 30, 60),
    ("Days 61-90", 60, 90),
    ("Days 91-120", 90, 120),
    ("Days 121-150", 120, 150),
    ("Days 151-181", 150, 181),
]

print(f"\n  {'Horizon':<20} {'Recursive MAPE':>15} {'Rows':>8}")
print(f"  {'-'*45}")
for label, lo, hi in horizon_bins:
    mask = (rc["days_out"] >= lo) & (rc["days_out"] < hi)
    if mask.sum() == 0:
        continue
    horizon_mape = mean_absolute_percentage_error(
        rc.loc[mask, "compute_hours"], rc.loc[mask, "recursive_P50"]
    ) * 100
    print(f"  {label:<20} {horizon_mape:>14.2f}% {mask.sum():>8,}")

# ---------------------------------------------------------------------------
# 6. First-day drop analysis
# ---------------------------------------------------------------------------
print("\n6. First forecast day analysis (the 87% drop question):")

last_actual_date = usage["date"].max()
first_forecast_date = TEST_START

last_actual_total = usage[usage["date"] == last_actual_date].groupby("date")["compute_hours"].sum().iloc[0]
first_recursive = rc[rc["date"] == first_forecast_date]["recursive_P50"].sum()
first_actual = rc[rc["date"] == first_forecast_date]["compute_hours"].sum()

print(f"  Last training day ({last_actual_date.date()}, {last_actual_date.day_name()}):")
print(f"    Actual total: {last_actual_total:,.0f}")
print(f"  First test day ({first_forecast_date.date()}, {first_forecast_date.day_name()}):")
print(f"    Actual total:    {first_actual:,.0f} ({first_actual/last_actual_total:.0%} of prior day)")
print(f"    Recursive P50:   {first_recursive:,.0f} ({first_recursive/last_actual_total:.0%} of prior day)")
print(f"    Recursive error: {abs(first_recursive - first_actual)/first_actual:.1%}")

# Check: is the drop just calendar effects?
# Compare to the same transition a year earlier
prev_year_last = usage[usage["date"] == pd.Timestamp("2025-06-30")].groupby("date")["compute_hours"].sum().iloc[0]
prev_year_first = usage[usage["date"] == pd.Timestamp("2025-07-01")].groupby("date")["compute_hours"].sum().iloc[0]
print(f"\n  Same transition one year earlier (2025):")
print(f"    Jun 30 actual: {prev_year_last:,.0f}")
print(f"    Jul 1 actual:  {prev_year_first:,.0f} ({prev_year_first/prev_year_last:.0%} of prior day)")

# Worst series by recursive MAPE
print("\n7. Top 5 worst series (recursive MAPE):")
series_mape = []
for (ct, seg), grp in rc.groupby(["compute_type", "customer_segment"]):
    m = mean_absolute_percentage_error(grp["compute_hours"], grp["recursive_P50"]) * 100
    series_mape.append({"Series": f"{ct} | {seg}", "Recursive MAPE": m})
series_mape_df = pd.DataFrame(series_mape).sort_values("Recursive MAPE", ascending=False)
for _, row in series_mape_df.head(5).iterrows():
    print(f"  {row['Series']:<40} {row['Recursive MAPE']:.2f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
