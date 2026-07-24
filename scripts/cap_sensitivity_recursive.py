"""Recursive variant of the trend-cap sensitivity sweep.

The single-step sweep (cap_sensitivity_sweep.py) scores the test window with
actual lag features (teacher forcing). This one instead runs the true recursive
forecast — feeding each day's P50 prediction back as the next day's lag, seeded
with actuals through the forecast origin (Dec 31 2025) — at each candidate cap,
using the same RecursiveFeatureBuilder as scripts/recursive_backtest.py and
notebook 03.

Question it answers: does the old 5% cap's penalty on the clipped series
(GPU Inference | Startup) *compound* in the recursive regime the capacity
decision actually uses, or does the hybrid's lag-robustness absorb it?

Runtime: recursive forecasting is sequential (each day depends on the previous),
so this is minutes, not seconds. P50 only (all that MAPE needs).

Run:  python scripts/cap_sensitivity_recursive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import compute_forecasting.features as feat  # noqa: E402
from compute_forecasting.features import (  # noqa: E402
    RecursiveFeatureBuilder,
    build_features,
    compute_residual_ratios,
    fit_series_trends,
    get_feature_columns,
    predict_with_trend,
)

CAPS = [0.04, 0.05, 0.06, 0.07, 10.0]  # 10.0 == "none" (never binds)
BASELINE_CAP = 0.05
SHIPPED_CAP = 0.06
WATCH = ("GPU Inference", "Startup")  # the series the old cap clipped

TARGET = "compute_hours"
CAT_FEATURES = ["compute_type", "customer_segment", "series_id"]
TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-06-30")

PARAMS = {
    "objective": "quantile", "alpha": 0.5, "metric": "quantile",
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "n_estimators": 2000, "random_state": 42, "verbose": -1,
}


def cap_label(cap: float) -> str:
    return "none" if cap >= 1.0 else f"{cap:.0%}"


def series_mape(y_true, y_pred, keys, target_key) -> float:
    """MAPE for rows whose (type, segment) equals target_key."""
    mask = np.array([k == target_key for k in keys])
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100


def run_one_cap(cap, df, feature_cols, train, val, test, usage, holidays,
                test_dates, actuals, types, segments):
    """Fit at `cap`, return (single_step_p50, recursive_df) plus MAPEs."""
    feat.MAX_MONTHLY_GROWTH_RATE = cap
    trends = fit_series_trends(train)

    y_train = np.log(compute_residual_ratios(train, trends))
    y_val = np.log(compute_residual_ratios(val, trends))
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        train[feature_cols], y_train,
        eval_set=[(val[feature_cols], y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=0)],
        categorical_feature=CAT_FEATURES,
    )

    # --- Single-step (actual lags) ---
    ss_p50 = predict_with_trend(test, model.predict(test[feature_cols]), trends)

    # --- Recursive (predictions feed forward), seeded through the origin ---
    builder = RecursiveFeatureBuilder(usage[usage["date"] <= VAL_END], holidays)
    rows = []
    for dt in test_dates:
        day = pd.DataFrame([builder.build_row(dt, ct, seg)
                            for ct in types for seg in segments])
        day["compute_type"] = day["compute_type"].astype("category")
        day["customer_segment"] = day["customer_segment"].astype("category")
        day["series_id"] = (day["compute_type"].astype(str) + " | "
                            + day["customer_segment"].astype(str)).astype("category")
        trend_vals = np.array([
            trends[(r["compute_type"], r["customer_segment"])].predict(pd.Series([dt]))[0]
            for _, r in day.iterrows()
        ])
        day["rec"] = np.maximum(trend_vals * np.exp(model.predict(day[feature_cols])), 0)
        for _, r in day.iterrows():
            builder.cache_prediction(dt, r["compute_type"], r["customer_segment"], r["rec"])
        day["date"] = dt
        rows.append(day[["date", "compute_type", "customer_segment", "rec"]])

    rc = pd.concat(rows, ignore_index=True).merge(
        actuals, on=["date", "compute_type", "customer_segment"], how="left")
    return ss_p50, rc


def main() -> None:
    print("Loading data and building features...")
    usage = pd.read_csv(ROOT / "data" / "compute_usage.csv", parse_dates=["date"])
    holidays = pd.read_csv(ROOT / "data" / "holiday_calendar.csv", parse_dates=["date"])
    df = build_features(usage, holidays)
    feature_cols = get_feature_columns(df)

    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] > VAL_END].copy()

    types = sorted(usage["compute_type"].unique())
    segments = sorted(usage["customer_segment"].unique())
    test_dates = pd.date_range(TEST_START, TEST_END, freq="D")
    actuals = usage[usage["date"] >= TEST_START][
        ["date", "compute_type", "customer_segment", "compute_hours"]]

    test_keys = list(zip(test["compute_type"], test["customer_segment"]))
    y_test = test[TARGET].to_numpy()

    results = {}
    for cap in CAPS:
        print(f"  cap={cap_label(cap):>5}: fitting + recursive forecast "
              f"({len(test_dates)} days x {len(types)*len(segments)} series)...")
        ss_p50, rc = run_one_cap(cap, df, feature_cols, train, val, test,
                                 usage, holidays, test_dates, actuals, types, segments)
        rc_keys = list(zip(rc["compute_type"], rc["customer_segment"]))
        rc_true = rc["compute_hours"].to_numpy()
        rc_pred = rc["rec"].to_numpy()
        results[cap] = {
            "ss_overall": mean_absolute_percentage_error(y_test, ss_p50) * 100,
            "rc_overall": mean_absolute_percentage_error(rc_true, rc_pred) * 100,
            "ss_watch": series_mape(y_test, ss_p50, test_keys, WATCH),
            "rc_watch": series_mape(rc_true, rc_pred, rc_keys, WATCH),
        }

    feat.MAX_MONTHLY_GROWTH_RATE = SHIPPED_CAP  # restore

    # ---- Report ----
    print("\n" + "=" * 72)
    print("SINGLE-STEP vs RECURSIVE, BY CAP")
    print("=" * 72)
    print(f"{'cap':>6} | {'SS overall':>10} {'REC overall':>11} {'degrade':>8} "
          f"| {'SS Startup':>10} {'REC Startup':>11} {'degrade':>8}")
    print("-" * 72)
    for cap in CAPS:
        r = results[cap]
        tag = {"5%": "  <- original", "6%": "  <- shipped"}.get(cap_label(cap), "")
        print(f"{cap_label(cap):>6} | {r['ss_overall']:>9.2f}% {r['rc_overall']:>10.2f}% "
              f"{r['rc_overall']-r['ss_overall']:>+7.2f} | "
              f"{r['ss_watch']:>9.1f}% {r['rc_watch']:>10.1f}% "
              f"{r['rc_watch']-r['ss_watch']:>+7.1f}{tag}")

    # ---- The headline question: does the cap penalty compound recursively? ----
    b, s = results[BASELINE_CAP], results[SHIPPED_CAP]
    print("\n" + "=" * 72)
    print(f"CAP PENALTY (5% -> 6%) FOR {WATCH[0]} | {WATCH[1]}")
    print("=" * 72)
    print(f"  Single-step: {b['ss_watch']:.1f}%  ->  {s['ss_watch']:.1f}%   "
          f"(cap cost {b['ss_watch']-s['ss_watch']:+.1f}pp)")
    print(f"  Recursive:   {b['rc_watch']:.1f}%  ->  {s['rc_watch']:.1f}%   "
          f"(cap cost {b['rc_watch']-s['rc_watch']:+.1f}pp)")
    ss_cost = b["ss_watch"] - s["ss_watch"]
    rc_cost = b["rc_watch"] - s["rc_watch"]
    if rc_cost > ss_cost + 0.5:
        verdict = "COMPOUNDS: the cap hurt more in the deployment (recursive) regime."
    elif rc_cost < ss_cost - 0.5:
        verdict = "ABSORBED: the hybrid's lag-robustness muted the cap penalty recursively."
    else:
        verdict = "SIMILAR: the cap penalty is about the same in both regimes."
    print(f"\n  Verdict: {verdict}")
    print(f"  Overall recursive: 5% {b['rc_overall']:.2f}% -> 6% {s['rc_overall']:.2f}% "
          f"({s['rc_overall']-b['rc_overall']:+.2f}pp)")


if __name__ == "__main__":
    main()
