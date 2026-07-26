"""Sensitivity sweep of the trend growth cap (MAX_MONTHLY_GROWTH_RATE).

Reuses the production training recipe verbatim (production/train.py's
fit_and_evaluate) and only varies the cap by patching the module constant that
fit_series_trends reads at call time. For each cap it refits the 16 per-series
trends, retrains the P10/P50/P90 boosters on log residual ratios, conformally
calibrates on validation, and scores the held-out test window (2026-01-01+).

Reports overall test MAPE / bias / coverage plus per-series P50 MAPE, with the
series where the cap actually binds called out. Single-step (teacher-forcing)
evaluation, matching the notebook's headline recipe. The cap was moved from 5%
to 6% on the strength of this sweep; run it to reproduce the comparison.

Run:  python scripts/cap_sensitivity_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "production"))

# Reuse the exact production pipeline pieces.
from train import (  # noqa: E402
    TARGET,
    TRAIN_END,
    VAL_END,
    fit_and_evaluate,
    load_data,
    predict_quantiles,
)

import compute_forecasting.features as feat  # noqa: E402
from compute_forecasting.features import build_features, fit_series_trends  # noqa: E402

# Caps to test, in monthly-growth terms. 10.0 (1000%/mo) never binds = "no cap".
CAPS = [0.04, 0.05, 0.06, 0.07, 10.0]
BASELINE_CAP = 0.05  # the shipped value

# True generating monthly rates (from data/generate_synthetic_data.py) for context.
GROWTH_RATES_BY_TYPE = {
    "GPU Training": 0.025,
    "GPU Inference": 0.040,
    "CPU Batch": 0.010,
    "CPU Interactive": 0.015,
}
SEGMENT_MULT = {
    "Enterprise": 0.85,
    "Mid-Market": 1.10,
    "Startup": 1.50,
    "Research/Academic": 0.30,
}


def cap_label(cap: float) -> str:
    return "none" if cap >= 1.0 else f"{cap:.0%}"


def per_series_mape(models, trends, test_df) -> dict[tuple[str, str], float]:
    """P50 test MAPE for each (compute_type, customer_segment) series."""
    y = test_df[TARGET].to_numpy()
    p50 = predict_quantiles(models, test_df, trends)["p50"]
    out = {}
    for (ct, seg), grp in test_df.groupby(["compute_type", "customer_segment"], observed=True):
        mask = grp.index
        pos = test_df.index.get_indexer(mask)
        out[(ct, seg)] = mean_absolute_percentage_error(y[pos], p50[pos]) * 100
    return out


def fitted_uncapped_rates(fit_df) -> dict[tuple[str, str], float]:
    """Monthly growth each series fits with NO cap (patch to a huge cap)."""
    saved = feat.MAX_MONTHLY_GROWTH_RATE
    feat.MAX_MONTHLY_GROWTH_RATE = 10.0
    try:
        trends = fit_series_trends(fit_df)
    finally:
        feat.MAX_MONTHLY_GROWTH_RATE = saved
    return {k: np.exp(t.daily_growth_rate * 30) - 1 for k, t in trends.items()}


def main() -> None:
    print("Loading data and building features...")
    usage, holidays = load_data()
    df = build_features(usage, holidays)

    fit_df = df[df["date"] <= TRAIN_END]
    val_df = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test_df = df[df["date"] > VAL_END]
    print(
        f"  train {fit_df['date'].min().date()}..{fit_df['date'].max().date()}  "
        f"test {test_df['date'].min().date()}..{test_df['date'].max().date()}  "
        f"({len(test_df)} test rows)\n"
    )

    # Which series genuinely want to grow faster than each cap?
    uncapped = fitted_uncapped_rates(fit_df)
    print("Series with fastest uncapped fits (monthly):")
    for k in sorted(uncapped, key=uncapped.get, reverse=True)[:4]:
        true_rate = GROWTH_RATES_BY_TYPE[k[0]] * SEGMENT_MULT[k[1]]
        print(f"  {k[0] + ' | ' + k[1]:<34} fitted {uncapped[k]:5.2%}   true {true_rate:5.2%}")
    print()

    overall_rows = []
    per_series_by_cap: dict[float, dict] = {}

    for cap in CAPS:
        feat.MAX_MONTHLY_GROWTH_RATE = cap
        n_binding = sum(1 for r in uncapped.values() if r > cap + 1e-9)
        metrics, models, trends, _ = fit_and_evaluate(fit_df, val_df, test_df)
        per_series_by_cap[cap] = per_series_mape(models, trends, test_df)
        overall_rows.append((cap, n_binding, metrics))
        print(
            f"cap={cap_label(cap):>5}  binds on {n_binding:>2} series  |  "
            f"test MAPE {metrics['mape_p50']:5.2f}%   "
            f"bias {metrics['bias_pct']:+5.2f}%   "
            f"cover {metrics['coverage_p10_p90']:4.1f}%"
        )

    feat.MAX_MONTHLY_GROWTH_RATE = BASELINE_CAP  # restore

    # ---- Overall comparison table ----
    base = next(m for c, _, m in overall_rows if c == BASELINE_CAP)["mape_p50"]
    print("\n" + "=" * 64)
    print("OVERALL TEST MAPE BY CAP")
    print("=" * 64)
    print(f"{'cap':>6} {'binds':>6} {'MAPE':>8} {'vs 5%':>8} {'bias':>8} {'cover':>7}")
    for cap, n_binding, m in overall_rows:
        delta = m["mape_p50"] - base
        mark = "  <- shipped" if cap == BASELINE_CAP else ""
        print(
            f"{cap_label(cap):>6} {n_binding:>6} {m['mape_p50']:>7.2f}% "
            f"{delta:>+7.2f} {m['bias_pct']:>+7.2f}% {m['coverage_p10_p90']:>6.1f}%{mark}"
        )

    # ---- Per-series MAPE for the series the cap touches ----
    watch = [
        ("GPU Inference", "Startup"),
        ("GPU Inference", "Mid-Market"),
        ("GPU Inference", "Enterprise"),
        ("GPU Training", "Startup"),
    ]
    print("\n" + "=" * 64)
    print("PER-SERIES P50 TEST MAPE (series the cap can touch)")
    print("=" * 64)
    header = f"{'series':<30}" + "".join(f"{cap_label(c):>8}" for c in CAPS)
    print(header)
    for key in watch:
        row = f"{key[0] + ' | ' + key[1]:<30}"
        for c in CAPS:
            row += f"{per_series_by_cap[c][key]:>7.1f}%"
        print(row)
    print("\n(none = uncapped. True fastest series = GPU Inference|Startup at 6.00%/mo.)")


if __name__ == "__main__":
    main()
