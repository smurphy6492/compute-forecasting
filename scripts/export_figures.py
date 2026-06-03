"""Export hero figures from notebooks for README embedding."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from compute_forecasting.features import (
    build_features,
    get_feature_columns,
    fit_series_trends,
    compute_residual_ratios,
    predict_with_trend,
)

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 14, "axes.labelsize": 12})

TYPE_COLORS = {
    "GPU Training": "#2196F3",
    "GPU Inference": "#FF9800",
    "CPU Batch": "#4CAF50",
    "CPU Interactive": "#9C27B0",
}
EVENT_COLORS = {"step_change": "#2E7D32", "spike": "#E65100", "outage": "#C62828", "churn": "#C62828"}

# Load data
usage = pd.read_csv(ROOT / "data" / "compute_usage.csv", parse_dates=["date"])
holidays = pd.read_csv(ROOT / "data" / "holiday_calendar.csv", parse_dates=["date"])
events = pd.read_csv(ROOT / "data" / "event_log.csv", parse_dates=["event_date", "event_end_date"])

daily_total = usage.groupby("date")["compute_hours"].sum().reset_index()

# ============================================================
# Figure 1: 3-Year Overview with Events
# ============================================================
print("Generating: overview_with_events.png")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_total["date"], daily_total["compute_hours"], alpha=0.2, color="#90CAF9", linewidth=0.5)
rolling_28 = daily_total.set_index("date")["compute_hours"].rolling(28, center=True).mean()
ax.plot(rolling_28.index, rolling_28.values, color="#1565C0", linewidth=2.5, label="28-day rolling avg")

y_max = daily_total["compute_hours"].max()
offsets = [0.97, 0.91, 0.85, 0.79, 0.73, 0.67, 0.61]
for i, (_, evt) in enumerate(events.iterrows()):
    color = EVENT_COLORS.get(evt["event_type"], "gray")
    ax.axvline(evt["event_date"], color=color, linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate(evt["event_name"], xy=(evt["event_date"], y_max * offsets[i % len(offsets)]),
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-12-31")
ax.axvline(TRAIN_END, color="#C62828", linestyle="-", linewidth=1.5, alpha=0.5)
ax.axvline(VAL_END, color="#E65100", linestyle="-", linewidth=1.5, alpha=0.5)
ax.axvspan(VAL_END, daily_total["date"].max(), alpha=0.08, color="orange")
ax.text(VAL_END + pd.Timedelta(days=10), y_max * 0.15, "Test\nPeriod", fontsize=9, color="#E65100", fontweight="bold")

ax.set_title("Total Daily Compute Hours - 3-Year Overview with Business Events", fontweight="bold")
ax.set_ylabel("Compute Hours")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000:,.0f}K"))
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(OUT / "overview_with_events.png", bbox_inches="tight")
plt.close()

# ============================================================
# Build Hybrid Model
# ============================================================
print("Training hybrid model...")
df = build_features(usage, holidays)
feature_cols = get_feature_columns(df)
TARGET = "compute_hours"
CAT_FEATURES = ["compute_type", "customer_segment", "series_id"]

train = df[df["date"] <= TRAIN_END].copy()
val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test = df[df["date"] > VAL_END].copy()

X_train, y_train = train[feature_cols], train[TARGET]
X_val, y_val = val[feature_cols], val[TARGET]
X_test, y_test = test[feature_cols], test[TARGET]

PARAMS = {
    "objective": "quantile", "alpha": 0.5, "metric": "quantile",
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "n_estimators": 2000, "random_state": 42, "verbose": -1,
}

# Fit trends and compute residual ratios
trends = fit_series_trends(train)
train_ratios = compute_residual_ratios(train, trends)
val_ratios = compute_residual_ratios(val, trends)

y_train_log_resid = np.log(train_ratios)
y_val_log_resid = np.log(val_ratios)

# Train hybrid models
models = {}
for quantile, label in [(0.5, "P50"), (0.1, "P10"), (0.9, "P90")]:
    params_q = PARAMS.copy()
    params_q["alpha"] = quantile
    m = lgb.LGBMRegressor(**params_q)
    m.fit(X_train, y_train_log_resid, eval_set=[(X_val, y_val_log_resid)],
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
          categorical_feature=CAT_FEATURES)
    models[label] = m

# Generate predictions via trend reconstruction
test_p50 = predict_with_trend(test, models["P50"].predict(X_test), trends)
test_p10_raw = predict_with_trend(test, models["P10"].predict(X_test), trends)
test_p90_raw = predict_with_trend(test, models["P90"].predict(X_test), trends)

# Per-type proportional conformal calibration
val_p50 = predict_with_trend(val, models["P50"].predict(X_val), trends)
val_p10_raw = predict_with_trend(val, models["P10"].predict(X_val), trends)
val_p90_raw = predict_with_trend(val, models["P90"].predict(X_val), trends)

type_pct_adj = {}
for ctype in ["GPU Training", "GPU Inference", "CPU Batch", "CPU Interactive"]:
    mask = (val["compute_type"] == ctype).values
    raw_scores = np.maximum(val_p10_raw[mask] - y_val.values[mask], y_val.values[mask] - val_p90_raw[mask])
    pct_scores = raw_scores / np.maximum(val_p50[mask], 1)
    n_t = len(pct_scores)
    q_t = min(np.ceil((n_t + 1) * 0.80) / n_t, 1.0)
    type_pct_adj[ctype] = np.quantile(pct_scores, q_t)

test_cal_p10 = np.zeros(len(test))
test_cal_p90 = np.zeros(len(test))
for ctype, pct_adj in type_pct_adj.items():
    mask = (test["compute_type"] == ctype).values
    test_cal_p10[mask] = test_p10_raw[mask] - pct_adj * test_p50[mask]
    test_cal_p90[mask] = test_p90_raw[mask] + pct_adj * test_p50[mask]

test_preds = {"P50": test_p50, "P10": test_cal_p10, "P90": test_cal_p90}

# ============================================================
# Figure 2: Forecast vs Actual (4-panel) - Hybrid Model
# ============================================================
print("Generating: forecast_vs_actual.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
series_to_plot = [
    ("GPU Training", "Enterprise"), ("GPU Inference", "Enterprise"),
    ("CPU Batch", "Enterprise"), ("GPU Training", "Mid-Market"),
]
for ax, (ctype, seg) in zip(axes.flat, series_to_plot):
    mask = (test["compute_type"] == ctype) & (test["customer_segment"] == seg)
    idx = mask.values
    dates = test[mask]["date"].values
    actual = test[mask]["compute_hours"].values
    mape = mean_absolute_percentage_error(actual, test_preds["P50"][idx]) * 100
    ax.fill_between(dates, test_preds["P10"][idx], test_preds["P90"][idx],
                    alpha=0.2, color="#1565C0", label="P10-P90")
    ax.plot(dates, actual, color="#333333", linewidth=1, alpha=0.7, label="Actual")
    ax.plot(dates, test_preds["P50"][idx], color="#1565C0", linewidth=1.5, label="P50 forecast")
    ax.set_title(f"{ctype} - {seg} (MAPE: {mape:.1f}%)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000:,.0f}K"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig.suptitle("Hybrid Model Forecast vs. Actual - Test Set (Jan-Jun 2026)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT / "forecast_vs_actual.png", bbox_inches="tight")
plt.close()

# ============================================================
# Figure 3: Feature Importance - Hybrid Model
# ============================================================
print("Generating: feature_importance.png")
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": models["P50"].feature_importances_,
}).sort_values("importance", ascending=False)
importance["pct"] = importance["importance"] / importance["importance"].sum() * 100

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#1565C0" if pct > 5 else "#90CAF9" for pct in importance["pct"]]
ax.barh(importance["feature"], importance["pct"], color=colors, edgecolor="white")
ax.set_xlabel("Importance (% of total gain)")
ax.set_title("Feature Importance - Hybrid Model (residual target)", fontweight="bold")
ax.invert_yaxis()
for i, (_, row) in enumerate(importance.iterrows()):
    if row["pct"] > 1:
        ax.text(row["pct"] + 0.3, i, f"{row['pct']:.1f}%", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "feature_importance.png", bbox_inches="tight")
plt.close()

# ============================================================
# Figure 4: Hybrid Model Improvement (Enterprise GPU Training)
# ============================================================
print("Generating: hybrid_improvement.png")

# Also train original model for comparison
orig_models = {}
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
for quantile, label in [(0.5, "P50"), (0.1, "P10"), (0.9, "P90")]:
    params_q = PARAMS.copy()
    params_q["alpha"] = quantile
    m = lgb.LGBMRegressor(**params_q)
    m.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
          categorical_feature=CAT_FEATURES)
    orig_models[label] = m

orig_test_p50 = np.expm1(orig_models["P50"].predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
ctype, seg = "GPU Training", "Enterprise"
mask = ((test["compute_type"] == ctype) & (test["customer_segment"] == seg)).values
dates = test.loc[mask, "date"].values
actual = y_test.values[mask]

# Original
ax = axes[0]
orig_mape = mean_absolute_percentage_error(actual, orig_test_p50[mask]) * 100
ax.plot(dates, actual, color="#333", linewidth=1, alpha=0.7, label="Actual")
ax.plot(dates, orig_test_p50[mask], color="#C62828", linewidth=1.5, label="Original P50")
ax.set_title(f"Original LightGBM - MAPE: {orig_mape:.1f}%", fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000:,.0f}K"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Compute Hours")

# Hybrid
ax = axes[1]
hyb_mape = mean_absolute_percentage_error(actual, test_preds["P50"][mask]) * 100
ax.fill_between(dates, test_preds["P10"][mask], test_preds["P90"][mask], alpha=0.2, color="#4CAF50")
ax.plot(dates, actual, color="#333", linewidth=1, alpha=0.7, label="Actual")
ax.plot(dates, test_preds["P50"][mask], color="#4CAF50", linewidth=1.5, label="Hybrid P50")
ax.set_title(f"Hybrid (Trend + Residual) - MAPE: {hyb_mape:.1f}%", fontweight="bold")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig.suptitle("Enterprise GPU Training - Extrapolation Problem Solved",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "hybrid_improvement.png", bbox_inches="tight")
plt.close()

# Summary
test_mape = mean_absolute_percentage_error(y_test, test_preds["P50"]) * 100
print(f"\nHybrid Model Test MAPE: {test_mape:.2f}%")
print(f"Enterprise GPU Training: {hyb_mape:.1f}% (was {orig_mape:.1f}%)")
print(f"Figures saved to: {OUT}")
