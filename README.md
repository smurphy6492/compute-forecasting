# Compute Capacity Forecasting

ML-driven capacity planning for a fast-growing AI compute infrastructure company. Translates 36 months of usage data into actionable forecasts with confidence intervals and scenario analysis for GPU procurement decisions.

## The Problem

AI compute providers face a planning paradox: GPU procurement lead times are 3-6 months, but demand is growing 47% annually with high variance across customer segments. Order too late and customers churn to competitors. Order too early and capital sits idle.

**The question this project answers:** *When do we need to buy more GPUs, and how confident should we be in that timeline?*

## Key Results

![3-Year Overview](outputs/figures/overview_with_events.png)
*Total daily compute hours across 16 series (4 compute types x 4 customer segments), with documented business events.*

| Metric | Result |
|--------|--------|
| **Test MAPE** | 8.8% (vs. 10.5% growth-adjusted seasonal naive) |
| **Backtesting** | Consistent across 3 walk-forward folds (7.9%-13.1%) |
| **Coverage (P10-P90)** | 76.9% overall; GPU series under-covered at 63-67% |
| **Top predictors** | Yesterday's value, weekly lags, rolling averages, trend |

![Forecast vs Actual](outputs/figures/forecast_vs_actual.png)
*LightGBM P50 forecast (blue) with P10-P90 confidence bands vs. actuals (black) on held-out test set (Jan-Jun 2026).*

## Approach

**Global LightGBM model** trained on all 16 series simultaneously, using compute type and customer segment as categorical features. One model learns shared patterns (weekly cycles, holiday effects) while categoricals capture series-specific growth rates and volatility.

**Why gradient boosting over ARIMA/Prophet:** Handles 16 related series in a single fit, natively supports categorical features, and produces quantile forecasts (P10/P50/P90) without distributional assumptions.

### Pipeline

```
EDA (51 cells)              Forecasting (50 cells)         Scenario Planning
--------------------        ----------------------         -----------------
Trend decomposition         24-feature engineering         60-day daily forecast
ACF/PACF + stationarity     LightGBM quantile regression   3-6 month envelopes
Event impact analysis       Conformal calibration (CQR)    Base/High/Low scenarios
Outlier quantification      Walk-forward backtesting       Capacity threshold charts
Correlation structure       SHAP interpretability
                            Coverage gap investigation
```

### Feature Engineering (24 features)

| Group | Features | Signal |
|-------|----------|--------|
| Calendar | day_of_week, month, quarter, is_weekend, is_quarter_end | Weekly/quarterly patterns |
| Holiday | is_holiday, days_to_next, days_from_last | 15-55% demand suppression |
| Lag | 1, 7, 14, 28, 365 days | Autoregressive structure |
| Rolling | 7d/28d mean and std | Local level and volatility |
| Trend | days_since_start | 47% annualized growth |
| Categorical | compute_type, customer_segment | Series-specific patterns |

![Feature Importance](outputs/figures/feature_importance.png)

## Honest Limitations

- **GPU series coverage gap** -- P10-P90 intervals cover only 63-67% of GPU actuals (target: 80%). Root cause: fast-growing series systematically exceed P90. Per-type conformal calibration helps but doesn't fully solve it.
- **Single-step evaluation** -- metrics use actual lag values. True 30-90 day recursive forecasts would degrade as errors accumulate.
- **Synthetic data** -- uses realistic synthetic data with documented events, variable growth rates, and segment-specific patterns. Not production telemetry.
- **Hyperparameters not tuned** -- sensible defaults, not optimized via grid search.

## Dataset

36 months of daily compute hours (Jul 2023 - Jun 2026) across 4 compute types and 4 customer segments (17,536 rows). Key characteristics:

- **GPU Inference** grew 5.4x (inference boom); **CPU Batch** grew 1.7x (mature)
- **Startups** grew 5.6x from small base; **Research/Academic** near-flat at 1.6x
- 4 step-changes (3 onboardings + 1 churn), 2 conference spikes, 1 GPU outage
- Weekly seasonality (30-50% weekend drop), tiered holiday effects, quarterly pushes

## Project Structure

```
compute-forecasting/
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis (51 cells)
│   └── 02_forecasting.ipynb            # LightGBM model + evaluation (50 cells)
├── src/compute_forecasting/
│   └── features.py                     # Reusable feature engineering pipeline
├── data/
│   ├── generate_synthetic_data.py      # Deterministic data generator (seed=42)
│   ├── compute_usage.csv               # 17,536 rows
│   ├── event_log.csv                   # 7 documented business events
│   └── holiday_calendar.csv            # US federal holidays (tiered severity)
└── outputs/figures/                    # Key visualizations
```

## How to Run

```bash
pip install -e ".[dev]"
python data/generate_synthetic_data.py    # regenerate data (deterministic)
jupyter notebook notebooks/01_eda.ipynb   # run notebooks in order
```

## Tech Stack

Python 3.11 | LightGBM | pandas | scikit-learn | SHAP | statsmodels | matplotlib/seaborn
