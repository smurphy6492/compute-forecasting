# Compute Capacity Forecasting

ML-driven capacity planning for a fast-growing AI compute infrastructure company. Forecasts daily compute hours using gradient boosting (LightGBM) with quantile regression for confidence intervals, plus scenario planning for board-level capacity decisions.

## Status

Work in progress — Session 1 (data generation) complete.

## Business Context

You're the analytics lead at a cloud GPU/CPU compute provider growing rapidly. Leadership needs to answer: **"When do we need to buy more GPUs?"**

This project builds a forecasting system that produces:

- **60-day daily forecast** with P10/P50/P90 confidence bands — actionable for near-term capacity management
- **Months 3-6 planning envelope** — monthly summary stats (avg, median, min, max) at each confidence level, suitable for procurement decisions with 3-6 month GPU lead times
- **Three scenarios** layered on top:
  - **Base:** Current trends continue
  - **High:** Sales pipeline converts (new large customers sign)
  - **Low:** Efficiency improvements reduce compute demand (a "DeepSeek moment")

## Dataset

36 months of synthetic daily compute hours (2023-07 to 2026-06) across:

- **4 compute types:** GPU Training, GPU Inference, CPU Batch, CPU Interactive
- **4 customer segments:** Enterprise, Mid-Market, Startup, Research/Academic

The data is synthetic but designed to feel real — every anomaly has a documented business reason:

| Signal | How It Appears |
|--------|---------------|
| Weekly seasonality | Weekdays ~33% higher than weekends; GPU Training has flatter weekends (jobs run overnight) |
| Annual seasonality | End-of-quarter spikes, summer dips, holiday slowdowns |
| Growth trends | GPU Inference growing 5.4x over 3 years (inference boom); CPU Batch at 1.7x (mature) |
| Segment differences | Startups grew 5.6x (small base, scaling fast); Research near-flat (budget-constrained) |
| Step-changes | 3 new customer onboardings causing permanent level shifts |
| One-time spikes | ML conference driving 2-3 day training bursts |
| Outage | GPU cluster failure — 2-day outage + 3-day recovery, CPU unaffected |
| Noise | Enterprise = low variance (SLA workloads), Startup = high variance (bursty) |

## Project Structure

```
compute-forecasting/
├── README.md
├── SESSION_GUIDE.md                    # Session-by-session build guide
├── pyproject.toml
├── data/
│   ├── generate_synthetic_data.py      # Deterministic, seeded data generator
│   ├── compute_usage.csv               # 17,536 rows (1,096 days x 16 series)
│   ├── upcoming_events.csv             # 5 sales pipeline deals with probabilities
│   ├── holiday_calendar.csv            # 55 US federal holidays
│   └── event_log.csv                   # 6 documented historical events
├── notebooks/
│   ├── 01_eda.ipynb                    # Part 1: Exploratory Data Analysis
│   ├── 02_forecasting.ipynb            # Part 2: LightGBM Forecast Model
│   └── 03_scenarios.ipynb              # Part 3: Scenario Planning & Capacity
├── src/
│   └── compute_forecasting/
│       ├── features.py                 # Reusable feature engineering
│       ├── evaluation.py               # Metrics and diagnostics
│       └── visualization.py            # Shared chart functions
└── outputs/
    └── figures/                        # Saved PNGs for portfolio
```

## Approach

### Part 1: Exploratory Data Analysis
Understand the data before modeling. Trend analysis by compute type and segment, seasonality decomposition, weekly pattern heatmaps, event impact quantification, and correlation analysis. Produces the insights that drive feature engineering decisions.

### Part 2: Forecasting with LightGBM
A global model (one LightGBM for all 16 series) with compute type and segment as categorical features. Why gradient boosting over ARIMA or Prophet:

- Handles multiple related time series in a single model
- Naturally incorporates external features (holidays, events, calendar)
- Non-linear relationships without manual specification
- Feature importance tells the business story

**Confidence intervals** via quantile regression — train separate models for P10, P50, P90. No distributional assumptions; directly learns conditional quantiles.

**Feature engineering:** Calendar features, holiday indicators, lag features (1/7/14/28/365 days), rolling statistics (7d/28d mean and std), trend proxy, and categorical interactions.

### Part 3: Scenario Planning
The executive deliverable. Three scenarios visualized against capacity thresholds:

- **Base:** Model forecast assuming current trends continue
- **High:** Sales pipeline converts — layer in expected customer signings, probability-weighted
- **Low:** Efficiency improvements — model a 20% reduction in GPU Inference demand

Includes sensitivity analysis (which assumptions move the needle most) and a capacity decision matrix.

## Tech Stack

- Python 3.11
- LightGBM (gradient boosting + quantile regression)
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- statsmodels (seasonal decomposition)
- Jupyter notebooks

## How to Run

```bash
# Install dependencies
pip install -e ".[dev]"

# Generate synthetic data (deterministic, seeded)
python data/generate_synthetic_data.py

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

## Key Design Decisions

1. **Global model** over per-series models — 16 series with only 1,096 days each would starve individual models. A global model shares patterns (all series drop on weekends) while categoricals capture differences.
2. **Two-tier forecast output** — daily precision for 60 days (actionable), summary stats for months 3-6 (planning). Daily forecasts 6 months out imply false confidence.
3. **Quantile regression** over bootstrap or Gaussian CIs — faster, no distributional assumptions, properly calibrated for asymmetric distributions.
4. **Narrative-driven synthetic data** — every anomaly is documented in `event_log.csv`. This mirrors real-world practice where the data team maintains an event log to explain historical anomalies to the model.
5. **Educational documentation** — every notebook section explains *why*, not just *what*. Targets a Director-level audience who needs to evaluate and communicate forecasting methodology.
