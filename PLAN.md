# Compute Capacity Forecasting — Project Plan

## Context

Sean had a technical interview involving forecasting daily compute hours. This project recreates that exercise as a portfolio piece: synthetic but realistic data, gradient boosting forecasting with confidence intervals, scenario planning for capacity decisions, and heavy educational documentation throughout. The goal is both a learning artifact and a demonstration of Director-level forecasting expertise.

**Business framing:** This is a fast-growing AI compute company presenting capacity plans to the board.

---

## File Structure

```
projects/compute-forecasting/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   ├── generate_synthetic_data.py      # Deterministic, seeded data generator
│   ├── compute_usage.csv               # 36 months daily (generated)
│   ├── upcoming_events.csv             # Sales pipeline with probabilities
│   ├── holiday_calendar.csv            # US federal holidays 2023-2027
│   └── event_log.csv                   # Step-changes, spikes, outage log
├── notebooks/
│   ├── 01_eda.ipynb                    # Part 1: Exploratory Data Analysis
│   ├── 02_forecasting.ipynb            # Part 2: LightGBM Forecast Model
│   └── 03_scenarios.ipynb             # Part 3: Scenario Planning & Capacity
├── src/
│   └── compute_forecasting/
│       ├── __init__.py
│       ├── features.py                 # Reusable feature engineering
│       ├── evaluation.py               # Metrics helpers
│       └── visualization.py            # Shared chart functions
└── outputs/
    └── figures/                        # Saved PNGs for portfolio
```

---

## Executive Deliverable Framing

The forecast output mirrors real-world capacity planning:

- **Days 1-60 (near-term):** Daily granularity with P10/P50/P90 fan chart. Actionable — you can spin up instances, shift workloads, negotiate spot capacity.
- **Months 3-6 (planning horizon):** Summary statistics per month — avg, median, min, max daily compute hours at P10/P50/P90. Daily precision at this horizon is false confidence. Instead, give leadership a planning envelope: "we need capacity for between X and Y daily hours by month 6."
- **Scenarios layer on top of both horizons:** High/Base/Low matter most in months 3-6 where uncertainty is widest and where procurement lead times (3-6 months for GPUs) require decisions now.

---

## Phases

### Phase 0: Scaffolding
- `pyproject.toml` with deps: pandas, numpy, lightgbm, scikit-learn, matplotlib, seaborn, plotly, statsmodels, jupyter
- `.gitignore`, directory structure, `README.md` stub

### Phase 1: Synthetic Data Generation
Script: `data/generate_synthetic_data.py` — layered, multiplicative signal composition:

| Layer | What | Details |
|-------|------|---------|
| Base | Per (compute_type, segment) baseline | 4 types x 4 segments = 16 series |
| Trend | Compound growth, different rates | GPU Inference ~4%/mo, CPU Batch ~1%/mo, Research flat |
| Weekly | Day-of-week multipliers | Weekday ~1.0-1.15, Sat ~0.70, Sun ~0.60; varies by type |
| Annual | Seasonal pattern | EOQ spikes, holiday dips, summer slowdown |
| Holidays | US federal holidays | 30-50% reduction on holidays |
| Step-changes | 3 documented events | New large customer onboards -> permanent level shift |
| Spikes | 2 documented events | ML conference -> 3-day burst |
| Outage | 1 event | GPU cluster down 2 days, 3-day recovery |
| Noise | Log-normal multiplicative | Enterprise=low, Startup=high |

**Compute types:** GPU Training, GPU Inference, CPU Batch, CPU Interactive
**Customer segments:** Enterprise, Mid-Market, Startup, Research/Academic
**Date range:** 2023-07-01 to 2026-06-30 (36 months)

Supporting CSVs: `upcoming_events.csv` (5 pipeline deals with probabilities), `holiday_calendar.csv`, `event_log.csv` (all documented events)

### Phase 2: EDA Notebook (`01_eda.ipynb`)
1. Data loading & quality checks
2. Overall trend with event annotations
3. Trend by compute_type (4 subplots + growth rate table)
4. Trend by customer_segment (concentration analysis)
5. Seasonality decomposition (statsmodels)
6. Weekly patterns — heatmap + boxplots by day-of-week per type
7. Event impact quantification (before/after comparison)
8. Distribution analysis (histograms, heavy tails)
9. Correlation between segments and types
10. Key findings summary -> "what this means for forecasting"

### Phase 3: Forecasting Notebook (`02_forecasting.ipynb`)
1. **Why gradient boosting** — comparison table vs ARIMA/Prophet
2. **Data prep** — model at (date, compute_type) level with segment as feature (global model)
3. **Feature engineering:**
   - Calendar: day_of_week, month, quarter, is_weekend, week_of_year
   - Holiday: is_holiday, days_since/until_holiday
   - Lags: 1, 7, 14, 28, 365
   - Rolling: 7d/28d mean and std
   - Trend: days_since_start
   - Categoricals: compute_type, customer_segment
4. **Time-based split:** Train (24mo) / Val (6mo) / Test (6mo)
5. **Baselines:** Naive, seasonal naive, 28d moving average
6. **LightGBM training** with explanation of hyperparameters
7. **Quantile regression** — 3 models (P10, P50, P90) for confidence intervals
   - Deep explanation of quantile loss function
8. **Evaluation:** MAPE, MAE, RMSE, coverage of P10-P90 interval
9. **Feature importance** analysis
10. **Residual analysis** — patterns, autocorrelation, bias by segment

### Phase 4: Scenario Planning Notebook (`03_scenarios.ipynb`)
1. **60-day daily forecast** — P10/P50/P90 fan chart, actionable near-term view
2. **Months 3-6 summary** — avg, median, min, max daily hours per month at each confidence level
3. **Base scenario:** Model forecast (current trends continue)
4. **High scenario:** Sales pipeline converts (layer in expected signings, probability-weighted)
5. **Low scenario:** Efficiency gains reduce GPU Inference ~20% (DeepSeek moment)
6. **Capacity threshold chart** — all 3 scenarios + capacity line, "when do we run out?"
7. **Sensitivity analysis** — tornado chart of key assumptions
8. **Recommendations** — capacity decision matrix, monitoring plan
9. **Executive summary** — 5 bullets + hero chart

### Phase 5: Shared Code Extraction
Extract common patterns from notebooks into `src/compute_forecasting/`:
- `features.py` — calendar, lag, rolling, holiday feature functions
- `evaluation.py` — metrics, coverage, model comparison
- `visualization.py` — forecast fan chart, scenario comparison, style defaults

### Phase 6: Documentation & Portfolio
- `README.md` — overview, how to run, key findings preview
- Save hero figures to `outputs/figures/`
- Portfolio website integration (future step via `/portfolio-updater`)

---

## Key Design Decisions

1. **Global model** (one LightGBM for all 16 series) — more training data, shared patterns, categorical features capture differences
2. **LightGBM over XGBoost** — faster, native categorical support, cleaner quantile API
3. **Quantile regression for CIs** — no distributional assumptions, properly calibrated, one model per quantile
4. **Daily granularity** — preserves weekly seasonality signal, ~1,100 obs per series
5. **Narrative-driven synthetic data** — every anomaly has a documented business reason in event_log.csv
6. **Educational density** — every section explains *why*, not just *what*, targeting Director-level audience
7. **Two-tier forecast output** — 60-day daily + months 3-6 summary stats; mirrors real capacity planning cadence

## Challenge: Multi-Step Forecasting
When forecasting 6 months out, short lags (lag_1, lag_7) won't be available. Approach: recursive forecasting (predict day 1 -> use as lag for day 2) for short lags, plus calendar features and lag_365 for structure. Document the limitation clearly.

---

## Verification
- Data generator produces correct shape, no negatives, all types/segments present
- Notebooks run top-to-bottom without errors
- Model beats all baselines on val and test sets
- P10-P90 coverage is ~80%
- Figures save cleanly to outputs/figures/
- All markdown cells render correctly (escape `\$` in Jupyter)

---

## Session Guide

Context management is critical — each notebook is substantial with educational markdown, code, and outputs. Here's how to split the work across sessions for maximum efficiency.

### Session 1: Scaffolding + Data Generation
**Scope:** Phase 0 + Phase 1
**Initial prompt:**
> We're building the compute capacity forecasting project. Start with Phase 0 (scaffolding) and Phase 1 (data generation). The plan is at `.claude/plans/gentle-doodling-sphinx.md`. Create the project structure, pyproject.toml, and the synthetic data generator script. Generate all CSVs. Validate the data looks realistic — spot check trends, seasonality, and events. Show me sample rows and basic summary stats so I can confirm the data feels right before we move to EDA.

**Agent orchestration:** No agents needed — this is single-thread work (sequential file creation + validation).
**Expected output:** All files in `data/`, working `pyproject.toml`, validated CSVs.

---

### Session 2: EDA Notebook
**Scope:** Phase 2
**Initial prompt:**
> Continue the compute forecasting project. The data is generated at `projects/compute-forecasting/data/`. Build the EDA notebook (`notebooks/01_eda.ipynb`) following the plan at `.claude/plans/gentle-doodling-sphinx.md`. This is Part 1 — exploratory analysis with heavy educational markdown. Cover: overall trends with event annotations, trends by compute_type and customer_segment, seasonality decomposition, weekly patterns, event impact quantification, distribution and correlation analysis, and a key findings summary. Use consistent styling (seaborn whitegrid, 100 dpi). Remember to escape `\$` in markdown cells.

**Agent orchestration:** Can optionally use `web-developer` agent in parallel to draft visualization helper functions in `src/compute_forecasting/visualization.py` while you build the notebook — but only if context is getting tight. Usually better to keep this single-threaded and extract shared code later.
**Expected output:** Complete `01_eda.ipynb` with all cells populated and outputs.

---

### Session 3: Forecasting Model
**Scope:** Phase 3 + start of Phase 5 (extract features.py)
**Initial prompt:**
> Continue the compute forecasting project. EDA is complete at `projects/compute-forecasting/notebooks/01_eda.ipynb`. Build the forecasting notebook (`notebooks/02_forecasting.ipynb`) following the plan. This is Part 2 — LightGBM with quantile regression. Cover: why gradient boosting (comparison table), feature engineering with explanations, time-based train/val/test split, baseline models, LightGBM training, quantile regression for P10/P50/P90, evaluation metrics, feature importance, and residual analysis. Also extract reusable feature engineering into `src/compute_forecasting/features.py`. Remember to escape `\$` in markdown cells.

**Agent orchestration:** This is the most complex session. Options:
- **Option A (recommended):** Single-thread. The notebook is sequential and each section builds on the last.
- **Option B (if context is tight):** Use a background agent to build `features.py` and `evaluation.py` in `src/` while you build the notebook, then import from those modules.

**Expected output:** Complete `02_forecasting.ipynb`, `src/compute_forecasting/features.py`, trained models.

---

### Session 4: Scenario Planning + Polish
**Scope:** Phase 4 + remainder of Phase 5 + Phase 6
**Initial prompt:**
> Continue the compute forecasting project. The forecasting model is built at `projects/compute-forecasting/notebooks/02_forecasting.ipynb`. Build the scenario planning notebook (`notebooks/03_scenarios.ipynb`) following the plan. This is Part 3 — the exec deliverable. Structure: 60-day daily forecast with P10/P50/P90 fan chart, then months 3-6 summary table (avg, median, min, max at each confidence level). Three scenarios: Base (current trends), High (sales pipeline converts), Low (efficiency improvements). Build the capacity threshold chart — all 3 scenarios + capacity line showing "when do we run out?" Include sensitivity analysis (tornado chart), recommendations, and executive summary. Save hero figures to `outputs/figures/`. Also extract visualization helpers to `src/compute_forecasting/visualization.py` and finalize `README.md`. Escape `\$` in markdown cells.

**Agent orchestration:** Can parallelize here:
- **Main thread:** Build `03_scenarios.ipynb`
- **Background agent (`content-writer`):** Draft the `README.md` and executive summary text while you build the notebook
- **Background agent (`web-developer`):** If portfolio integration is in scope, start drafting the portfolio case study page

**Expected output:** Complete `03_scenarios.ipynb`, all `src/` modules, `README.md`, figures in `outputs/figures/`.

---

### Session 5 (Optional): Portfolio Integration
**Scope:** Website case study + PR
**Initial prompt:**
> The compute forecasting project is complete at `projects/compute-forecasting/`. Use `/portfolio-updater` to add it to the personal website. The hero image is `outputs/figures/capacity_threshold_chart.png`. Frame it as "Compute Capacity Forecasting — ML-driven capacity planning for an AI infrastructure company" targeting Director of Analytics roles.

**Agent orchestration:** Use `/portfolio-updater` skill which chains `content-writer` + `web-developer` + `github-workflow`.

---

### When to Start a New Session
- **Start fresh** when the previous session's notebook is complete and validated
- **Don't split** a single notebook across sessions — each notebook should be built in one session
- **Context pressure signals:** If you're past ~60% of context and still have major work left, wrap up the current deliverable and note what's left for the next session

### Agent Team Patterns
| Pattern | When to Use |
|---------|-------------|
| Single-thread | Default for notebook building — cells are sequential |
| Background `content-writer` | README, case study, executive summary text — can run while you code |
| Background `web-developer` | Portfolio page scaffolding — only in Session 5 |
| Background code extraction | `src/` module extraction — only if main notebook is done and you want to parallelize cleanup |
| `python-reviewer` at end | Run on completed `.py` files before committing (Session 4 wrap-up) |
