# Production Pipeline

MLflow tracking and a local model registry for the hybrid forecast model.
This is the first phase of taking the published notebook model to a full production loop (track → register → deploy → monitor → retrain).
The notebooks stay untouched; this directory re-implements their training recipe as tracked scripts.

## What's here

| File | Purpose |
|---|---|
| `train.py` | Walk-forward backtest + final model training, all logged to MLflow; registers the model with the `@champion` alias |
| `predict.py` | Loads `models:/compute-forecast-hybrid@champion` from the registry and forecasts the next day for all 16 series |
| `pyfunc_model.py` | Custom pyfunc that packages the hybrid model's three components as one loadable model |
| `tracking.py` | Shared MLflow config: sqlite backend, experiment and model names |

## Quickstart

```bash
pip install -e ".[production]"
python production/train.py     # trains, logs, registers @champion
python production/predict.py   # loads champion from registry, forecasts
mlflow ui --backend-store-uri sqlite:///mlflow.db   # from repo root
```

The tracking store is a repo-local `mlflow.db` (sqlite) with artifacts under `mlruns/`.
Both are gitignored — a fresh clone rebuilds them by running `train.py`.
Sqlite rather than the default file store because MLflow 3.x deprecated the file backend and the model registry requires a database.

## Why a custom pyfunc

The model is not one estimator.
A forecast is `trend(t) * exp(booster(features))`, conformally widened per compute type:

- three LightGBM boosters (P10/P50/P90) trained on log residual ratios
- per-series exponential trend parameters for real-scale reconstruction
- per-compute-type conformal adjustments fitted on the validation window

None of these serializes with `mlflow.lightgbm.log_model` alone, so `pyfunc_model.py` bundles them: boosters as native LightGBM text files (saved at their early-stopping iteration), trends and conformal adjustments as JSON, plus the feature-column order.
`code_paths` ships `compute_forecasting` and the wrapper module inside the model artifact, so the registered model loads in a process with no repo checkout on `sys.path`.

## Model contract

Input: a DataFrame with a `date` column plus the 25 feature columns from `compute_forecasting.features.FEATURE_COLUMNS` (numerics as float64, categoricals as strings).
Feature building stays outside the model — use `build_features` for batch scoring or `RecursiveFeatureBuilder` for day-by-day forecasting.

Output: one row per input row with `date`, `compute_type`, `customer_segment`, and calibrated `p10` / `p50` / `p90` on the compute-hours scale.

## What gets logged per training run

- **Params:** LightGBM params, split dates, quantiles, conformal target coverage, feature and row counts
- **Metrics:** per-fold backtest MAPE/coverage, final test MAPE/bias/coverage, per-compute-type MAPE, conformal adjustment per type, best iteration per quantile
- **Artifacts:** SHAP summary plot (P50 model), backtest fold table, trend and conformal JSONs, and the registered pyfunc model

## Split design

The registered champion trains on data through 2025-06-30, early-stops and calibrates on Jul–Dec 2025, and reports metrics on the held-out Jan–Jun 2026 window (test MAPE ~8.0%, matching the published notebook result).
The walk-forward backtest uses three expanding-window folds ending 2024-06-30, 2024-12-31, and 2025-06-30, each scoring the following six months.

One deliberate simplification for this phase: the champion is not refit through the latest data before serving.
Refit-after-selection arrives with the retraining loop in the promotion-gate phase, where it matters.
