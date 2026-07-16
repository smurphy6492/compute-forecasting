"""Train the hybrid forecast model with MLflow tracking and registry.

Reproduces the training recipe from notebook 02 (and scripts/recursive_backtest.py)
as a single tracked pipeline:

1. Walk-forward backtest over 3 expanding-window folds, metrics logged per fold.
2. Final model: fit on data through TRAIN_END, early-stop and conformally
   calibrate on the validation window, evaluate on the held-out test window.
3. Log params, metrics, SHAP summary, and component artifacts to MLflow.
4. Package trends + boosters + conformal adjustments as a single pyfunc and
   register it as ``compute-forecast-hybrid`` with the ``@champion`` alias.

Run from anywhere:

    python production/train.py

then inspect with ``mlflow ui --backend-store-uri sqlite:///mlflow.db`` from
the repo root.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_percentage_error

PRODUCTION_DIR = Path(__file__).resolve().parent
ROOT = PRODUCTION_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PRODUCTION_DIR))

from pyfunc_model import HybridForecastModel  # noqa: E402
from tracking import CHAMPION_ALIAS, MODEL_NAME, configure_mlflow  # noqa: E402

from compute_forecasting.features import (  # noqa: E402
    FEATURE_COLUMNS,
    SeriesTrend,
    apply_conformal_calibration,
    build_features,
    compute_conformal_adjustments,
    compute_residual_ratios,
    fit_series_trends,
    predict_with_trend,
)

TARGET = "compute_hours"
CAT_FEATURES = ["compute_type", "customer_segment", "series_id"]
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
TARGET_COVERAGE = 0.80

TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-12-31")

# Expanding-window walk-forward folds. Each fold early-stops and calibrates on
# the last FOLD_VAL_DAYS of its training window, then scores the next 6 months.
BACKTEST_FOLDS = [
    ("fold_1", pd.Timestamp("2024-06-30"), pd.Timestamp("2024-12-31")),
    ("fold_2", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-06-30")),
    ("fold_3", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-12-31")),
]
FOLD_VAL_DAYS = 90

LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50
SHAP_SAMPLE_SIZE = 2000


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    usage = pd.read_csv(ROOT / "data" / "compute_usage.csv", parse_dates=["date"])
    holidays = pd.read_csv(ROOT / "data" / "holiday_calendar.csv", parse_dates=["date"])
    return usage, holidays


def train_quantile_models(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    trends: dict[tuple[str, ...], SeriesTrend],
) -> dict[str, lgb.LGBMRegressor]:
    """Train P10/P50/P90 boosters on log residual ratios with early stopping."""
    y_fit = np.log(compute_residual_ratios(fit_df, trends))
    y_val = np.log(compute_residual_ratios(val_df, trends))

    models = {}
    for label, alpha in QUANTILES.items():
        params = LGB_PARAMS | {"alpha": alpha}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fit_df[FEATURE_COLUMNS],
            y_fit,
            eval_set=[(val_df[FEATURE_COLUMNS], y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
            categorical_feature=CAT_FEATURES,
        )
        models[label] = model
    return models


def predict_quantiles(
    models: dict[str, lgb.LGBMRegressor],
    df: pd.DataFrame,
    trends: dict[tuple[str, ...], SeriesTrend],
) -> dict[str, np.ndarray]:
    """Real-scale quantile predictions: trend(t) * exp(log-residual prediction)."""
    return {
        label: np.maximum(predict_with_trend(df, model.predict(df[FEATURE_COLUMNS]), trends), 0.0)
        for label, model in models.items()
    }


def fit_and_evaluate(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict, dict[str, lgb.LGBMRegressor], dict, dict[str, float]]:
    """Full pipeline on one split: trends, boosters, conformal, test metrics."""
    trends = fit_series_trends(fit_df)
    models = train_quantile_models(fit_df, val_df, trends)

    val_preds = predict_quantiles(models, val_df, trends)
    adjustments = compute_conformal_adjustments(
        val_df,
        val_preds["p10"],
        val_preds["p50"],
        val_preds["p90"],
        val_df[TARGET].to_numpy(),
        target_coverage=TARGET_COVERAGE,
    )

    test_preds = predict_quantiles(models, test_df, trends)
    cal_p10, cal_p90 = apply_conformal_calibration(
        test_df,
        test_preds["p10"],
        test_preds["p50"],
        test_preds["p90"],
        adjustments,
    )

    y_test = test_df[TARGET].to_numpy()
    p50 = test_preds["p50"]
    metrics = {
        "mape_p50": mean_absolute_percentage_error(y_test, p50) * 100,
        "bias_pct": float(np.mean((p50 - y_test) / y_test) * 100),
        "coverage_p10_p90": float(np.mean((y_test >= cal_p10) & (y_test <= cal_p90)) * 100),
        "n_test_rows": len(test_df),
    }
    return metrics, models, trends, adjustments


def run_backtest(df: pd.DataFrame) -> list[dict]:
    """Walk-forward backtest; returns one metrics dict per fold."""
    results = []
    for name, train_end, test_end in BACKTEST_FOLDS:
        val_start = train_end - pd.Timedelta(days=FOLD_VAL_DAYS)
        fit_df = df[df["date"] <= val_start]
        val_df = df[(df["date"] > val_start) & (df["date"] <= train_end)]
        test_df = df[(df["date"] > train_end) & (df["date"] <= test_end)]

        metrics, _, _, _ = fit_and_evaluate(fit_df, val_df, test_df)
        metrics = {
            "fold": name,
            "train_end": str(train_end.date()),
            "test_end": str(test_end.date()),
            **metrics,
        }
        results.append(metrics)
        print(
            f"  {name}: MAPE {metrics['mape_p50']:.2f}%  "
            f"coverage {metrics['coverage_p10_p90']:.1f}%  "
            f"bias {metrics['bias_pct']:+.2f}%"
        )
    return results


def save_shap_summary(model: lgb.LGBMRegressor, X_fit: pd.DataFrame, out_path: Path) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_fit), size=min(SHAP_SAMPLE_SIZE, len(X_fit)), replace=False)
    sample = X_fit.iloc[idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.title("SHAP summary — P50 model (log residual ratio scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_component_artifacts(
    out_dir: Path,
    models: dict[str, lgb.LGBMRegressor],
    trends: dict[tuple[str, ...], SeriesTrend],
    adjustments: dict[str, float],
) -> dict[str, str]:
    """Serialize model components to files; returns pyfunc artifact map."""
    artifacts = {}

    for label, model in models.items():
        path = out_dir / f"booster_{label}.txt"
        # Saving at best_iteration_ bakes early stopping into the file, so the
        # loaded Booster predicts with the selected tree count by default.
        model.booster_.save_model(str(path), num_iteration=model.best_iteration_)
        artifacts[f"booster_{label}"] = str(path)

    origin = next(iter(trends.values())).origin_date
    trends_payload = {
        "origin_date": str(origin.date()),
        "series": [
            {
                "compute_type": key[0],
                "customer_segment": key[1],
                "log_intercept": t.log_intercept,
                "daily_growth_rate": t.daily_growth_rate,
            }
            for key, t in trends.items()
        ],
    }
    trends_path = out_dir / "trends.json"
    trends_path.write_text(json.dumps(trends_payload, indent=2), encoding="utf-8")
    artifacts["trends"] = str(trends_path)

    conformal_path = out_dir / "conformal_adjustments.json"
    conformal_path.write_text(
        json.dumps({"target_coverage": TARGET_COVERAGE, "adjustments": adjustments}, indent=2),
        encoding="utf-8",
    )
    artifacts["conformal"] = str(conformal_path)

    features_path = out_dir / "feature_columns.json"
    features_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    artifacts["feature_columns"] = str(features_path)

    return artifacts


def make_input_example(test_df: pd.DataFrame) -> pd.DataFrame:
    """Signature example: id columns + features, categoricals as plain strings."""
    example = test_df[["date"] + FEATURE_COLUMNS].head(5).copy()
    for col in CAT_FEATURES:
        example[col] = example[col].astype(str)
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CAT_FEATURES]
    example[numeric_cols] = example[numeric_cols].astype("float64")
    return example


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    print("Loading data and building features...")
    usage, holidays = load_data()
    df = build_features(usage, holidays)
    test_end = df["date"].max()

    configure_mlflow(ROOT)

    with mlflow.start_run(run_name="train-hybrid") as run:
        mlflow.log_params(LGB_PARAMS)
        mlflow.log_params(
            {
                "quantiles": list(QUANTILES.values()),
                "target_coverage": TARGET_COVERAGE,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "train_end": str(TRAIN_END.date()),
                "val_end": str(VAL_END.date()),
                "test_end": str(test_end.date()),
                "fold_val_days": FOLD_VAL_DAYS,
                "n_features": len(FEATURE_COLUMNS),
                "n_rows": len(df),
                "n_series": df["series_id"].nunique(),
            }
        )

        print("\nWalk-forward backtest:")
        fold_results = run_backtest(df)
        for fold in fold_results:
            mlflow.log_metric(f"backtest_mape_{fold['fold']}", fold["mape_p50"])
            mlflow.log_metric(f"backtest_coverage_{fold['fold']}", fold["coverage_p10_p90"])
        fold_mapes = [f["mape_p50"] for f in fold_results]
        mlflow.log_metric("backtest_mape_mean", float(np.mean(fold_mapes)))
        mlflow.log_metric("backtest_mape_std", float(np.std(fold_mapes)))

        print(
            "\nTraining final model (train through "
            f"{TRAIN_END.date()}, validate through {VAL_END.date()})..."
        )
        fit_df = df[df["date"] <= TRAIN_END]
        val_df = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
        test_df = df[df["date"] > VAL_END]

        metrics, models, trends, adjustments = fit_and_evaluate(fit_df, val_df, test_df)
        mlflow.log_metric("test_mape_p50", metrics["mape_p50"])
        mlflow.log_metric("test_bias_pct", metrics["bias_pct"])
        mlflow.log_metric("test_coverage_p10_p90", metrics["coverage_p10_p90"])
        for label, model in models.items():
            mlflow.log_metric(f"best_iteration_{label}", model.best_iteration_)
        for ctype, adj in adjustments.items():
            mlflow.log_metric(f"conformal_adj_{slug(ctype)}", adj)

        y_test = test_df[TARGET].to_numpy()
        p50 = predict_quantiles(models, test_df, trends)["p50"]
        for ctype in test_df["compute_type"].unique():
            mask = (test_df["compute_type"] == ctype).to_numpy()
            type_mape = mean_absolute_percentage_error(y_test[mask], p50[mask]) * 100
            mlflow.log_metric(f"test_mape_{slug(str(ctype))}", type_mape)

        print(
            f"  Test MAPE {metrics['mape_p50']:.2f}%  "
            f"coverage {metrics['coverage_p10_p90']:.1f}%  "
            f"bias {metrics['bias_pct']:+.2f}%"
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            print("\nComputing SHAP summary...")
            shap_path = tmp_dir / "shap_summary_p50.png"
            save_shap_summary(models["p50"], fit_df[FEATURE_COLUMNS], shap_path)
            mlflow.log_artifact(str(shap_path), artifact_path="explainability")

            folds_path = tmp_dir / "backtest_folds.json"
            folds_path.write_text(json.dumps(fold_results, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(folds_path), artifact_path="evaluation")

            artifacts = write_component_artifacts(tmp_dir, models, trends, adjustments)
            mlflow.log_artifact(artifacts["trends"], artifact_path="components")
            mlflow.log_artifact(artifacts["conformal"], artifact_path="components")

            print("Logging and registering pyfunc model...")
            input_example = make_input_example(test_df)
            wrapper = HybridForecastModel()
            head = test_df.head(5)
            head_preds = predict_quantiles(models, head, trends)
            cal_p10, cal_p90 = apply_conformal_calibration(
                head, head_preds["p10"], head_preds["p50"], head_preds["p90"], adjustments
            )
            output_example = pd.DataFrame(
                {
                    "date": head["date"].values,
                    "compute_type": head["compute_type"].astype(str).values,
                    "customer_segment": head["customer_segment"].astype(str).values,
                    "p10": cal_p10,
                    "p50": head_preds["p50"],
                    "p90": cal_p90,
                }
            )
            signature = infer_signature(input_example, output_example)

            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper,
                artifacts=artifacts,
                code_paths=[
                    str(PRODUCTION_DIR / "pyfunc_model.py"),
                    str(ROOT / "src" / "compute_forecasting"),
                ],
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
            )

        version = model_info.registered_model_version
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

        print("\n" + "=" * 60)
        print(f"Run ID:            {run.info.run_id}")
        print(f"Registered model:  {MODEL_NAME} v{version} (@{CHAMPION_ALIAS})")
        print(f"Test MAPE (P50):   {metrics['mape_p50']:.2f}%")
        print(f"Test coverage:     {metrics['coverage_p10_p90']:.1f}% (target 80%)")
        print("Inspect with:      mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print("=" * 60)


if __name__ == "__main__":
    main()
