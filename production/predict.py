"""Load the registered champion model and forecast the next day.

Demonstrates the registry contract end to end: resolve
``models:/compute-forecast-hybrid@champion``, build feature rows for the day
after the last observed actuals, and print calibrated P10/P50/P90 forecasts
for all 16 series.

    python production/predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd

PRODUCTION_DIR = Path(__file__).resolve().parent
ROOT = PRODUCTION_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PRODUCTION_DIR))

from tracking import CHAMPION_ALIAS, MODEL_NAME, configure_mlflow  # noqa: E402

from compute_forecasting.features import (  # noqa: E402
    FEATURE_COLUMNS,
    RecursiveFeatureBuilder,
)

CAT_FEATURES = ["compute_type", "customer_segment", "series_id"]


def build_next_day_features(
    usage: pd.DataFrame, holidays: pd.DataFrame
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Feature rows for the day after the last observed actuals, all series."""
    forecast_date = usage["date"].max() + pd.Timedelta(days=1)
    builder = RecursiveFeatureBuilder(usage, holidays)

    rows = [
        builder.build_row(forecast_date, ct, seg)
        for ct in sorted(usage["compute_type"].unique())
        for seg in sorted(usage["customer_segment"].unique())
    ]
    features = pd.DataFrame(rows)
    features["date"] = forecast_date

    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CAT_FEATURES]
    features[numeric_cols] = features[numeric_cols].astype("float64")
    return forecast_date, features[["date"] + FEATURE_COLUMNS]


def main() -> None:
    configure_mlflow(ROOT)

    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    print(f"Loading {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)

    usage = pd.read_csv(ROOT / "data" / "compute_usage.csv", parse_dates=["date"])
    holidays = pd.read_csv(ROOT / "data" / "holiday_calendar.csv", parse_dates=["date"])

    forecast_date, features = build_next_day_features(usage, holidays)
    forecast = model.predict(features)

    print(f"\nForecast for {forecast_date.date()} ({forecast_date.day_name()}):\n")
    print(
        forecast.sort_values(["compute_type", "customer_segment"]).to_string(
            index=False, float_format=lambda x: f"{x:,.0f}"
        )
    )
    print(f"\nTotal P50: {forecast['p50'].sum():,.0f} compute hours")
    print(f"Total P90: {forecast['p90'].sum():,.0f} compute hours")


if __name__ == "__main__":
    main()
