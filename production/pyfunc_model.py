"""MLflow pyfunc wrapper for the hybrid trend + residual forecast model.

The hybrid model is three components, none of which is a single sklearn
estimator MLflow can log natively:

- three LightGBM boosters (P10/P50/P90) predicting log residual ratios
- per-series exponential trend parameters for real-scale reconstruction
- per-compute-type conformal adjustments for interval calibration

This wrapper bundles them into one pyfunc so a caller can load
``models:/compute-forecast-hybrid@champion`` and get calibrated quantile
forecasts without knowing the decomposition exists.

Input contract: a DataFrame of feature rows as produced by
``compute_forecasting.features.build_features`` or
``RecursiveFeatureBuilder.build_row``, plus a ``date`` column. Numeric
features should be float64; categoricals may be plain strings.

Output: DataFrame with [date, compute_type, customer_segment, p10, p50, p90]
on the real compute-hours scale, conformally calibrated.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd

QUANTILE_LABELS = ("p10", "p50", "p90")
ID_COLUMNS = ["date", "compute_type", "customer_segment"]
CAT_COLUMNS = ["compute_type", "customer_segment", "series_id"]


class HybridForecastModel(mlflow.pyfunc.PythonModel):
    """Trend * exp(LightGBM log-residual) with conformal interval calibration."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # Deferred import: resolved from the model's bundled code_paths copy
        # of the package, not the training repo.
        from compute_forecasting.features import SeriesTrend

        self._boosters = {
            label: lgb.Booster(model_file=context.artifacts[f"booster_{label}"])
            for label in QUANTILE_LABELS
        }

        with open(context.artifacts["trends"], encoding="utf-8") as f:
            raw = json.load(f)
        origin_date = pd.Timestamp(raw["origin_date"])
        self._trends = {}
        for entry in raw["series"]:
            key = (entry["compute_type"], entry["customer_segment"])
            self._trends[key] = SeriesTrend(
                series_key=key,
                log_intercept=entry["log_intercept"],
                daily_growth_rate=entry["daily_growth_rate"],
                origin_date=origin_date,
            )

        with open(context.artifacts["conformal"], encoding="utf-8") as f:
            self._conformal = json.load(f)["adjustments"]

        with open(context.artifacts["feature_columns"], encoding="utf-8") as f:
            self._feature_columns = json.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:
        df = model_input.copy().reset_index(drop=True)

        missing = [c for c in ID_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        if "series_id" not in df.columns:
            df["series_id"] = (
                df["compute_type"].astype(str) + " | " + df["customer_segment"].astype(str)
            )
        # The saved boosters carry LightGBM's pandas_categorical mapping, so
        # string -> category casting here reproduces the training encoding.
        for col in CAT_COLUMNS:
            df[col] = df[col].astype("category")

        missing_features = [c for c in self._feature_columns if c not in df.columns]
        if missing_features:
            raise ValueError(f"Input is missing feature columns: {missing_features}")

        X = df[self._feature_columns]
        trend_values = self._trend_values(df)

        raw = {
            label: np.maximum(trend_values * np.exp(booster.predict(X)), 0.0)
            for label, booster in self._boosters.items()
        }

        adjustment = (
            df["compute_type"].astype(str).map(lambda ct: self._conformal.get(ct, 0.0)).to_numpy()
        )
        p50 = raw["p50"]
        p10 = np.maximum(raw["p10"] - adjustment * p50, 0.0)
        p90 = raw["p90"] + adjustment * p50
        # Guard against quantile crossing after calibration
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        return pd.DataFrame(
            {
                "date": df["date"].values,
                "compute_type": df["compute_type"].astype(str).values,
                "customer_segment": df["customer_segment"].astype(str).values,
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )

    def _trend_values(self, df: pd.DataFrame) -> np.ndarray:
        values = np.zeros(len(df))
        for key, grp in df.groupby(["compute_type", "customer_segment"], observed=True):
            key = tuple(str(k) for k in key)
            trend = self._trends.get(key)
            if trend is None:
                raise ValueError(f"No fitted trend for series {key}")
            positions = df.index.get_indexer(grp.index)
            values[positions] = trend.predict(grp["date"])
        return values
