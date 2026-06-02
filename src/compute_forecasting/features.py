"""Feature engineering for compute capacity forecasting.

Builds calendar, holiday, lag, rolling, and trend features from raw usage data.
Designed for a global LightGBM model across all (compute_type, customer_segment) series.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features from a date column.

    Features: day_of_week, month, day_of_month, week_of_year, is_weekend,
    quarter, day_of_quarter, is_quarter_end (last 10 days of quarter).
    """
    df = df.copy()
    dt = df[date_col]

    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = dt.dt.quarter
    df["day_of_year"] = dt.dt.dayofyear

    # Day within the quarter (1-based)
    quarter_start = dt.dt.to_period("Q").dt.start_time
    df["day_of_quarter"] = (dt - quarter_start).dt.days + 1

    # Is this in the last 10 days of a quarter-end month (Mar, Jun, Sep, Dec)?
    quarter_end = dt.dt.to_period("Q").dt.end_time.dt.date
    days_to_quarter_end = (pd.to_datetime(quarter_end) - dt).dt.days
    df["is_quarter_end"] = (days_to_quarter_end <= 10).astype(int)

    return df


def add_holiday_features(
    df: pd.DataFrame,
    holidays: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add holiday-related features.

    Features: is_holiday, days_to_next_holiday, days_from_last_holiday.
    """
    df = df.copy()
    holiday_dates = sorted(holidays["date"].dt.date.unique())

    df["is_holiday"] = df[date_col].dt.date.isin(holiday_dates).astype(int)

    # Pre-compute holiday proximity for all unique dates
    unique_dates = sorted(df[date_col].dt.date.unique())
    holiday_arr = np.array([pd.Timestamp(d) for d in holiday_dates])

    days_to_next = {}
    days_from_last = {}
    for d in unique_dates:
        ts = pd.Timestamp(d)
        future = holiday_arr[holiday_arr >= ts]
        past = holiday_arr[holiday_arr <= ts]
        days_to_next[d] = (future[0] - ts).days if len(future) > 0 else 30
        days_from_last[d] = (ts - past[-1]).days if len(past) > 0 else 30

    df["days_to_next_holiday"] = df[date_col].dt.date.map(days_to_next)
    df["days_from_last_holiday"] = df[date_col].dt.date.map(days_from_last)

    # Cap at 30 — beyond that, holidays have negligible effect
    df["days_to_next_holiday"] = df["days_to_next_holiday"].clip(upper=30)
    df["days_from_last_holiday"] = df["days_from_last_holiday"].clip(upper=30)

    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "compute_hours",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target variable per series.

    Default lags: 1, 7, 14, 28, 365 days.
    """
    if lags is None:
        lags = [1, 7, 14, 28, 365]

    df = df.sort_values(group_cols + ["date"]).copy()

    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "compute_hours",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std of the target variable per series.

    Default windows: 7 and 28 days. Uses shift(1) to avoid data leakage.
    """
    if windows is None:
        windows = [7, 28]

    df = df.sort_values(group_cols + ["date"]).copy()

    for w in windows:
        shifted = df.groupby(group_cols)[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.groupby(df[group_cols].apply(tuple, axis=1)).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = shifted.groupby(df[group_cols].apply(tuple, axis=1)).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )

    return df


def add_trend_feature(
    df: pd.DataFrame,
    date_col: str = "date",
    origin: str | None = None,
) -> pd.DataFrame:
    """Add a linear trend proxy (days since the start of the dataset)."""
    df = df.copy()
    if origin is None:
        origin_date = df[date_col].min()
    else:
        origin_date = pd.Timestamp(origin)
    df["days_since_start"] = (df[date_col] - origin_date).dt.days
    return df


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert categorical columns to pandas Categorical type for LightGBM."""
    if cat_cols is None:
        cat_cols = ["compute_type", "customer_segment"]

    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def build_features(
    usage: pd.DataFrame,
    holidays: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "compute_hours",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Applies all feature groups in order: calendar, holiday, lag, rolling,
    trend, and categorical encoding.

    Parameters
    ----------
    usage : DataFrame with columns [date, compute_type, customer_segment, compute_hours]
    holidays : DataFrame with a 'date' column of holiday dates
    group_cols : columns that define a series (default: compute_type + customer_segment)
    target_col : name of the target column
    lags : lag periods (default: [1, 7, 14, 28, 365])
    rolling_windows : rolling window sizes (default: [7, 28])

    Returns
    -------
    DataFrame with all features added, sorted by group_cols + date.
    """
    if group_cols is None:
        group_cols = ["compute_type", "customer_segment"]

    df = usage.copy()
    df = add_calendar_features(df)
    df = add_holiday_features(df, holidays)
    df = add_lag_features(df, group_cols, target_col, lags)
    df = add_rolling_features(df, group_cols, target_col, rolling_windows)
    df = add_trend_feature(df)

    # Series interaction: compute_type × customer_segment as a single categorical
    # Lets the model split directly on specific series (e.g., "Enterprise GPU Training")
    # without requiring multiple tree splits through separate categoricals
    df["series_id"] = df["compute_type"].astype(str) + " | " + df["customer_segment"].astype(str)

    df = encode_categoricals(df, cat_cols=["compute_type", "customer_segment", "series_id"])

    return df.sort_values(group_cols + ["date"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return the list of feature columns (everything except date and target)."""
    if exclude is None:
        exclude = ["date", "compute_hours"]
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Hybrid Trend + Residual Model Support
# ---------------------------------------------------------------------------

MAX_MONTHLY_GROWTH_RATE = 0.05  # Cap: 5% per month (~80% annualized)
RESIDUAL_CLIP_RANGE = (0.1, 10.0)  # Clip residual ratios to prevent extremes


@dataclass
class SeriesTrend:
    """Fitted exponential trend parameters for a single series."""

    series_key: tuple[str, str]
    log_intercept: float  # log(a) in log(y) = log(a) + b*t
    daily_growth_rate: float  # b (already capped)
    origin_date: pd.Timestamp  # t=0 reference date

    def predict(self, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """Extrapolate the trend to arbitrary dates."""
        delta = pd.to_datetime(dates) - self.origin_date
        # .dt.days for Series, .days for DatetimeIndex/TimedeltaIndex
        t = delta.dt.days.values if hasattr(delta, "dt") else delta.days.values
        return np.exp(self.log_intercept + self.daily_growth_rate * t)


def fit_series_trends(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "compute_hours",
    date_col: str = "date",
) -> dict[tuple[str, ...], SeriesTrend]:
    """Fit per-series exponential trends via log-linear regression.

    For each series: log(y) = log(a) + b*t, where t = days since origin.
    Growth rate is capped at MAX_MONTHLY_GROWTH_RATE / 30 per day.

    Parameters
    ----------
    df : Training data (only pass training rows to avoid data leakage)
    group_cols : Columns defining each series
    target_col : Target column name
    date_col : Date column name

    Returns
    -------
    Dict mapping series key tuple to fitted SeriesTrend.
    """
    if group_cols is None:
        group_cols = ["compute_type", "customer_segment"]

    max_daily_rate = np.log(1 + MAX_MONTHLY_GROWTH_RATE) / 30
    origin_date = df[date_col].min()
    trends: dict[tuple[str, ...], SeriesTrend] = {}

    for key, grp in df.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)

        t = (grp[date_col] - origin_date).dt.days.values.astype(float)
        y = grp[target_col].values.astype(float)

        # Filter out zeros/negatives for log transform
        valid = y > 0
        t_valid, log_y = t[valid], np.log(y[valid])

        # Fit: log(y) = c0 + c1*t (polyfit degree 1, coefficients in ascending order)
        c0, c1 = polyfit(t_valid, log_y, deg=1)

        # Cap growth rate
        c1_capped = min(c1, max_daily_rate)

        trends[key] = SeriesTrend(
            series_key=key,
            log_intercept=c0,
            daily_growth_rate=c1_capped,
            origin_date=origin_date,
        )

    return trends


def compute_residual_ratios(
    df: pd.DataFrame,
    trends: dict[tuple[str, ...], SeriesTrend],
    group_cols: list[str] | None = None,
    target_col: str = "compute_hours",
    date_col: str = "date",
) -> pd.Series:
    """Compute residual ratio = actual / trend(t) for each row.

    Result is clipped to RESIDUAL_CLIP_RANGE to prevent extreme values.
    """
    if group_cols is None:
        group_cols = ["compute_type", "customer_segment"]

    ratios = pd.Series(np.nan, index=df.index, dtype=float)

    for key, grp in df.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        trend = trends[key]
        trend_values = trend.predict(grp[date_col])
        raw_ratio = grp[target_col].values / np.maximum(trend_values, 1.0)
        clipped = np.clip(raw_ratio, *RESIDUAL_CLIP_RANGE)
        ratios.loc[grp.index] = clipped

    return ratios


def predict_with_trend(
    df: pd.DataFrame,
    lgb_log_residual_preds: np.ndarray,
    trends: dict[tuple[str, ...], SeriesTrend],
    group_cols: list[str] | None = None,
    date_col: str = "date",
) -> np.ndarray:
    """Reconstruct real-scale predictions: trend(t) * exp(lgb_prediction).

    Parameters
    ----------
    df : DataFrame with date and group columns (for trend lookup)
    lgb_log_residual_preds : LightGBM predictions of log(residual_ratio)
    trends : Fitted trend dict from fit_series_trends
    group_cols : Series grouping columns
    date_col : Date column name

    Returns
    -------
    Real-scale predictions array aligned with df index order.
    """
    if group_cols is None:
        group_cols = ["compute_type", "customer_segment"]

    result = np.zeros(len(df))

    for key, grp in df.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        trend = trends[key]
        trend_values = trend.predict(grp[date_col])
        idx = grp.index
        # Map DataFrame positions to array positions
        positions = [df.index.get_loc(i) for i in idx]
        residual_preds = np.exp(lgb_log_residual_preds[positions])
        result[positions] = trend_values * residual_preds

    return result
