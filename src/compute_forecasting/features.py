"""Feature engineering for compute capacity forecasting.

Builds calendar, holiday, lag, rolling, and trend features from raw usage data.
Designed for a global LightGBM model across all (compute_type, customer_segment) series.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


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
    df = encode_categoricals(df)

    return df.sort_values(group_cols + ["date"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return the list of feature columns (everything except date and target)."""
    if exclude is None:
        exclude = ["date", "compute_hours"]
    return [c for c in df.columns if c not in exclude]
