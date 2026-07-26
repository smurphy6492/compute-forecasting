"""Tests for feature engineering functions.

Integration-style tests using actual project data for realistic validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compute_forecasting.features import (
    FEATURE_COLUMNS,
    RecursiveFeatureBuilder,
    add_calendar_features,
    add_lag_features,
    apply_conformal_calibration,
    build_features,
    compute_conformal_adjustments,
    compute_residual_ratios,
    fit_series_trends,
    get_feature_columns,
    predict_with_trend,
)

# Test data paths
DATA_DIR = Path(__file__).parent.parent / "data"
USAGE_PATH = DATA_DIR / "compute_usage.csv"
HOLIDAYS_PATH = DATA_DIR / "holiday_calendar.csv"


@pytest.fixture
def usage_data() -> pd.DataFrame:
    """Load actual compute usage data."""
    return pd.read_csv(USAGE_PATH, parse_dates=["date"])


@pytest.fixture
def holidays_data() -> pd.DataFrame:
    """Load actual holiday calendar."""
    return pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Minimal sample data for unit tests."""
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "compute_type": ["GPU Training"] * 400,
            "customer_segment": ["Enterprise"] * 400,
            "compute_hours": np.random.default_rng(42).uniform(1000, 5000, 400),
        }
    )


def test_build_features_output_shape(usage_data, holidays_data):
    """Verify build_features produces expected number of columns."""
    df = build_features(usage_data, holidays_data)

    # Expected columns:
    # - Original: date, compute_type, customer_segment, compute_hours (4)
    # - Calendar: day_of_week, month, day_of_month, week_of_year, is_weekend,
    #             quarter, day_of_year, day_of_quarter, is_quarter_end (9)
    # - Holiday: is_holiday, days_to_next_holiday, days_from_last_holiday (3)
    # - Lags (default): lag_1, lag_7, lag_14, lag_28, lag_365 (5)
    # - Rolling (default): rolling_mean_7, rolling_std_7, rolling_mean_28, rolling_std_28 (4)
    # - Trend: days_since_start (1)
    # - Series: series_id (1)
    # Total: 4 + 9 + 3 + 5 + 4 + 1 + 1 = 27
    expected_cols = 27
    assert len(df.columns) == expected_cols, (
        f"Expected {expected_cols} columns, got {len(df.columns)}. Columns: {sorted(df.columns)}"
    )

    # Core columns must be present
    required = {"date", "compute_type", "customer_segment", "compute_hours"}
    assert required.issubset(df.columns)


def test_build_features_no_unexpected_nans(usage_data, holidays_data):
    """Verify NaN patterns are as expected (lag_365 = first year, lag_1 = first day)."""
    df = build_features(usage_data, holidays_data)

    # Check lag_1: should have NaN only for first day of each series
    group_cols = ["compute_type", "customer_segment"]
    first_rows = df.groupby(group_cols).head(1).index
    assert df.loc[first_rows, "lag_1"].isna().all(), "lag_1 should be NaN for first row per series"
    assert df.loc[~df.index.isin(first_rows), "lag_1"].notna().all(), (
        "lag_1 should be non-NaN for all other rows"
    )

    # Check lag_365: should have NaN for first 365 days of each series
    first_year_rows = df.groupby(group_cols).head(365).index
    assert df.loc[first_year_rows, "lag_365"].isna().all(), (
        "lag_365 should be NaN for first 365 days per series"
    )

    # No unexpected NaNs in calendar features
    calendar_features = [
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "is_weekend",
        "quarter",
        "day_of_year",
        "day_of_quarter",
        "is_quarter_end",
    ]
    for feat in calendar_features:
        assert df[feat].notna().all(), f"Calendar feature {feat} should have no NaNs"


def test_add_calendar_features_day_of_week():
    """Verify day_of_week is correct for known dates."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-02",  # Monday
                    "2023-01-07",  # Saturday
                    "2023-01-08",  # Sunday
                ]
            ),
        }
    )
    result = add_calendar_features(df)

    assert result["day_of_week"].tolist() == [0, 5, 6], "day_of_week should be 0=Mon, 5=Sat, 6=Sun"
    assert result["is_weekend"].tolist() == [
        0,
        1,
        1,
    ], "is_weekend should be 1 for Sat/Sun, 0 otherwise"


def test_add_lag_features_correctness(sample_data):
    """Spot check that lag_7 produces correct value."""
    df = sample_data.copy()
    df = df.sort_values("date").reset_index(drop=True)

    result = add_lag_features(
        df,
        group_cols=["compute_type", "customer_segment"],
        target_col="compute_hours",
        lags=[7],
    )

    # Check lag_7 at row 7: should equal row 0
    assert result.loc[7, "lag_7"] == result.loc[0, "compute_hours"], (
        "lag_7 at row 7 should equal compute_hours at row 0"
    )

    # Check lag_7 at row 0: should be NaN
    assert pd.isna(result.loc[0, "lag_7"]), "lag_7 at row 0 should be NaN"


def test_fit_series_trends_returns_one_per_series(usage_data):
    """Verify fit_series_trends returns exactly one trend per series."""
    trends = fit_series_trends(usage_data)

    # Count unique series in data
    n_series = usage_data.groupby(["compute_type", "customer_segment"]).ngroups
    assert len(trends) == n_series, f"Expected {n_series} trends, got {len(trends)}"

    # Check structure
    for key, trend in trends.items():
        assert isinstance(key, tuple), "Trend keys should be tuples"
        assert len(key) == 2, "Trend keys should have 2 elements (compute_type, customer_segment)"
        assert hasattr(trend, "predict"), "Trend should have predict method"
        assert hasattr(trend, "daily_growth_rate"), "Trend should have daily_growth_rate"


def test_compute_residual_ratios_centered_around_one(sample_data):
    """Verify residual ratios are centered around 1.0."""
    trends = fit_series_trends(sample_data)
    ratios = compute_residual_ratios(sample_data, trends)

    # Ratios should be clipped to (0.1, 10.0) range
    assert ratios.min() >= 0.1, "Residual ratios should be >= 0.1"
    assert ratios.max() <= 10.0, "Residual ratios should be <= 10.0"

    # Mean should be close to 1.0 (since trend is fit to this data)
    mean_ratio = ratios.mean()
    assert 0.8 <= mean_ratio <= 1.2, f"Mean residual ratio should be ~1.0, got {mean_ratio:.3f}"


def test_predict_with_trend_round_trip(sample_data):
    """Verify predict_with_trend round-trips correctly when residual is 0 in log space."""
    trends = fit_series_trends(sample_data)

    # Create zero log-residual predictions (exp(0) = 1.0)
    zero_log_residuals = np.zeros(len(sample_data))

    predictions = predict_with_trend(
        sample_data,
        zero_log_residuals,
        trends,
        group_cols=["compute_type", "customer_segment"],
    )

    # Predictions should equal trend values (since residual factor is 1.0)
    for key, grp in sample_data.groupby(["compute_type", "customer_segment"]):
        trend = trends[(key,) if not isinstance(key, tuple) else key]
        expected = trend.predict(grp["date"])
        actual = predictions[grp.index]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-10,
            err_msg="predict_with_trend should equal trend when log-residual is 0",
        )


def test_feature_columns_alignment(usage_data, holidays_data):
    """Verify build_features output matches FEATURE_COLUMNS constant.

    If this test fails, FEATURE_COLUMNS and the build_row_features()
    function in notebooks/03_scenarios.ipynb need to be updated to match.
    """
    df = build_features(usage_data, holidays_data)
    actual_features = set(get_feature_columns(df))
    expected_features = set(FEATURE_COLUMNS)

    missing = expected_features - actual_features
    extra = actual_features - expected_features

    assert not missing, f"FEATURE_COLUMNS lists columns not produced by build_features: {missing}"
    assert not extra, (
        f"build_features produces columns not in FEATURE_COLUMNS: {extra}. "
        f"Update FEATURE_COLUMNS in features.py and build_row_features() in 03_scenarios.ipynb."
    )


def test_predict_with_trend_vectorization(sample_data):
    """Verify predict_with_trend uses vectorized indexing (performance regression test)."""
    trends = fit_series_trends(sample_data)
    log_residuals = np.random.randn(len(sample_data))

    # This should complete quickly even for large datasets
    # If O(n²) behavior returns, this will timeout
    predictions = predict_with_trend(
        sample_data,
        log_residuals,
        trends,
        group_cols=["compute_type", "customer_segment"],
    )

    assert len(predictions) == len(sample_data)
    assert not np.isnan(predictions).any(), "Predictions should not contain NaN"


# ---------------------------------------------------------------------------
# Conformal Calibration Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def conformal_data():
    """Create synthetic validation data for conformal calibration tests."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "compute_type": (["GPU Training"] * 100) + (["CPU Batch"] * 100),
        }
    )
    y_val = rng.uniform(500, 5000, n)
    # P50 close to actuals, P10/P90 as symmetric bands
    p50 = y_val + rng.normal(0, 100, n)
    p10 = p50 - rng.uniform(200, 400, n)
    p90 = p50 + rng.uniform(200, 400, n)
    return df, p10, p50, p90, y_val


def test_conformal_adjustments_are_finite(conformal_data):
    """Conformal adjustments should be finite real numbers."""
    df, p10, p50, p90, y_val = conformal_data
    adjustments = compute_conformal_adjustments(df, p10, p50, p90, y_val)

    for ctype, adj in adjustments.items():
        assert np.isfinite(adj), f"Adjustment for {ctype} should be finite, got {adj}"


def test_conformal_adjustments_per_type(conformal_data):
    """Should return one adjustment per compute type."""
    df, p10, p50, p90, y_val = conformal_data
    adjustments = compute_conformal_adjustments(df, p10, p50, p90, y_val)

    expected_types = set(df["compute_type"].unique())
    assert set(adjustments.keys()) == expected_types, (
        f"Expected adjustments for {expected_types}, got {set(adjustments.keys())}"
    )


def test_calibrated_intervals_change_symmetrically(conformal_data):
    """Calibrated intervals should widen/narrow symmetrically around the raw bands."""
    df, p10, p50, p90, y_val = conformal_data
    adjustments = compute_conformal_adjustments(df, p10, p50, p90, y_val)
    cal_p10, cal_p90 = apply_conformal_calibration(df, p10, p50, p90, adjustments)

    # For each type, the P10 shift and P90 shift should be equal magnitude
    for ctype in df["compute_type"].unique():
        mask = (df["compute_type"] == ctype).values
        p10_shift = p10[mask] - cal_p10[mask]  # positive = P10 moved down (wider)
        p90_shift = cal_p90[mask] - p90[mask]  # positive = P90 moved up (wider)
        np.testing.assert_allclose(
            p10_shift, p90_shift, rtol=1e-10, err_msg=f"Adjustments for {ctype} should be symmetric"
        )


def test_conformal_coverage_meets_target():
    """On a well-behaved dataset, calibrated coverage should meet the target."""
    rng = np.random.default_rng(123)
    n = 500
    df = pd.DataFrame({"compute_type": ["GPU Training"] * n})

    # Generate data where actuals follow predictions closely
    y_val = rng.uniform(1000, 10000, n)
    noise = rng.normal(0, 200, n)
    p50 = y_val + noise
    p10 = p50 - 500
    p90 = p50 + 500

    target = 0.80
    adjustments = compute_conformal_adjustments(df, p10, p50, p90, y_val, target_coverage=target)
    cal_p10, cal_p90 = apply_conformal_calibration(df, p10, p50, p90, adjustments)

    # Check coverage on the same validation set (should be >= target)
    covered = ((y_val >= cal_p10) & (y_val <= cal_p90)).mean()
    assert covered >= target - 0.02, (
        f"Calibrated coverage {covered:.1%} should be near target {target:.0%}"
    )


def test_conformal_with_single_observation():
    """Conformal calibration should handle a type with very few observations."""
    df = pd.DataFrame({"compute_type": ["GPU Training"] * 3})
    y_val = np.array([100.0, 200.0, 300.0])
    p10 = np.array([80.0, 180.0, 280.0])
    p50 = np.array([100.0, 200.0, 300.0])
    p90 = np.array([120.0, 220.0, 320.0])

    # Should not crash with very small sample
    adjustments = compute_conformal_adjustments(df, p10, p50, p90, y_val)
    assert "GPU Training" in adjustments
    assert np.isfinite(adjustments["GPU Training"])


# ---------------------------------------------------------------------------
# RecursiveFeatureBuilder Tests
# ---------------------------------------------------------------------------


def test_recursive_builder_feature_alignment(usage_data, holidays_data):
    """Verify RecursiveFeatureBuilder.build_row produces the same features as build_features."""
    builder = RecursiveFeatureBuilder(usage_data, holidays_data)

    # Pick a date in the middle of the dataset
    test_date = pd.Timestamp("2024-06-15")
    ct, seg = "GPU Training", "Enterprise"

    row_features = builder.build_row(test_date, ct, seg)

    # Compare to build_features output for the same row
    df = build_features(usage_data, holidays_data)
    batch_row = df[
        (df["date"] == test_date) & (df["compute_type"] == ct) & (df["customer_segment"] == seg)
    ].iloc[0]

    # Check all FEATURE_COLUMNS match
    mismatches = []
    for col in FEATURE_COLUMNS:
        if col in ("compute_type", "customer_segment", "series_id"):
            # Categorical columns: compare string values
            if str(row_features[col]) != str(batch_row[col]):
                mismatches.append(f"{col}: {row_features[col]} vs {batch_row[col]}")
        else:
            rv = row_features[col]
            bv = batch_row[col]
            if pd.isna(rv) and pd.isna(bv):
                continue
            if not np.isclose(rv, bv, rtol=1e-6):
                mismatches.append(f"{col}: {rv} vs {bv}")

    assert not mismatches, f"Feature mismatches: {mismatches}"
