"""
Synthetic data generator for compute capacity forecasting.

Generates 36 months of daily compute hours across 4 compute types and 4 customer
segments. Uses layered multiplicative signal composition to produce realistic data
with documented business events (step-changes, spikes, outages).

All randomness is seeded for reproducibility.

Usage:
    python generate_synthetic_data.py
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

START_DATE = date(2023, 7, 1)
END_DATE = date(2026, 6, 30)

COMPUTE_TYPES = ["GPU Training", "GPU Inference", "CPU Batch", "CPU Interactive"]
CUSTOMER_SEGMENTS = ["Enterprise", "Mid-Market", "Startup", "Research/Academic"]

# Base daily compute hours per (compute_type, segment).
# These represent starting levels on day 0. Units = compute-hours per day.
BASE_HOURS: dict[tuple[str, str], float] = {
    # GPU Training — large training jobs, Enterprise dominates
    ("GPU Training", "Enterprise"): 11_847,
    ("GPU Training", "Mid-Market"): 4_623,
    ("GPU Training", "Startup"): 1_891,
    ("GPU Training", "Research/Academic"): 3_174,
    # GPU Inference — serving models, fast-growing across all segments
    ("GPU Inference", "Enterprise"): 7_832,
    ("GPU Inference", "Mid-Market"): 3_417,
    ("GPU Inference", "Startup"): 2_156,
    ("GPU Inference", "Research/Academic"): 1_243,
    # CPU Batch — ETL, data processing, mature workload
    ("CPU Batch", "Enterprise"): 5_918,
    ("CPU Batch", "Mid-Market"): 2_973,
    ("CPU Batch", "Startup"): 814,
    ("CPU Batch", "Research/Academic"): 1_537,
    # CPU Interactive — notebooks, dev environments, tied to developer activity
    ("CPU Interactive", "Enterprise"): 3_946,
    ("CPU Interactive", "Mid-Market"): 2_087,
    ("CPU Interactive", "Startup"): 1_178,
    ("CPU Interactive", "Research/Academic"): 2_461,
}

# Monthly compound growth rates by compute_type
GROWTH_RATES_BY_TYPE: dict[str, float] = {
    "GPU Training": 0.025,      # 2.5% monthly — strong but maturing
    "GPU Inference": 0.040,     # 4.0% monthly — inference boom
    "CPU Batch": 0.010,         # 1.0% monthly — mature, stable
    "CPU Interactive": 0.015,   # 1.5% monthly — moderate
}

# Growth rate multipliers by segment (applied on top of type growth)
SEGMENT_GROWTH_MULTIPLIERS: dict[str, float] = {
    "Enterprise": 0.85,         # Large base, steady growth
    "Mid-Market": 1.10,         # Healthy growth
    "Startup": 1.50,            # Small base, scaling fast
    "Research/Academic": 0.30,  # Budget-constrained, near-flat
}

# Monthly growth variation — growth rates aren't perfectly constant
# Standard deviation of monthly growth noise (multiplicative on effective rate)
GROWTH_NOISE_STD = 0.35  # ±35% variation around base growth rate each month

# ---------------------------------------------------------------------------
# Weekly seasonality — day-of-week multipliers (Mon=0 ... Sun=6)
# ---------------------------------------------------------------------------
WEEKLY_PATTERNS: dict[str, list[float]] = {
    # GPU Training: flatter weekend — training runs continue overnight/weekends
    "GPU Training": [1.00, 1.05, 1.10, 1.10, 1.05, 0.80, 0.75],
    # GPU Inference: moderate weekend dip — some inference is automated
    "GPU Inference": [1.00, 1.05, 1.08, 1.08, 1.03, 0.72, 0.65],
    # CPU Batch: batch jobs often scheduled weekdays, some weekend runs
    "CPU Batch": [1.02, 1.05, 1.08, 1.08, 1.05, 0.70, 0.60],
    # CPU Interactive: steepest weekend drop — developers go home
    "CPU Interactive": [1.00, 1.08, 1.12, 1.15, 1.05, 0.58, 0.50],
}

# Segment-specific weekend modifiers — Startups work more weekends, Enterprise less
# Applied multiplicatively to the weekend days (Sat=5, Sun=6) only
SEGMENT_WEEKEND_MODIFIER: dict[str, float] = {
    "Enterprise": 0.90,         # Enterprise weekends are even quieter
    "Mid-Market": 1.00,         # Baseline
    "Startup": 1.15,            # Startups work more weekends
    "Research/Academic": 1.05,  # Grad students work some weekends
}

# ---------------------------------------------------------------------------
# Annual seasonality — month multipliers (Jan=1 ... Dec=12)
# ---------------------------------------------------------------------------
ANNUAL_PATTERN: dict[int, float] = {
    1: 0.88,    # January — slow start, recovery from holidays
    2: 0.95,    # February — ramping back
    3: 1.08,    # March — end of Q1 push
    4: 0.98,    # April — post-quarter normalization
    5: 1.00,    # May — steady
    6: 1.06,    # June — end of Q2 push
    7: 0.93,    # July — summer dip
    8: 0.91,    # August — peak summer slowdown
    9: 1.05,    # September — back from summer, Q3 push begins
    10: 1.03,   # October — steady
    11: 1.02,   # November — pre-holiday, slight dip at Thanksgiving
    12: 1.10,   # December — end of Q4 / year-end push (first half), holiday dip (second half)
}

# Additional end-of-quarter boost for last 10 days of quarter-end months
EOQ_MONTHS = {3, 6, 9, 12}
EOQ_BOOST = 1.08  # multiplicative on top of the monthly pattern

# December holiday suppression: Dec 20-31 gets pulled down
DEC_HOLIDAY_SUPPRESSION = 0.72

# ---------------------------------------------------------------------------
# US Federal Holidays (date -> name)
# ---------------------------------------------------------------------------


def _get_us_holidays(year: int) -> list[tuple[date, str]]:
    """Return US federal holidays for a given year."""
    holidays = []

    # New Year's Day
    holidays.append((date(year, 1, 1), "New Year's Day"))

    # MLK Day — 3rd Monday of January
    jan1 = date(year, 1, 1)
    first_monday = jan1 + timedelta(days=(7 - jan1.weekday()) % 7)
    if first_monday.month != 1:
        first_monday = date(year, 1, 1) + timedelta(days=(0 - jan1.weekday()) % 7)
    mlk = date(year, 1, 1)
    # Find first Monday in Jan
    d = date(year, 1, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    mlk = d + timedelta(weeks=2)  # 3rd Monday
    holidays.append((mlk, "MLK Day"))

    # Presidents' Day — 3rd Monday of February
    d = date(year, 2, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    presidents = d + timedelta(weeks=2)
    holidays.append((presidents, "Presidents' Day"))

    # Memorial Day — last Monday of May
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays.append((d, "Memorial Day"))

    # Independence Day
    holidays.append((date(year, 7, 4), "Independence Day"))

    # Labor Day — 1st Monday of September
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    holidays.append((d, "Labor Day"))

    # Columbus Day — 2nd Monday of October
    d = date(year, 10, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    columbus = d + timedelta(weeks=1)
    holidays.append((columbus, "Columbus Day"))

    # Veterans Day
    holidays.append((date(year, 11, 11), "Veterans Day"))

    # Thanksgiving — 4th Thursday of November
    d = date(year, 11, 1)
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    thanksgiving = d + timedelta(weeks=3)
    holidays.append((thanksgiving, "Thanksgiving"))
    holidays.append((thanksgiving + timedelta(days=1), "Day After Thanksgiving"))

    # Christmas
    holidays.append((date(year, 12, 25), "Christmas Day"))

    return holidays


def _build_holiday_set() -> dict[date, str]:
    """Build holiday lookup for all years in the data range."""
    holidays = {}
    for year in range(START_DATE.year, END_DATE.year + 2):
        for d, name in _get_us_holidays(year):
            holidays[d] = name
    return holidays


# Holiday impact varies by holiday severity AND segment
# Tier 1 (major): Christmas, Thanksgiving, Independence Day, New Year's
# Tier 2 (moderate): Memorial Day, Labor Day, Day After Thanksgiving
# Tier 3 (minor): MLK Day, Presidents' Day, Columbus Day, Veterans Day
MAJOR_HOLIDAYS = {"Christmas Day", "Thanksgiving", "Independence Day", "New Year's Day"}
MODERATE_HOLIDAYS = {"Memorial Day", "Labor Day", "Day After Thanksgiving"}
# Everything else is minor

HOLIDAY_IMPACT: dict[str, dict[str, float]] = {
    # (segment -> {tier -> reduction})
    "Enterprise": {"major": 0.55, "moderate": 0.35, "minor": 0.15},
    "Mid-Market": {"major": 0.50, "moderate": 0.30, "minor": 0.12},
    "Startup": {"major": 0.30, "moderate": 0.15, "minor": 0.05},
    "Research/Academic": {"major": 0.60, "moderate": 0.40, "minor": 0.20},
}

# ---------------------------------------------------------------------------
# Step-changes — permanent level shifts from new customers
# ---------------------------------------------------------------------------
# Step-changes ramp up over ramp_days instead of instant jumps
STEP_CHANGES = [
    {
        "date": date(2024, 2, 15),
        "name": "MegaCorp Enterprise Onboarding",
        "description": "Large enterprise customer 'MegaCorp' signs multi-year contract. "
                       "Onboards GPU Training and GPU Inference workloads.",
        "ramp_days": 14,  # 2-week onboarding ramp
        "affected": [
            ("GPU Training", "Enterprise", 0.15),   # +15% permanent
            ("GPU Inference", "Enterprise", 0.15),
        ],
    },
    {
        "date": date(2024, 12, 1),
        "name": "ScaleAI-Partner Mid-Market Signing",
        "description": "Mid-market AI company 'ScaleAI-Partner' begins heavy inference "
                       "workloads after pilot program converts.",
        "ramp_days": 10,
        "affected": [
            ("GPU Inference", "Mid-Market", 0.08),   # +8% permanent
        ],
    },
    {
        "date": date(2025, 6, 15),
        "name": "DataFlow Startup Churn",
        "description": "Startup customer 'DataFlow' migrates to competitor platform "
                       "after Series A pivot. Gradual wind-down.",
        "ramp_days": 21,  # 3-week wind-down
        "affected": [
            ("GPU Training", "Startup", -0.10),   # -10% permanent (churn)
            ("GPU Inference", "Startup", -0.06),   # -6%
        ],
    },
    {
        "date": date(2025, 12, 10),
        "name": "University Research Grant Cycle",
        "description": "New federal research grants bring 3 universities onto the platform "
                       "for large-scale training experiments.",
        "ramp_days": 14,
        "affected": [
            ("GPU Training", "Research/Academic", 0.12),  # +12% permanent
        ],
    },
]

# ---------------------------------------------------------------------------
# One-time spikes — events that cause temporary surges
# ---------------------------------------------------------------------------
SPIKE_EVENTS = [
    {
        "start_date": date(2024, 8, 12),
        "end_date": date(2024, 8, 14),
        "name": "ML Conference 2024 (NeurIPS Prep)",
        "description": "Major ML conference drives burst of training activity from "
                       "Startups and Research labs submitting last-minute experiments.",
        "affected": [
            ("GPU Training", "Startup", 1.8),          # 1.8x multiplier (80% spike)
            ("GPU Training", "Research/Academic", 2.0),  # 2x multiplier
        ],
    },
    {
        "start_date": date(2025, 8, 11),
        "end_date": date(2025, 8, 14),
        "name": "ML Conference 2025 (NeurIPS Prep)",
        "description": "Same annual conference, larger spike as the platform has more "
                       "Research and Startup customers.",
        "affected": [
            ("GPU Training", "Startup", 2.2),
            ("GPU Training", "Research/Academic", 2.5),
            ("GPU Inference", "Startup", 1.5),
        ],
    },
]

# ---------------------------------------------------------------------------
# Outage event
# ---------------------------------------------------------------------------
OUTAGE = {
    "start_date": date(2025, 4, 7),
    "end_date": date(2025, 4, 8),
    "recovery_end": date(2025, 4, 11),
    "name": "GPU Cluster Outage",
    "description": "Hardware failure in primary GPU cluster causes 2-day outage for "
                   "GPU workloads. 3-day recovery period at reduced capacity. "
                   "CPU workloads on separate infrastructure are unaffected.",
    "outage_factor": 0.10,    # 10% of normal during outage
    "recovery_factor": 0.60,  # 60% of normal during recovery
    "affected_types": {"GPU Training", "GPU Inference"},
}

# ---------------------------------------------------------------------------
# Noise parameters by segment
# ---------------------------------------------------------------------------
NOISE_STD: dict[str, float] = {
    "Enterprise": 0.04,         # Low noise — predictable SLA workloads
    "Mid-Market": 0.07,         # Moderate noise
    "Startup": 0.12,            # High noise — bursty, unpredictable
    "Research/Academic": 0.08,  # Moderate — correlated with academic calendar
}

# ---------------------------------------------------------------------------
# Upcoming events (sales pipeline) — for scenario planning
# ---------------------------------------------------------------------------
UPCOMING_EVENTS = [
    {
        "event_name": "TechGiant Corp GPU Training Contract",
        "expected_date": "2026-08-01",
        "compute_type": "GPU Training",
        "customer_segment": "Enterprise",
        "estimated_daily_hours": 2500,
        "probability": 0.70,
        "notes": "In final contract negotiations. CTO has signed off internally.",
    },
    {
        "event_name": "FinanceAI Inference Platform Migration",
        "expected_date": "2026-09-15",
        "compute_type": "GPU Inference",
        "customer_segment": "Enterprise",
        "estimated_daily_hours": 1800,
        "probability": 0.55,
        "notes": "Migrating from on-prem. Security review in progress.",
    },
    {
        "event_name": "AI Startup Cohort (YC S2026 Batch)",
        "expected_date": "2026-08-15",
        "compute_type": "GPU Training",
        "customer_segment": "Startup",
        "estimated_daily_hours": 800,
        "probability": 0.85,
        "notes": "Partnership with YC to provide compute credits. High confidence.",
    },
    {
        "event_name": "Federal Research Computing Grant Wave",
        "expected_date": "2026-10-01",
        "compute_type": "GPU Training",
        "customer_segment": "Research/Academic",
        "estimated_daily_hours": 1200,
        "probability": 0.40,
        "notes": "Grant applications submitted. Awards announced Sept 2026.",
    },
    {
        "event_name": "MedTech Mid-Market Inference Expansion",
        "expected_date": "2026-11-01",
        "compute_type": "GPU Inference",
        "customer_segment": "Mid-Market",
        "estimated_daily_hours": 600,
        "probability": 0.60,
        "notes": "Existing customer expanding from pilot to production inference.",
    },
]


# ===========================================================================
# Generator
# ===========================================================================


def _months_since_start(d: date) -> float:
    """Fractional months since START_DATE."""
    delta = d - START_DATE
    return delta.days / 30.44  # average days per month


def _build_growth_schedule() -> dict[tuple[str, str, int], float]:
    """Pre-compute per-series monthly growth rates with random variation."""
    schedule = {}
    for ct in COMPUTE_TYPES:
        for seg in CUSTOMER_SEGMENTS:
            base_rate = GROWTH_RATES_BY_TYPE[ct] * SEGMENT_GROWTH_MULTIPLIERS[seg]
            for month_idx in range(48):  # enough months to cover the range
                # Add random variation to monthly growth rate
                noise = RNG.normal(0, GROWTH_NOISE_STD * base_rate)
                monthly_rate = max(base_rate + noise, -0.02)  # floor at -2% to avoid collapse
                schedule[(ct, seg, month_idx)] = monthly_rate
    return schedule


GROWTH_SCHEDULE = _build_growth_schedule()


def _compute_trend(d: date, compute_type: str, segment: str) -> float:
    """Compound growth factor using pre-computed variable monthly rates."""
    months = _months_since_start(d)
    full_months = int(months)
    frac = months - full_months

    # Compound the variable monthly rates
    factor = 1.0
    for m in range(full_months):
        rate = GROWTH_SCHEDULE[(compute_type, segment, m)]
        factor *= (1 + rate)

    # Partial month
    if frac > 0 and full_months < 48:
        rate = GROWTH_SCHEDULE[(compute_type, segment, full_months)]
        factor *= (1 + rate * frac)

    return factor


def _weekly_factor(d: date, compute_type: str, segment: str) -> float:
    """Day-of-week multiplier with segment-specific weekend adjustment."""
    dow = d.weekday()
    factor = WEEKLY_PATTERNS[compute_type][dow]
    # Apply segment weekend modifier on Sat/Sun only
    if dow >= 5:
        modifier = SEGMENT_WEEKEND_MODIFIER[segment]
        # Blend toward modifier: weekend factor adjusted proportionally
        factor *= modifier
    return factor


def _annual_factor(d: date) -> float:
    """Month-level seasonality plus end-of-quarter and December holiday effects."""
    factor = ANNUAL_PATTERN[d.month]

    # End-of-quarter boost for last 10 days of quarter-end months
    if d.month in EOQ_MONTHS and d.day >= 21:
        factor *= EOQ_BOOST

    # December holiday suppression: Dec 20-31
    if d.month == 12 and d.day >= 20:
        factor *= DEC_HOLIDAY_SUPPRESSION

    return factor


def _holiday_factor(d: date, segment: str, holidays: dict[date, str]) -> float:
    """Holiday reduction factor with severity tiers. Returns 1.0 if not a holiday."""
    if d not in holidays:
        return 1.0
    name = holidays[d]
    if name in MAJOR_HOLIDAYS:
        tier = "major"
    elif name in MODERATE_HOLIDAYS:
        tier = "moderate"
    else:
        tier = "minor"
    return 1.0 - HOLIDAY_IMPACT[segment][tier]


def _step_change_factor(d: date, compute_type: str, segment: str) -> float:
    """Cumulative step-change multiplier with gradual ramp-up/down."""
    factor = 1.0
    for event in STEP_CHANGES:
        event_date = event["date"]
        ramp_days = event.get("ramp_days", 1)
        ramp_end = event_date + timedelta(days=ramp_days)
        for ct, seg, magnitude in event["affected"]:
            if ct != compute_type or seg != segment:
                continue
            if d >= ramp_end:
                # Fully ramped
                factor *= (1.0 + magnitude)
            elif d >= event_date:
                # During ramp: linear interpolation
                progress = (d - event_date).days / ramp_days
                factor *= (1.0 + magnitude * progress)
    return factor


def _spike_factor(d: date, compute_type: str, segment: str) -> float:
    """One-time spike multiplier with buildup and decay shape."""
    factor = 1.0
    for event in SPIKE_EVENTS:
        start = event["start_date"]
        end = event["end_date"]
        duration = (end - start).days + 1
        # Extend window: 1 day buildup before, 1 day decay after
        if start - timedelta(days=1) <= d <= end + timedelta(days=1):
            for ct, seg, mult in event["affected"]:
                if ct != compute_type or seg != segment:
                    continue
                excess = mult - 1.0  # e.g., 1.8 -> 0.8 excess
                if d < start:
                    # Buildup day: 40% of peak
                    factor *= 1.0 + excess * 0.4
                elif d > end:
                    # Decay day: 30% of peak
                    factor *= 1.0 + excess * 0.3
                elif duration >= 3:
                    # Multi-day: ramp up, peak, ramp down
                    day_idx = (d - start).days
                    if day_idx == 0:
                        factor *= 1.0 + excess * 0.7
                    elif day_idx == duration - 1:
                        factor *= 1.0 + excess * 0.8
                    else:
                        factor *= mult  # full peak on middle days
                else:
                    factor *= mult  # short spikes: full peak
    return factor


def _outage_factor(d: date, compute_type: str) -> float:
    """Outage impact. Only affects GPU workloads."""
    if compute_type not in OUTAGE["affected_types"]:
        return 1.0
    if OUTAGE["start_date"] <= d <= OUTAGE["end_date"]:
        return OUTAGE["outage_factor"]
    if OUTAGE["end_date"] < d <= OUTAGE["recovery_end"]:
        # Gradual recovery: linearly interpolate from recovery_factor to 1.0
        recovery_days = (OUTAGE["recovery_end"] - OUTAGE["end_date"]).days
        days_into_recovery = (d - OUTAGE["end_date"]).days
        progress = days_into_recovery / recovery_days
        return OUTAGE["recovery_factor"] + (1.0 - OUTAGE["recovery_factor"]) * progress
    return 1.0


def generate_compute_usage() -> pd.DataFrame:
    """Generate the main compute usage dataset."""
    holidays = _build_holiday_set()
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []
    for d_ts in dates:
        d = d_ts.date()
        for ct in COMPUTE_TYPES:
            for seg in CUSTOMER_SEGMENTS:
                base = BASE_HOURS[(ct, seg)]
                trend = _compute_trend(d, ct, seg)
                weekly = _weekly_factor(d, ct, seg)
                annual = _annual_factor(d)
                holiday = _holiday_factor(d, seg, holidays)
                step = _step_change_factor(d, ct, seg)
                spike = _spike_factor(d, ct, seg)
                outage = _outage_factor(d, ct)

                # Symmetric multiplicative noise: exp(N(0, sigma))
                # Median = 1.0, no upward bias unlike raw lognormal
                noise = np.exp(RNG.normal(0.0, NOISE_STD[seg]))

                value = base * trend * weekly * annual * holiday * step * spike * outage * noise
                value = max(0, round(value))

                rows.append({
                    "date": d,
                    "compute_type": ct,
                    "customer_segment": seg,
                    "compute_hours": value,
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_holiday_calendar() -> pd.DataFrame:
    """Generate US federal holiday calendar for the full date range + forecast period."""
    rows = []
    for year in range(START_DATE.year, END_DATE.year + 2):  # +2 for forecast period
        for d, name in _get_us_holidays(year):
            rows.append({"date": d, "holiday_name": name})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def generate_event_log() -> pd.DataFrame:
    """Generate the historical event log documenting all known events."""
    rows = []

    # Step-changes (including churn)
    for event in STEP_CHANGES:
        affected_types = sorted({ct for ct, _, _ in event["affected"]})
        affected_segs = sorted({seg for _, seg, _ in event["affected"]})
        magnitudes = [f"{mag:+.0%}" for _, _, mag in event["affected"]]
        ramp_days = event.get("ramp_days", 1)
        ramp_end = event["date"] + timedelta(days=ramp_days)
        event_type = "churn" if any(m < 0 for _, _, m in event["affected"]) else "step_change"
        rows.append({
            "event_date": event["date"],
            "event_end_date": ramp_end,
            "event_type": event_type,
            "event_name": event["name"],
            "description": event["description"],
            "affected_compute_types": "; ".join(affected_types),
            "affected_segments": "; ".join(affected_segs),
            "magnitude": "; ".join(magnitudes),
        })

    # Spikes
    for event in SPIKE_EVENTS:
        affected_types = sorted({ct for ct, _, _ in event["affected"]})
        affected_segs = sorted({seg for _, seg, _ in event["affected"]})
        magnitudes = [f"{mult:.1f}x" for _, _, mult in event["affected"]]
        rows.append({
            "event_date": event["start_date"],
            "event_end_date": event["end_date"],
            "event_type": "spike",
            "event_name": event["name"],
            "description": event["description"],
            "affected_compute_types": "; ".join(affected_types),
            "affected_segments": "; ".join(affected_segs),
            "magnitude": "; ".join(magnitudes),
        })

    # Outage
    rows.append({
        "event_date": OUTAGE["start_date"],
        "event_end_date": OUTAGE["recovery_end"],
        "event_type": "outage",
        "event_name": OUTAGE["name"],
        "description": OUTAGE["description"],
        "affected_compute_types": "; ".join(sorted(OUTAGE["affected_types"])),
        "affected_segments": "All",
        "magnitude": f"{OUTAGE['outage_factor']:.0%} during outage; "
                     f"{OUTAGE['recovery_factor']:.0%} during recovery",
    })

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["event_end_date"] = pd.to_datetime(df["event_end_date"])
    return df.sort_values("event_date").reset_index(drop=True)


def generate_upcoming_events() -> pd.DataFrame:
    """Generate the sales pipeline / upcoming events dataset."""
    df = pd.DataFrame(UPCOMING_EVENTS)
    df["expected_date"] = pd.to_datetime(df["expected_date"])
    return df


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    output_dir = Path(__file__).parent

    print("Generating compute usage data...")
    usage = generate_compute_usage()
    usage.to_csv(output_dir / "compute_usage.csv", index=False)
    print(f"  -> {len(usage):,} rows, {usage['date'].nunique()} days")
    print(f"  -> Date range: {usage['date'].min().date()} to {usage['date'].max().date()}")
    print(f"  -> Compute types: {usage['compute_type'].nunique()}")
    print(f"  -> Segments: {usage['customer_segment'].nunique()}")
    print(f"  -> Total compute hours: {usage['compute_hours'].sum():,.0f}")

    print("\nGenerating holiday calendar...")
    holidays = generate_holiday_calendar()
    holidays.to_csv(output_dir / "holiday_calendar.csv", index=False)
    print(f"  -> {len(holidays)} holidays")

    print("\nGenerating event log...")
    events = generate_event_log()
    events.to_csv(output_dir / "event_log.csv", index=False)
    print(f"  -> {len(events)} events")

    print("\nGenerating upcoming events (sales pipeline)...")
    upcoming = generate_upcoming_events()
    upcoming.to_csv(output_dir / "upcoming_events.csv", index=False)
    print(f"  -> {len(upcoming)} pipeline deals")

    print("\nDone! All CSVs written to:", output_dir)


if __name__ == "__main__":
    main()
