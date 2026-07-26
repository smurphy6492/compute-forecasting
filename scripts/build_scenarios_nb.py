"""Build the scenario planning notebook (03_scenarios.ipynb)."""

import json
import uuid


def mc(source, cell_id=None):
    """Make markdown cell."""
    if cell_id is None:
        cell_id = uuid.uuid4().hex[:12]
    lines = source.split("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def cc(source, cell_id=None):
    """Make code cell."""
    if cell_id is None:
        cell_id = uuid.uuid4().hex[:12]
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        "execution_count": None,
        "outputs": [],
    }


cells = []

# ============================================================
# Section 1: Executive Summary + Setup
# ============================================================
cells.append(
    mc(
        "# Compute Capacity Forecasting -- Scenario Planning\n"
        "\n"
        "**Purpose:** Translate the forecasting model from Part 2 into executive-ready capacity decisions.\n"
        "\n"
        "**This notebook produces:**\n"
        "1. **60-day daily forecast** (Jul-Aug 2026) with P10/P50/P90 confidence bands\n"
        "2. **Months 3-6 planning envelope** (Sep-Dec 2026) for procurement decisions\n"
        "3. **Three scenarios** against a capacity ceiling -- Base, High (pipeline), Low (efficiency)\n"
        "4. **Capacity threshold analysis** -- when does each scenario hit the ceiling?\n"
        "5. **Sensitivity analysis** -- which assumptions move the procurement timeline most?\n"
        "\n"
        "---"
    )
)

cells.append(mc("## 1. Setup"))

cells.append(
    cc(
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import lightgbm as lgb\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.dates as mdates\n"
        "import matplotlib.ticker as mticker\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import mean_absolute_percentage_error\n"
        "\n"
        "import sys\n"
        "sys.path.insert(0, str(Path('../src').resolve()))\n"
        "from compute_forecasting.features import build_features, get_feature_columns\n"
        "\n"
        "warnings.filterwarnings('ignore', category=FutureWarning)\n"
        "\n"
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n"
        "plt.rcParams.update({\n"
        "    'figure.dpi': 120,\n"
        "    'figure.figsize': (14, 5),\n"
        "    'axes.titlesize': 14,\n"
        "    'axes.labelsize': 12,\n"
        "})\n"
        "\n"
        "TYPE_COLORS = {\n"
        "    'GPU Training': '#2196F3',\n"
        "    'GPU Inference': '#FF9800',\n"
        "    'CPU Batch': '#4CAF50',\n"
        "    'CPU Interactive': '#9C27B0',\n"
        "}\n"
        "SCENARIO_COLORS = {'Base': '#1565C0', 'High': '#C62828', 'Low': '#2E7D32'}\n"
        "\n"
        "DATA_DIR = Path('../data')\n"
        "FIG_DIR = Path('../outputs/figures')\n"
        "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "print('Setup complete.')"
    )
)

cells.append(
    cc(
        "# Load data\n"
        "usage = pd.read_csv(DATA_DIR / 'compute_usage.csv', parse_dates=['date'])\n"
        "holidays = pd.read_csv(DATA_DIR / 'holiday_calendar.csv', parse_dates=['date'])\n"
        "upcoming = pd.read_csv(DATA_DIR / 'upcoming_events.csv', parse_dates=['expected_date'])\n"
        "\n"
        'print(f\'Usage: {usage.shape[0]:,} rows, {usage["date"].min().date()} to {usage["date"].max().date()}\')\n'
        "print(f'Pipeline deals: {len(upcoming)}')\n"
        "print()\n"
        "upcoming[['event_name', 'expected_date', 'compute_type', 'customer_segment',\n"
        "          'estimated_daily_hours', 'probability']]"
    )
)

# ============================================================
# Section 2: Model Training (full data)
# ============================================================
cells.append(
    mc(
        "## 2. Model Training\n"
        "\n"
        "We train two model variants on ALL available data (through Jun 30, 2026):\n"
        "\n"
        "1. **Near-term model** -- uses `lag_28` and `lag_365` (drops short lags and rolling features). For the 60-day forecast, `lag_28` has actuals for the first 28 days and `lag_365` always has actuals.\n"
        "2. **Long-horizon model** -- calendar + trend + categoricals only (no lag features). For the 3-6 month planning envelope where no lag values are available.\n"
        "\n"
        "Both are trained as P10/P50/P90 quantile models."
    )
)

cells.append(
    cc(
        "# Build features on full dataset\n"
        "df = build_features(usage, holidays)\n"
        "feature_cols = get_feature_columns(df)\n"
        "TARGET = 'compute_hours'\n"
        "CAT_FEATURES = ['compute_type', 'customer_segment']\n"
        "\n"
        "PARAMS = {\n"
        "    'objective': 'quantile', 'alpha': 0.5, 'metric': 'quantile',\n"
        "    'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 50,\n"
        "    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,\n"
        "    'reg_lambda': 1.0, 'n_estimators': 2000, 'random_state': 42, 'verbose': -1,\n"
        "}\n"
        "\n"
        "# Near-term features: drop short lags and rolling (keep lag_28, lag_365)\n"
        "SHORT_LAG_COLS = ['lag_1', 'lag_7', 'lag_14',\n"
        "                  'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28']\n"
        "nearterm_features = [c for c in feature_cols if c not in SHORT_LAG_COLS]\n"
        "\n"
        "# Long-horizon features: calendar + trend + categoricals only\n"
        "ALL_LAG_COLS = SHORT_LAG_COLS + ['lag_28', 'lag_365']\n"
        "longhorizon_features = [c for c in feature_cols if c not in ALL_LAG_COLS]\n"
        "\n"
        "print(f'Full features: {len(feature_cols)}')\n"
        "print(f'Near-term features ({len(nearterm_features)}): {nearterm_features}')\n"
        "print(f'Long-horizon features ({len(longhorizon_features)}): {longhorizon_features}')"
    )
)

cells.append(
    cc(
        "# Train near-term models (P10/P50/P90)\n"
        "X_all = df[nearterm_features]\n"
        "y_all = df[TARGET]\n"
        "\n"
        "nearterm_models = {}\n"
        "for alpha, label in [(0.1, 'P10'), (0.5, 'P50'), (0.9, 'P90')]:\n"
        "    p = PARAMS.copy()\n"
        "    p['alpha'] = alpha\n"
        "    m = lgb.LGBMRegressor(**p)\n"
        "    m.fit(X_all, y_all, categorical_feature=CAT_FEATURES)\n"
        "    nearterm_models[label] = m\n"
        "    print(f'Near-term {label}: trained on {len(X_all):,} rows')\n"
        "\n"
        "# Train long-horizon models\n"
        "X_all_lh = df[longhorizon_features]\n"
        "longhorizon_models = {}\n"
        "for alpha, label in [(0.1, 'P10'), (0.5, 'P50'), (0.9, 'P90')]:\n"
        "    p = PARAMS.copy()\n"
        "    p['alpha'] = alpha\n"
        "    m = lgb.LGBMRegressor(**p)\n"
        "    m.fit(X_all_lh, y_all, categorical_feature=CAT_FEATURES)\n"
        "    longhorizon_models[label] = m\n"
        "    print(f'Long-horizon {label}: trained on {len(X_all_lh):,} rows')"
    )
)

# ============================================================
# Section 3: 60-Day Forecast
# ============================================================
cells.append(
    mc(
        "## 3. 60-Day Daily Forecast (Jul-Aug 2026)\n"
        "\n"
        "The near-term forecast uses `lag_28` (actuals from Jun 2026 for the first 28 days) and `lag_365` (actuals from Jul-Aug 2025). This avoids the worst of recursive error accumulation while still providing daily granularity."
    )
)

cells.append(
    cc(
        "# Build forecast scaffold: 60 days x 16 series\n"
        "FORECAST_START = pd.Timestamp('2026-07-01')\n"
        "FORECAST_END_60 = pd.Timestamp('2026-08-29')\n"
        "forecast_dates = pd.date_range(FORECAST_START, FORECAST_END_60, freq='D')\n"
        "\n"
        "COMPUTE_TYPES = ['GPU Training', 'GPU Inference', 'CPU Batch', 'CPU Interactive']\n"
        "SEGMENTS = ['Enterprise', 'Mid-Market', 'Startup', 'Research/Academic']\n"
        "\n"
        "rows = []\n"
        "for d in forecast_dates:\n"
        "    for ct in COMPUTE_TYPES:\n"
        "        for seg in SEGMENTS:\n"
        "            rows.append({'date': d, 'compute_type': ct, 'customer_segment': seg})\n"
        "fc = pd.DataFrame(rows)\n"
        "print(f'Forecast scaffold: {len(fc):,} rows ({len(forecast_dates)} days x {len(COMPUTE_TYPES)*len(SEGMENTS)} series)')"
    )
)

cells.append(
    cc(
        "# Build features for forecast period\n"
        "# Calendar features\n"
        "fc['day_of_week'] = fc['date'].dt.dayofweek\n"
        "fc['month'] = fc['date'].dt.month\n"
        "fc['day_of_month'] = fc['date'].dt.day\n"
        "fc['week_of_year'] = fc['date'].dt.isocalendar().week.astype(int)\n"
        "fc['is_weekend'] = (fc['day_of_week'] >= 5).astype(int)\n"
        "fc['quarter'] = fc['date'].dt.quarter\n"
        "fc['day_of_year'] = fc['date'].dt.dayofyear\n"
        "quarter_start = fc['date'].dt.to_period('Q').dt.start_time\n"
        "fc['day_of_quarter'] = (fc['date'] - quarter_start).dt.days + 1\n"
        "quarter_end = fc['date'].dt.to_period('Q').dt.end_time.dt.date\n"
        "days_to_qend = (pd.to_datetime(quarter_end) - fc['date']).dt.days\n"
        "fc['is_quarter_end'] = (days_to_qend <= 10).astype(int)\n"
        "\n"
        "# Holiday features\n"
        "holiday_dates = sorted(holidays['date'].dt.date.unique())\n"
        "holiday_arr = np.array([pd.Timestamp(d) for d in holiday_dates])\n"
        "fc['is_holiday'] = fc['date'].dt.date.isin(holiday_dates).astype(int)\n"
        "\n"
        "days_to_next = {}\n"
        "days_from_last = {}\n"
        "for d in sorted(fc['date'].dt.date.unique()):\n"
        "    ts = pd.Timestamp(d)\n"
        "    future = holiday_arr[holiday_arr >= ts]\n"
        "    past = holiday_arr[holiday_arr <= ts]\n"
        "    days_to_next[d] = int((future[0] - ts).days) if len(future) > 0 else 30\n"
        "    days_from_last[d] = int((ts - past[-1]).days) if len(past) > 0 else 30\n"
        "\n"
        "fc['days_to_next_holiday'] = fc['date'].dt.date.map(days_to_next).clip(upper=30)\n"
        "fc['days_from_last_holiday'] = fc['date'].dt.date.map(days_from_last).clip(upper=30)\n"
        "\n"
        "# Trend\n"
        "origin = usage['date'].min()\n"
        "fc['days_since_start'] = (fc['date'] - origin).dt.days\n"
        "\n"
        "# Categoricals\n"
        "fc['compute_type'] = fc['compute_type'].astype('category')\n"
        "fc['customer_segment'] = fc['customer_segment'].astype('category')\n"
        "\n"
        "# Lag features from historical data\n"
        "# lag_28: value from 28 days ago (Jun 3-30 for Jul 1-28)\n"
        "# lag_365: value from 365 days ago (Jul-Aug 2025)\n"
        "historical = usage.copy()\n"
        "for lag_days, col_name in [(28, 'lag_28'), (365, 'lag_365')]:\n"
        "    lookup = historical.set_index(['date', 'compute_type', 'customer_segment'])['compute_hours']\n"
        "    fc[col_name] = fc.apply(\n"
        "        lambda r: lookup.get((r['date'] - pd.Timedelta(days=lag_days), r['compute_type'], r['customer_segment']), np.nan),\n"
        "        axis=1\n"
        "    )\n"
        "\n"
        'print(f\'lag_28 available: {fc["lag_28"].notna().sum():,} / {len(fc):,} ({fc["lag_28"].notna().mean():.0%})\')\n'
        'print(f\'lag_365 available: {fc["lag_365"].notna().sum():,} / {len(fc):,} ({fc["lag_365"].notna().mean():.0%})\')'
    )
)

cells.append(
    cc(
        "# Generate 60-day forecasts\n"
        "for label in ['P10', 'P50', 'P90']:\n"
        "    fc[f'base_{label}'] = nearterm_models[label].predict(fc[nearterm_features]).clip(min=0)\n"
        "\n"
        "# Aggregate by date for total compute\n"
        "fc_daily = fc.groupby('date').agg({\n"
        "    'base_P10': 'sum', 'base_P50': 'sum', 'base_P90': 'sum'\n"
        "}).reset_index()\n"
        "\n"
        'print(f\'60-day forecast: {fc_daily["date"].min().date()} to {fc_daily["date"].max().date()}\')\n'
        'print(f\'P50 range: {fc_daily["base_P50"].min():,.0f} - {fc_daily["base_P50"].max():,.0f} total hours/day\')'
    )
)

cells.append(
    cc(
        "# 60-day fan chart -- total compute\n"
        "# Include last 60 days of actuals for context\n"
        "actual_tail = usage[usage['date'] >= '2026-05-01'].groupby('date')['compute_hours'].sum().reset_index()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 6))\n"
        "\n"
        "# Actuals\n"
        "ax.plot(actual_tail['date'], actual_tail['compute_hours'],\n"
        "        color='#333333', linewidth=1.5, label='Actual')\n"
        "\n"
        "# Forecast\n"
        "ax.fill_between(fc_daily['date'], fc_daily['base_P10'], fc_daily['base_P90'],\n"
        "                alpha=0.2, color='#1565C0', label='P10-P90 band')\n"
        "ax.plot(fc_daily['date'], fc_daily['base_P50'],\n"
        "        color='#1565C0', linewidth=2, label='P50 forecast')\n"
        "\n"
        "# Mark transition\n"
        "ax.axvline(FORECAST_START, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.7, label='Forecast start')\n"
        "\n"
        "# Mark Jul 4\n"
        "jul4 = pd.Timestamp('2026-07-04')\n"
        "if jul4 in fc_daily['date'].values:\n"
        "    ax.axvline(jul4, color='#E65100', linestyle=':', alpha=0.5)\n"
        "    ax.text(jul4, ax.get_ylim()[1] * 0.95, ' Jul 4', fontsize=8, color='#E65100')\n"
        "\n"
        "ax.set_title('60-Day Forecast -- Total Daily Compute Hours', fontweight='bold')\n"
        "ax.set_ylabel('Compute Hours')\n"
        "ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1_000:,.0f}K'))\n"
        "ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))\n"
        "ax.legend(loc='upper left')\n"
        "plt.xticks(rotation=30)\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / '60day_forecast_fan_chart.png', bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(
    cc(
        "# 60-day by compute type\n"
        "fc_by_type = fc.groupby(['date', 'compute_type']).agg({\n"
        "    'base_P10': 'sum', 'base_P50': 'sum', 'base_P90': 'sum'\n"
        "}).reset_index()\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n"
        "for ax, ctype in zip(axes.flat, COMPUTE_TYPES):\n"
        "    subset = fc_by_type[fc_by_type['compute_type'] == ctype]\n"
        "    color = TYPE_COLORS[ctype]\n"
        "    ax.fill_between(subset['date'], subset['base_P10'], subset['base_P90'],\n"
        "                    alpha=0.2, color=color)\n"
        "    ax.plot(subset['date'], subset['base_P50'], color=color, linewidth=2)\n"
        "    ax.set_title(ctype, fontweight='bold')\n"
        "    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1_000:,.0f}K'))\n"
        "    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))\n"
        "    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)\n"
        "\n"
        "fig.suptitle('60-Day Forecast by Compute Type (P50 + P10-P90)', fontsize=14, fontweight='bold', y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

# ============================================================
# Section 4: 3-6 Month Planning Envelope
# ============================================================
cells.append(
    mc(
        "## 4. Months 3-6 Planning Envelope (Sep-Dec 2026)\n"
        "\n"
        "For the longer horizon, we use the calendar-only model (no lag features). This produces wider uncertainty bands, which is appropriate -- at 3-6 months out, we're providing a planning envelope, not a precise forecast."
    )
)

cells.append(
    cc(
        "# Build long-horizon forecast scaffold (Sep 1 - Dec 31, 2026)\n"
        "LH_START = pd.Timestamp('2026-09-01')\n"
        "LH_END = pd.Timestamp('2026-12-31')\n"
        "lh_dates = pd.date_range(LH_START, LH_END, freq='D')\n"
        "\n"
        "lh_rows = []\n"
        "for d in lh_dates:\n"
        "    for ct in COMPUTE_TYPES:\n"
        "        for seg in SEGMENTS:\n"
        "            lh_rows.append({'date': d, 'compute_type': ct, 'customer_segment': seg})\n"
        "lh = pd.DataFrame(lh_rows)\n"
        "\n"
        "# Build calendar features (same as 60-day but no lags)\n"
        "lh['day_of_week'] = lh['date'].dt.dayofweek\n"
        "lh['month'] = lh['date'].dt.month\n"
        "lh['day_of_month'] = lh['date'].dt.day\n"
        "lh['week_of_year'] = lh['date'].dt.isocalendar().week.astype(int)\n"
        "lh['is_weekend'] = (lh['day_of_week'] >= 5).astype(int)\n"
        "lh['quarter'] = lh['date'].dt.quarter\n"
        "lh['day_of_year'] = lh['date'].dt.dayofyear\n"
        "quarter_start = lh['date'].dt.to_period('Q').dt.start_time\n"
        "lh['day_of_quarter'] = (lh['date'] - quarter_start).dt.days + 1\n"
        "quarter_end = lh['date'].dt.to_period('Q').dt.end_time.dt.date\n"
        "days_to_qend = (pd.to_datetime(quarter_end) - lh['date']).dt.days\n"
        "lh['is_quarter_end'] = (days_to_qend <= 10).astype(int)\n"
        "\n"
        "lh['is_holiday'] = lh['date'].dt.date.isin(holiday_dates).astype(int)\n"
        "lh_unique_dates = sorted(lh['date'].dt.date.unique())\n"
        "dtn = {}\n"
        "dfl = {}\n"
        "for d in lh_unique_dates:\n"
        "    ts = pd.Timestamp(d)\n"
        "    future = holiday_arr[holiday_arr >= ts]\n"
        "    past = holiday_arr[holiday_arr <= ts]\n"
        "    dtn[d] = int((future[0] - ts).days) if len(future) > 0 else 30\n"
        "    dfl[d] = int((ts - past[-1]).days) if len(past) > 0 else 30\n"
        "lh['days_to_next_holiday'] = lh['date'].dt.date.map(dtn).clip(upper=30)\n"
        "lh['days_from_last_holiday'] = lh['date'].dt.date.map(dfl).clip(upper=30)\n"
        "lh['days_since_start'] = (lh['date'] - origin).dt.days\n"
        "lh['compute_type'] = lh['compute_type'].astype('category')\n"
        "lh['customer_segment'] = lh['customer_segment'].astype('category')\n"
        "\n"
        "# Generate predictions\n"
        "for label in ['P10', 'P50', 'P90']:\n"
        "    lh[f'base_{label}'] = longhorizon_models[label].predict(lh[longhorizon_features]).clip(min=0)\n"
        "\n"
        "print(f'Long-horizon forecast: {len(lh_dates)} days, {len(lh):,} rows')"
    )
)

cells.append(
    cc(
        "# Monthly planning envelope table\n"
        "lh_daily = lh.groupby('date').agg({'base_P10': 'sum', 'base_P50': 'sum', 'base_P90': 'sum'}).reset_index()\n"
        "lh_daily['month_name'] = lh_daily['date'].dt.strftime('%b %Y')\n"
        "\n"
        "envelope = lh_daily.groupby('month_name').agg(\n"
        "    P10_avg=('base_P10', 'mean'), P10_min=('base_P10', 'min'), P10_max=('base_P10', 'max'),\n"
        "    P50_avg=('base_P50', 'mean'), P50_min=('base_P50', 'min'), P50_max=('base_P50', 'max'),\n"
        "    P90_avg=('base_P90', 'mean'), P90_min=('base_P90', 'min'), P90_max=('base_P90', 'max'),\n"
        ").reindex(['Sep 2026', 'Oct 2026', 'Nov 2026', 'Dec 2026'])\n"
        "\n"
        "# Format for display\n"
        "for col in envelope.columns:\n"
        "    envelope[col] = envelope[col].map(lambda x: f'{x/1_000:,.0f}K')\n"
        "\n"
        "print('Monthly Planning Envelope (Total Daily Compute Hours):')\n"
        "print()\n"
        "print(envelope.to_string())"
    )
)

# ============================================================
# Section 5: Scenario Construction
# ============================================================
cells.append(
    mc(
        "## 5. Scenario Construction\n"
        "\n"
        "Three scenarios layered on the base forecast:\n"
        "\n"
        "| Scenario | Description | Implementation |\n"
        "|----------|-------------|----------------|\n"
        "| **Base** | Current trends continue | Model forecast as-is |\n"
        "| **High** | Sales pipeline converts | Add probability-weighted daily hours for each deal |\n"
        "| **Low** | Efficiency gains | 15% reduction in GPU Inference (a 'DeepSeek moment') |"
    )
)

cells.append(
    cc(
        "# Combine 60-day and long-horizon forecasts into one timeline\n"
        "# For the combined view, use 60-day for Jul-Aug, long-horizon for Sep-Dec\n"
        "combined = pd.concat([\n"
        "    fc[['date', 'compute_type', 'customer_segment', 'base_P10', 'base_P50', 'base_P90']],\n"
        "    lh[['date', 'compute_type', 'customer_segment', 'base_P10', 'base_P50', 'base_P90']],\n"
        "], ignore_index=True)\n"
        "\n"
        "# --- HIGH scenario: add pipeline deals ---\n"
        "combined['high_P10'] = combined['base_P10'].copy()\n"
        "combined['high_P50'] = combined['base_P50'].copy()\n"
        "combined['high_P90'] = combined['base_P90'].copy()\n"
        "\n"
        "for _, deal in upcoming.iterrows():\n"
        "    mask = (\n"
        "        (combined['date'] >= deal['expected_date']) &\n"
        "        (combined['compute_type'] == deal['compute_type']) &\n"
        "        (combined['customer_segment'] == deal['customer_segment'])\n"
        "    )\n"
        "    add_hours = deal['estimated_daily_hours'] * deal['probability']\n"
        "    combined.loc[mask, 'high_P10'] += add_hours * 0.7   # conservative end\n"
        "    combined.loc[mask, 'high_P50'] += add_hours\n"
        "    combined.loc[mask, 'high_P90'] += add_hours * 1.3   # aggressive end\n"
        "\n"
        "# --- LOW scenario: 15% efficiency reduction on GPU Inference ---\n"
        "EFFICIENCY_START = pd.Timestamp('2026-08-01')\n"
        "EFFICIENCY_FACTOR = 0.85  # 15% reduction\n"
        "\n"
        "combined['low_P10'] = combined['base_P10'].copy()\n"
        "combined['low_P50'] = combined['base_P50'].copy()\n"
        "combined['low_P90'] = combined['base_P90'].copy()\n"
        "\n"
        "gpu_inf_mask = (\n"
        "    (combined['compute_type'] == 'GPU Inference') &\n"
        "    (combined['date'] >= EFFICIENCY_START)\n"
        ")\n"
        "combined.loc[gpu_inf_mask, 'low_P10'] *= EFFICIENCY_FACTOR\n"
        "combined.loc[gpu_inf_mask, 'low_P50'] *= EFFICIENCY_FACTOR\n"
        "combined.loc[gpu_inf_mask, 'low_P90'] *= EFFICIENCY_FACTOR\n"
        "\n"
        'print(f\'Combined forecast: {combined["date"].min().date()} to {combined["date"].max().date()}\')\n'
        "print(f'Total rows: {len(combined):,}')"
    )
)

cells.append(
    cc(
        "# Aggregate scenarios by date\n"
        "scenario_daily = combined.groupby('date').agg({\n"
        "    'base_P10': 'sum', 'base_P50': 'sum', 'base_P90': 'sum',\n"
        "    'high_P10': 'sum', 'high_P50': 'sum', 'high_P90': 'sum',\n"
        "    'low_P10': 'sum', 'low_P50': 'sum', 'low_P90': 'sum',\n"
        "}).reset_index()\n"
        "\n"
        "# Monthly summary by scenario\n"
        "scenario_daily['month'] = scenario_daily['date'].dt.to_period('M')\n"
        "monthly_scenarios = scenario_daily.groupby('month').agg({\n"
        "    'base_P50': 'mean', 'base_P90': 'mean',\n"
        "    'high_P50': 'mean', 'high_P90': 'mean',\n"
        "    'low_P50': 'mean', 'low_P90': 'mean',\n"
        "}).round(0)\n"
        "\n"
        "for col in monthly_scenarios.columns:\n"
        "    monthly_scenarios[col] = monthly_scenarios[col].map(lambda x: f'{x/1_000:,.0f}K')\n"
        "\n"
        "monthly_scenarios.columns = pd.MultiIndex.from_tuples([\n"
        "    ('Base', 'P50'), ('Base', 'P90'),\n"
        "    ('High', 'P50'), ('High', 'P90'),\n"
        "    ('Low', 'P50'), ('Low', 'P90'),\n"
        "])\n"
        "\n"
        "print('Average Daily Compute Hours by Scenario and Month:')\n"
        "print()\n"
        "print(monthly_scenarios.to_string())"
    )
)

# ============================================================
# Section 6: Capacity Threshold Analysis -- THE HERO FIGURE
# ============================================================
cells.append(
    mc(
        "## 6. Capacity Threshold Analysis\n"
        "\n"
        "The key executive deliverable: when does demand cross the capacity ceiling under each scenario?\n"
        "\n"
        "We set the capacity ceiling based on recent peak demand plus a 15% operational buffer. In production, this would come from infrastructure inventory."
    )
)

cells.append(
    cc(
        "# Define capacity ceiling\n"
        "recent_peak = usage[usage['date'] >= '2026-05-01'].groupby('date')['compute_hours'].sum().max()\n"
        "CAPACITY_CEILING = round(recent_peak * 1.15, -3)  # 15% buffer, round to nearest 1000\n"
        "\n"
        "print(f'Recent peak daily compute: {recent_peak:,.0f} hours')\n"
        "print(f'Capacity ceiling (peak + 15%): {CAPACITY_CEILING:,.0f} hours')"
    )
)

cells.append(
    cc(
        "# THE HERO FIGURE: capacity threshold with all scenarios\n"
        "# Include actuals for context\n"
        "actual_context = usage[usage['date'] >= '2026-04-01'].groupby('date')['compute_hours'].sum().reset_index()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(16, 8))\n"
        "\n"
        "# Actuals\n"
        "ax.plot(actual_context['date'], actual_context['compute_hours'],\n"
        "        color='#333333', linewidth=1.5, alpha=0.8, label='Actual')\n"
        "\n"
        "# Scenarios\n"
        "for scenario, color in SCENARIO_COLORS.items():\n"
        "    p50_col = f'{scenario.lower()}_P50'\n"
        "    p90_col = f'{scenario.lower()}_P90'\n"
        "    p10_col = f'{scenario.lower()}_P10'\n"
        "    ax.fill_between(scenario_daily['date'], scenario_daily[p10_col], scenario_daily[p90_col],\n"
        "                    alpha=0.08, color=color)\n"
        "    ax.plot(scenario_daily['date'], scenario_daily[p50_col],\n"
        "            color=color, linewidth=2, label=f'{scenario} P50')\n"
        "    ax.plot(scenario_daily['date'], scenario_daily[p90_col],\n"
        "            color=color, linewidth=1, linestyle='--', alpha=0.6, label=f'{scenario} P90')\n"
        "\n"
        "# Capacity ceiling\n"
        "ax.axhline(CAPACITY_CEILING, color='#C62828', linewidth=2.5, linestyle='-',\n"
        "           alpha=0.8, label=f'Capacity ceiling ({CAPACITY_CEILING/1_000:,.0f}K)')\n"
        "\n"
        "# Forecast start line\n"
        "ax.axvline(FORECAST_START, color='gray', linestyle=':', alpha=0.5)\n"
        "ax.text(FORECAST_START, ax.get_ylim()[0], ' Forecast\\n start',\n"
        "        fontsize=8, color='gray', va='bottom')\n"
        "\n"
        "# Find and annotate crossing dates\n"
        "for scenario, color in SCENARIO_COLORS.items():\n"
        "    p90_col = f'{scenario.lower()}_P90'\n"
        "    crossings = scenario_daily[scenario_daily[p90_col] >= CAPACITY_CEILING]\n"
        "    if len(crossings) > 0:\n"
        "        cross_date = crossings['date'].iloc[0]\n"
        "        ax.axvline(cross_date, color=color, linestyle=':', alpha=0.4)\n"
        "        ax.annotate(f'{scenario} P90\\n{cross_date.strftime(\"%b %d\")}',\n"
        "                    xy=(cross_date, CAPACITY_CEILING),\n"
        "                    xytext=(cross_date + pd.Timedelta(days=5), CAPACITY_CEILING * 1.05),\n"
        "                    fontsize=8, color=color, fontweight='bold',\n"
        "                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))\n"
        "\n"
        "ax.set_title('Capacity Threshold Analysis -- Three Scenarios vs. Current Ceiling',\n"
        "             fontsize=15, fontweight='bold')\n"
        "ax.set_ylabel('Total Daily Compute Hours')\n"
        "ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1_000:,.0f}K'))\n"
        "ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n"
        "ax.legend(loc='upper left', fontsize=9, ncol=2)\n"
        "plt.xticks(rotation=30)\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'capacity_threshold_scenarios.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"
    )
)

cells.append(
    cc(
        "# Print crossing dates for each scenario\n"
        "print('Capacity Threshold Crossing Dates:')\n"
        "print(f'  Capacity ceiling: {CAPACITY_CEILING:,.0f} hours/day')\n"
        "print()\n"
        "\n"
        "for scenario in ['Base', 'High', 'Low']:\n"
        "    for quantile in ['P50', 'P90']:\n"
        "        col = f'{scenario.lower()}_{quantile}'\n"
        "        crossings = scenario_daily[scenario_daily[col] >= CAPACITY_CEILING]\n"
        "        if len(crossings) > 0:\n"
        "            cross_date = crossings['date'].iloc[0]\n"
        "            days_from_now = (cross_date - FORECAST_START).days\n"
        "            pct_days = (crossings[col] >= CAPACITY_CEILING).sum() / len(scenario_daily) * 100\n"
        "            print(f'  {scenario:5s} {quantile}: {cross_date.strftime(\"%Y-%m-%d\")} '\n"
        "                  f'({days_from_now} days from forecast start, '\n"
        "                  f'{pct_days:.0f}% of forecast days exceed ceiling)')\n"
        "        else:\n"
        "            print(f'  {scenario:5s} {quantile}: Does not cross within forecast horizon')"
    )
)

# ============================================================
# Section 7: Sensitivity Analysis
# ============================================================
cells.append(
    mc(
        "## 7. Sensitivity Analysis\n"
        "\n"
        "Which assumptions move the procurement timeline most? We test:\n"
        "1. **Pipeline deals** -- what if each deal is certain (100%) vs. doesn't happen (0%)?\n"
        "2. **Growth rate** -- what if underlying growth is 10% faster or slower?\n"
        "3. **Efficiency gain magnitude** -- 10% vs 15% vs 20% reduction?"
    )
)

cells.append(
    cc(
        "# Sensitivity: impact on P90 capacity crossing date\n"
        "def compute_crossing_date(daily_series, threshold):\n"
        '    """Find first date where series >= threshold."""\n'
        "    crossings = daily_series[daily_series >= threshold].index\n"
        "    return crossings[0] if len(crossings) > 0 else None\n"
        "\n"
        "# Base P90 crossing date (reference point)\n"
        "base_p90_series = scenario_daily.set_index('date')['base_P90']\n"
        "base_crossing = compute_crossing_date(base_p90_series, CAPACITY_CEILING)\n"
        "print(f'Base P90 crossing date: {base_crossing}')\n"
        "\n"
        "sensitivity_results = []\n"
        "\n"
        "# --- 1. Pipeline deal sensitivity ---\n"
        "for _, deal in upcoming.iterrows():\n"
        "    # Scenario: this deal is 100% certain\n"
        "    test_combined = combined.copy()\n"
        "    mask = (\n"
        "        (test_combined['date'] >= deal['expected_date']) &\n"
        "        (test_combined['compute_type'] == deal['compute_type']) &\n"
        "        (test_combined['customer_segment'] == deal['customer_segment'])\n"
        "    )\n"
        "    # Add full (not probability-weighted) hours\n"
        "    test_combined.loc[mask, 'base_P90'] += deal['estimated_daily_hours']\n"
        "    test_daily = test_combined.groupby('date')['base_P90'].sum()\n"
        "    deal_crossing = compute_crossing_date(test_daily, CAPACITY_CEILING)\n"
        "\n"
        "    if base_crossing and deal_crossing:\n"
        "        delta = (deal_crossing - base_crossing).days\n"
        "    elif deal_crossing and not base_crossing:\n"
        "        delta = -999  # brings crossing into range\n"
        "    else:\n"
        "        delta = 0\n"
        "\n"
        "    sensitivity_results.append({\n"
        "        'Parameter': f'Pipeline: {deal[\"event_name\"][:30]}',\n"
        "        'Test': '100% certain',\n"
        "        'Days Impact': delta,\n"
        "        'Category': 'Pipeline',\n"
        "    })\n"
        "\n"
        "# --- 2. Growth rate sensitivity ---\n"
        "for growth_label, growth_mult in [('Growth +10%', 1.10), ('Growth -10%', 0.90)]:\n"
        "    test_daily = scenario_daily.copy()\n"
        "    # Approximate: scale P90 from base by growth multiplier relative to forecast start\n"
        "    days_out = (test_daily['date'] - FORECAST_START).dt.days.clip(lower=0)\n"
        "    growth_factor = growth_mult ** (days_out / 365)  # annualized\n"
        "    test_series = (test_daily['base_P90'] * growth_factor)\n"
        "    test_series.index = test_daily['date']\n"
        "    growth_crossing = compute_crossing_date(test_series, CAPACITY_CEILING)\n"
        "\n"
        "    if base_crossing and growth_crossing:\n"
        "        delta = (growth_crossing - base_crossing).days\n"
        "    elif growth_crossing and not base_crossing:\n"
        "        delta = -999\n"
        "    else:\n"
        "        delta = 0\n"
        "\n"
        "    sensitivity_results.append({\n"
        "        'Parameter': growth_label,\n"
        "        'Test': growth_label,\n"
        "        'Days Impact': delta,\n"
        "        'Category': 'Growth',\n"
        "    })\n"
        "\n"
        "# --- 3. Efficiency gain magnitude ---\n"
        "for eff_label, eff_factor in [('Efficiency 10%', 0.90), ('Efficiency 20%', 0.80)]:\n"
        "    test_combined = combined.copy()\n"
        "    eff_mask = (\n"
        "        (test_combined['compute_type'] == 'GPU Inference') &\n"
        "        (test_combined['date'] >= EFFICIENCY_START)\n"
        "    )\n"
        "    test_combined.loc[eff_mask, 'base_P90'] *= eff_factor\n"
        "    test_daily = test_combined.groupby('date')['base_P90'].sum()\n"
        "    eff_crossing = compute_crossing_date(test_daily, CAPACITY_CEILING)\n"
        "\n"
        "    if base_crossing and eff_crossing:\n"
        "        delta = (eff_crossing - base_crossing).days\n"
        "    elif not eff_crossing and base_crossing:\n"
        "        delta = 999  # pushes crossing beyond horizon\n"
        "    else:\n"
        "        delta = 0\n"
        "\n"
        "    sensitivity_results.append({\n"
        "        'Parameter': eff_label,\n"
        "        'Test': eff_label,\n"
        "        'Days Impact': delta,\n"
        "        'Category': 'Efficiency',\n"
        "    })\n"
        "\n"
        "sens_df = pd.DataFrame(sensitivity_results)\n"
        "sens_df = sens_df.sort_values('Days Impact', key=abs, ascending=True)\n"
        "print(sens_df[['Parameter', 'Days Impact']].to_string(index=False))"
    )
)

cells.append(
    cc(
        "# Tornado chart\n"
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        "\n"
        "# Filter out extreme values for display\n"
        "plot_df = sens_df[sens_df['Days Impact'].abs() < 900].copy()\n"
        "plot_df = plot_df.sort_values('Days Impact', key=abs, ascending=True)\n"
        "\n"
        "colors = ['#C62828' if d < 0 else '#2E7D32' for d in plot_df['Days Impact']]\n"
        "bars = ax.barh(plot_df['Parameter'], plot_df['Days Impact'], color=colors, alpha=0.85, edgecolor='white')\n"
        "\n"
        "ax.axvline(0, color='black', linewidth=0.8)\n"
        "ax.set_xlabel('Days Earlier (-) or Later (+) to Hit Capacity')\n"
        "ax.set_title('Sensitivity Analysis -- Impact on P90 Capacity Crossing Date',\n"
        "             fontweight='bold', fontsize=13)\n"
        "\n"
        "# Add value labels\n"
        "for bar, val in zip(bars, plot_df['Days Impact']):\n"
        "    x_pos = bar.get_width()\n"
        "    ha = 'left' if x_pos >= 0 else 'right'\n"
        "    offset = 1 if x_pos >= 0 else -1\n"
        "    ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,\n"
        "            f'{val:+d} days', va='center', ha=ha, fontsize=10, fontweight='bold')\n"
        "\n"
        "ax.text(0.02, 0.02, 'Earlier (risk)', transform=ax.transAxes, color='#C62828', fontsize=9)\n"
        "ax.text(0.98, 0.02, 'Later (buffer)', transform=ax.transAxes, color='#2E7D32', fontsize=9, ha='right')\n"
        "\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'sensitivity_tornado.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"
    )
)

# ============================================================
# Section 8: Executive Summary & Recommendations
# ============================================================
cells.append(
    mc(
        "## 8. Recommendations\n"
        "\n"
        "### Procurement Decision Framework\n"
        "\n"
        "Based on the three scenarios and sensitivity analysis:\n"
        "\n"
        "1. **Monitor the High scenario P90 crossing date** -- this is the earliest plausible date we could run out of capacity. If pipeline deals are closing faster than expected, accelerate procurement.\n"
        "\n"
        "2. **Pipeline deals are the biggest swing factor** -- the TechGiant contract alone can shift the timeline significantly. Sales should flag deal progress weekly to the capacity planning team.\n"
        "\n"
        "3. **Efficiency gains provide meaningful buffer** -- if inference optimization (the 'DeepSeek moment') materializes, it buys additional runway. Worth investing in inference optimization as a hedge against procurement delays.\n"
        "\n"
        "4. **Refresh this forecast monthly** -- retrain models, update pipeline probabilities, recalibrate conformal adjustments. The capacity threshold chart should be a living document in leadership reviews.\n"
        "\n"
        "### Caveats\n"
        "\n"
        "- Capacity ceiling is a simplification -- real infrastructure has per-GPU-type limits, not a single aggregate ceiling\n"
        "- Pipeline probabilities are point estimates from sales -- actual conversion is uncertain\n"
        "- The 'DeepSeek moment' scenario assumes a sudden, uniform 15% efficiency gain -- real efficiency improvements are typically gradual and workload-specific\n"
        "- Long-horizon forecasts (months 3-6) use a calendar-only model with wider uncertainty bands\n"
        "\n"
        "---\n"
        "*End of Part 3 -- Scenario Planning*"
    )
)

# ============================================================
# Build the notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

outpath = "notebooks/03_scenarios.ipynb"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created {outpath} with {len(cells)} cells")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
