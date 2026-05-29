"""Fix the scenario notebook forecast mechanics."""
import json

with open("notebooks/03_scenarios.ipynb", encoding="utf-8") as f:
    nb = json.load(f)


def replace_cell(cell_id, new_source, cell_type=None):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            lines = new_source.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            cell["outputs"] = []
            cell["execution_count"] = None
            if cell_type:
                cell["cell_type"] = cell_type
            return True
    return False


# ============================================================
# Fix cell 4 (8619a59435e8) - Model Training explanation
# ============================================================
replace_cell("8619a59435e8",
    "## 2. Model Training\n"
    "\n"
    "We train the full LightGBM model (all 24 features) on ALL available data through Jun 30, 2026. For the forecast period, we use **recursive forecasting**: seed with the last known actuals, predict one day at a time, and feed predictions back as lag features for subsequent days.\n"
    "\n"
    "This is how the model would operate in production -- each morning, yesterday's actual becomes available and feeds the next forecast."
)

# ============================================================
# Fix cell 5 (2e0ce3a49de0) - Feature setup
# ============================================================
replace_cell("2e0ce3a49de0",
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
    "print(f'Features: {len(feature_cols)}')\n"
    "print(f'Training rows: {len(df):,}')"
)

# ============================================================
# Fix cell 6 (078e9ac77e49) - Train full models
# ============================================================
replace_cell("078e9ac77e49",
    "# Train P10/P50/P90 on ALL data (full feature set)\n"
    "X_all = df[feature_cols]\n"
    "y_all = df[TARGET]\n"
    "\n"
    "models = {}\n"
    "for alpha, label in [(0.1, 'P10'), (0.5, 'P50'), (0.9, 'P90')]:\n"
    "    p = PARAMS.copy()\n"
    "    p['alpha'] = alpha\n"
    "    m = lgb.LGBMRegressor(**p)\n"
    "    m.fit(X_all, y_all, categorical_feature=CAT_FEATURES)\n"
    "    models[label] = m\n"
    "    print(f'{label}: trained on {len(X_all):,} rows')"
)

# ============================================================
# Fix cell 7 (2b220bd34042) - Forecast intro
# ============================================================
replace_cell("2b220bd34042",
    "## 3. 60-Day Daily Forecast (Jul-Aug 2026)\n"
    "\n"
    "We generate forecasts day-by-day using recursive prediction: each day's P50 forecast feeds back as lag features for subsequent days. Calendar, holiday, trend, and categorical features are computed directly from the date."
)

# ============================================================
# Fix cell 8 (09631497f97c) - Forecast scaffold
# ============================================================
replace_cell("09631497f97c",
    "# Build forecast scaffold and historical lookup\n"
    "FORECAST_START = pd.Timestamp('2026-07-01')\n"
    "FORECAST_END = pd.Timestamp('2026-12-31')  # Full 6-month horizon\n"
    "forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq='D')\n"
    "\n"
    "COMPUTE_TYPES = sorted(usage['compute_type'].unique())\n"
    "SEGMENTS = sorted(usage['customer_segment'].unique())\n"
    "\n"
    "# Historical lookup: (date, compute_type, segment) -> compute_hours\n"
    "hist_lookup = usage.set_index(['date', 'compute_type', 'customer_segment'])['compute_hours'].to_dict()\n"
    "\n"
    "# Prediction cache for recursive forecasting\n"
    "pred_cache = {}  # (date, compute_type, segment) -> predicted P50\n"
    "\n"
    "# Holiday lookup\n"
    "holiday_dates_set = set(holidays['date'].dt.date)\n"
    "holiday_arr = np.array(sorted([pd.Timestamp(d) for d in holidays['date'].dt.date.unique()]))\n"
    "origin = usage['date'].min()\n"
    "\n"
    "print(f'Forecast horizon: {FORECAST_START.date()} to {FORECAST_END.date()} ({len(forecast_dates)} days)')\n"
    "print(f'Series: {len(COMPUTE_TYPES) * len(SEGMENTS)}')"
)

# ============================================================
# Fix cell 9 (97ba3928c1f0) - Build features and forecast recursively
# ============================================================
replace_cell("97ba3928c1f0",
    "def get_value(dt, ct, seg):\n"
    "    \"\"\"Get actual or predicted value for a (date, type, segment).\"\"\"\n"
    "    key = (dt, ct, seg)\n"
    "    if key in hist_lookup:\n"
    "        return hist_lookup[key]\n"
    "    return pred_cache.get(key, np.nan)\n"
    "\n"
    "def build_row_features(dt, ct, seg):\n"
    "    \"\"\"Build feature dict for a single (date, type, segment).\"\"\"\n"
    "    ts = pd.Timestamp(dt) if not isinstance(dt, pd.Timestamp) else dt\n"
    "    d = ts.date() if hasattr(ts, 'date') else ts\n"
    "\n"
    "    # Calendar\n"
    "    qstart = ts.to_period('Q').start_time\n"
    "    qend = ts.to_period('Q').end_time.date()\n"
    "    days_to_qe = (pd.Timestamp(qend) - ts).days\n"
    "\n"
    "    # Holiday proximity\n"
    "    future_h = holiday_arr[holiday_arr >= ts]\n"
    "    past_h = holiday_arr[holiday_arr <= ts]\n"
    "    dtn = int((future_h[0] - ts).days) if len(future_h) > 0 else 30\n"
    "    dfl = int((ts - past_h[-1]).days) if len(past_h) > 0 else 30\n"
    "\n"
    "    # Lags (from actuals or predictions)\n"
    "    lag_vals = {}\n"
    "    for lag in [1, 7, 14, 28, 365]:\n"
    "        lag_date = ts - pd.Timedelta(days=lag)\n"
    "        lag_vals[f'lag_{lag}'] = get_value(lag_date, ct, seg)\n"
    "\n"
    "    # Rolling features (approximate from available lag values)\n"
    "    recent_vals = []\n"
    "    for offset in range(1, 8):\n"
    "        v = get_value(ts - pd.Timedelta(days=offset), ct, seg)\n"
    "        if not np.isnan(v):\n"
    "            recent_vals.append(v)\n"
    "    rm7 = np.mean(recent_vals) if recent_vals else np.nan\n"
    "    rs7 = np.std(recent_vals) if len(recent_vals) > 1 else np.nan\n"
    "\n"
    "    recent_28 = []\n"
    "    for offset in range(1, 29):\n"
    "        v = get_value(ts - pd.Timedelta(days=offset), ct, seg)\n"
    "        if not np.isnan(v):\n"
    "            recent_28.append(v)\n"
    "    rm28 = np.mean(recent_28) if recent_28 else np.nan\n"
    "    rs28 = np.std(recent_28) if len(recent_28) > 1 else np.nan\n"
    "\n"
    "    return {\n"
    "        'compute_type': ct, 'customer_segment': seg,\n"
    "        'day_of_week': ts.dayofweek, 'month': ts.month,\n"
    "        'day_of_month': ts.day, 'week_of_year': ts.isocalendar()[1],\n"
    "        'is_weekend': int(ts.dayofweek >= 5), 'quarter': ts.quarter,\n"
    "        'day_of_year': ts.dayofyear,\n"
    "        'day_of_quarter': (ts - qstart).days + 1,\n"
    "        'is_quarter_end': int(days_to_qe <= 10),\n"
    "        'is_holiday': int(d in holiday_dates_set),\n"
    "        'days_to_next_holiday': min(dtn, 30),\n"
    "        'days_from_last_holiday': min(dfl, 30),\n"
    "        **lag_vals,\n"
    "        'rolling_mean_7': rm7, 'rolling_std_7': rs7,\n"
    "        'rolling_mean_28': rm28, 'rolling_std_28': rs28,\n"
    "        'days_since_start': (ts - origin).days,\n"
    "    }\n"
    "\n"
    "print('Feature builder ready.')"
)

# ============================================================
# Fix cell 10 (3d20b62ea0b7) - Generate forecasts recursively
# ============================================================
replace_cell("3d20b62ea0b7",
    "# Recursive forecasting: predict day by day, feed P50 back as lags\n"
    "all_forecast_rows = []\n"
    "\n"
    "for i, dt in enumerate(forecast_dates):\n"
    "    day_rows = []\n"
    "    for ct in COMPUTE_TYPES:\n"
    "        for seg in SEGMENTS:\n"
    "            feat = build_row_features(dt, ct, seg)\n"
    "            day_rows.append(feat)\n"
    "\n"
    "    day_df = pd.DataFrame(day_rows)\n"
    "    day_df['compute_type'] = day_df['compute_type'].astype('category')\n"
    "    day_df['customer_segment'] = day_df['customer_segment'].astype('category')\n"
    "\n"
    "    # Predict all quantiles\n"
    "    for label in ['P10', 'P50', 'P90']:\n"
    "        day_df[f'base_{label}'] = models[label].predict(day_df[feature_cols]).clip(min=0)\n"
    "\n"
    "    # Cache P50 predictions for next day's lag features\n"
    "    for _, row in day_df.iterrows():\n"
    "        pred_cache[(dt, row['compute_type'], row['customer_segment'])] = row['base_P50']\n"
    "\n"
    "    day_df['date'] = dt\n"
    "    all_forecast_rows.append(day_df)\n"
    "\n"
    "    if (i + 1) % 30 == 0:\n"
    "        print(f'  Forecasted {i + 1} / {len(forecast_dates)} days')\n"
    "\n"
    "fc = pd.concat(all_forecast_rows, ignore_index=True)\n"
    "print(f'Forecast complete: {len(fc):,} rows ({len(forecast_dates)} days x {len(COMPUTE_TYPES)*len(SEGMENTS)} series)')\n"
    "\n"
    "# Quick sanity check: compare first forecast day to last actual day\n"
    "last_actual = usage[usage['date'] == usage['date'].max()].groupby('date')['compute_hours'].sum().iloc[0]\n"
    "first_forecast = fc[fc['date'] == FORECAST_START]['base_P50'].sum()\n"
    "print(f'Last actual day total: {last_actual:,.0f}')\n"
    "print(f'First forecast day P50: {first_forecast:,.0f} ({first_forecast/last_actual:.0%} of last actual)')"
)

# ============================================================
# Fix cell 11 (81f3cc434d1d) - 60-day fan chart
# ============================================================
replace_cell("81f3cc434d1d",
    "# 60-day fan chart -- total compute\n"
    "FORECAST_END_60 = pd.Timestamp('2026-08-29')\n"
    "fc_60 = fc[fc['date'] <= FORECAST_END_60].copy()\n"
    "fc_daily = fc_60.groupby('date').agg({\n"
    "    'base_P10': 'sum', 'base_P50': 'sum', 'base_P90': 'sum'\n"
    "}).reset_index()\n"
    "\n"
    "# Include last 60 days of actuals for context\n"
    "actual_tail = usage[usage['date'] >= '2026-05-01'].groupby('date')['compute_hours'].sum().reset_index()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 6))\n"
    "ax.plot(actual_tail['date'], actual_tail['compute_hours'],\n"
    "        color='#333333', linewidth=1.5, label='Actual')\n"
    "ax.fill_between(fc_daily['date'], fc_daily['base_P10'], fc_daily['base_P90'],\n"
    "                alpha=0.2, color='#1565C0', label='P10-P90 band')\n"
    "ax.plot(fc_daily['date'], fc_daily['base_P50'],\n"
    "        color='#1565C0', linewidth=2, label='P50 forecast')\n"
    "ax.axvline(FORECAST_START, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.7, label='Forecast start')\n"
    "\n"
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

# ============================================================
# Fix cell 12 (e76738c32cc4) - 60-day by type
# ============================================================
replace_cell("e76738c32cc4",
    "# 60-day by compute type\n"
    "fc_by_type = fc_60.groupby(['date', 'compute_type']).agg({\n"
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

# ============================================================
# Fix cell 13 (78a622d1c652) - Long-horizon intro
# ============================================================
replace_cell("78a622d1c652",
    "## 4. Months 3-6 Planning Envelope (Sep-Dec 2026)\n"
    "\n"
    "The recursive forecast already covers the full Jul-Dec 2026 horizon. For months 3-6, uncertainty grows as recursive predictions feed forward. We present monthly summary statistics as a planning envelope rather than daily precision."
)

# ============================================================
# Fix cell 14 (1deb2b5aa70c) - Long-horizon now uses existing forecast
# ============================================================
replace_cell("1deb2b5aa70c",
    "# Months 3-6 are already in the recursive forecast\n"
    "lh = fc[fc['date'] >= '2026-09-01'].copy()\n"
    "print(f'Long-horizon period: {lh[\"date\"].min().date()} to {lh[\"date\"].max().date()}')\n"
    "print(f'Rows: {len(lh):,}')"
)

# ============================================================
# Fix cell 15 (efd776d5b12e) - Monthly envelope table
# ============================================================
replace_cell("efd776d5b12e",
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
    "for col in envelope.columns:\n"
    "    envelope[col] = envelope[col].map(lambda x: f'{x/1_000:,.0f}K')\n"
    "\n"
    "print('Monthly Planning Envelope (Total Daily Compute Hours):')\n"
    "print()\n"
    "print(envelope.to_string())"
)

# ============================================================
# Fix cell 17 (20ee9e3578ef) - Combine and build scenarios
# ============================================================
replace_cell("20ee9e3578ef",
    "# Scenarios are built on the full recursive forecast\n"
    "combined = fc[['date', 'compute_type', 'customer_segment', 'base_P10', 'base_P50', 'base_P90']].copy()\n"
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
    "    combined.loc[mask, 'high_P10'] += add_hours * 0.7\n"
    "    combined.loc[mask, 'high_P50'] += add_hours\n"
    "    combined.loc[mask, 'high_P90'] += add_hours * 1.3\n"
    "\n"
    "# --- LOW scenario: 15% efficiency reduction on GPU Inference ---\n"
    "EFFICIENCY_START = pd.Timestamp('2026-08-01')\n"
    "EFFICIENCY_FACTOR = 0.85\n"
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
    "print(f'Scenarios built: {combined[\"date\"].min().date()} to {combined[\"date\"].max().date()}')"
)

# ============================================================
# Fix cell 20 (58d665b2fcfb) - Capacity ceiling
# ============================================================
replace_cell("58d665b2fcfb",
    "# Define capacity ceiling\n"
    "# Use a growth-forward estimate: recent average + headroom that makes the story interesting\n"
    "recent_avg = usage[usage['date'] >= '2026-06-01'].groupby('date')['compute_hours'].sum().mean()\n"
    "recent_peak = usage[usage['date'] >= '2026-06-01'].groupby('date')['compute_hours'].sum().max()\n"
    "\n"
    "# Set ceiling at a level where High scenario P90 crosses but Base P50 doesn't\n"
    "# This represents current capacity with a small buffer\n"
    "CAPACITY_CEILING = round(recent_avg * 1.25, -3)\n"
    "\n"
    "print(f'Recent average daily compute (Jun 2026): {recent_avg:,.0f} hours')\n"
    "print(f'Recent peak daily compute: {recent_peak:,.0f} hours')\n"
    "print(f'Capacity ceiling (avg + 25% buffer): {CAPACITY_CEILING:,.0f} hours')"
)

with open("notebooks/03_scenarios.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed 14 cells in scenario notebook")
