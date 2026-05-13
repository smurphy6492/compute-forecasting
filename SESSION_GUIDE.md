# Session Guide — Compute Capacity Forecasting

Context management is critical — each notebook is substantial with educational markdown, code, and outputs. Here's how to split the work across sessions for maximum efficiency.

The full project plan is at `.claude/plans/gentle-doodling-sphinx.md`.

---

## Session 1: Scaffolding + Data Generation [COMPLETE]
**Scope:** Phase 0 + Phase 1
**Initial prompt:**
> We're building the compute capacity forecasting project. Start with Phase 0 (scaffolding) and Phase 1 (data generation). The plan is at `.claude/plans/gentle-doodling-sphinx.md`. Create the project structure, pyproject.toml, and the synthetic data generator script. Generate all CSVs. Validate the data looks realistic — spot check trends, seasonality, and events. Show me sample rows and basic summary stats so I can confirm the data feels right before we move to EDA.

**Agent orchestration:** No agents needed — this is single-thread work (sequential file creation + validation).
**Expected output:** All files in `data/`, working `pyproject.toml`, validated CSVs.

---

## Session 2: EDA Notebook
**Scope:** Phase 2
**Initial prompt:**
> Continue the compute forecasting project. The data is generated at `projects/compute-forecasting/data/`. Build the EDA notebook (`notebooks/01_eda.ipynb`) following the plan at `.claude/plans/gentle-doodling-sphinx.md`. This is Part 1 — exploratory analysis with heavy educational markdown. Cover: overall trends with event annotations, trends by compute_type and customer_segment, seasonality decomposition, weekly patterns, event impact quantification, distribution and correlation analysis, and a key findings summary. Use consistent styling (seaborn whitegrid, 100 dpi). Remember to escape `$` as `\$` in markdown cells.

**Agent orchestration:** Can optionally use `web-developer` agent in parallel to draft visualization helper functions in `src/compute_forecasting/visualization.py` while you build the notebook — but only if context is getting tight. Usually better to keep this single-threaded and extract shared code later.
**Expected output:** Complete `01_eda.ipynb` with all cells populated and outputs.

---

## Session 3: Forecasting Model
**Scope:** Phase 3 + start of Phase 5 (extract features.py)
**Initial prompt:**
> Continue the compute forecasting project. EDA is complete at `projects/compute-forecasting/notebooks/01_eda.ipynb`. Build the forecasting notebook (`notebooks/02_forecasting.ipynb`) following the plan at `.claude/plans/gentle-doodling-sphinx.md`. This is Part 2 — LightGBM with quantile regression. Cover: why gradient boosting (comparison table), feature engineering with explanations, time-based train/val/test split, baseline models, LightGBM training, quantile regression for P10/P50/P90, evaluation metrics, feature importance, and residual analysis. Also extract reusable feature engineering into `src/compute_forecasting/features.py`. Remember to escape `$` as `\$` in markdown cells.

**Agent orchestration:** This is the most complex session. Options:
- **Option A (recommended):** Single-thread. The notebook is sequential and each section builds on the last.
- **Option B (if context is tight):** Use a background agent to build `features.py` and `evaluation.py` in `src/` while you build the notebook, then import from those modules.

**Expected output:** Complete `02_forecasting.ipynb`, `src/compute_forecasting/features.py`, trained models.

---

## Session 4: Scenario Planning + Polish
**Scope:** Phase 4 + remainder of Phase 5 + Phase 6
**Initial prompt:**
> Continue the compute forecasting project. The forecasting model is built at `projects/compute-forecasting/notebooks/02_forecasting.ipynb`. Build the scenario planning notebook (`notebooks/03_scenarios.ipynb`) following the plan at `.claude/plans/gentle-doodling-sphinx.md`. This is Part 3 — the exec deliverable. Structure: 60-day daily forecast with P10/P50/P90 fan chart, then months 3-6 summary table (avg, median, min, max at each confidence level). Three scenarios: Base (current trends), High (sales pipeline converts), Low (efficiency improvements). Build the capacity threshold chart — all 3 scenarios + capacity line showing "when do we run out?" Include sensitivity analysis (tornado chart), recommendations, and executive summary. Save hero figures to `outputs/figures/`. Also extract visualization helpers to `src/compute_forecasting/visualization.py` and finalize `README.md`. Escape `$` as `\$` in markdown cells.

**Agent orchestration:** Can parallelize here:
- **Main thread:** Build `03_scenarios.ipynb`
- **Background agent (`content-writer`):** Draft the `README.md` and executive summary text while you build the notebook
- **Background agent (`web-developer`):** If portfolio integration is in scope, start drafting the portfolio case study page

**Expected output:** Complete `03_scenarios.ipynb`, all `src/` modules, `README.md`, figures in `outputs/figures/`.

---

## Session 5 (Optional): Portfolio Integration
**Scope:** Website case study + PR
**Initial prompt:**
> The compute forecasting project is complete at `projects/compute-forecasting/`. Use `/portfolio-updater` to add it to the personal website. The hero image is `outputs/figures/capacity_threshold_chart.png`. Frame it as "Compute Capacity Forecasting — ML-driven capacity planning for an AI infrastructure company" targeting Director of Analytics roles.

**Agent orchestration:** Use `/portfolio-updater` skill which chains `content-writer` + `web-developer` + `github-workflow`.

---

## When to Start a New Session
- **Start fresh** when the previous session's notebook is complete and validated
- **Don't split** a single notebook across sessions — each notebook should be built in one session
- **Context pressure signals:** If you're past ~60% of context and still have major work left, wrap up the current deliverable and note what's left for the next session

## Agent Team Patterns
| Pattern | When to Use |
|---------|-------------|
| Single-thread | Default for notebook building — cells are sequential |
| Background `content-writer` | README, case study, executive summary text — can run while you code |
| Background `web-developer` | Portfolio page scaffolding — only in Session 5 |
| Background code extraction | `src/` module extraction — only if main notebook is done and you want to parallelize cleanup |
| `python-reviewer` at end | Run on completed `.py` files before committing (Session 4 wrap-up) |
