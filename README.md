# Residential Launch-to-Sales Velocity and Cement Demand

This project asks whether how quickly new homes sell in Indian cities predicts
the **direction of national cement demand one to two months before the
official figure is published**.

When new homes start selling faster, builders soon need more cement to keep
building. The project tracks that early signal across cities and tests whether
it calls the direction of the cement line of the Index of Eight Core Industries
before that index is released.

## Team

- George Deng
- TODO

## Project scope

- **Primary:** a launch-to-sales index that calls national cement demand
  direction one to two months early.
- **Secondary (stretch):** show that the result holds after controlling for
  bank lending rates.
- **Methodological question:** the benchmark exists only at the national level,
  so city signals must be aggregated. Equal-weighted and
  construction-share-weighted aggregation are both built and compared.

## Data

All four datasets come from Thurro:

| Dataset | Role | Geography |
|---|---|---|
| Launch & sales data | Leading signal | City |
| Project registrations | Earlier-stage pipeline signal | City |
| Bank lending rates | Control | National |
| Core Industries Index (cement) | Target | National |

Thurro data is licensed, so the exports are **not committed**. Each teammate
exports them as CSV into:

```text
data/raw/thurro/launch_sales/
data/raw/thurro/project_registrations/
data/raw/thurro/lending_rates/
data/raw/thurro/core_industries/
```

Name each export so it sorts by date (for example
`launch_sales_2026-10-08.csv`). The pipeline reads the newest one.

## Repository structure

- `src/cement_demand/pipeline/`: data collection and preparation
- `src/cement_demand/features/`: feature engineering (city aggregation lives here)
- `src/cement_demand/models/`: model implementations
- `src/cement_demand/evaluation/`: model evaluation
- `tests/unit/`, `tests/integration/`: tests
- `config/datasets.json`: dataset column mappings, release lags, city aliases
- `data/raw/`, `data/processed/`: source exports and cleaned tables
- `artifacts/`, `reports/`, `notebooks/`, `docs/`: outputs and analysis

## Setup

```bash
python -m venv .VENV
source .VENV/bin/activate        # Windows: .VENV\Scripts\Activate.ps1
pip install -r requirements.txt
pre-commit install --hook-type commit-msg
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/),
for example `feat: add launch-to-sales index`.

## Data cleaning

```bash
PYTHONPATH=src python -m cement_demand.pipeline.clean_data
```

Pass `--dataset launch_sales` (repeatable) to clean a subset, and
`--overwrite` to replace existing outputs. The run writes to `data/processed/`:

- `observations.parquet`: every screened row in one long format
- `city_panel.parquet`: city metrics on a complete city × month grid
- `national_series.parquet`: the cement index and lending rate, by month
- `screening_report.csv`: what each cleaning rule removed

### How cleaning works

Each Thurro export has its own subclass of `DatasetSource`
(`pipeline/sources.py`). The shared base class locates the export, selects the
relevant rows, dates them, places them in a city or nationally, and reshapes
the metric columns into one row per (dataset, metric, geography, month).
Column names live in `config/datasets.json`, so a renamed column in a new
export is a config edit rather than a code change, and a missing column fails
with a message listing the columns the export does have.

Three design choices matter for the modeling that follows:

1. **Two clocks on every row.** `period` is the month a number describes;
   `available_at` is the first day it was public (`period` end plus the
   dataset's `release_lag_days`). A feature for a forecast made on a given day
   may only use rows already available on that day. This is what makes
   "calls the official print one to two months early" testable rather than
   accidental.
2. **One name per city.** `pipeline/cities.py` maps spellings such as
   Bangalore/Bengaluru to one canonical city. An unrecognised spelling is
   reported and dropped rather than counted as a new market, which would
   otherwise give one market two votes in an equal-weighted index.
3. **No aggregation and no filling in cleaning.** Missing city-months stay
   missing in `city_panel.parquet`, and cities are not combined into a national
   signal here. Both are modeling choices and belong in `features/`.

Screening (`pipeline/quality.py`) removes rows with an unknown city, an
unparseable month, a month outside the study window, a missing value, or an
implausible value (for example negative unit counts or a lending rate above
25 %), and keeps the later of two rows with the same key. Every removal is
counted in `screening_report.csv`.

### Open items before the first real run

- Confirm each export's column names and update `config/datasets.json`
  (the current names are placeholders).
- Confirm each dataset's frequency. Launch and sales data may be quarterly;
  if so, its source needs a period rule for quarters.
- Confirm `release_lag_days` for each dataset against the publishers' release
  calendars.
- Check whether Thurro stores first-release or revised index values. Core
  Industries figures are revised after first publication; if only revised
  values are available, the backtest will see numbers that were not known at
  the time, which must be stated as a limitation.
