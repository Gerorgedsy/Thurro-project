"""Canonical column vocabulary for every cleaned dataset.

The four Thurro exports arrive with their own column names, date formats and
geographies. Each is translated into one long format, a row per
(dataset, metric, geography, period), so downstream code never needs to know
which export a number came from.

Two clocks travel with every row, for the same reason project 1 kept a GFS
run's initialisation time beside its valid time:

``period``
    The month the number describes, as the first day of that month.
``available_at``
    The first day the number was public. A month's cement index is not known
    on the last day of that month; it is printed weeks later. A feature built
    for a forecast origin may use a row only if ``available_at`` is on or
    before that origin. Keeping the two apart is what lets the project claim
    to call the official print *before* it lands, rather than after.
"""

from __future__ import annotations

DATASET = "dataset"
METRIC = "metric"
GEOGRAPHY_LEVEL = "geography_level"
GEOGRAPHY = "geography"
PERIOD = "period"
AVAILABLE_AT = "available_at"
VALUE = "value"
UNIT = "unit"

OBSERVATION_COLUMNS: tuple[str, ...] = (
    DATASET,
    METRIC,
    GEOGRAPHY_LEVEL,
    GEOGRAPHY,
    PERIOD,
    AVAILABLE_AT,
    VALUE,
    UNIT,
)

# Rows keyed on these must be unique once cleaned.
OBSERVATION_KEY: tuple[str, ...] = (DATASET, METRIC, GEOGRAPHY, PERIOD)

CITY = "city"
NATIONAL = "national"
GEOGRAPHY_LEVELS: frozenset[str] = frozenset({CITY, NATIONAL})

# National series carry this as their geography, so every row has one.
INDIA = "India"
