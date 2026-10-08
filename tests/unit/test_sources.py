import pandas as pd
import pytest

from cement_demand.pipeline import schema
from cement_demand.pipeline.cities import CityRegistry
from cement_demand.pipeline.sources import (
    CoreIndustriesSource,
    DatasetSpec,
    LaunchSalesSource,
    available_from,
    build_sources,
    month_start,
)

CITIES = CityRegistry.from_config({"Bengaluru": ["Bangalore"], "Pune": []})

LAUNCH_SALES = DatasetSpec.from_config(
    "launch_sales",
    {
        "role": "feature",
        "geography": "city",
        "release_lag_days": 15,
        "unit": "units",
        "columns": {
            "period": "Relevant_Date",
            "geography": "City",
            "metrics": {"units_launched": "Launches", "units_sold": "Sales"},
        },
    },
)

CORE = DatasetSpec.from_config(
    "core_industries",
    {
        "role": "target",
        "geography": "national",
        "release_lag_days": 20,
        "unit": "index",
        "columns": {
            "period": "Relevant_Date",
            "filter": {"column": "Industry", "equals": "Cement"},
            "metrics": {"cement_index": "Index_Value"},
        },
    },
)


def test_month_start_floors_any_day_to_the_first():
    dates = pd.Series(["2025-03-31", "2025-03-01", "2025-03-15 10:00"])
    assert month_start(dates).dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2025-03-01"
    ]


def test_a_month_is_available_only_after_it_ends_plus_the_lag():
    periods = pd.Series(pd.to_datetime(["2025-03-01", "2025-12-01"]))
    assert available_from(periods, 20).dt.strftime("%Y-%m-%d").tolist() == [
        "2025-04-21",
        "2026-01-21",
    ]


def test_city_export_melts_into_one_row_per_metric():
    raw = pd.DataFrame(
        {
            "Relevant_Date": ["2025-03-31", "2025-03-31"],
            "City": ["Bangalore", "Pune"],
            "Launches": [1200, 800],
            "Sales": [1000, "n/a"],
        }
    )
    long = LaunchSalesSource(LAUNCH_SALES).normalize(raw, CITIES)

    assert list(long.columns) == list(schema.OBSERVATION_COLUMNS)
    assert len(long) == 4
    assert set(long[schema.GEOGRAPHY]) == {"Bengaluru", "Pune"}
    assert (long[schema.GEOGRAPHY_LEVEL] == schema.CITY).all()
    sold = long.loc[long[schema.METRIC] == "units_sold"].set_index(schema.GEOGRAPHY)
    assert sold.loc["Bengaluru", schema.VALUE] == 1000
    # Text in a numeric column becomes missing, for screening to count.
    assert pd.isna(sold.loc["Pune", schema.VALUE])


def test_national_export_keeps_only_the_filtered_line():
    raw = pd.DataFrame(
        {
            "Relevant_Date": ["2025-03-31"] * 3,
            "Industry": ["Cement", "Steel", " cement "],
            "Index_Value": [190.0, 210.0, 191.0],
        }
    )
    long = CoreIndustriesSource(CORE).normalize(raw, CITIES)

    assert long[schema.VALUE].tolist() == [190.0, 191.0]
    assert (long[schema.GEOGRAPHY] == schema.INDIA).all()
    assert (long[schema.AVAILABLE_AT] == pd.Timestamp("2025-04-21")).all()


def test_a_renamed_export_column_fails_with_the_columns_it_has():
    raw = pd.DataFrame({"Date": ["2025-03-31"], "City": ["Pune"], "Launches": [1]})
    with pytest.raises(KeyError, match="config/datasets.json"):
        LaunchSalesSource(LAUNCH_SALES).normalize(raw, CITIES)


def test_a_source_refuses_a_spec_of_the_wrong_geography():
    with pytest.raises(ValueError, match="expects national"):
        CoreIndustriesSource(LAUNCH_SALES)


def test_every_configured_dataset_gets_its_source(tmp_path):
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "datasets.json").read_text()
    )
    names = [source.name for source in build_sources(config)]
    assert names == list(config["datasets"])
