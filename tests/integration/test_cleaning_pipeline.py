import json

import pandas as pd
import pytest

from cement_demand.pipeline import schema
from cement_demand.pipeline.clean_data import CleaningApp

pytest.importorskip("pyarrow")

CONFIG = {
    "window": {"start": "2025-01-01", "end": "2025-05-01"},
    "cities": {"aliases": {"Bengaluru": ["Bangalore"], "Pune": []}},
    "datasets": {
        "launch_sales": {
            "role": "feature",
            "geography": "city",
            "release_lag_days": 15,
            "unit": "units",
            "columns": {
                "period": "Relevant_Date",
                "geography": "City",
                "metrics": {"units_launched": "Launches", "units_sold": "Sales"},
            },
            "plausible": {"min": 0},
        },
        "core_industries": {
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
    },
}


@pytest.fixture
def project(tmp_path):
    config_path = tmp_path / "datasets.json"
    config_path.write_text(json.dumps(CONFIG))
    raw = tmp_path / "raw"
    (raw / "launch_sales").mkdir(parents=True)
    (raw / "core_industries").mkdir(parents=True)
    pd.DataFrame(
        {
            "Relevant_Date": ["2025-01-31", "2025-02-28", "2025-01-31", "2025-03-31"],
            "City": ["Bangalore", "Bengaluru", "Pune", "Atlantis"],
            "Launches": [100, 120, 50, 9],
            "Sales": [90, 110, -1, 9],
        }
    ).to_csv(raw / "launch_sales" / "launch_sales_2026-10-08.csv", index=False)
    pd.DataFrame(
        {
            "Relevant_Date": pd.date_range("2025-01-31", periods=4, freq="ME"),
            "Industry": "Cement",
            "Index_Value": [180.0, 185.0, 190.0, 188.0],
        }
    ).to_csv(raw / "core_industries" / "core_2026-10-08.csv", index=False)
    return config_path, raw, tmp_path / "processed"


def test_the_pipeline_writes_every_output(project):
    config_path, raw, out = project
    outputs = CleaningApp(config_path, raw, out).run()

    assert all(path.exists() for path in outputs.values())
    report = pd.read_csv(outputs["report"])
    assert set(report["rule"]) == {"unknown_geography", "implausible_value"}


def test_the_city_panel_is_a_complete_grid_with_gaps_left_missing(project):
    config_path, raw, out = project
    outputs = CleaningApp(config_path, raw, out).run()
    panel = pd.read_parquet(outputs["city_panel"])

    # Two cities x four months, Bangalore and Bengaluru merged into one.
    assert sorted(panel[schema.CITY].unique()) == ["Bengaluru", "Pune"]
    assert len(panel) == 8
    pune = panel.loc[panel[schema.CITY] == "Pune"].set_index(schema.PERIOD)
    assert pune.loc["2025-01-01", "units_launched"] == 50
    # The negative sales figure was screened, and nothing was invented.
    assert pd.isna(pune.loc["2025-01-01", "units_sold"])
    assert pune.loc["2025-02-01", ["units_launched", "units_sold"]].isna().all()


def test_national_series_carries_each_months_release_date(project):
    config_path, raw, out = project
    outputs = CleaningApp(config_path, raw, out).run()
    national = pd.read_parquet(outputs["national_series"]).set_index(schema.PERIOD)

    assert national["cement_index"].tolist() == [180.0, 185.0, 190.0, 188.0]
    assert national.loc["2025-03-01", "cement_index_available_at"] == pd.Timestamp(
        "2025-04-21"
    )


def test_existing_outputs_are_protected(project):
    config_path, raw, out = project
    CleaningApp(config_path, raw, out).run()
    with pytest.raises(FileExistsError):
        CleaningApp(config_path, raw, out).run()
