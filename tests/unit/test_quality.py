import pandas as pd

from cement_demand.pipeline import schema
from cement_demand.pipeline.quality import Screener, Window

WINDOW = Window(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01"))


def _row(geography="Pune", period="2025-03-01", value=10.0, dataset="launch_sales"):
    return {
        schema.DATASET: dataset,
        schema.METRIC: "units_sold",
        schema.GEOGRAPHY_LEVEL: schema.CITY,
        schema.GEOGRAPHY: geography,
        schema.PERIOD: pd.Timestamp(period) if period else pd.NaT,
        schema.AVAILABLE_AT: pd.Timestamp("2025-04-16"),
        schema.VALUE: value,
        schema.UNIT: "units",
    }


def _screen(rows, plausible=None):
    return Screener(WINDOW, plausible or {}).screen(pd.DataFrame(rows))


def _removed(result, rule):
    report = result.report
    return int(report.loc[report["rule"] == rule, "rows"].sum())


def test_clean_rows_pass_untouched():
    result = _screen([_row(), _row(period="2025-04-01")])
    assert len(result.frame) == 2
    assert result.removed == 0


def test_each_rule_removes_and_counts_its_rows():
    result = _screen(
        [
            _row(),
            _row(geography=None),
            _row(period=None),
            _row(period="2024-12-01"),
            _row(period="2025-07-01"),
            _row(period="2025-05-01", value=None),
            _row(period="2025-06-01", value=-5.0),
        ],
        plausible={"launch_sales": {"min": 0}},
    )
    assert len(result.frame) == 1
    assert _removed(result, "unknown_geography") == 1
    assert _removed(result, "unparseable_period") == 1
    assert _removed(result, "outside_window") == 2
    assert _removed(result, "missing_value") == 1
    assert _removed(result, "implausible_value") == 1


def test_the_end_of_the_window_is_excluded():
    # Half-open, as in project 1: the end month belongs to the forecast.
    assert WINDOW.months[-1] == pd.Timestamp("2025-06-01")
    assert len(WINDOW.months) == 6


def test_a_duplicated_key_keeps_the_later_row():
    result = _screen([_row(value=10.0), _row(value=12.0)])
    assert result.frame[schema.VALUE].tolist() == [12.0]
    assert _removed(result, "duplicate_key") == 1


def test_bounds_apply_only_to_their_own_dataset():
    result = _screen(
        [
            _row(value=40.0, dataset="lending_rates"),
            _row(value=40.0, dataset="launch_sales"),
        ],
        plausible={"lending_rates": {"min": 0, "max": 25}},
    )
    assert result.frame[schema.DATASET].tolist() == ["launch_sales"]
