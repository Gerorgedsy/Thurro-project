"""Read each Thurro export and express it in the canonical long format.

Run through ``clean_data``; this module only defines the sources.

Every export goes through the same lifecycle, implemented once in
:class:`DatasetSource`:

1. **locate** the newest CSV beneath ``data/raw/thurro/<dataset>/``;
2. **select** the rows the dataset is about (for example, only the cement
   line of the core-industries table);
3. **date** each row: the month it describes and the day it became public;
4. **place** it geographically (a canonical city, or India);
5. **melt** the configured metric columns into one row per metric.

Subclasses only state what differs: whether the geography is a city or the
whole country. The column names themselves live in ``config/datasets.json``,
not here, so a renamed column in a new export is a config edit rather than a
code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from cement_demand.pipeline import schema
from cement_demand.pipeline.cities import CityRegistry


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's entry in ``config/datasets.json``."""

    name: str
    role: str
    geography: str
    release_lag_days: int
    period_column: str
    metrics: Mapping[str, str]
    unit: str
    geography_column: str | None = None
    row_filter: Mapping[str, Any] | None = None
    plausible: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, entry: Mapping[str, Any]) -> DatasetSpec:
        columns = entry["columns"]
        if entry["geography"] not in schema.GEOGRAPHY_LEVELS:
            raise ValueError(
                f"{name}: geography must be one of {sorted(schema.GEOGRAPHY_LEVELS)}."
            )
        if not columns.get("metrics"):
            raise ValueError(f"{name}: at least one metric column is required.")
        return cls(
            name=name,
            role=entry["role"],
            geography=entry["geography"],
            release_lag_days=int(entry["release_lag_days"]),
            period_column=columns["period"],
            metrics=dict(columns["metrics"]),
            unit=entry["unit"],
            geography_column=columns.get("geography"),
            row_filter=columns.get("filter"),
            plausible=dict(entry.get("plausible", {})),
        )


def month_start(values: pd.Series) -> pd.Series:
    """Return each date as the first day of its month, timezone-naive.

    Sources disagree about which day stands for a month -- the first, the
    last, or a release date inside it. Flooring to the month makes them
    agree without guessing which convention each one used.
    """
    # ``format="mixed"`` parses each value on its own. Without it pandas infers
    # one format from the first row and silently turns every differently
    # formatted date into a missing value.
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.to_period("M").dt.to_timestamp()


def available_from(periods: pd.Series, release_lag_days: int) -> pd.Series:
    """The first day a month's figure could have been known.

    A figure for March cannot be read before March ends, so the lag is
    counted from the first day of the following month.
    """
    next_month = periods + pd.offsets.MonthBegin(1)
    return next_month + pd.Timedelta(days=release_lag_days)


class DatasetSource(ABC):
    """Shared read-and-normalize lifecycle for one Thurro export."""

    geography: ClassVar[str]

    def __init__(self, spec: DatasetSpec) -> None:
        if spec.geography != self.geography:
            raise ValueError(
                f"{type(self).__name__} expects {self.geography} data, but "
                f"{spec.name} is configured as {spec.geography}."
            )
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def locate(self, input_dir: Path) -> Path:
        """Return the newest export for this dataset.

        Export filenames are expected to sort by date (for example
        ``launch_sales_2026-10-08.csv``), so the last one is the newest.
        """
        candidates = sorted((input_dir / self.name).glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No export for {self.name} beneath {input_dir / self.name}. "
                "Export it from Thurro as CSV and place it there."
            )
        return candidates[-1]

    def read(self, input_dir: Path) -> pd.DataFrame:
        return pd.read_csv(self.locate(input_dir), low_memory=False)

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the rows this dataset is about, if the config narrows them."""
        if not self.spec.row_filter:
            return frame
        column = self.spec.row_filter["column"]
        wanted = str(self.spec.row_filter["equals"]).casefold()
        self._require(frame, [column])
        keep = frame[column].astype("string").str.strip().str.casefold() == wanted
        return frame.loc[keep.fillna(False)]

    @abstractmethod
    def place(self, frame: pd.DataFrame, cities: CityRegistry) -> pd.Series:
        """Return the canonical geography of every row."""

    def normalize(self, frame: pd.DataFrame, cities: CityRegistry) -> pd.DataFrame:
        """Express the raw export in the canonical long format."""
        self._require(frame, [self.spec.period_column, *self.spec.metrics.values()])
        frame = self.select(frame)

        periods = month_start(frame[self.spec.period_column])
        identity = pd.DataFrame(
            {
                schema.GEOGRAPHY: self.place(frame, cities),
                schema.PERIOD: periods,
                schema.AVAILABLE_AT: available_from(
                    periods, self.spec.release_lag_days
                ),
            },
            index=frame.index,
        )

        long_rows = []
        for metric, column in self.spec.metrics.items():
            long_rows.append(
                identity.assign(
                    **{
                        schema.METRIC: metric,
                        schema.VALUE: pd.to_numeric(frame[column], errors="coerce"),
                    }
                )
            )
        long = pd.concat(long_rows, ignore_index=True).assign(
            **{
                schema.DATASET: self.name,
                schema.GEOGRAPHY_LEVEL: self.geography,
                schema.UNIT: self.spec.unit,
            }
        )
        return long.loc[:, list(schema.OBSERVATION_COLUMNS)]

    def load(self, input_dir: Path, cities: CityRegistry) -> pd.DataFrame:
        return self.normalize(self.read(input_dir), cities)

    def _require(self, frame: pd.DataFrame, columns: list[str]) -> None:
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise KeyError(
                f"{self.name}: export is missing column(s) {missing}. Update "
                f"config/datasets.json to the export's names; it has "
                f"{sorted(frame.columns)}."
            )


class CitySource(DatasetSource):
    """A dataset reported per city."""

    geography = schema.CITY

    def place(self, frame: pd.DataFrame, cities: CityRegistry) -> pd.Series:
        if not self.spec.geography_column:
            raise ValueError(f"{self.name}: a city dataset needs a geography column.")
        self._require(frame, [self.spec.geography_column])
        return cities.resolve(frame[self.spec.geography_column])


class NationalSource(DatasetSource):
    """A dataset reported for India as a whole."""

    geography = schema.NATIONAL

    def place(self, frame: pd.DataFrame, cities: CityRegistry) -> pd.Series:
        del cities
        return pd.Series(schema.INDIA, index=frame.index, dtype="string")


class LaunchSalesSource(CitySource):
    """New residential launches and sales, by city: the leading signal."""


class ProjectRegistrationsSource(CitySource):
    """RERA project registrations, by city: an earlier-stage pipeline signal."""


class LendingRatesSource(NationalSource):
    """Bank lending rates: the control that separates credit from demand."""


class CoreIndustriesSource(NationalSource):
    """The cement line of the Index of Eight Core Industries: the target."""


SOURCES: Mapping[str, type[DatasetSource]] = {
    "launch_sales": LaunchSalesSource,
    "project_registrations": ProjectRegistrationsSource,
    "lending_rates": LendingRatesSource,
    "core_industries": CoreIndustriesSource,
}


def build_sources(config: Mapping[str, Any]) -> list[DatasetSource]:
    """Instantiate one source per configured dataset."""
    sources: list[DatasetSource] = []
    for name, entry in config["datasets"].items():
        if name not in SOURCES:
            raise ValueError(f"No source class for dataset {name!r}.")
        sources.append(SOURCES[name](DatasetSpec.from_config(name, entry)))
    return sources
