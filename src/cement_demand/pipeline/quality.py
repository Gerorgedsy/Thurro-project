"""Screen the normalized observations before anything is built from them.

Every rule removes or masks rows *and counts what it did*, so the effect of
cleaning can be stated in the write-up rather than assumed -- the same
discipline as project 1's screening report.

The rules are deliberately mechanical. They catch rows that cannot be used
(no city, no date, no number, outside the study window) and values that
cannot be true (negative unit counts, a 300 per cent lending rate). They do
not trim genuine extremes: a city whose sales doubled in a month is exactly
the signal this project is looking for.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from cement_demand.pipeline import schema

REPORT_COLUMNS: tuple[str, ...] = ("rule", schema.DATASET, schema.METRIC, "rows")


@dataclass(frozen=True)
class Window:
    """The half-open range of months the study uses."""

    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, periods: pd.Series) -> pd.Series:
        return (periods >= self.start) & (periods < self.end)

    @property
    def months(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.start, self.end, freq="MS", inclusive="left", name=schema.PERIOD
        )


@dataclass(frozen=True)
class ScreeningResult:
    frame: pd.DataFrame
    report: pd.DataFrame

    @property
    def removed(self) -> int:
        return int(self.report["rows"].sum()) if len(self.report) else 0


@dataclass
class Screener:
    """Apply every rule in order and record what each one removed."""

    window: Window
    # dataset -> {"min": ..., "max": ...}
    plausible: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def screen(self, frame: pd.DataFrame) -> ScreeningResult:
        records: list[dict[str, object]] = []
        frame = frame.copy()

        frame = self._drop(
            frame, records, "unknown_geography", frame[schema.GEOGRAPHY].isna()
        )
        frame = self._drop(
            frame, records, "unparseable_period", frame[schema.PERIOD].isna()
        )
        frame = self._drop(
            frame,
            records,
            "outside_window",
            ~self.window.contains(frame[schema.PERIOD]),
        )
        frame = self._drop(frame, records, "missing_value", frame[schema.VALUE].isna())
        frame = self._drop(
            frame, records, "implausible_value", self._implausible(frame)
        )

        # A key seen twice is most often a revised figure exported beside the
        # original. The later row in the export wins, and the count is
        # reported so a systematic duplication is noticed, not absorbed.
        duplicated = frame.duplicated(list(schema.OBSERVATION_KEY), keep="last")
        frame = self._drop(frame, records, "duplicate_key", duplicated)

        report = pd.DataFrame(records, columns=list(REPORT_COLUMNS))
        ordered = frame.sort_values(
            [schema.DATASET, schema.METRIC, schema.GEOGRAPHY, schema.PERIOD],
            kind="stable",
        ).reset_index(drop=True)
        return ScreeningResult(frame=ordered, report=report)

    def _implausible(self, frame: pd.DataFrame) -> pd.Series:
        outside = pd.Series(False, index=frame.index)
        for dataset, bounds in self.plausible.items():
            rows = frame[schema.DATASET] == dataset
            values = frame[schema.VALUE]
            if "min" in bounds:
                outside |= rows & (values < bounds["min"])
            if "max" in bounds:
                outside |= rows & (values > bounds["max"])
        return outside

    @staticmethod
    def _drop(
        frame: pd.DataFrame,
        records: list[dict[str, object]],
        rule: str,
        mask: pd.Series,
    ) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            counts = (
                frame.loc[mask]
                .groupby([schema.DATASET, schema.METRIC], sort=True)
                .size()
            )
            records.extend(
                {"rule": rule, schema.DATASET: d, schema.METRIC: m, "rows": int(n)}
                for (d, m), n in counts.items()
            )
        return frame.loc[~mask]
