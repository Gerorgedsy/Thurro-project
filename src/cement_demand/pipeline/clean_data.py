"""Clean the Thurro exports into the tables every model reads.

Run from the repository root with:

    PYTHONPATH=src python -m cement_demand.pipeline.clean_data

Inputs are CSV exports placed in ``data/raw/thurro/<dataset>/``. Outputs go
to ``data/processed/``:

- ``observations.parquet`` -- every screened row, in the long format of
  ``schema.py``, with both the month it describes and the day it became
  public;
- ``city_panel.parquet`` -- city-level metrics on a complete city x month
  grid, missing months left missing rather than filled;
- ``national_series.parquet`` -- national metrics (the cement index and
  lending rates), one row per month;
- ``screening_report.csv`` -- what each cleaning rule removed, by dataset.

This stage does not aggregate cities into a national signal. Equal-weighted
versus construction-share-weighted aggregation is the project's central
methodological question, so it belongs in the feature layer where both can
be built and compared, not baked in here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from cement_demand.pipeline import schema
from cement_demand.pipeline.cities import CityRegistry
from cement_demand.pipeline.quality import Screener, Window
from cement_demand.pipeline.sources import DatasetSource, build_sources

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.json"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "thurro"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Wide tables carry one column per metric, indexed by these.
CITY_PANEL_INDEX: tuple[str, ...] = (schema.GEOGRAPHY, schema.PERIOD)


def city_panel(observations: pd.DataFrame, window: Window) -> pd.DataFrame:
    """City metrics on the full city x month grid.

    Every city carries every month so that a gap is visible as a missing
    value. Filling it here would decide, silently, that a city with no
    report sold nothing -- a modeling choice, not a cleaning one.
    """
    city_rows = observations.loc[observations[schema.GEOGRAPHY_LEVEL] == schema.CITY]
    if city_rows.empty:
        return pd.DataFrame(columns=list(CITY_PANEL_INDEX))
    wide = city_rows.pivot_table(
        index=list(CITY_PANEL_INDEX),
        columns=schema.METRIC,
        values=schema.VALUE,
        aggfunc="last",
    )
    grid = pd.MultiIndex.from_product(
        [sorted(city_rows[schema.GEOGRAPHY].unique()), window.months],
        names=list(CITY_PANEL_INDEX),
    )
    wide = wide.reindex(grid)
    wide.columns.name = None
    return wide.reset_index().rename(columns={schema.GEOGRAPHY: schema.CITY})


def national_series(observations: pd.DataFrame, window: Window) -> pd.DataFrame:
    """National metrics, one row per month, plus each metric's release date."""
    national = observations.loc[observations[schema.GEOGRAPHY_LEVEL] == schema.NATIONAL]
    if national.empty:
        return pd.DataFrame(columns=[schema.PERIOD])
    values = national.pivot_table(
        index=schema.PERIOD, columns=schema.METRIC, values=schema.VALUE, aggfunc="last"
    )
    released = national.pivot_table(
        index=schema.PERIOD,
        columns=schema.METRIC,
        values=schema.AVAILABLE_AT,
        aggfunc="last",
    ).add_suffix("_available_at")
    wide = values.join(released).reindex(window.months)
    wide.columns.name = None
    return wide.reset_index()


class CleaningApp:
    """Load configuration and run every cleaning stage end to end."""

    filenames: ClassVar[dict[str, str]] = {
        "observations": "observations.parquet",
        "city_panel": "city_panel.parquet",
        "national_series": "national_series.parquet",
        "report": "screening_report.csv",
    }

    def __init__(self, config_path: Path, input_dir: Path, output_dir: Path) -> None:
        with config_path.open(encoding="utf-8") as config_file:
            self.config: dict[str, Any] = json.load(config_file)
        self.cities = CityRegistry.from_config(self.config["cities"]["aliases"])
        self.window = Window(
            start=pd.Timestamp(self.config["window"]["start"]),
            end=pd.Timestamp(self.config["window"]["end"]),
        )
        self.sources = build_sources(self.config)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def select(self, names: Sequence[str] | None) -> list[DatasetSource]:
        if names is None:
            return self.sources
        known = {source.name: source for source in self.sources}
        unknown = sorted(set(names).difference(known))
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}. Known: {sorted(known)}.")
        return [known[name] for name in names]

    def run(
        self, *, datasets: Sequence[str] | None = None, overwrite: bool = False
    ) -> dict[str, Path]:
        outputs = {key: self.output_dir / name for key, name in self.filenames.items()}
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{existing[0]} already exists; pass --overwrite to replace it."
            )

        frames = []
        for source in self.select(datasets):
            print(f"Normalizing {source.name}...", flush=True)
            raw = source.read(self.input_dir)
            if source.spec.geography_column:
                unknown = self.cities.unknown(raw[source.spec.geography_column])
                if unknown:
                    print(
                        f"  {len(unknown)} unrecognised city spelling(s), dropped "
                        f"until added to config: {unknown[:10]}",
                        flush=True,
                    )
            frames.append(source.normalize(raw, self.cities))

        plausible = {s.name: s.spec.plausible for s in self.sources if s.spec.plausible}
        screened = Screener(self.window, plausible).screen(
            pd.concat(frames, ignore_index=True)
        )
        print(f"Screening removed {screened.removed:,} row(s).", flush=True)
        self._summarize(screened.frame)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write(screened.frame.to_parquet, outputs["observations"])
        _write(
            city_panel(screened.frame, self.window).to_parquet, outputs["city_panel"]
        )
        _write(
            national_series(screened.frame, self.window).to_parquet,
            outputs["national_series"],
        )
        _write(lambda p: screened.report.to_csv(p, index=False), outputs["report"])
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}", flush=True)
        return outputs

    @staticmethod
    def _summarize(observations: pd.DataFrame) -> None:
        if observations.empty:
            print("No observations survived screening.", flush=True)
            return
        summary = observations.groupby(schema.DATASET).agg(
            first=(schema.PERIOD, "min"),
            last=(schema.PERIOD, "max"),
            geographies=(schema.GEOGRAPHY, "nunique"),
            rows=(schema.VALUE, "size"),
        )
        for dataset, row in summary.iterrows():
            print(
                f"  {dataset:<22s} {row['first']:%Y-%m} to {row['last']:%Y-%m}  "
                f"{int(row['geographies'])} geography(ies)  {int(row['rows']):,} rows",
                flush=True,
            )


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset to clean; repeat for several. Defaults to all configured.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    CleaningApp(args.config, args.input_dir, args.output_dir).run(
        datasets=args.dataset, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
