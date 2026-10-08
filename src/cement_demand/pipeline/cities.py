"""Resolve the many spellings of an Indian city to one canonical name.

City-level aggregation is the methodological heart of the project, and it is
only as good as the city key. "Bangalore" and "Bengaluru" counted as two
markets would split one market's sales in half and give it two votes in an
equal-weighted index. The registry therefore fails loudly: a spelling it has
not been told about is reported rather than passed through.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd


def _key(name: str) -> str:
    """Case- and whitespace-insensitive lookup key."""
    return re.sub(r"\s+", " ", name).strip().casefold()


@dataclass(frozen=True)
class CityRegistry:
    """Map every known spelling to its canonical city name."""

    lookup: Mapping[str, str]

    @classmethod
    def from_config(cls, aliases: Mapping[str, Sequence[str]]) -> CityRegistry:
        lookup: dict[str, str] = {}
        for canonical, spellings in aliases.items():
            for spelling in (canonical, *spellings):
                key = _key(spelling)
                existing = lookup.get(key)
                if existing is not None and existing != canonical:
                    raise ValueError(
                        f"{spelling!r} is listed under both {existing!r} and "
                        f"{canonical!r}; an alias must name one city."
                    )
                lookup[key] = canonical
        return cls(lookup=lookup)

    @property
    def cities(self) -> list[str]:
        return sorted(set(self.lookup.values()))

    def resolve(self, names: pd.Series) -> pd.Series:
        """Return canonical names, with unknown spellings left missing."""
        keys = names.astype("string").map(lambda name: _key(name), na_action="ignore")
        return keys.map(self.lookup).astype("string")

    def unknown(self, names: pd.Series) -> list[str]:
        """The distinct spellings this registry cannot resolve, for reporting."""
        present = names.dropna().astype("string")
        resolved = self.resolve(present)
        return sorted(present[resolved.isna()].unique().tolist())
