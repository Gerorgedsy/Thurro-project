import pandas as pd
import pytest

from cement_demand.pipeline.cities import CityRegistry

REGISTRY = CityRegistry.from_config(
    {"Bengaluru": ["Bangalore"], "Gurugram": ["Gurgaon"], "Pune": []}
)


def test_spellings_resolve_to_one_canonical_city():
    names = pd.Series(["Bangalore", " bengaluru ", "BANGALORE", "Gurgaon", "Pune"])
    assert REGISTRY.resolve(names).tolist() == [
        "Bengaluru",
        "Bengaluru",
        "Bengaluru",
        "Gurugram",
        "Pune",
    ]


def test_unknown_spellings_stay_missing_and_are_reported():
    names = pd.Series(["Pune", "Atlantis", None])
    resolved = REGISTRY.resolve(names)
    assert resolved.isna().tolist() == [False, True, True]
    assert REGISTRY.unknown(names) == ["Atlantis"]


def test_an_alias_cannot_belong_to_two_cities():
    with pytest.raises(ValueError, match="one city"):
        CityRegistry.from_config({"Delhi": ["NCR"], "Noida": ["NCR"]})
