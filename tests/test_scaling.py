import numpy as np
import pytest
from cpbm.scaling.aggregator import RegionalAggregator


def test_aggregate_basic():
    agg = RegionalAggregator()
    for i in range(3):
        agg.add_community(f"GEO_{i}", npi=100.0 + i*10,
                          D=0.15, tau_min=7.0, tau_max=21.0, population=300+i*50)
    features = agg.aggregate()
    assert features.shape[0] == 5


def test_aggregate_with_macro():
    agg = RegionalAggregator()
    agg.add_community("GEO_0", npi=120.0, D=0.12, tau_min=5.0, tau_max=18.0, population=400)
    macro = {"cpi_growth": 0.08, "unemployment": 0.10, "gdp_growth": 0.03}
    features = agg.aggregate(macro)
    assert features.shape[0] == 8


def test_aggregate_empty_raises():
    agg = RegionalAggregator()
    with pytest.raises(RuntimeError):
        agg.aggregate()


def test_reset_clears_records():
    agg = RegionalAggregator()
    agg.add_community("GEO_0", npi=100.0, D=0.15, tau_min=7.0, tau_max=21.0, population=300)
    agg.reset()
    assert len(agg.community_records_) == 0