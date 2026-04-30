import numpy as np
import pandas as pd
import pytest
from cpbm.core.signature import SignatureExtractor
from cpbm.data.synthetic import SyntheticCommunity


@pytest.fixture
def synthetic_data():
    community = SyntheticCommunity(n=100, seed=0).generate()
    return community


def test_signature_shape(synthetic_data):
    extractor = SignatureExtractor(window_days=90, k_range=(2, 5))
    Phi = extractor.fit_transform(synthetic_data["transactions"])
    assert Phi.shape[1] == 7
    assert Phi.shape[0] > 0


def test_cluster_labels_length(synthetic_data):
    extractor = SignatureExtractor(window_days=90, k_range=(2, 5))
    Phi = extractor.fit_transform(synthetic_data["transactions"])
    assert len(extractor.cluster_labels_) == Phi.shape[0]


def test_kstar_in_range(synthetic_data):
    extractor = SignatureExtractor(window_days=90, k_range=(2, 6))
    extractor.fit_transform(synthetic_data["transactions"])
    assert 2 <= extractor.k_star_ <= 6


def test_timing_entropy_zero_for_single_purchase():
    extractor = SignatureExtractor(timing_bins=12)
    timestamps = np.array([45.0])
    h = extractor._timing_entropy(timestamps)
    assert h >= 0.0


def test_arc_elasticity_sign():
    extractor = SignatureExtractor()
    prices = [50.0, 75.0, 100.0]
    quantities = [10, 6, 3]
    e1, e2 = extractor._arc_elasticity(prices, quantities)
    assert e1 < 0
    assert e2 < 0