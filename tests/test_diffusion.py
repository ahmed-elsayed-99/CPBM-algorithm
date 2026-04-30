import numpy as np
import pytest
from cpbm.core.diffusion import DiffusionLayer


def test_fit_returns_params():
    dl = DiffusionLayer()
    t = np.linspace(0, 60, 61)
    adoption = 0.4 * (1 - np.exp(-0.05 * t)) * np.exp(-0.005 * t**2)
    dl.fit(t, adoption)
    assert dl.params_ is not None
    assert "D" in dl.params_
    assert dl.params_["D"] >= 0


def test_build_graph_node_count():
    dl = DiffusionLayer(random_state=42)
    G = dl.build_graph(n_nodes=50)
    assert G.number_of_nodes() == 50


def test_simulate_returns_dataframe():
    dl = DiffusionLayer(random_state=42)
    dl.params_ = {"D": 0.15, "alpha": 0.5, "beta": 0.04}
    dl.build_graph(50)
    proba = np.random.uniform(0.2, 0.8, 50)
    result = dl.simulate(50, [0, 1, 2], proba, steps=10)
    assert len(result) == 10
    assert "penetration_pct" in result.columns


def test_hub_seed_candidates():
    dl = DiffusionLayer(random_state=42)
    dl.build_graph(100)
    hubs = dl.hub_seed_candidates(top_k=5)
    assert len(hubs) == 5