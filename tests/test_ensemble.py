import numpy as np
import pytest
from cpbm.models.ensemble import CPBMEnsemble
from cpbm.data.synthetic import SyntheticCommunity


@pytest.fixture
def small_community():
    return SyntheticCommunity(n=150, seed=1).generate()


def test_ensemble_fit_returns_eval(small_community):
    Phi = small_community["phi_matrix"]
    labels = small_community["labels"]
    model = CPBMEnsemble(random_state=42)
    model.fit(Phi, labels)
    assert "auc_ensemble" in model.eval_results_
    assert model.eval_results_["auc_ensemble"] > 0.5


def test_ensemble_predict_proba_shape(small_community):
    Phi = small_community["phi_matrix"]
    labels = small_community["labels"]
    model = CPBMEnsemble(random_state=42)
    model.fit(Phi, labels)
    proba = model.predict_proba(Phi)
    assert proba.shape == (len(Phi),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_ensemble_evaluate_keys(small_community):
    Phi = small_community["phi_matrix"]
    labels = small_community["labels"]
    model = CPBMEnsemble(random_state=42)
    model.fit(Phi, labels)
    result = model.evaluate(Phi, labels)
    for key in ["auc", "accuracy", "f1", "precision", "recall"]:
        assert key in result