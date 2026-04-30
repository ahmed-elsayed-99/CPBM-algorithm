import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

from cpbm.models.xgb_layer import XGBLayer
from cpbm.models.lstm_layer import LSTMLayer
from cpbm.models.gat_layer import GATLayer


class CPBMEnsemble:

    WEIGHTS_DEFAULT = (0.45, 0.30, 0.25)

    def __init__(
        self,
        weights: Tuple[float, float, float] = WEIGHTS_DEFAULT,
        test_size: float = 0.20,
        seq_len: int = 6,
        random_state: int = 42,
        xgb_kwargs: Optional[Dict] = None,
        lstm_kwargs: Optional[Dict] = None,
        gat_kwargs: Optional[Dict] = None,
    ):
        self.weights = weights
        self.test_size = test_size
        self.seq_len = seq_len
        self.random_state = random_state

        self.xgb = XGBLayer(**(xgb_kwargs or {}))
        self.lstm = LSTMLayer(**(lstm_kwargs or {}))
        self.gat = GATLayer(**(gat_kwargs or {}))

        self.is_fitted_ = False
        self.eval_results_: Dict = {}
        self.feature_names_: Optional[List[str]] = None

    def _prepare_tabular(
        self,
        Phi: np.ndarray,
        tau: float,
        npi: float,
        cluster_labels: np.ndarray,
    ) -> np.ndarray:
        tau_col = np.full((len(Phi), 1), tau)
        npi_col = np.full((len(Phi), 1), npi)
        cluster_col = cluster_labels.reshape(-1, 1)
        return np.hstack([Phi, tau_col, npi_col, cluster_col])

    def _prepare_sequences(self, Phi: np.ndarray) -> np.ndarray:
        n, d = Phi.shape
        sequences = []
        for i in range(n):
            if i < self.seq_len:
                pad = np.zeros((self.seq_len - i - 1, d))
                seq = np.vstack([pad, Phi[: i + 1]])
            else:
                seq = Phi[i - self.seq_len + 1: i + 1]
            sequences.append(seq)
        return np.array(sequences)

    def fit(
        self,
        Phi: np.ndarray,
        labels: np.ndarray,
        tau: float = 0.0,
        npi: float = 100.0,
        cluster_labels: Optional[np.ndarray] = None,
        G=None,
        diffusion_params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "CPBMEnsemble":
        if cluster_labels is None:
            cluster_labels = np.zeros(len(Phi), dtype=int)
        self.feature_names_ = feature_names

        X_tab = self._prepare_tabular(Phi, tau, npi, cluster_labels)
        X_seq = self._prepare_sequences(Phi)

        idx_tr, idx_va = train_test_split(
            np.arange(len(labels)),
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state,
        )

        X_tab_tr, X_tab_va = X_tab[idx_tr], X_tab[idx_va]
        X_seq_tr, X_seq_va = X_seq[idx_tr], X_seq[idx_va]
        y_tr, y_va = labels[idx_tr], labels[idx_va]

        fn = (feature_names or [f"f{i}" for i in range(Phi.shape[1])]) + ["tau", "npi", "cluster"]
        self.xgb.fit(X_tab_tr, y_tr, X_tab_va, y_va, feature_names=fn)
        self.lstm.fit(X_seq_tr, y_tr, X_seq_va, y_va)

        if G is not None:
            Phi_gat = Phi
            from torch_geometric.data import Data
            gat_data_tr = self.gat.build_pyg_data(Phi[idx_tr], G, y_tr)
            gat_data_va = self.gat.build_pyg_data(Phi[idx_va], G, y_va)
            self.gat.fit(gat_data_tr, gat_data_va)
            self._use_gat = True
        else:
            self._use_gat = False

        self.idx_va_ = idx_va
        self.X_tab_va_ = X_tab_va
        self.X_seq_va_ = X_seq_va
        self.y_va_ = y_va
        self.is_fitted_ = True

        p_xgb = self.xgb.predict_proba(X_tab_va)
        p_lstm = self.lstm.predict_proba(X_seq_va)

        if self._use_gat:
            from torch_geometric.data import Data
            gat_va = self.gat.build_pyg_data(Phi[idx_va], G, y_va)
            p_gat = self.gat.predict_proba(gat_va)
        else:
            p_gat = p_xgb * 0.0

        w = self.weights
        if not self._use_gat:
            w_adj = (w[0] + w[2] * 0.5, w[1] + w[2] * 0.5, 0.0)
        else:
            w_adj = w

        p_ens = w_adj[0] * p_xgb + w_adj[1] * p_lstm + w_adj[2] * p_gat

        self.eval_results_ = {
            "auc_xgb": float(roc_auc_score(y_va, p_xgb)),
            "auc_lstm": float(roc_auc_score(y_va, p_lstm)),
            "auc_ensemble": float(roc_auc_score(y_va, p_ens)),
            "ap_ensemble": float(average_precision_score(y_va, p_ens)),
        }
        return self

    def predict_proba(
        self,
        Phi: np.ndarray,
        tau: float = 0.0,
        npi: float = 100.0,
        cluster_labels: Optional[np.ndarray] = None,
        G=None,
    ) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit first.")
        if cluster_labels is None:
            cluster_labels = np.zeros(len(Phi), dtype=int)

        X_tab = self._prepare_tabular(Phi, tau, npi, cluster_labels)
        X_seq = self._prepare_sequences(Phi)

        p_xgb = self.xgb.predict_proba(X_tab)
        p_lstm = self.lstm.predict_proba(X_seq)

        if self._use_gat and G is not None:
            dummy_y = np.zeros(len(Phi))
            data = self.gat.build_pyg_data(Phi, G, dummy_y)
            p_gat = self.gat.predict_proba(data)
            w = self.weights
        else:
            p_gat = np.zeros(len(Phi))
            w = (self.weights[0] + self.weights[2] * 0.5,
                 self.weights[1] + self.weights[2] * 0.5, 0.0)

        return w[0] * p_xgb + w[1] * p_lstm + w[2] * p_gat

    def evaluate(
        self,
        Phi: np.ndarray,
        labels: np.ndarray,
        tau: float = 0.0,
        npi: float = 100.0,
        cluster_labels: Optional[np.ndarray] = None,
        G=None,
        threshold: float = 0.50,
    ) -> Dict:
        proba = self.predict_proba(Phi, tau, npi, cluster_labels, G)
        preds = (proba >= threshold).astype(int)
        report = classification_report(labels, preds, output_dict=True)
        return {
            "auc": float(roc_auc_score(labels, proba)),
            "average_precision": float(average_precision_score(labels, proba)),
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"] if "1" in report else 0.0,
            "recall": report["1"]["recall"] if "1" in report else 0.0,
            "f1": report["1"]["f1-score"] if "1" in report else 0.0,
            "classification_report": report,
        }