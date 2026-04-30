import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings("ignore")


class SignatureExtractor:

    def __init__(
        self,
        window_days: int = 90,
        price_sentinels: int = 3,
        timing_bins: int = 12,
        k_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
    ):
        self.window_days = window_days
        self.price_sentinels = price_sentinels
        self.timing_bins = timing_bins
        self.k_range = k_range
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.kmeans_ = None
        self.k_star_ = None
        self.silhouette_scores_: Dict[int, float] = {}
        self.feature_names_ = [
            "f_score",
            "s_score",
            "sigma_timing",
            "e_low",
            "e_high",
            "c_max",
            "h_entropy",
        ]

    def _purchase_frequency(self, txn_df: pd.DataFrame) -> float:
        return len(txn_df) / max(self.window_days, 1)

    def _expenditure_share(self, txn_df: pd.DataFrame) -> float:
        cat_spend = txn_df["amount"].sum()
        total_spend = txn_df["total_spend"].sum()
        return float(cat_spend / (total_spend + 1e-12))

    def _timing_entropy(self, timestamps: np.ndarray) -> float:
        if len(timestamps) == 0:
            return 0.0
        counts, _ = np.histogram(timestamps, bins=self.timing_bins,
                                  range=(0, self.window_days))
        probs = counts / (counts.sum() + 1e-12)
        return float(entropy(probs, base=2))

    def _timing_std(self, timestamps: np.ndarray) -> float:
        if len(timestamps) < 2:
            return 0.0
        gaps = np.diff(np.sort(timestamps))
        return float(np.std(gaps)) if len(gaps) > 0 else 0.0

    def _arc_elasticity(
        self,
        prices: List[float],
        quantities: List[float],
    ) -> Tuple[float, float]:
        if len(prices) < 3 or len(quantities) < 3:
            return -1.0, -1.5
        e = []
        for k in range(len(prices) - 1):
            dq = (quantities[k + 1] - quantities[k]) / (quantities[k] + 1e-12)
            dp = (prices[k + 1] - prices[k]) / (prices[k] + 1e-12)
            e.append(float(dq / (dp + 1e-12)))
        return e[0], e[1]

    def _compensation_max(
        self,
        category_shares: pd.DataFrame,
        target_col: str,
    ) -> float:
        if target_col not in category_shares.columns:
            return 0.0
        if len(category_shares) < 3:
            return 0.0
        monthly_changes = category_shares.pct_change().dropna()
        if monthly_changes.shape[0] < 2:
            return 0.0
        corr = monthly_changes.corr()
        if target_col not in corr.columns:
            return 0.0
        comp = -corr[target_col].drop(target_col, errors="ignore")
        return float(comp.max()) if len(comp) > 0 else 0.0

    def _build_phi(
        self,
        txn_df: pd.DataFrame,
        prices: List[float],
        quantities: List[float],
        category_shares: pd.DataFrame,
        target_col: str,
    ) -> np.ndarray:
        f = self._purchase_frequency(txn_df)
        s = self._expenditure_share(txn_df)
        timestamps = txn_df["day"].values if "day" in txn_df.columns else np.array([])
        sigma_t = self._timing_std(timestamps)
        h = self._timing_entropy(timestamps)
        e_low, e_high = self._arc_elasticity(prices, quantities)
        c_max = self._compensation_max(category_shares, target_col)
        return np.array([f, s, sigma_t, e_low, e_high, c_max, h])

    def fit_transform(
        self,
        transactions: pd.DataFrame,
        prices: Optional[List[float]] = None,
        quantities: Optional[Dict[str, List[float]]] = None,
        category_shares: Optional[pd.DataFrame] = None,
        target_col: str = "target",
    ) -> np.ndarray:
        if prices is None:
            price_range = (
                transactions["amount"].quantile(0.10),
                transactions["amount"].quantile(0.50),
                transactions["amount"].quantile(0.90),
            )
            prices = list(price_range)

        individuals = transactions["ind_id"].unique()
        Phi = []

        for ind in individuals:
            ind_txn = transactions[transactions["ind_id"] == ind].copy()
            if "total_spend" not in ind_txn.columns:
                ind_txn["total_spend"] = ind_txn["amount"].sum()

            q_ind = quantities.get(str(ind), [5, 3, 1]) if quantities else [5, 3, 1]
            cs = category_shares if category_shares is not None else pd.DataFrame()

            phi = self._build_phi(ind_txn, prices, q_ind, cs, target_col)
            Phi.append(phi)

        Phi = np.array(Phi)
        Phi_scaled = self.scaler_.fit_transform(Phi)

        k_range = range(self.k_range[0], self.k_range[1])
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(Phi_scaled)
            if len(np.unique(labels)) < k:
                self.silhouette_scores_[k] = -1.0
            else:
                self.silhouette_scores_[k] = silhouette_score(Phi_scaled, labels)

        self.k_star_ = max(self.silhouette_scores_, key=self.silhouette_scores_.get)
        self.kmeans_ = KMeans(
            n_clusters=self.k_star_, random_state=self.random_state, n_init=10
        )
        cluster_labels = self.kmeans_.fit_predict(Phi_scaled)

        self.Phi_ = Phi
        self.Phi_scaled_ = Phi_scaled
        self.cluster_labels_ = cluster_labels
        self.individuals_ = individuals

        return Phi

    def transform(self, transactions: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("Call fit_transform first.")
        return self.fit_transform(transactions, **kwargs)

    def get_cluster_profiles(self) -> pd.DataFrame:
        if self.Phi_ is None:
            raise RuntimeError("Call fit_transform first.")
        df = pd.DataFrame(self.Phi_, columns=self.feature_names_)
        df["cluster"] = self.cluster_labels_
        return df.groupby("cluster").mean().round(4)

    def get_signature_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.Phi_, columns=self.feature_names_)
        df["ind_id"] = self.individuals_
        df["cluster"] = self.cluster_labels_
        return df