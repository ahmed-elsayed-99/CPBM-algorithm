import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import curve_fit
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class DiffusionLayer:

    def __init__(
        self,
        D_init: float = 0.15,
        alpha_init: float = 0.50,
        beta_init: float = 0.05,
        gamma: float = 0.01,
        graph_model: str = "barabasi_albert",
        ba_m: int = 2,
        random_state: int = 42,
    ):
        self.D_init = D_init
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.gamma = gamma
        self.graph_model = graph_model
        self.ba_m = ba_m
        self.random_state = random_state
        self.params_: Optional[Dict[str, float]] = None
        self.G_: Optional[nx.Graph] = None

    @staticmethod
    def _scalar_diffusion(t: np.ndarray, D: float, alpha: float, beta: float) -> np.ndarray:
        return (alpha / (beta + 1e-12)) * (1 - np.exp(-beta * t)) * np.exp(-D * t ** 2)

    def fit(
        self,
        time_axis: np.ndarray,
        adoption_curve: np.ndarray,
    ) -> "DiffusionLayer":
        p0 = [self.D_init, self.alpha_init, self.beta_init]
        bounds = (0, [2.0, 10.0, 2.0])
        try:
            popt, _ = curve_fit(
                self._scalar_diffusion,
                time_axis,
                adoption_curve,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            self.params_ = {"D": popt[0], "alpha": popt[1], "beta": popt[2]}
        except RuntimeError:
            self.params_ = {"D": self.D_init, "alpha": self.alpha_init, "beta": self.beta_init}
        return self

    def build_graph(self, n_nodes: int, edge_list: Optional[List[Tuple]] = None) -> nx.Graph:
        np.random.seed(self.random_state)
        if edge_list is not None:
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            for u, v, w in edge_list:
                G.add_edge(u, v, weight=w)
        elif self.graph_model == "barabasi_albert":
            G = nx.barabasi_albert_graph(n_nodes, self.ba_m, seed=self.random_state)
            for u, v in G.edges():
                G[u][v]["weight"] = np.random.uniform(0.3, 1.0)
        else:
            G = nx.erdos_renyi_graph(n_nodes, 0.05, seed=self.random_state)
            for u, v in G.edges():
                G[u][v]["weight"] = np.random.uniform(0.3, 1.0)
        self.G_ = G
        return G

    def simulate(
        self,
        n_total: int,
        seed_nodes: List[int],
        purchase_proba: np.ndarray,
        steps: int = 60,
        D: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> pd.DataFrame:
        if D is None:
            D = self.params_["D"] if self.params_ else self.D_init
        if beta is None:
            beta = self.params_["beta"] if self.params_ else self.beta_init

        np.random.seed(self.random_state)

        if self.G_ is None:
            self.build_graph(n_total)

        G = self.G_
        active = set(seed_nodes)
        records = []

        for step in range(1, steps + 1):
            new_active = set()
            for node in range(n_total):
                if node in active:
                    continue
                if G.degree(node) == 0:
                    continue
                nbr_pressure = sum(
                    G[node][nbr].get("weight", 1.0)
                    for nbr in G.neighbors(node)
                    if nbr in active
                ) / (G.degree(node) + 1e-9)
                p_adopt = (1.0 - np.exp(-D * nbr_pressure)) * purchase_proba[node]
                if np.random.random() < p_adopt:
                    new_active.add(node)

            decayed = {n for n in active if np.random.random() < beta}
            active = (active | new_active) - decayed

            records.append(
                {
                    "day": step,
                    "n_active": len(active),
                    "penetration_pct": 100.0 * len(active) / n_total,
                    "new_adopters": len(new_active),
                    "decayed": len(decayed),
                }
            )

        return pd.DataFrame(records)

    def predict_curve(self, time_axis: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Call fit first.")
        return self._scalar_diffusion(
            time_axis,
            self.params_["D"],
            self.params_["alpha"],
            self.params_["beta"],
        )

    def hub_seed_candidates(self, top_k: int = 10) -> List[int]:
        if self.G_ is None:
            raise RuntimeError("Call build_graph first.")
        centrality = nx.degree_centrality(self.G_)
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
        return sorted_nodes[:top_k]