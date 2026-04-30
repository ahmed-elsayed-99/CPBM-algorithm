import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class SyntheticCommunity:

    STRATUM_PARAMS = {
        1: {"f_mean": 3.0, "s_mean": 0.12, "h_mean": 2.8,
            "e1_mean": -1.1, "e2_mean": -1.9, "tau_mean": 5.5,
            "buy_prob_base": 0.30, "n_frac": 0.25},
        2: {"f_mean": 6.0, "s_mean": 0.16, "h_mean": 2.2,
            "e1_mean": -0.9, "e2_mean": -1.5, "tau_mean": 3.5,
            "buy_prob_base": 0.45, "n_frac": 0.50},
        3: {"f_mean": 9.0, "s_mean": 0.20, "h_mean": 1.6,
            "e1_mean": -0.6, "e2_mean": -1.1, "tau_mean": 1.5,
            "buy_prob_base": 0.61, "n_frac": 0.25},
    }

    def __init__(self, n: int = 500, seed: int = 42, window_days: int = 90):
        self.n = n
        self.seed = seed
        self.window_days = window_days

    def generate(self) -> Dict:
        np.random.seed(self.seed)
        strata = np.random.choice(
            [1, 2, 3], size=self.n,
            p=[p["n_frac"] for p in self.STRATUM_PARAMS.values()]
        )
        records = []
        for i in range(self.n):
            s = strata[i]
            p = self.STRATUM_PARAMS[s]
            noise = 0.30
            f = max(0.1, np.random.normal(p["f_mean"], p["f_mean"] * noise))
            s_share = np.clip(np.random.normal(p["s_mean"], p["s_mean"] * noise), 0.01, 0.99)
            h = np.clip(np.random.normal(p["h_mean"], 0.5), 0.0, 3.58)
            e1 = np.random.normal(p["e1_mean"], 0.20)
            e2 = np.random.normal(p["e2_mean"], 0.30)
            tau = max(0.5, np.random.normal(p["tau_mean"], 1.0))
            c_max = np.random.normal(0.3 + 0.1 * (3 - s), 0.10)
            sigma_t = np.random.normal(15 - 3 * s, 3.0)
            npi = np.random.normal(50 + 10 * s, 15)
            base_prob = p["buy_prob_base"] + 0.01 * f + 0.05 * s_share
            label = int(np.random.random() < min(max(base_prob, 0.01), 0.99))
            records.append({
                "ind_id": i, "stratum": s,
                "f_score": f, "s_score": s_share,
                "sigma_timing": max(0.0, sigma_t),
                "e_low": e1, "e_high": e2,
                "c_max": c_max, "h_entropy": h,
                "tau": tau, "npi": npi,
                "label": label,
            })

        df = pd.DataFrame(records)
        transactions = self._generate_transactions(df)
        stratum_adoption = self._generate_stratum_adoption()
        adoption_history = self._generate_adoption_history()

        return {
            "phi_matrix": df[["f_score", "s_score", "sigma_timing",
                               "e_low", "e_high", "c_max", "h_entropy"]].values,
            "labels": df["label"].values,
            "strata": df["stratum"].values,
            "tau_values": df["tau"].values,
            "npi_values": df["npi"].values,
            "dataframe": df,
            "transactions": transactions,
            "stratum_adoption": stratum_adoption,
            "adoption_history": adoption_history,
        }

    def _generate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            n_txns = max(1, int(row["f_score"]))
            for _ in range(n_txns):
                day = np.random.randint(0, self.window_days)
                amount = np.random.lognormal(3.5 + 0.3 * row["stratum"], 0.5)
                total = amount / max(row["s_score"], 0.01)
                rows.append({
                    "ind_id": row["ind_id"],
                    "day": day,
                    "amount": round(amount, 2),
                    "total_spend": round(total, 2),
                    "category": "target",
                    "stratum": row["stratum"],
                })
        return pd.DataFrame(rows)

    def _generate_stratum_adoption(self) -> Dict:
        t = np.linspace(0, 90, 91)
        from scipy.integrate import odeint

        def bass_sol(t, p, q, M, M_val):
            def ode(y, t):
                F = y[0] / M
                return [(p + q * F) * (1 - F) * M]
            sol = odeint(ode, [0.0], t)
            return sol[:, 0]

        return {
            3: (t, bass_sol(t, 0.05, 0.45, 125, 125), 125.0),
            2: (t, bass_sol(t, 0.03, 0.35, 250, 250), 250.0),
            1: (t, bass_sol(t, 0.02, 0.25, 125, 125), 125.0),
        }

    def _generate_adoption_history(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 60, 61)
        adoption = (0.5 / 0.05) * (1 - np.exp(-0.05 * t)) * np.exp(-0.01 * t ** 2)
        adoption = adoption / adoption.max() * 0.44
        return t, adoption