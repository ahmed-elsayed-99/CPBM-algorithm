import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class RegionalAggregator:

    def __init__(self):
        self.regional_features_: Optional[np.ndarray] = None
        self.community_records_: List[Dict] = []

    def add_community(
        self,
        geo_code: str,
        npi: float,
        D: float,
        tau_min: float,
        tau_max: float,
        population: int,
    ) -> None:
        self.community_records_.append({
            "geo_code": geo_code,
            "npi": npi,
            "D": D,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "population": population,
        })

    def aggregate(self, macro_signals: Optional[Dict[str, float]] = None) -> np.ndarray:
        if len(self.community_records_) == 0:
            raise RuntimeError("No communities added. Call add_community first.")

        npi_arr = np.array([c["npi"] for c in self.community_records_])
        D_arr = np.array([c["D"] for c in self.community_records_])
        tau_min_arr = np.array([c["tau_min"] for c in self.community_records_])
        tau_max_arr = np.array([c["tau_max"] for c in self.community_records_])
        pop_arr = np.array([c["population"] for c in self.community_records_], dtype=float)
        w = pop_arr / pop_arr.sum()

        x = [
            float(np.average(npi_arr, weights=w)),
            float(npi_arr.std()),
            float(np.average(D_arr, weights=w)),
            float(tau_min_arr.min()),
            float(tau_max_arr.max() - tau_min_arr.min()),
        ]

        if macro_signals:
            for key in ["cpi_growth", "unemployment", "gdp_growth",
                        "consumer_confidence", "median_income_growth"]:
                x.append(macro_signals.get(key, 0.0))

        self.regional_features_ = np.array(x)
        return self.regional_features_

    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.community_records_)

    def reset(self) -> None:
        self.community_records_ = []
        self.regional_features_ = None