import numpy as np
import pandas as pd

from cpbm.data.synthetic import SyntheticCommunity
from cpbm.core.diffusion import DiffusionLayer
from cpbm.core.npi import NPICalculator
from cpbm.core.stratum import StratumGradient
from cpbm.scaling.aggregator import RegionalAggregator


GEO_CODES = ["GEO_A", "GEO_B", "GEO_C", "GEO_D", "GEO_E"]
COMMUNITY_SIZES = [300, 450, 500, 380, 420]


def process_community(geo_code: str, n: int, seed: int):
    community = SyntheticCommunity(n=n, seed=seed).generate()
    Phi = community["phi_matrix"]
    transactions = community["transactions"]

    diffusion = DiffusionLayer(random_state=seed)
    t_h, a_h = community["adoption_history"]
    diffusion.fit(t_h, a_h)
    D = diffusion.params_["D"]

    stratum = StratumGradient()
    stratum.fit(community["stratum_adoption"])
    tau_vals = list(stratum.tau_.values())
    tau_min = float(np.nanmin(tau_vals)) if tau_vals else 7.0
    tau_max = float(np.nanmax(tau_vals)) if tau_vals else 21.0

    npi_calc = NPICalculator()
    npi = npi_calc.compute_from_dataframe(transactions, "target")

    return {
        "geo_code": geo_code,
        "npi": npi,
        "D": D,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "population": n,
    }


def run():
    print("=" * 60)
    print("CPBM Exponential Scaling Demo")
    print("=" * 60)

    aggregator = RegionalAggregator()

    for i, (geo, n) in enumerate(zip(GEO_CODES, COMMUNITY_SIZES)):
        print(f"Processing community {geo} (N={n})...")
        params = process_community(geo, n, seed=i * 7 + 1)
        aggregator.add_community(**params)
        print(f"  NPI={params['npi']:.1f}, D={params['D']:.4f}, "
              f"tau=[{params['tau_min']:.1f},{params['tau_max']:.1f}]d")

    macro = {
        "cpi_growth": 0.087,
        "unemployment": 0.112,
        "gdp_growth": 0.034,
        "consumer_confidence": 0.61,
        "median_income_growth": 0.022,
    }

    regional_features = aggregator.aggregate(macro_signals=macro)

    feature_names = [
        "mean_NPI", "std_NPI", "weighted_mean_D",
        "min_tau_days", "tau_range_days",
        "cpi_growth", "unemployment", "gdp_growth",
        "consumer_confidence", "median_income_growth",
    ]

    print("\nRegional feature vector (L2 input):")
    for name, val in zip(feature_names, regional_features):
        print(f"  {name:30s}: {val:.6f}")

    print("\nCommunity summary:")
    print(aggregator.get_summary().to_string(index=False))

    print("\nScaling hierarchy:")
    print("  L1 communities -> L2 regional features -> L3 national model")
    print(f"  Total individuals covered: {sum(COMMUNITY_SIZES):,}")
    print(f"  Regional feature vector dimension: {len(regional_features)}")


if __name__ == "__main__":
    run()