import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")

from cpbm.data.loader import CPBMDataLoader
from cpbm.core.signature import SignatureExtractor
from cpbm.core.diffusion import DiffusionLayer
from cpbm.core.stratum import StratumGradient
from cpbm.core.npi import NPICalculator
from cpbm.models.ensemble import CPBMEnsemble
from cpbm.evaluation.metrics import CPBMEvaluator
from cpbm.viz.plots import plot_full_dashboard, plot_feature_importance


TARGET_CATEGORY = "packaged_juice"
WINDOW_DAYS = 90
FORECAST_HORIZON = 30
DATA_PATH = os.environ.get("CPBM_DATA_PATH", "data/fmcg_transactions.csv")
RESULTS_DIR = "results/real_data"


def run():
    print("=" * 60)
    print("CPBM Real-Data FMCG Experiment")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}.")
        print("Expected CSV columns: ind_id, amount, txn_timestamp, category, stratum")
        print("Falling back to synthetic data with real-data pipeline structure.\n")
        from cpbm.data.synthetic import SyntheticCommunity
        community = SyntheticCommunity(n=1000, seed=7).generate()
        transactions = community["transactions"]
        transactions["category"] = TARGET_CATEGORY
        transactions["txn_timestamp"] = pd.date_range(
            start="2023-01-01", periods=len(transactions), freq="h"
        )
        labels = community["labels"]
        strata = community["strata"]
    else:
        loader = CPBMDataLoader()
        transactions = loader.load_transactions_csv(DATA_PATH)
        transactions = loader.compute_total_spend_per_individual(transactions)
        label_series = loader.build_label_vector(
            transactions, TARGET_CATEGORY, forecast_horizon_days=FORECAST_HORIZON
        )
        transactions["label"] = transactions["ind_id"].map(label_series).fillna(0).astype(int)
        labels = transactions.groupby("ind_id")["label"].first().values
        strata = transactions.groupby("ind_id")["stratum"].first().values if "stratum" in transactions.columns else np.ones(len(labels), dtype=int)

    extractor = SignatureExtractor(window_days=WINDOW_DAYS, k_range=(2, 8))
    Phi = extractor.fit_transform(transactions)
    print(f"Signatures extracted: shape={Phi.shape}, K*={extractor.k_star_}")

    diffusion_layer = DiffusionLayer(random_state=42)
    t_dummy = np.linspace(0, 60, 61)
    adop_dummy = 0.4 * (1 - np.exp(-0.05 * t_dummy)) * np.exp(-0.005 * t_dummy ** 2)
    diffusion_layer.fit(t_dummy, adop_dummy)
    diffusion_layer.build_graph(n_nodes=len(Phi))
    seed_nodes = np.where(Phi[:, 0] >= np.percentile(Phi[:, 0], 95))[0].tolist()
    approx_proba = labels * 0.65 + 0.15
    diff_results = diffusion_layer.simulate(
        n_total=len(Phi),
        seed_nodes=seed_nodes,
        purchase_proba=approx_proba,
        steps=60,
    )

    npi_calc = NPICalculator()
    npi_val = npi_calc.compute_from_dataframe(transactions, TARGET_CATEGORY)
    print(f"NPI: {npi_val:.2f} ({npi_calc.interpret(npi_val)})")

    stratum_layer = StratumGradient()
    t_axis = np.linspace(0, WINDOW_DAYS, WINDOW_DAYS + 1)
    unique_strata = np.unique(strata)
    stratum_adoption = {}
    for s in unique_strata:
        mask = strata == s
        M = float(mask.sum())
        from scipy.integrate import odeint
        p_s = max(0.01, 0.05 - 0.01 * (3 - s))
        q_s = max(0.10, 0.35 - 0.05 * (3 - s))
        def ode(y, t, p=p_s, q=q_s, Mv=M):
            F = y[0] / Mv
            return [(p + q * F) * (1 - F) * Mv]
        sol = odeint(ode, [0.0], t_axis)
        stratum_adoption[int(s)] = (t_axis, sol[:, 0], M)
    stratum_layer.fit(stratum_adoption)
    tau_values = list(stratum_layer.tau_.values())
    tau_mean = float(np.nanmean(tau_values)) if tau_values else 14.0
    print("\nStratum gradient summary:")
    print(stratum_layer.summary())

    ensemble = CPBMEnsemble(random_state=42)
    ensemble.fit(
        Phi=Phi,
        labels=labels,
        tau=tau_mean,
        npi=npi_val,
        cluster_labels=extractor.cluster_labels_,
    )
    all_proba = ensemble.predict_proba(
        Phi, tau=tau_mean, npi=npi_val,
        cluster_labels=extractor.cluster_labels_
    )

    evaluator = CPBMEvaluator()
    result = evaluator.full_report(labels, all_proba, "CPBM Real Data")
    print(f"\nAUC-ROC: {result['auc_roc']:.4f}")
    print(f"AUC-PR : {result['auc_pr']:.4f}")
    print(f"F1     : {result['f1']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    imp_df = ensemble.xgb.get_feature_importance_df()
    plot_feature_importance(imp_df, save_path=f"{RESULTS_DIR}/feature_importance.png")

    stratum_proba_dict = {int(s): all_proba[strata == s] for s in unique_strata}
    stratum_curves_pred = {
        int(s): stratum_layer.predict_adoption(int(s), t_axis)
        for s in unique_strata
    }
    plot_full_dashboard(
        diff_results, imp_df, Phi,
        extractor.cluster_labels_,
        stratum_proba_dict, strata,
        save_path=f"{RESULTS_DIR}/dashboard.png",
    )
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()