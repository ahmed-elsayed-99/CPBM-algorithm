import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from cpbm.data.synthetic import SyntheticCommunity
from cpbm.core.signature import SignatureExtractor
from cpbm.core.diffusion import DiffusionLayer
from cpbm.core.stratum import StratumGradient
from cpbm.core.npi import NPICalculator
from cpbm.models.ensemble import CPBMEnsemble
from cpbm.evaluation.metrics import CPBMEvaluator
from cpbm.viz.plots import (
    plot_diffusion_curves,
    plot_stratum_adoption,
    plot_feature_importance,
    plot_pulse_clusters_pca,
    plot_full_dashboard,
)


def run():
    print("=" * 60)
    print("CPBM Synthetic Baseline Experiment")
    print("=" * 60)

    community = SyntheticCommunity(n=500, seed=42).generate()
    df = community["dataframe"]
    Phi = community["phi_matrix"]
    labels = community["labels"]
    strata = community["strata"]
    transactions = community["transactions"]

    print(f"\nCommunity: N={len(df)}")
    print(df.groupby("stratum")["label"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "purchase_rate"}
    ).round(3))

    extractor = SignatureExtractor(window_days=90, k_range=(2, 8))
    Phi_extracted = extractor.fit_transform(transactions)
    print(f"\nOptimal clusters: K* = {extractor.k_star_}")
    print("\nCluster profiles:")
    print(extractor.get_cluster_profiles())

    diffusion_layer = DiffusionLayer(random_state=42)
    t_hist, adop_hist = community["adoption_history"]
    diffusion_layer.fit(t_hist, adop_hist)
    print(f"\nDiffusion params: {diffusion_layer.params_}")

    diffusion_layer.build_graph(n_nodes=len(df))
    seed_nodes = np.where(
        Phi[:, 0] >= np.percentile(Phi[:, 0], 95)
    )[0].tolist()
    all_proba_approx = labels * 0.7 + 0.15
    diff_results = diffusion_layer.simulate(
        n_total=len(df),
        seed_nodes=seed_nodes,
        purchase_proba=all_proba_approx,
        steps=60,
    )
    peak = diff_results.loc[diff_results["penetration_pct"].idxmax()]
    print(f"\nDiffusion peak: day {int(peak['day'])}, {peak['penetration_pct']:.1f}%")

    stratum_layer = StratumGradient()
    stratum_layer.fit(community["stratum_adoption"])
    print("\nStratum gradient summary:")
    print(stratum_layer.summary())

    npi_calc = NPICalculator()
    npi_val = npi_calc.compute_from_dataframe(transactions, "target")
    print(f"\nNPI value: {npi_val:.2f} — {npi_calc.interpret(npi_val)}")

    tau_32 = stratum_layer.get_tau(3, 2)
    tau_21 = stratum_layer.get_tau(2, 1)
    tau_mean = np.nanmean([tau_32, tau_21])

    ensemble = CPBMEnsemble(random_state=42)
    ensemble.fit(
        Phi=Phi,
        labels=labels,
        tau=tau_mean,
        npi=npi_val,
        cluster_labels=extractor.cluster_labels_,
    )
    print("\nEnsemble training results:")
    for k, v in ensemble.eval_results_.items():
        print(f"  {k}: {v:.4f}")

    evaluator = CPBMEvaluator()
    all_proba = ensemble.predict_proba(
        Phi, tau=tau_mean, npi=npi_val,
        cluster_labels=extractor.cluster_labels_
    )
    full = evaluator.full_report(labels, all_proba, "CPBM Ensemble")
    print(f"\nFull evaluation:")
    print(f"  AUC-ROC : {full['auc_roc']:.4f}")
    print(f"  AUC-PR  : {full['auc_pr']:.4f}")
    print(f"  F1      : {full['f1']:.4f}")
    print(f"  Accuracy: {full['accuracy']:.4f}")

    stratum_report = evaluator.stratum_report(labels, all_proba, strata)
    print("\nPer-stratum performance:")
    print(stratum_report.to_string(index=False))

    imp_df = ensemble.xgb.get_feature_importance_df()
    print("\nTop-5 features:")
    print(imp_df.head(5).to_string(index=False))

    stratum_proba_dict = {
        s: all_proba[strata == s] for s in np.unique(strata)
    }
    stratum_curves = {}
    t_axis = np.linspace(0, 90, 91)
    for s_id in [1, 2, 3]:
        stratum_curves[s_id] = stratum_layer.predict_adoption(s_id, t_axis)

    plot_diffusion_curves(diff_results, save_path="results/diffusion.png")
    plot_stratum_adoption(stratum_curves, t_axis,
                          tau_dict=stratum_layer.tau_,
                          save_path="results/stratum_adoption.png")
    plot_feature_importance(imp_df, save_path="results/feature_importance.png")
    plot_pulse_clusters_pca(Phi, extractor.cluster_labels_, strata,
                            save_path="results/pca_clusters.png")
    plot_full_dashboard(diff_results, imp_df, Phi,
                        extractor.cluster_labels_,
                        stratum_proba_dict, strata,
                        save_path="results/dashboard.png")
    print("\nPlots saved to results/")
    print("\nExperiment complete.")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()