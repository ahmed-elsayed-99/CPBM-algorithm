import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional, Dict, List


PALETTE = {"stratum_1": "#e74c3c", "stratum_2": "#2ecc71", "stratum_3": "#3498db",
           "teal": "#1abc9c", "dark": "#2c3e50", "light": "#ecf0f1"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


def plot_diffusion_curves(
    diffusion_df: pd.DataFrame,
    diffusion_df_random: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diffusion_df["day"], diffusion_df["penetration_pct"],
            color=PALETTE["teal"], linewidth=2.5, label="Targeted seed (top 5%)")
    if diffusion_df_random is not None:
        ax.plot(diffusion_df_random["day"], diffusion_df_random["penetration_pct"],
                color=PALETTE["stratum_1"], linewidth=2.0, linestyle="--",
                label="Random seed")
    peak_row = diffusion_df.loc[diffusion_df["penetration_pct"].idxmax()]
    ax.axvline(peak_row["day"], color="gray", linestyle=":", linewidth=1.5)
    ax.annotate(
        f"Peak: day {int(peak_row['day'])}\n{peak_row['penetration_pct']:.1f}%",
        xy=(peak_row["day"], peak_row["penetration_pct"]),
        xytext=(peak_row["day"] + 3, peak_row["penetration_pct"] - 5),
        fontsize=9, color=PALETTE["dark"],
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Community penetration (%)", fontsize=11)
    ax.set_title("CPBM — Purchase Diffusion Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stratum_adoption(
    stratum_curves: Dict[int, np.ndarray],
    time_axis: np.ndarray,
    tau_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    colors = {3: PALETTE["stratum_3"], 2: PALETTE["stratum_2"], 1: PALETTE["stratum_1"]}
    labels = {3: "Stratum 3 (upper)", 2: "Stratum 2 (middle)", 1: "Stratum 1 (lower)"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in [3, 2, 1]:
        if s in stratum_curves:
            ax.plot(time_axis, stratum_curves[s], color=colors[s],
                    linewidth=2.5, label=labels[s])
    if tau_dict:
        for key, tau in tau_dict.items():
            parts = key.split("_")
            ax.annotate(
                f"τ={tau:.1f}d",
                xy=(float(time_axis[len(time_axis) // 2]), 0),
                fontsize=8, color="gray",
            )
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Cumulative adoption (units)", fontsize=11)
    ax.set_title("CPBM — Stratum Adoption Cascade", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    df = importance_df.sort_values("importance", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df["feature"], df["importance"],
                   color=PALETTE["teal"], edgecolor=PALETTE["dark"],
                   linewidth=0.6, height=0.6)
    for bar, val in zip(bars, df["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=PALETTE["dark"])
    ax.set_xlabel("Feature importance (XGBoost gain)", fontsize=11)
    ax.set_title("CPBM — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlim(0, df["importance"].max() * 1.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pulse_clusters_pca(
    Phi: np.ndarray,
    cluster_labels: np.ndarray,
    strata: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(Phi)
    var_explained = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2 if strata is not None else 1,
                             figsize=(14 if strata is not None else 7, 5))
    if strata is None:
        axes = [axes]
    sc0 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                          cmap="tab10", alpha=0.65, s=25, linewidths=0.3)
    axes[0].set_title("Pulse Clusters (Behavioural Archetypes)", fontsize=11)
    axes[0].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=9)
    axes[0].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=9)
    plt.colorbar(sc0, ax=axes[0], label="Cluster")
    if strata is not None:
        sc1 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=strata,
                              cmap="RdYlGn", alpha=0.65, s=25, linewidths=0.3)
        axes[1].set_title("Socioeconomic Strata", fontsize=11)
        axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=9)
        axes[1].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=9)
        plt.colorbar(sc1, ax=axes[1], label="Stratum")
    plt.suptitle("CPBM — Pulse Signature Space (PCA)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_full_dashboard(
    diffusion_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    Phi: np.ndarray,
    cluster_labels: np.ndarray,
    stratum_proba: Dict[int, np.ndarray],
    strata: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(diffusion_df["day"], diffusion_df["penetration_pct"],
             color=PALETTE["teal"], linewidth=2.5)
    peak = diffusion_df.loc[diffusion_df["penetration_pct"].idxmax()]
    ax1.axvline(peak["day"], color="gray", linestyle=":", linewidth=1.2)
    ax1.set_title("Community Purchase Diffusion", fontweight="bold")
    ax1.set_xlabel("Day"); ax1.set_ylabel("Penetration (%)")

    ax2 = fig.add_subplot(gs[0, 2])
    imp = importance_df.sort_values("importance", ascending=True).tail(7)
    ax2.barh(imp["feature"], imp["importance"], color=PALETTE["teal"],
             height=0.55, edgecolor=PALETTE["dark"], linewidth=0.5)
    ax2.set_title("Feature Importance", fontweight="bold")
    ax2.set_xlabel("Gain")

    ax3 = fig.add_subplot(gs[1, :2])
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(Phi)
    sc = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                     cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax3, label="Cluster")
    ax3.set_title("Pulse Signature Clusters (PCA)", fontweight="bold")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")

    ax4 = fig.add_subplot(gs[1, 2])
    colors_s = {1: PALETTE["stratum_1"], 2: PALETTE["stratum_2"], 3: PALETTE["stratum_3"]}
    for s, proba_arr in stratum_proba.items():
        ax4.hist(proba_arr, bins=20, alpha=0.55,
                 color=colors_s.get(s, "gray"),
                 label=f"Stratum {s}", density=True)
    ax4.set_title("Purchase Probability by Stratum", fontweight="bold")
    ax4.set_xlabel("P(purchase)"); ax4.set_ylabel("Density")
    ax4.legend(fontsize=8)

    fig.suptitle("CPBM — Full Model Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig