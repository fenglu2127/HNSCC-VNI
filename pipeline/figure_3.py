from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats
N_GENES = 15

ANTIGEN_PRESENTATION = {"CD74", "HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DPA1", "HLA-DQA1"}

INTERFERON_INFLAMMATION = {"IFITM1", "IFITM3", "OASL", "RSAD2", "GBP4", "BST2", "UBD"}

ENDOTHELIAL_VASCULAR = {"AQP1", "ACKR1", "ALPL", "CLEC3B", "SLCO2A1", "RNASE1", "PLVAP", "RAMP3", "VWF", "PRCP", "A2M"}

GROUP_ORDER = [
    "Antigen presentation",
    "Interferon / inflammation",
    "Endothelial / vascular",
    "Other / unclassified",
]

COLORS = {
    "Antigen presentation": "#0072B2",
    "Interferon / inflammation": "#D55E00",
    "Endothelial / vascular": "#009E73",
    "Other / unclassified": "#7A7A7A",
}

def functional_group(gene: str) -> str:
    if gene in ANTIGEN_PRESENTATION:
        return "Antigen presentation"
    if gene in INTERFERON_INFLAMMATION:
        return "Interferon / inflammation"
    if gene in ENDOTHELIAL_VASCULAR:
        return "Endothelial / vascular"
    return "Other / unclassified"

def main() -> None:
    de = pd.read_csv(INPUT)
    required = {"names", "logfoldchanges", "pvals_adj", "used_in_vni_construction"}
    missing = required.difference(de.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    eligible = de.loc[
        (~de["used_in_vni_construction"].astype(bool))
        & (de["logfoldchanges"] > 0)
        & (de["pvals_adj"] < 0.05)
        & (~de["names"].str.startswith(("MT-", "RP", "AP00"), na=False))
    ].copy()

    # Select 15 genes by Wilcoxon score, using adjusted P value to break ties.
    plotted = eligible.sort_values(
        ["scores", "pvals_adj"], ascending=[False, True]
    ).head(N_GENES).copy()

    # Display the selected genes in ascending adjusted-P order.
    plotted = plotted.sort_values(
        ["pvals_adj", "scores"], ascending=[True, False]
    ).reset_index(drop=True)
    plotted["functional_group"] = plotted["names"].map(functional_group)
    plotted["rank_by_adjusted_p"] = np.arange(1, len(plotted) + 1)

    # Save the exact plotted rows and annotations for auditability.
    plotted.to_csv(OUTPUT / "Figure3_plotted_genes.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    y = np.arange(len(plotted))
    values = plotted["logfoldchanges"].to_numpy()
    colors = plotted["functional_group"].map(COLORS).to_numpy()

    for yi, value, color in zip(y, values, colors):
        ax.hlines(yi, 0, value, color=color, linewidth=2.4, alpha=0.9, zorder=1)
    ax.scatter(values, y, c=colors, s=72, edgecolors="white", linewidths=0.8, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(plotted["names"])
    ax.invert_yaxis()  # smallest adjusted P value appears at the top
    ax.set_xlabel("log2 fold change (VNI-High vs VNI-Low)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(values) * 1.10)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.3)

    present_groups = set(plotted["functional_group"])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[group],
               markeredgecolor="white", markersize=8, label=group)
        for group in GROUP_ORDER if group in present_groups
    ]
    ax.legend(handles=handles, title="Functional group", frameon=False,
              loc="lower right", bbox_to_anchor=(1.0, 0.02))

    fig.tight_layout()
    png = OUTPUT / "Figure3.png"
    pdf = OUTPUT / "Figure3.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(plotted[["rank_by_adjusted_p", "names", "logfoldchanges", "pvals_adj", "functional_group"]].to_string(index=False))
