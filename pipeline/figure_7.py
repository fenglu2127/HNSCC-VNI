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
import common
from common import save
def export_single_panel(fig,name):
    save(fig,name)
def make_figure_7a() -> None:
    corr = pd.read_csv(common.D("review_axis_spearman_correlation.csv"), index_col=0)
    idx = [
        "review_normalization_index",
        "immune_gate_reopening_index",
        "endothelial_immune_barrier_index",
        "stromal_compression_index",
        "metabolic_immunosuppression_index",
    ]
    labels = [
        "VN Index",
        "Immune Gate\nReopening",
        "Endothelial\nImmune Barrier",
        "Stromal\nCompression",
        "Metabolic\nImmunosuppression",
    ]
    values = corr.loc[idx, idx].values
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold" if (abs(value) > 0.4 and i != j) else "normal",
                color="white" if abs(value) > 0.55 else "black",
            )
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title(
        "Correlations Among Mechanism-Informed Ecosystem Indices\n"
        "(Spearman rho, 232,015 cells)",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman rho")
    fig.tight_layout()
    export_single_panel(fig, "Figure7A")

def make_figure_7b() -> None:
    data = pd.read_csv(common.D("scores_by_celltype.csv"), index_col=0)
    order = [
        "Endothelial cells",
        "Pericytes",
        "CD8+ T-cells",
        "NK cells",
        "CD4+ T-cells",
        "Macrophages",
        "Fibroblasts",
        "Epithelial cells",
    ]
    idx = [
        "review_normalization_index",
        "immune_gate_reopening_index",
        "endothelial_immune_barrier_index",
        "stromal_compression_index",
        "metabolic_immunosuppression_index",
    ]
    labels = [
        "VN Index",
        "Immune Gate\nReopening",
        "Endothelial\nImmune Barrier",
        "Stromal\nCompression",
        "Metabolic\nImmunosuppression",
    ]
    matrix = data.loc[order, idx]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    for i in range(len(order)):
        for j in range(len(idx)):
            value = matrix.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.7 else "black",
                fontsize=10,
            )
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{name}\n(n={int(data.loc[name, 'n']):,})" for name in order], fontsize=9)
    ax.set_title(
        "Cell-Type Distribution of Mechanism-Informed Ecosystem Indices",
        fontweight="bold",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Mean score (z-scaled)")
    fig.tight_layout()
    export_single_panel(fig, "Figure7B")

def make_figure_7c() -> None:
    modules = [
        "EC_maturation_phalanx_score",
        "EC_tip_sprouting_score",
        "EC_stalk_Notch_score",
        "EC_leukocyte_adhesion_score",
        "EC_death_ligand_barrier_score",
        "pericyte_coverage_score",
        "ANG1_TIE2_stabilization_score",
        "ANG2_destabilization_score",
        "basement_membrane_integrity_score",
        "CXCL9_10_Tcell_recruitment_score",
        "pathological_TEC_embryonic_score",
        "CAF_mechanical_compression_score",
        "hypoxia_acidosis_score",
        "adenosine_immunosuppression_score",
    ]
    labels = [
        "EC maturation (Phalanx)",
        "EC tip sprouting",
        "Stalk/Notch state",
        "Leukocyte adhesion",
        "Death-ligand barrier",
        "Pericyte coverage",
        "ANG1/TIE2 stabilization",
        "ANG2 destabilization",
        "Basement-membrane integrity",
        "CXCL9/10 T-cell recruitment",
        "Pathological TEC embryonic",
        "CAF mechanical compression",
        "Hypoxia/acidosis",
        "Adenosine immunosuppression",
    ]
    scores = pd.read_csv(
        common.D("cell_review_module_scores.csv"),
        index_col=0,
        usecols=["cell"] + modules,
    )
    ec_index = common.ec_vni_split().index
    ec_mean = scores.loc[scores.index.intersection(ec_index), modules].mean()
    atlas_mean = scores[modules].mean()
    x = np.arange(len(modules))
    width = 0.4
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.bar(x - width / 2, ec_mean.values, width, color="#e8637c", label="Endothelial cells")
    ax.bar(x + width / 2, atlas_mean.values, width, color="#b0b0b0", label="Atlas mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean module score (AddModuleScore units)")
    ax.set_title(
        "Endothelial Mechanism-Informed Module Profile vs Atlas Mean",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend()
    fig.tight_layout()
    export_single_panel(fig, "Figure7C")
