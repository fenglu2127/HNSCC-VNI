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
def make_figure(obs: pd.DataFrame, umap: np.ndarray) -> None:
    columns = ["VM_Score", "PC_Score_adj", "HDA_Score", "IVI_Score", "EBI_Score_adj", "VNI"]
    labels = ["VM", "PC (adjusted)", "HDA", "IVI", "EBI (adjusted)", "VNI"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax, column, label in zip(axes.ravel(), columns, labels):
        values = obs[column].to_numpy()
        limit = float(np.nanpercentile(np.abs(values), 99))
        points = ax.scatter(
            umap[:, 0],
            umap[:, 1],
            c=values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
            s=2,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Unified vascular UMAP1")
        ax.set_ylabel("Unified vascular UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Five-Axis Vascular Normalization Landscape\n"
        f"Unified vascular-ecosystem UMAP, n = {len(obs):,} cells",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"Figure1.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
