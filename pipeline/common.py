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
import os, tempfile
ROOT=Path(__file__).resolve().parents[1]
FIG=None
_INDEX={}
_META=None
BLUE, RED, GREEN, GREY = "#3b78c3", "#e24a4a", "#4a9d5b", "#9e9e9e"

def D(name):
    return _INDEX[name]

def save(fig, name):
    for ext in ('png','pdf'):
        fig.savefig(Path(FIG)/(Path(name).stem+'.'+ext), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def meta():
    global _META
    if _META is None:
        _META = pd.read_csv(D("vascular_cell_metadata.csv"), index_col=0, low_memory=False)
    return _META

def ec_vni_split():
    """Endothelial cells split into VN-High / VN-Low by the EC VNI median (6698/6698)."""
    m = meta(); ec = m[m["EC_subtype"].notna()].copy()
    v = pd.to_numeric(ec["VNI"], errors="coerce")
    ec["VN_split"] = np.where(v > v.median(), "VN-High", "VN-Low")
    return ec

def knn_vni(cell_df, k=10):
    """Mean VNI of k nearest ECs in the shared atlas transcriptomic UMAP."""
    from sklearn.neighbors import NearestNeighbors
    ec = ec_vni_split()
    coord_cols = ["AtlasUMAP1", "AtlasUMAP2"]
    ecx = ec[coord_cols].apply(pd.to_numeric, errors="coerce")
    ecv = pd.to_numeric(ec["VNI"], errors="coerce"); ok = ecx.notna().all(1) & ecv.notna()
    x = cell_df[coord_cols].apply(pd.to_numeric, errors="coerce")
    m = x.notna().all(1)
    nn = NearestNeighbors(n_neighbors=k).fit(ecx[ok].values)
    _, idx = nn.kneighbors(x[m].values)
    out = np.full(len(cell_df), np.nan); out[m.values] = ecv[ok].values[idx].mean(1)
    return out

def _ec_module_bar(ax, mods, labels, title):
    df = ec_review_split(mods); hi = df[df.VN_split == "VN-High"]; lo = df[df.VN_split == "VN-Low"]
    mh = [hi[m].mean() for m in mods]; sh = [hi[m].sem() for m in mods]
    ml = [lo[m].mean() for m in mods]; sl = [lo[m].sem() for m in mods]
    ps = [stats.mannwhitneyu(hi[m].dropna(), lo[m].dropna()).pvalue for m in mods]
    x = np.arange(len(mods)); w = 0.38
    ax.bar(x - w/2, mh, w, yerr=sh, capsize=2.5, color=BLUE, label="VN-High EC")
    ax.bar(x + w/2, ml, w, yerr=sl, capsize=2.5, color=RED, label="VN-Low EC")
    ax.axhline(0, color="k", lw=0.8)
    for i, p in enumerate(ps):
        ax.text(x[i], max(mh[i]+sh[i], ml[i]+sl[i]) + 0.015, stars(p), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel("Mean module score ± SEM")
    ax.set_title(title, fontsize=11.5, fontweight="bold"); ax.legend(); ax.grid(axis="y", ls="--", alpha=0.3)
def ec_five_axis_split(modules: list[str]) -> pd.DataFrame:
    """Join review-module scores to ECs grouped by the unified VNI median."""
    ec = meta()
    ec = ec[ec["EC_subtype"].notna()].copy()
    ec["VNI"] = pd.to_numeric(ec["VNI"], errors="coerce")
    ec = ec.dropna(subset=["VNI"])
    threshold = float(ec["VNI"].median())
    ec["VN_split"] = np.where(ec["VNI"] >= threshold, "VN-High", "VN-Low")

    use = list(dict.fromkeys(modules))
    scores = pd.read_csv(
        D("cell_review_module_scores.csv"),
        index_col=0,
        usecols=["cell"] + use,
    )
    return ec[["VNI", "VN_split"]].join(scores, how="inner")

ec_review_split=ec_five_axis_split
