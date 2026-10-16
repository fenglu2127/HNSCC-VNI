#!/usr/bin/env python3
"""
T-cell & macrophage subset analysis: VN-High vs VN-Low vascular neighborhoods
=======================================================================
Two levels:
  1. Cell level: in the shared atlas transcriptomic UMAP, find each T cell /
               macrophage's nearest 10 endothelial cells and use their mean VNI.
               This is transcriptomic UMAP proximity, not physical spatial proximity.
  2. Dataset level: for each of the 6 datasets, correlate VNI vs immune subset proportions
"""

import os, warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(_PKG, "outputs", "immune_subtype")
os.makedirs(OUT, exist_ok=True)

def ps(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

FUNC_SIGS = {
    "CD8_Cytotoxic": ["CD8A","CD8B","GZMB","GZMA","PRF1","IFNG","GNLY","NKG7"],
    "T_Exhaustion":  ["PDCD1","LAG3","HAVCR2","TIGIT","TOX","ENTPD1","CTLA4"],
    "Treg_Activity": ["FOXP3","IL2RA","IKZF2","IL10","TGFB1"],
    "T_Effector":    ["TBX21","EOMES","CX3CR1","KLRG1","GZMB","PRF1"],
    "M1_like":       ["CD86","NOS2","IL6","TNF","CXCL9","CXCL10","STAT1","IL12B"],
    "M2_like":       ["CD163","MRC1","ARG1","IL10","CCL22","CD209","TGFB1"],
    "SPP1_TAM":      ["SPP1","APOE","C1QA","C1QB","C1QC","GPNMB","FTL"],
}

# ============================================================
# SECTION 1  Load data
# ============================================================
ps("SECTION 1 · Load data")

adata = sc.read_h5ad(os.path.join(_PKG, "raw_data", "HNSCC_integrated_atlas.h5ad"))
vascular = sc.read_h5ad(os.path.join(_PKG, "outputs", "vascular_ecosystem_vn_scored.h5ad"))
if "VN_state" not in vascular.obs.columns:
    vascular.obs["VN_state"] = np.where(
        vascular.obs["VNI"] >= vascular.obs["VNI"].median(), "VN-High", "VN-Low"
    )
# Sync VNI and VN_state back to the atlas
adata.obs["VNI"]      = np.nan
adata.obs["VN_state"] = None
adata.obs.loc[vascular.obs.index, "VNI"]      = vascular.obs["VNI"].values
adata.obs.loc[vascular.obs.index, "VN_state"] = vascular.obs["VN_state"].values
print(f"Atlas: {adata.n_obs:,} cells, VNI synchronized")

umap_all = adata.obsm["X_umap"]

# ============================================================
# SECTION 2  KNN: find the nearest ECs for each T cell / macrophage
#             use the mean VNI of these ECs to define the "vascular neighborhood VNI"
# ============================================================
ps("SECTION 2 · Compute vascular neighborhood VNI of immune cells (KNN)")

ec_mask    = adata.obs["atlas_classification"] == "Endothelial"
t_mask     = adata.obs["atlas_classification"] == "T cells"
my_mask    = adata.obs["atlas_classification"] == "Immune/Myeloid"

ec_umap    = umap_all[ec_mask.values]
ec_vni     = adata.obs.loc[ec_mask, "VNI"].values.astype(float)
t_umap     = umap_all[t_mask.values]
my_umap    = umap_all[my_mask.values]

print(f"Endothelial: {ec_mask.sum():,} | T cells: {t_mask.sum():,} | Myeloid: {my_mask.sum():,}")

# Keep only ECs that have a VNI
valid_ec = ~np.isnan(ec_vni)
ec_umap_v = ec_umap[valid_ec]
ec_vni_v  = ec_vni[valid_ec]
print(f"ECs with VNI: {valid_ec.sum():,}")

# Build KNN index for ECs
K_EC = 10    # find the nearest 10 ECs for each immune cell
print(f"Building EC KNN (K={K_EC})...")
nbrs = NearestNeighbors(n_neighbors=K_EC, algorithm="ball_tree", n_jobs=4)
nbrs.fit(ec_umap_v)

cache_t  = f"{OUT}/nbr_vni_tcell.npy"
cache_my = f"{OUT}/nbr_vni_myeloid.npy"

# Always recompute: VNI weights and the EC-median definition may change between
# runs, and an unversioned cache can silently mix incompatible analyses.
print("Computing atlas-UMAP neighborhood VNI for T cells...")
_, idx_t  = nbrs.kneighbors(t_umap)
nbr_vni_t = ec_vni_v[idx_t].mean(axis=1)
np.save(cache_t, nbr_vni_t)

print("Computing atlas-UMAP neighborhood VNI for myeloid cells...")
_, idx_my = nbrs.kneighbors(my_umap)
nbr_vni_my = ec_vni_v[idx_my].mean(axis=1)
np.save(cache_my, nbr_vni_my)

print(f"T cell neighborhood VNI: mean={nbr_vni_t.mean():.3f}, std={nbr_vni_t.std():.3f}")
print(f"Myeloid neighborhood VNI:  mean={nbr_vni_my.mean():.3f}, std={nbr_vni_my.std():.3f}")

# Group by neighborhood VNI: Q1 (low) vs Q3 (high), drop the middle 50% to make the contrast more extreme
q25_t = np.percentile(nbr_vni_t, 25)
q75_t = np.percentile(nbr_vni_t, 75)
q25_m = np.percentile(nbr_vni_my, 25)
q75_m = np.percentile(nbr_vni_my, 75)

t_zone  = np.where(nbr_vni_t >= q75_t, "High-VNI neighborhood (Q4)",
          np.where(nbr_vni_t <= q25_t, "Low-VNI neighborhood (Q1)", "Middle"))
my_zone = np.where(nbr_vni_my >= q75_m, "High-VNI neighborhood (Q4)",
          np.where(nbr_vni_my <= q25_m, "Low-VNI neighborhood (Q1)", "Middle"))

print(f"\nT cell grouping: High-VNI(Q4)={( t_zone=='High-VNI neighborhood (Q4)').sum():,} | Low-VNI(Q1)={( t_zone=='Low-VNI neighborhood (Q1)').sum():,}")
print(f"Myeloid grouping:  High-VNI(Q4)={(my_zone=='High-VNI neighborhood (Q4)').sum():,} | Low-VNI(Q1)={(my_zone=='Low-VNI neighborhood (Q1)').sum():,}")

# ============================================================
# SECTION 3  Immune subset composition comparison
# ============================================================
ps("SECTION 3 · Immune subset composition comparison (cell level)")

# T cell subsets
t_obs = adata.obs[t_mask.values].copy()
t_obs["VN_zone"]    = t_zone
t_obs["nbr_VNI"]    = nbr_vni_t
t_core = t_obs[t_obs["VN_zone"] != "Middle"]

t_comp = (t_core.groupby(["VN_zone","Immune_Classification"])
          .size().unstack(fill_value=0))
t_pct  = t_comp.div(t_comp.sum(axis=1), axis=0) * 100

print("\nT cell subset proportions (%):")
print(t_pct.round(2).to_string())

# Myeloid subsets
my_obs = adata.obs[my_mask.values].copy()
my_obs["VN_zone"] = my_zone
my_obs["nbr_VNI"] = nbr_vni_my
my_core = my_obs[my_obs["VN_zone"] != "Middle"]

my_comp = (my_core.groupby(["VN_zone","Immune_Classification"])
           .size().unstack(fill_value=0))
my_pct  = my_comp.div(my_comp.sum(axis=1), axis=0) * 100

print("\nMyeloid subset proportions (%):")
print(my_pct.round(2).to_string())

# Statistical test (per subset)
def chi2_pval(comp_df, subtype):
    if subtype not in comp_df.columns: return 1.0
    ct = comp_df[subtype].values
    total = comp_df.sum(axis=1).values
    try:
        _, p, _, _ = stats.chi2_contingency(
            [[ct[0], total[0]-ct[0]], [ct[1], total[1]-ct[1]]]
        )
        return p
    except: return 1.0

# ============================================================
# SECTION 4  Functional scoring
# ============================================================
ps("SECTION 4 · Functional scoring (cell level)")

def safe_score(ad, genes, name):
    present = [g for g in genes if g in ad.var_names]
    if len(present) < 3:
        ad.obs[name] = 0.0; return
    sc.tl.score_genes(ad, present, score_name=name,
                      ctrl_size=min(50, len(ad.var_names)//20))

# T cell functional scoring
t_cells = adata[t_mask.values].copy()
sc.pp.normalize_total(t_cells, target_sum=1e4)
sc.pp.log1p(t_cells)
for k, genes in FUNC_SIGS.items():
    if k.startswith("M") or k.startswith("S") or k.startswith("I"): continue
    safe_score(t_cells, genes, k)
t_cells.obs["VN_zone"] = t_zone
t_cells.obs["nbr_VNI"] = nbr_vni_t
t_cells.obs["AtlasUMAP1"] = t_umap[:, 0]
t_cells.obs["AtlasUMAP2"] = t_umap[:, 1]

# Myeloid functional scoring
my_cells = adata[my_mask.values].copy()
sc.pp.normalize_total(my_cells, target_sum=1e4)
sc.pp.log1p(my_cells)
for k, genes in FUNC_SIGS.items():
    if k.startswith("CD8") or k.startswith("T_") or k.startswith("Treg"): continue
    safe_score(my_cells, genes, k)
my_cells.obs["VN_zone"] = my_zone
my_cells.obs["nbr_VNI"] = nbr_vni_my
my_cells.obs["AtlasUMAP1"] = my_umap[:, 0]
my_cells.obs["AtlasUMAP2"] = my_umap[:, 1]

# Export the per-cell scored obs tables (T-cell / myeloid functional scores + UMAP coords).
# These feed the Figure 5D / 8B panels, which recompute the endothelial-neighbourhood VNI
# themselves; the columns needed are the functional scores plus the Compartment-UMAP coords.
t_cells.obs.to_csv(f"{OUT}/tcell_scored.csv")
my_cells.obs.to_csv(f"{OUT}/myeloid_scored.csv")

t_scored  = t_cells.obs[t_cells.obs["VN_zone"] != "Middle"]
my_scored = my_cells.obs[my_cells.obs["VN_zone"] != "Middle"]

t_func_keys  = ["CD8_Cytotoxic","T_Exhaustion","Treg_Activity","T_Effector"]
my_func_keys = ["M1_like","M2_like","SPP1_TAM"]

print("\nT cell functional scores:")
for k in t_func_keys:
    if k not in t_scored.columns: continue
    h = t_scored[t_scored["VN_zone"]=="High-VNI neighborhood (Q4)"][k]
    l = t_scored[t_scored["VN_zone"]=="Low-VNI neighborhood (Q1)"][k]
    _, p = stats.mannwhitneyu(h, l, alternative="two-sided")
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    direction = "↑High-VNI" if h.mean()>l.mean() else "↓High-VNI"
    print(f"  {k:20s}: High={h.mean():.4f}  Low={l.mean():.4f}  p={p:.2e} {sig}  {direction}")

print("\nMyeloid cell functional scores:")
for k in my_func_keys:
    if k not in my_scored.columns: continue
    h = my_scored[my_scored["VN_zone"]=="High-VNI neighborhood (Q4)"][k]
    l = my_scored[my_scored["VN_zone"]=="Low-VNI neighborhood (Q1)"][k]
    _, p = stats.mannwhitneyu(h, l, alternative="two-sided")
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    direction = "↑High-VNI" if h.mean()>l.mean() else "↓High-VNI"
    print(f"  {k:20s}: High={h.mean():.4f}  Low={l.mean():.4f}  p={p:.2e} {sig}  {direction}")

# ============================================================
# SECTION 5  Dataset-level analysis
# ============================================================
ps("SECTION 5 · Dataset-level analysis")

ds_stats = []
for ds in adata.obs["dataset"].unique():
    m = adata.obs["dataset"] == ds
    cells = adata.obs[m]
    n_total = len(cells)

    # Proportions of each subset
    tc   = cells[cells["atlas_classification"]=="T cells"]
    my_c = cells[cells["atlas_classification"]=="Immune/Myeloid"]
    ec_c = cells[cells["atlas_classification"]=="Endothelial"]

    vni_mean = ec_c["VNI"].mean()
    row = {"dataset": ds, "VNI_mean": vni_mean, "n_cells": n_total,
           "T_frac": len(tc)/n_total, "My_frac": len(my_c)/n_total}

    # T cell subsets
    if len(tc) > 0:
        for sub in ["CD8T","Dysfunctional","Treg","CytotoxicT/NK","Naive","Tcm"]:
            row[f"T_{sub}"] = (tc["Immune_Classification"]==sub).mean()
    # Myeloid subsets
    if len(my_c) > 0:
        for sub in ["Macrophages","CD14+ Mono","Mac/Mono/Neut","cDC","CD16+ Mono"]:
            row[f"My_{sub}"] = (my_c["Immune_Classification"]==sub).mean()

    ds_stats.append(row)

ds_df = pd.DataFrame(ds_stats).set_index("dataset")
ds_df_valid = ds_df.dropna(subset=["VNI_mean"])
ds_df.to_csv(f"{OUT}/dataset_immune_stats.csv")

print("\nDataset-level correlation: VNI vs immune subset proportions:")
corr_results = {}
for col in [c for c in ds_df_valid.columns if c.startswith("T_") or c.startswith("My_")]:
    valid = ds_df_valid[["VNI_mean",col]].dropna()
    if len(valid) < 4: continue
    r, p = stats.pearsonr(valid["VNI_mean"], valid[col])
    corr_results[col] = (r, p)
    sig = "*" if p<0.05 else ("†" if p<0.1 else "")
    print(f"  {col:25s}: r={r:+.3f}  p={p:.3f} {sig}")

# ============================================================
# SECTION 6  Visualization
# ============================================================
ps("SECTION 6 · Visualization")

COLORS_ZONE = {"High-VNI neighborhood (Q4)": "#1565C0", "Low-VNI neighborhood (Q1)": "#E53935"}
fig = plt.figure(figsize=(24, 20))
fig.suptitle("T cell & macrophage subsets: High-VNI vs Low-VNI vascular neighborhoods\n(grouped by the mean VNI of each immune cell's nearest 10 ECs, Q1 vs Q4)",
             fontsize=13, fontweight="bold", y=0.99)

# ── 1. T cell subset proportion comparison ─────────────────────────
ax1 = fig.add_subplot(3, 4, 1)
key_t_subtypes = ["CD8T","CytotoxicT/NK","Dysfunctional","Treg","Naive","Tcm","Tfh","Proliferating"]
t_plot = t_pct[[s for s in key_t_subtypes if s in t_pct.columns]]
x = np.arange(len(t_plot.columns))
w = 0.35
for i, (zone, color) in enumerate([("High-VNI neighborhood (Q4)","#1565C0"),("Low-VNI neighborhood (Q1)","#E53935")]):
    if zone not in t_plot.index: continue
    ax1.bar(x + (i-0.5)*w, t_plot.loc[zone], w, label=zone, color=color, alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(t_plot.columns, rotation=35, ha="right", fontsize=8)
ax1.set_ylabel("T cell proportion (%)"); ax1.legend(fontsize=7)
ax1.set_title("T cell subset proportions\n(High vs Low VNI neighborhood)", fontweight="bold", fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

# ── 2. Myeloid subset proportion comparison ──────────────────────────
ax2 = fig.add_subplot(3, 4, 2)
key_m_subtypes = ["Macrophages","CD14+ Mono","Mac/Mono/Neut","cDC","CD16+ Mono","Mast","pDC"]
my_plot = my_pct[[s for s in key_m_subtypes if s in my_pct.columns]]
x2 = np.arange(len(my_plot.columns))
for i, (zone, color) in enumerate([("High-VNI neighborhood (Q4)","#1565C0"),("Low-VNI neighborhood (Q1)","#E53935")]):
    if zone not in my_plot.index: continue
    ax2.bar(x2 + (i-0.5)*w, my_plot.loc[zone], w, label=zone, color=color, alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(my_plot.columns, rotation=35, ha="right", fontsize=8)
ax2.set_ylabel("Myeloid cell proportion (%)"); ax2.legend(fontsize=7)
ax2.set_title("Myeloid subset proportions\n(High vs Low VNI neighborhood)", fontweight="bold", fontsize=9)
ax2.spines[["top","right"]].set_visible(False)

# ── 3-6. T cell functional score violins ────────────────────────────
for idx, key in enumerate(t_func_keys, start=3):
    ax = fig.add_subplot(3, 4, idx)
    if key not in t_scored.columns: continue
    plot_data = t_scored[t_scored["VN_zone"].isin(["High-VNI neighborhood (Q4)","Low-VNI neighborhood (Q1)"])].copy()
    sns.violinplot(data=plot_data, x="VN_zone", y=key,
                   palette=COLORS_ZONE, inner="box", ax=ax, cut=0)
    h = plot_data[plot_data["VN_zone"]=="High-VNI neighborhood (Q4)"][key]
    l = plot_data[plot_data["VN_zone"]=="Low-VNI neighborhood (Q1)"][key]
    _, p = stats.mannwhitneyu(h, l, alternative="two-sided")
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else f"p={p:.2f}"))
    direction = "↑" if h.mean()>l.mean() else "↓"
    ax.set_title(f"{key}\n{direction}High-VNI neighborhood  {sig}", fontweight="bold", fontsize=9)
    ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

# ── 7-9. Myeloid functional score violins ─────────────────────────────
for idx, key in enumerate(my_func_keys, start=7):
    ax = fig.add_subplot(3, 4, idx)
    if key not in my_scored.columns: continue
    plot_data = my_scored[my_scored["VN_zone"].isin(["High-VNI neighborhood (Q4)","Low-VNI neighborhood (Q1)"])].copy()
    sns.violinplot(data=plot_data, x="VN_zone", y=key,
                   palette=COLORS_ZONE, inner="box", ax=ax, cut=0)
    h = plot_data[plot_data["VN_zone"]=="High-VNI neighborhood (Q4)"][key]
    l = plot_data[plot_data["VN_zone"]=="Low-VNI neighborhood (Q1)"][key]
    _, p = stats.mannwhitneyu(h, l, alternative="two-sided")
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else f"p={p:.2f}"))
    direction = "↑" if h.mean()>l.mean() else "↓"
    ax.set_title(f"{key}\n{direction}High-VNI neighborhood  {sig}", fontweight="bold", fontsize=9)
    ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

# ── 10. Dataset level: VNI vs CD8T correlation ─────────────────────────
ax10 = fig.add_subplot(3, 4, 10)
if "T_CD8T" in ds_df_valid.columns:
    valid = ds_df_valid[["VNI_mean","T_CD8T"]].dropna()
    ax10.scatter(valid["VNI_mean"], valid["T_CD8T"]*100, s=100,
                 color="#1e88e5", zorder=3)
    for ds_name, row in valid.iterrows():
        ax10.annotate(ds_name, (row["VNI_mean"], row["T_CD8T"]*100),
                      fontsize=7, ha="left", va="bottom")
    if len(valid) >= 3:
        z = np.polyfit(valid["VNI_mean"], valid["T_CD8T"]*100, 1)
        xl = np.linspace(valid["VNI_mean"].min(), valid["VNI_mean"].max(), 50)
        ax10.plot(xl, np.polyval(z,xl), "r--", alpha=0.6)
        r, p = stats.pearsonr(valid["VNI_mean"], valid["T_CD8T"]*100)
        ax10.text(0.05,0.9,f"r={r:.2f}\np={p:.3f}",transform=ax10.transAxes,fontsize=8)
    ax10.set_xlabel("Dataset mean VNI"); ax10.set_ylabel("CD8+ T cell proportion (%)")
    ax10.set_title("Dataset level\nVNI vs CD8T proportion", fontweight="bold", fontsize=9)
    ax10.spines[["top","right"]].set_visible(False)

# ── 11. Dataset level: VNI vs Macrophage ────────────────────────
ax11 = fig.add_subplot(3, 4, 11)
if "My_Macrophages" in ds_df_valid.columns:
    valid = ds_df_valid[["VNI_mean","My_Macrophages"]].dropna()
    ax11.scatter(valid["VNI_mean"], valid["My_Macrophages"]*100, s=100,
                 color="#e94560", zorder=3)
    for ds_name, row in valid.iterrows():
        ax11.annotate(ds_name, (row["VNI_mean"], row["My_Macrophages"]*100),
                      fontsize=7, ha="left", va="bottom")
    if len(valid) >= 3:
        z = np.polyfit(valid["VNI_mean"], valid["My_Macrophages"]*100, 1)
        xl = np.linspace(valid["VNI_mean"].min(), valid["VNI_mean"].max(), 50)
        ax11.plot(xl, np.polyval(z,xl), "r--", alpha=0.6)
        r, p = stats.pearsonr(valid["VNI_mean"], valid["My_Macrophages"]*100)
        ax11.text(0.05,0.9,f"r={r:.2f}\np={p:.3f}",transform=ax11.transAxes,fontsize=8)
    ax11.set_xlabel("Dataset mean VNI"); ax11.set_ylabel("Macrophage proportion (%)")
    ax11.set_title("Dataset level\nVNI vs Macrophage proportion", fontweight="bold", fontsize=9)
    ax11.spines[["top","right"]].set_visible(False)

# ── 12. Neighborhood VNI distribution (T cells) ─────────────────────────
ax12 = fig.add_subplot(3, 4, 12)
ax12.hist(nbr_vni_t, bins=50, color="#1e88e5", alpha=0.7, edgecolor="white")
ax12.axvline(q25_t, color="#E53935", linestyle="--", label=f"Q1={q25_t:.2f}")
ax12.axvline(q75_t, color="#1565C0", linestyle="--", label=f"Q4={q75_t:.2f}")
ax12.set_xlabel("T cell neighborhood VNI (mean VNI of nearest 10 ECs)")
ax12.set_ylabel("Number of T cells")
ax12.set_title("T cell neighborhood VNI distribution\n(Q1/Q4 boundaries)", fontweight="bold", fontsize=9)
ax12.legend(fontsize=8); ax12.spines[["top","right"]].set_visible(False)

plt.tight_layout()
for ext in ("png","pdf"):
    plt.savefig(f"{OUT}/immune_subtype_comparison.{ext}", dpi=180, bbox_inches="tight")
plt.close()

# ── Save data ──────────────────────────────────────────────────
t_pct.to_csv(f"{OUT}/tcell_subtype_proportion.csv")
my_pct.to_csv(f"{OUT}/myeloid_subtype_proportion.csv")
ds_df.to_csv(f"{OUT}/dataset_immune_stats.csv")

ps("Done")
print(f"Figure: {OUT}/immune_subtype_comparison.{{pdf,png}}")
