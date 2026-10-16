#!/usr/bin/env python3
"""
R3: Immune-Vascular Crosstalk Analysis
=======================================
Core question: Do tumors with a higher degree of vascular normalization show greater immune cell (T cell) infiltration?

Analysis at three levels:
  1. UMAP neighborhood analysis: do VN-High endothelial cells have more T cells around them (cell-level)?
  2. Dataset-level correlation: mean vascular VNI vs T cell fraction across the 6 datasets (sample-level).
  3. T cell functional scoring: do samples with better vascular status have more combat-capable T cells (cytotoxicity up, exhaustion down)?
"""

import os
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

sc.settings.verbosity = 1
np.random.seed(1234)

PIPELINE_DIR = _PKG
SC_DIR       = os.path.join(PIPELINE_DIR, "outputs")
OUT_DIR      = os.path.join(SC_DIR, "r3_immune_vascular")
os.makedirs(OUT_DIR, exist_ok=True)

def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def safe_score(adata, genes, name):
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 3:
        print(f"  SKIP {name}: only {len(present)} genes found")
        adata.obs[name] = 0.0
        return
    ctrl = min(50, len(adata.var_names) // 20)
    sc.tl.score_genes(adata, present, score_name=name, ctrl_size=ctrl)

# ── T cell functional gene sets ────────────────────────────────
T_SIGNATURES = {
    # Cytotoxic T cells: the ones that actually kill cancer cells
    "CD8_Cytotoxic": ["CD8A", "CD8B", "GZMB", "GZMA", "PRF1", "IFNG",
                       "GNLY", "NKG7", "FASLG"],
    # T cell exhaustion: "worn out" within the tumor, having lost their fighting capacity
    "T_Exhaustion":  ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4",
                       "TOX", "NR4A1", "ENTPD1"],
    # Regulatory T cells (Treg): suppress immunity, favorable for the tumor
    "T_Regulatory":  ["FOXP3", "IL2RA", "IKZF2", "CTLA4", "TIGIT",
                       "IL10", "TGFB1"],
    # Effector memory T cells: the combat-capable ones
    "T_Effector":    ["TBX21", "EOMES", "IFNG", "TNF", "IL2",
                       "CXCR3", "CCR5", "CX3CR1"],
}

# ============================================================
# Load the vascular cell data that has already been scored
# ============================================================
print_section("Loading data")

vascular_path = os.path.join(SC_DIR, "vascular_ecosystem_vn_scored.h5ad")
full_path      = os.path.join(_PKG, "raw_data", "HNSCC_integrated_atlas.h5ad")

print(f"Loading vascular cells (with existing VNI scores): {vascular_path}")
vascular = sc.read_h5ad(vascular_path)
print(f"  {vascular.n_obs:,} vascular cells, VNI scores ready")

print(f"\nLoading the full atlas (including T cells): {full_path}")
adata_full = sc.read_h5ad(full_path)
print(f"  {adata_full.n_obs:,} cells")

# Sync the vascular cells' VNI scores to the full atlas
vascular_idx = vascular.obs.index
adata_full.obs.loc[vascular_idx, "VNI"] = vascular.obs["VNI"].values

# VN_state may be absent from the h5ad (category columns are sometimes not saved); recompute it from VNI
if "VN_state" not in vascular.obs.columns:
    vni_median = vascular.obs["VNI"].median()
    vascular.obs["VN_state"] = np.where(
        vascular.obs["VNI"] >= vni_median, "VN-High", "VN-Low"
    )
    print("  VN_state recomputed from VNI")
adata_full.obs.loc[vascular_idx, "VN_state"] = vascular.obs["VN_state"].values

# Lowercase the cell type labels for easier matching
adata_full.obs["celltype_lower"] = adata_full.obs["atlas_classification"].str.lower()

# ============================================================
# SECTION 1: Dataset level -- vascular VNI vs T cell fraction
# ============================================================
print_section("SECTION 1 · Dataset level: vascular VNI vs immune infiltration")

# Statistics for each dataset
dataset_stats = []
for ds in adata_full.obs["dataset"].unique():
    mask = adata_full.obs["dataset"] == ds
    cells = adata_full.obs[mask]
    total = len(cells)

    # Immune cell fractions
    t_frac      = (cells["celltype_lower"].str.contains("t cell", na=False)).sum() / total
    myeloid_frac= (cells["celltype_lower"].str.contains("myeloid", na=False)).sum() / total
    b_frac      = (cells["celltype_lower"].str.contains("b cell", na=False)).sum() / total
    immune_frac = t_frac + myeloid_frac + b_frac

    # Mean VNI of the vascular cells
    ec_mask = mask & adata_full.obs["celltype_lower"].str.contains("endothelial", na=False)
    vni_mean = adata_full.obs.loc[ec_mask, "VNI"].mean()
    vni_high_frac = (adata_full.obs.loc[ec_mask, "VN_state"] == "VN-High").mean() \
                    if ec_mask.sum() > 0 else np.nan

    dataset_stats.append({
        "dataset":       ds,
        "n_cells":       total,
        "n_ec":          ec_mask.sum(),
        "VNI_mean":      vni_mean,
        "VN_High_frac":  vni_high_frac,
        "T_cell_frac":   t_frac,
        "Myeloid_frac":  myeloid_frac,
        "B_cell_frac":   b_frac,
        "Immune_frac":   immune_frac,
    })

ds_df = pd.DataFrame(dataset_stats).set_index("dataset")
ds_df.to_csv(os.path.join(OUT_DIR, "dataset_vni_immune.csv"))
print(ds_df.round(3).to_string())

# Compute correlations
for immune_col in ["T_cell_frac", "Myeloid_frac", "Immune_frac"]:
    valid = ds_df[["VNI_mean", immune_col]].dropna()
    if len(valid) >= 3:
        r, p = stats.pearsonr(valid["VNI_mean"], valid[immune_col])
        rs, ps = stats.spearmanr(valid["VNI_mean"], valid[immune_col])
        print(f"\n  VNI_mean vs {immune_col}:  "
              f"Pearson r={r:.3f} p={p:.3f}  |  Spearman r={rs:.3f} p={ps:.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, label in zip(axes,
    ["T_cell_frac", "Myeloid_frac", "Immune_frac"],
    ["T cell fraction", "Myeloid fraction", "Total immune fraction"]):

    valid = ds_df[["VNI_mean", col]].dropna()
    ax.scatter(valid["VNI_mean"], valid[col], s=120, color="steelblue", zorder=3)
    for ds_name, row in valid.iterrows():
        ax.annotate(ds_name, (row["VNI_mean"], row[col]),
                    fontsize=8, ha="left", va="bottom")

    # Trend line
    if len(valid) >= 3:
        z = np.polyfit(valid["VNI_mean"], valid[col], 1)
        x_line = np.linspace(valid["VNI_mean"].min(), valid["VNI_mean"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
        r, p = stats.pearsonr(valid["VNI_mean"], valid[col])
        ax.set_title(f"VNI vs {label}\nPearson r={r:.2f}, p={p:.3f}", fontsize=10)
    ax.set_xlabel("Mean VNI (endothelial cells)")
    ax.set_ylabel(label)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT_DIR, f"dataset_vni_vs_immune.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("\nSaved: dataset_vni_vs_immune.{pdf,png}")


# ============================================================
# SECTION 2: UMAP neighborhood analysis -- T cell fraction around VN-High ECs
# ============================================================
print_section("SECTION 2 · UMAP neighborhood analysis: Are there more T cells around VN-High ECs?")

# Use the shared atlas transcriptomic UMAP for KNN.  This measures similarity in
# the integrated expression embedding, not physical tissue proximity.
ec_mask_full = adata_full.obs["celltype_lower"].str.contains("endothelial", na=False)
t_mask_full  = adata_full.obs["celltype_lower"].str.contains("t cell", na=False)

n_ec = ec_mask_full.sum()
n_t  = t_mask_full.sum()
print(f"Endothelial cells: {n_ec:,}  |  T cells: {n_t:,}")

# Get UMAP coordinates
umap_coords = adata_full.obsm["X_umap"]
ec_coords   = umap_coords[ec_mask_full.values]
all_coords  = umap_coords

# Build KNN over all cells (30 neighbors)
knn_cache = os.path.join(OUT_DIR, "knn_tcell_fracs.npy")
print("Building atlas-UMAP KNN graph (30 neighbors)...")
nbrs = NearestNeighbors(n_neighbors=31, algorithm="ball_tree", n_jobs=4)
nbrs.fit(all_coords)
distances, indices = nbrs.kneighbors(ec_coords)
is_tcell = t_mask_full.values
t_fracs_per_ec = np.array([is_tcell[idx].mean() for idx in indices[:, 1:]])
np.save(knn_cache, t_fracs_per_ec)
print("KNN results refreshed")

# Add the T cell neighbor fraction to the EC obs
ec_obs_idx = adata_full.obs.index[ec_mask_full.values]
adata_full.obs.loc[ec_obs_idx, "neighbor_T_frac"] = t_fracs_per_ec

# Sync to the vascular object
shared_ec = [i for i in ec_obs_idx if i in vascular.obs.index]
if shared_ec:
    vascular.obs.loc[shared_ec, "neighbor_T_frac"] = \
        adata_full.obs.loc[shared_ec, "neighbor_T_frac"]

# Compare T cell neighbor fraction between VN-High and VN-Low
ec_vascular = adata_full.obs.loc[ec_obs_idx,
    ["dataset", "VNI", "VN_state", "neighbor_T_frac"]].copy()
# VN_state may be NaN (ECs not in the vascular subset); fill it in
if ec_vascular["VN_state"].isna().all():
    vni_med = ec_vascular["VNI"].median()
    ec_vascular["VN_state"] = np.where(
        ec_vascular["VNI"] >= vni_med, "VN-High", "VN-Low"
    )

vn_high = ec_vascular[ec_vascular["VN_state"] == "VN-High"]["neighbor_T_frac"].dropna()
vn_low  = ec_vascular[ec_vascular["VN_state"] == "VN-Low"]["neighbor_T_frac"].dropna()

print(f"\nT cell fraction among VN-High EC neighbors: mean={vn_high.mean():.4f} (n={len(vn_high)})")
print(f"T cell fraction among VN-Low  EC neighbors: mean={vn_low.mean():.4f}  (n={len(vn_low)})")

t_stat, p_val = stats.mannwhitneyu(vn_high, vn_low, alternative="two-sided")
print(f"Mann-Whitney U test: p = {p_val:.4e}")

# Violin plot
fig, ax = plt.subplots(figsize=(6, 5))
data_plot = pd.DataFrame({
    "T cell neighbor fraction": pd.concat([vn_high, vn_low]),
    "VN state": ["VN-High"] * len(vn_high) + ["VN-Low"] * len(vn_low)
})
sns.violinplot(data=data_plot, x="VN state", y="T cell neighbor fraction",
               palette={"VN-High": "#1565C0", "VN-Low": "#B71C1C"},
               inner="box", ax=ax)
ax.set_title(f"VN-High vs VN-Low endothelial cells\nT cell fraction among the 30 nearest neighbors\np = {p_val:.2e}",
             fontsize=10)
ax.set_ylabel("T cell fraction among neighbors")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT_DIR, f"ec_neighbor_tcell_fraction.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved: ec_neighbor_tcell_fraction.{pdf,png}")


# ============================================================
# SECTION 3: T cell functional scoring -- do high-VNI samples have stronger T cells?
# ============================================================
print_section("SECTION 3 · T cell functional scoring")

t_cells = adata_full[t_mask_full].copy()
print(f"Number of T cells: {t_cells.n_obs:,}")

# Log-normalization (if not already done)
sc.pp.normalize_total(t_cells, target_sum=1e4)
sc.pp.log1p(t_cells)

for sig_name, genes in T_SIGNATURES.items():
    safe_score(t_cells, genes, sig_name)

# Merge each dataset's mean T cell functional scores with the corresponding VNI
tcell_dataset = t_cells.obs.groupby("dataset")[list(T_SIGNATURES.keys())].mean()
tcell_dataset = tcell_dataset.join(ds_df[["VNI_mean", "VN_High_frac"]])
tcell_dataset.to_csv(os.path.join(OUT_DIR, "tcell_function_by_dataset.csv"))

print("\nT cell functional scores × dataset (mean):")
print(tcell_dataset.round(3).to_string())

# Correlation analysis
print("\nDataset VNI vs T cell function correlation:")
for sig in T_SIGNATURES.keys():
    valid = tcell_dataset[["VNI_mean", sig]].dropna()
    if len(valid) >= 3:
        r, p = stats.pearsonr(valid["VNI_mean"], valid[sig])
        direction = "↑" if r > 0 else "↓"
        print(f"  VNI vs {sig:20s}: r={r:+.3f}  p={p:.3f}  {direction}")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
plot_cols = list(T_SIGNATURES.keys()) + ["VNI_mean"]
corr_mat = tcell_dataset[plot_cols].corr()
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title("VNI × T cell functional score correlation matrix\n(dataset level)")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT_DIR, f"vni_tcell_function_corr.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()

# Scatter plots (VNI vs cytotoxicity vs exhaustion)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, sig, color, label in zip(
    axes,
    ["CD8_Cytotoxic", "T_Exhaustion"],
    ["#1565C0", "#B71C1C"],
    ["CD8 cytotoxicity score (higher is better)", "T cell exhaustion score (lower is better)"]
):
    valid = tcell_dataset[["VNI_mean", sig]].dropna()
    ax.scatter(valid["VNI_mean"], valid[sig], s=120, color=color, zorder=3)
    for ds_name, row in valid.iterrows():
        ax.annotate(ds_name, (row["VNI_mean"], row[sig]),
                    fontsize=8, ha="left", va="bottom")
    if len(valid) >= 3:
        z = np.polyfit(valid["VNI_mean"], valid[sig], 1)
        x_line = np.linspace(valid["VNI_mean"].min(), valid["VNI_mean"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color=color, alpha=0.7)
        r, p = stats.pearsonr(valid["VNI_mean"], valid[sig])
        ax.set_title(f"VNI vs {label}\nr={r:.2f}, p={p:.3f}", fontsize=10)
    ax.set_xlabel("Mean VNI (endothelial cells)")
    ax.set_ylabel(label)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT_DIR, f"vni_vs_tcell_cytotoxic_exhaustion.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved: vni_vs_tcell_cytotoxic_exhaustion.{pdf,png}")


# ============================================================
# SECTION 4: Combined summary figure
# ============================================================
print_section("SECTION 4 · Combined figure: VN state × immune cell composition")

# Immune cell fraction comparison between VN-High and VN-Low samples (grouped by dataset)
vn_high_ds = ds_df[ds_df["VN_High_frac"] >= ds_df["VN_High_frac"].median()]
vn_low_ds  = ds_df[ds_df["VN_High_frac"] <  ds_df["VN_High_frac"].median()]

summary = pd.DataFrame({
    "VN-High samples": vn_high_ds[["T_cell_frac", "Myeloid_frac", "B_cell_frac"]].mean(),
    "VN-Low samples":  vn_low_ds[["T_cell_frac", "Myeloid_frac", "B_cell_frac"]].mean(),
}).rename(index={"T_cell_frac": "T cells", "Myeloid_frac": "Myeloid cells", "B_cell_frac": "B cells"})

print("\nImmune composition differences between VN-High and VN-Low samples:")
print(summary.round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(summary))
w = 0.35
ax.bar(x - w/2, summary["VN-High samples"], w, label="VN-High samples", color="#1565C0", alpha=0.85)
ax.bar(x + w/2, summary["VN-Low samples"],  w, label="VN-Low samples",  color="#B71C1C", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(summary.index, fontsize=11)
ax.set_ylabel("Cell fraction")
ax.set_title("VN-High vs VN-Low samples\nImmune cell infiltration comparison")
ax.legend()
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT_DIR, f"vn_state_immune_composition.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close()

# ============================================================
# Save results
# ============================================================
print_section("Saving all outputs")
print(f"\nOutput directory: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
    print(f"  {f:55s}  {sz:>5} KB")

print("""
Summary of key findings:
  dataset_vni_immune.csv                ← VNI vs immune infiltration across the 6 datasets
  ec_neighbor_tcell_fraction.pdf        ← More T cells around VN-High ECs?
  vni_tcell_function_corr.pdf           ← Correlation heatmap of VNI vs T cell functional scores
  vni_vs_tcell_cytotoxic_exhaustion.pdf ← Higher VNI → cytotoxicity ↑, exhaustion ↓?
  vn_state_immune_composition.pdf       ← Higher overall immune infiltration in VN-High samples?
""")
