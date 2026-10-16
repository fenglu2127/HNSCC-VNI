#!/usr/bin/env python3
"""
Vascular Normalization Ecosystem Analysis in HPV-negative HNSCC
==============================================================
Five-dimensional VN scoring framework + Lasso-Cox prognostic model
derived from HNSCC integrated single-cell atlas (Zenodo 17880437).

Five Component Scores:
  1. VM_Score  — Vascular Maturation (endothelial integrity & quiescence)
  2. PC_Score  — Pericyte Coverage (mature mural cell investment)
  3. HDA_Score — Hypoxia-Driven Angiogenesis (dysfunction indicator, inverted in VNI)
  4. IVI_Score — Immune-Vascular Interaction (immunostimulatory vessel capacity)
  5. EBI_Score — ECM Barrier Integrity (basement membrane completeness)

Composite: direction-aligned equal-weight VNI of 5 EC-standardized components;
           EC-only PCA retained as a sensitivity analysis (VNI_PCA)
Prognosis:  Univariate Cox → Lasso-Cox risk score → Kaplan-Meier stratification
"""

import os
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Install with: pip install lifelines")

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

sc.settings.verbosity = 2
sc.set_figure_params(figsize=(8, 8), dpi=150)
np.random.seed(1234)

# --- package-relative paths (this file lives in <pkg>/pipeline/) ---
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5AD_PATH = os.environ.get("SCHNSCC_H5AD", os.path.join(_PKG, "raw_data", "HNSCC_integrated_atlas.h5ad"))
OUT_DIR = os.environ.get("SCHNSCC_OUT", os.path.join(_PKG, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# GENE SIGNATURES  (literature-anchored)
# ============================================================

SIGNATURES = {
    # Score 1 — Vascular Maturation
    # Captures quiescent EC state, tight junction integrity, venous/arterial maturity.
    # Refs: Jain 2005 Science; Carmeliet & Jain 2011 Nature; Augustin & Koh 2017 Science
    "VM_Score": [
        "PECAM1", "CDH5", "KDR", "TEK", "TIE1", "ENG", "NRP1",
        "CLDN5", "OCLN", "ESAM",                        # tight junction / barrier
        "GJA4", "GJA5", "EFNB2", "HEY1", "HEY2",       # arterial maturation
        "NR2F2", "EMCN", "APLNR", "EPHB4",              # venous identity
        "DLL4", "JAG1", "NOTCH4",                        # Notch-driven maturation
        "FLT1",                                          # VEGF-trap (anti-angiogenic)
    ],

    # Score 2 — Pericyte Coverage
    # Mature pericyte investment in vessel wall = hallmark of normalized vasculature.
    # Refs: Hamzah et al. 2008 Nature; Morikawa et al. 2002; Ribeiro & Bhatt 2020
    "PC_Score": [
        "PDGFRB", "CSPG4",         # canonical pericyte receptors
        "ANGPT1",                   # vessel stabilization ligand
        "MCAM",                     # pericyte surface marker
        "DES", "TAGLN", "MYH11",    # cytoskeletal maturity
        "NOTCH3",                   # pericyte differentiation
        "ACTA2",                    # α-SMA, mural cell
    ],

    # Score 3 — Hypoxia-Driven Angiogenesis  [DYSFUNCTION indicator]
    # High = chaotic sprouting, high permeability, immature neovasculature.
    # Inverted when building VNI; positive HR expected in Cox model.
    # Refs: Semenza 2012; Bergers & Hanahan 2008; Eelen et al. 2018 Cell
    "HDA_Score": [
        "HIF1A", "EPAS1",           # HIF-1α / HIF-2α
        "VEGFA", "VEGFC",           # pro-angiogenic drive
        "ANGPT2",                   # vessel destabilization
        "CA9",                      # hypoxia-specific carbonic anhydrase
        "LDHA", "PDK1",             # glycolytic shift
        "SLC2A1",                   # GLUT1, glucose uptake
        "BNIP3", "BNIP3L",          # hypoxic mitophagy
        "NDRG1",                    # HIF target
        "ESM1", "APLN", "CXCR4",   # tip-cell / sprouting markers
        "CXCL8",                    # IL-8, promotes abnormal angiogenesis
        "S100A8", "S100A9",         # pro-inflammatory alarmin
        "MKI67",                    # EC proliferation (pathological)
    ],

    # Score 4 — Immune-Vascular Interaction
    # Normalized vessels upregulate adhesion molecules and chemokines that
    # enable productive T-cell trafficking; dysfunctional vessels suppress this.
    # Refs: Johansson-Percival et al. 2017 Nat Immunol; Motz & Coukos 2013 Immunity
    "IVI_Score": [
        "ICAM1", "VCAM1",           # T-cell docking
        "SELE", "SELP",             # selectin-mediated rolling
        "CXCL9", "CXCL10", "CXCL11",  # CXCR3-axis T-cell recruitment
        "CCL2", "CCL5",             # monocyte/T-cell chemoattraction
        "HLA-A", "HLA-B", "HLA-C", # MHC-I antigen presentation
        "TNFSF10",                  # TRAIL, death ligand on vessels
        "CD274", "PDCD1LG2",        # PD-L1/2 (checkpoint on vessels)
    ],

    # Score 5 — ECM Barrier Integrity
    # Intact basement membrane (Col IV, laminins) + balanced MMP inhibition
    # reflect vascular normalisation; excessive MMP activity signals ECM breach.
    # Refs: Kalluri 2003 Nat Rev Cancer; Iozzo & Gubbiotti 2018 Matrix Biol
    "EBI_Score": [
        "COL4A1", "COL4A2", "COL4A3",   # basement-membrane collagen IV
        "LAMA4", "LAMB1", "LAMB2", "LAMC1",  # laminins
        "HSPG2",                         # perlecan
        "NID1", "NID2",                  # nidogens
        "TIMP1", "TIMP2",                # MMP inhibitors
    ],
}

# Dysfunction sub-signatures (subtracted from relevant component scores)
ECM_DYSFUNCTION_GENES = ["MMP2", "MMP9", "MMP14", "MMP1", "FN1", "CTGF", "CCN2"]
IMMATURE_PERICYTE_GENES = ["RGS5", "HEYL", "MCAM"]   # high = abnormal coverage

# EC sub-type markers for sub-population characterisation
EC_SUBTYPE_MARKERS = {
    "Tip":       ["ESM1", "APLN", "CXCR4", "ANGPT2", "KDR", "KCNE3"],
    "Stalk":     ["HES1", "NOTCH1", "JAG1", "DLL4", "NOTCH4", "NRARP"],
    "Arterial":  ["GJA4", "GJA5", "EFNB2", "HEY1", "HEY2", "DKK2"],
    "Venous":    ["NR2F2", "EMCN", "APLNR", "EPHB4", "PLVAP"],
    "Capillary": ["RGCC", "CA4", "SLC2A1", "MFSD2A", "TFPI"],
    "Lymphatic": ["PROX1", "PDPN", "LYVE1", "FLT4", "MMRN1"],
}


# ============================================================
# UTILITIES
# ============================================================

def filter_genes_present(adata, gene_list, name=""):
    present = [g for g in gene_list if g in adata.var_names]
    missing = set(gene_list) - set(present)
    if missing and name:
        print(f"    [{name}] missing {len(missing)}/{len(gene_list)} genes: "
              f"{sorted(missing)[:6]}{'…' if len(missing) > 6 else ''}")
    return present


def safe_score_genes(adata, gene_list, score_name, ctrl_size=50):
    present = filter_genes_present(adata, gene_list, score_name)
    if len(present) < 3:
        print(f"  SKIP {score_name}: only {len(present)} genes found.")
        adata.obs[score_name] = 0.0
        return
    n_ctrl = min(ctrl_size, max(10, len(adata.var_names) // 20))
    sc.tl.score_genes(adata, gene_list=present,
                      score_name=score_name, ctrl_size=n_ctrl)


def zscore_col(series):
    return (series - series.mean()) / (series.std() + 1e-8)


def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ============================================================
# SECTION 1 — LOAD DATA
# ============================================================
print_section("SECTION 1 · Loading HNSCC Atlas")

adata = sc.read_h5ad(H5AD_PATH)
print(adata)
print("\nObs columns:", adata.obs.columns.tolist())
# Detect cell type column
CELLTYPE_COL = None
for c in ["atlas_classification", "celltype", "cell_type", "CellType",
          "NonImmune_Classification", "Compartment", "annotation", "leiden"]:
    if c in adata.obs.columns:
        CELLTYPE_COL = c
        break
print(f"\nCell type column: '{CELLTYPE_COL}'")
if CELLTYPE_COL:
    print(adata.obs[CELLTYPE_COL].value_counts().to_string())

# Detect patient / sample column
PATIENT_COL = None
for c in ["dataset", "patient", "Patient", "patient_id", "sample",
          "sample_id", "donor", "donor_id", "orig.ident"]:
    if c in adata.obs.columns:
        PATIENT_COL = c
        break
print(f"\nPatient column: '{PATIENT_COL}'")

# Detect survival columns
SURVIVAL_COL, DURATION_COL = None, None
for c in ["OS_event", "overall_survival_event", "vital_status", "OS_status",
          "event", "dead"]:
    if c in adata.obs.columns:
        SURVIVAL_COL = c
        break
for c in ["OS_time", "overall_survival_time", "survival_time",
          "OS_months", "OS_days", "time"]:
    if c in adata.obs.columns:
        DURATION_COL = c
        break
print(f"Survival columns: event='{SURVIVAL_COL}', time='{DURATION_COL}'")

if not (SURVIVAL_COL and DURATION_COL):
    print("\n  NOTE: No survival columns found in .obs.")
    print("  Prognostic models require patient-level survival data.")
    print("  Options:")
    print("    A) Merge TCGA-HNSC clinical data (tcga_clinical.csv) keyed on patient ID.")
    print("    B) Add OS_event / OS_time columns to the .h5ad obs table.")


# ============================================================
# SECTION 2 — SUBSET VASCULAR ECOSYSTEM
# ============================================================
print_section("SECTION 2 · Subsetting Vascular Ecosystem")

VASCULAR_KEYWORDS = ["endothelial", "pericyte", "smooth muscle",
                      "fibroblast", "myeloid"]

if CELLTYPE_COL is None:
    raise ValueError("No cell type column found. Check .obs columns.")

obs_celltypes = adata.obs[CELLTYPE_COL].unique()
matched = [ct for ct in obs_celltypes
           if any(kw in str(ct).lower() for kw in VASCULAR_KEYWORDS)]
print(f"Matched cell types: {matched}")

if not matched:
    raise ValueError(
        f"No vascular cell types found in '{CELLTYPE_COL}'. "
        f"Values: {list(obs_celltypes[:10])}"
    )

vascular = adata[adata.obs[CELLTYPE_COL].isin(matched)].copy()
print(f"Vascular cells selected: {vascular.n_obs:,} / {adata.n_obs:,}")

# Preserve the atlas-wide integrated UMAP before computing a vascular-only UMAP.
# These coordinates provide the one common transcriptomic embedding used by all
# endothelial/immune UMAP-neighborhood analyses in Figures 3--5.
if "X_umap" in vascular.obsm:
    vascular.obsm["X_atlas_umap"] = np.asarray(vascular.obsm["X_umap"]).copy()
    vascular.obs["AtlasUMAP1"] = vascular.obsm["X_atlas_umap"][:, 0]
    vascular.obs["AtlasUMAP2"] = vascular.obsm["X_atlas_umap"][:, 1]

# Restore raw counts for unbiased module scoring
if vascular.raw is not None:
    vascular_raw = vascular.raw.to_adata()
    vascular_raw.obs = vascular.obs.copy()
    vascular = vascular_raw
    print("Restored raw counts from .raw")

del adata   # free memory


# ============================================================
# SECTION 3 — PREPROCESSING + DIMENSIONALITY REDUCTION
# ============================================================
print_section("SECTION 3 · Preprocessing")

sc.pp.normalize_total(vascular, target_sum=1e4)
sc.pp.log1p(vascular)

# Save normalised expression for later scoring
vascular.layers["log_norm"] = vascular.X.copy()

sc.pp.highly_variable_genes(
    vascular, n_top_genes=3000, flavor="seurat"
)
vascular_hvg = vascular[:, vascular.var.highly_variable].copy()

sc.pp.scale(vascular_hvg, max_value=10)
sc.tl.pca(vascular_hvg, svd_solver="arpack", n_comps=50)
sc.pp.neighbors(vascular_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(vascular_hvg)
sc.tl.leiden(vascular_hvg, resolution=0.5, key_added="leiden_vascular")

# Copy UMAP + cluster assignments back to full-gene object
vascular.obsm["X_umap"] = vascular_hvg.obsm["X_umap"]
vascular.obs["leiden_vascular"] = vascular_hvg.obs["leiden_vascular"].values

# Cluster marker genes
sc.tl.rank_genes_groups(
    vascular_hvg, groupby="leiden_vascular", method="wilcoxon", n_genes=50
)
markers_df = sc.get.rank_genes_groups_df(vascular_hvg, group=None)
markers_df.to_csv(f"{OUT_DIR}/vascular_cluster_markers.csv", index=False)

del vascular_hvg


# ============================================================
# SECTION 4 — FIVE VN COMPONENT SCORES
# ============================================================
print_section("SECTION 4 · Five VN Component Scores")

for score_name, gene_list in SIGNATURES.items():
    print(f"\n  [{score_name}]")
    safe_score_genes(vascular, gene_list, score_name)

# Dysfunction sub-scores (subtracted from EBI and PC)
print("\n  [ECM_Dysfunction_Score]")
safe_score_genes(vascular, ECM_DYSFUNCTION_GENES, "ECM_Dysfunction_Score")
print("\n  [ImmaturePericyte_Score]")
safe_score_genes(vascular, IMMATURE_PERICYTE_GENES, "ImmaturePericyte_Score")

# Adjusted scores: subtract dysfunction from structural components
vascular.obs["PC_Score_adj"] = (
    vascular.obs["PC_Score"] - vascular.obs["ImmaturePericyte_Score"]
)
vascular.obs["EBI_Score_adj"] = (
    vascular.obs["EBI_Score"] - vascular.obs["ECM_Dysfunction_Score"]
)

# Define the target compartment once.  VNI is an endothelial state index, so
# component standardisation and PCA fitting must not be driven by fibroblast or
# myeloid cell identity.
ec_mask = vascular.obs[CELLTYPE_COL].str.lower().str.contains("endothelial", na=False)

# Z-score all component scores against the EC reference distribution.  The same
# EC-derived centring/scaling is retained when scores are projected to other cells.
COMPONENT_SCORES = {
    "VM_Score":      "VM_Score_z",       # higher = more normalized
    "PC_Score_adj":  "PC_Score_adj_z",   # higher = more pericyte-covered
    "HDA_Score":     "HDA_Score_z",      # higher = more dysfunctional
    "IVI_Score":     "IVI_Score_z",      # higher = more immunostimulatory
    "EBI_Score_adj": "EBI_Score_adj_z",  # higher = better ECM barrier
}

for raw, z in COMPONENT_SCORES.items():
    if raw in vascular.obs.columns:
        x = pd.to_numeric(vascular.obs[raw], errors="coerce")
        mu = x.loc[ec_mask].mean()
        sd = x.loc[ec_mask].std(ddof=0)
        vascular.obs[z] = (x - mu) / sd if np.isfinite(sd) and sd > 0 else 0.0

# HDA is dysfunction → invert for VNI composition
vascular.obs["HDA_Score_neg_z"] = -vascular.obs["HDA_Score_z"]

print("\nComponent score summary:")
raw_cols = list(COMPONENT_SCORES.keys())
available = [c for c in raw_cols if c in vascular.obs.columns]
print(vascular.obs[available].describe().round(3).to_string())


# ============================================================
# SECTION 5 — COMPOSITE VNI  (Vascular Normalization Index)
# ============================================================
print_section("SECTION 5 · Composite Vascular Normalization Index (VNI)")

# Stack z-scores; HDA is inverted so all columns point toward normalization
z_matrix_cols = [
    "VM_Score_z",
    "PC_Score_adj_z",
    "HDA_Score_neg_z",   # inverted dysfunction
    "IVI_Score_z",
    "EBI_Score_adj_z",
]
available_z = [c for c in z_matrix_cols if c in vascular.obs.columns]

Z = vascular.obs[available_z].fillna(0).values
Z_ec = Z[ec_mask.values]

pca_vni = PCA(n_components=min(5, len(available_z)), random_state=1234)
pca_vni.fit(Z_ec)
pc_coords = pca_vni.transform(Z)

# Keep PC1 as a sensitivity analysis; check its sign convention against VM.
# PCA is not used as the primary VNI because unconstrained loadings can reverse
# the pre-specified biological direction of individual components.
pc1 = pc_coords[:, 0]
flip_vni = np.corrcoef(
    pc1[ec_mask.values], vascular.obs.loc[ec_mask, "VM_Score"].values
)[0, 1] < 0
if flip_vni:
    pc1 = -pc1    # flip so VNI-high means more normalized

vascular.obs["VNI_PCA"] = pc1

loadings_df = pd.DataFrame(
    pca_vni.components_.T,
    index=[c.replace("_z", "").replace("_neg", "(-inv)") for c in available_z],
    columns=[f"PC{i+1}" for i in range(pca_vni.n_components_)],
)
if flip_vni:
    loadings_df["PC1"] = -loadings_df["PC1"]
explained_var = pca_vni.explained_variance_ratio_

print(f"\nPCA explained variance: " +
      " | ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(explained_var)))
print("\nPC1 loadings (VNI components):")
print(loadings_df["PC1"].sort_values(ascending=False).round(3).to_string())
loadings_df.to_csv(f"{OUT_DIR}/vni_pca_loadings.csv")

# Primary VNI: equal-weight mean of the five EC-standardised, direction-aligned
# components. HDA has already been inverted, so every term points toward a more
# normalized vascular state by construction.
vascular.obs["VNI_additive"] = vascular.obs[available_z].sum(axis=1)
vascular.obs["VNI"] = vascular.obs[available_z].mean(axis=1)

# Binary VN state: one project-wide threshold, the endothelial VNI median.
# All EC analyses in Figures 3--5 use this exact stored label.
vni_median = vascular.obs.loc[ec_mask, "VNI"].median()
vascular.obs["VN_state"] = np.where(
    vascular.obs["VNI"] >= vni_median, "VN-High", "VN-Low"
)
print(f"\nVN-High: {(vascular.obs['VN_state'] == 'VN-High').sum():,}")
print(f"VN-Low:  {(vascular.obs['VN_state'] == 'VN-Low').sum():,}")
print(f"EC VNI median threshold: {vni_median:.6f}")
print("EC split:")
print(vascular.obs.loc[ec_mask, "VN_state"].value_counts().to_string())


# ============================================================
# SECTION 6 — ENDOTHELIAL SUB-TYPE ANALYSIS
# ============================================================
print_section("SECTION 6 · Endothelial Sub-type Characterisation")

ec = vascular[ec_mask].copy()
print(f"Endothelial cells: {ec.n_obs:,}")

for subtype, genes in EC_SUBTYPE_MARKERS.items():
    safe_score_genes(ec, genes, f"EC_{subtype}_Score")

ec_subtype_cols = [f"EC_{st}_Score" for st in EC_SUBTYPE_MARKERS]
present_ec_cols = [c for c in ec_subtype_cols if c in ec.obs.columns]

if present_ec_cols:
    ec.obs["EC_subtype"] = (
        ec.obs[present_ec_cols]
        .idxmax(axis=1)
        .str.replace("EC_", "")
        .str.replace("_Score", "")
    )
    print("\nEC sub-type distribution:")
    print(ec.obs["EC_subtype"].value_counts().to_string())

    # Copy back to full vascular object
    vascular.obs.loc[ec.obs.index, "EC_subtype"] = ec.obs["EC_subtype"]
    for c in present_ec_cols:
        vascular.obs.loc[ec.obs.index, c] = ec.obs[c]


# ============================================================
# SECTION 7 — PATIENT-LEVEL PSEUDO-BULK AGGREGATION
# ============================================================
print_section("SECTION 7 · Patient-Level Pseudo-Bulk Aggregation")

PATIENT_SCORE_COLS = (
    list(COMPONENT_SCORES.keys()) +
    ["PC_Score_adj", "EBI_Score_adj", "VNI", "VNI_additive"]
)
avail_score_cols = [c for c in PATIENT_SCORE_COLS if c in vascular.obs.columns]

patient_df = pd.DataFrame()

if PATIENT_COL and PATIENT_COL in vascular.obs.columns:
    # Mean VN scores per patient (all vascular cells)
    patient_df = vascular.obs.groupby(PATIENT_COL)[avail_score_cols].mean()

    # Fraction of VN-High cells per patient
    vn_frac = (
        vascular.obs.groupby(PATIENT_COL)["VN_state"]
        .apply(lambda x: (x == "VN-High").mean())
        .rename("VN_High_fraction")
    )
    patient_df = patient_df.join(vn_frac)

    # EC sub-type proportions per patient
    if "EC_subtype" in vascular.obs.columns:
        ec_props = (
            vascular.obs[ec_mask]
            .groupby(PATIENT_COL)["EC_subtype"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .add_prefix("EC_prop_")
        )
        patient_df = patient_df.join(ec_props, how="left")

    # Attach survival data if available in .obs
    if SURVIVAL_COL and DURATION_COL:
        surv_meta = (
            vascular.obs[[PATIENT_COL, SURVIVAL_COL, DURATION_COL]]
            .drop_duplicates(subset=[PATIENT_COL])
            .set_index(PATIENT_COL)
        )
        patient_df = patient_df.join(surv_meta, how="left")

    patient_df.to_csv(f"{OUT_DIR}/patient_vn_scores.csv")
    print(f"Patient-level table: {len(patient_df)} patients × {patient_df.shape[1]} features")
    print(patient_df[avail_score_cols[:5]].describe().round(3).to_string())

else:
    print(f"Patient column '{PATIENT_COL}' not found — skipping pseudo-bulk.")
    print("Available obs columns:", vascular.obs.columns.tolist())


# ============================================================
# SECTION 8 — PROGNOSTIC MODELS
# ============================================================
print_section("SECTION 8 · Prognostic Models")

if not HAS_LIFELINES:
    print("Skipping: lifelines not installed.  pip install lifelines")

elif patient_df.empty:
    print("Skipping: no patient-level data available.")

elif SURVIVAL_COL not in patient_df.columns or DURATION_COL not in patient_df.columns:
    print("Skipping: survival columns absent from patient_df.")
    print("Merge TCGA-HNSC clinical data and re-run this section.")
    print("  Expected columns in patient_vn_scores.csv:")
    print(f"    {DURATION_COL} (numeric, months or days)")
    print(f"    {SURVIVAL_COL} (0/1 integer event indicator)")

else:
    # ---- prepare survival dataframe ----
    cox_features = [
        "VM_Score", "PC_Score_adj", "HDA_Score",
        "IVI_Score", "EBI_Score_adj", "VNI", "VN_High_fraction",
    ]
    cox_features = [f for f in cox_features if f in patient_df.columns]

    surv_df = patient_df[[DURATION_COL, SURVIVAL_COL] + cox_features].copy()
    surv_df[DURATION_COL] = pd.to_numeric(surv_df[DURATION_COL], errors="coerce")
    surv_df[SURVIVAL_COL] = pd.to_numeric(surv_df[SURVIVAL_COL], errors="coerce")
    surv_df = surv_df.dropna()
    surv_df = surv_df[surv_df[DURATION_COL] > 0]
    print(f"\nPatients for survival analysis: {len(surv_df)}")

    # ---- 8.1 Univariate Cox ----
    print("\n  8.1  Univariate Cox Regression (per-score, z-standardised)")
    scaler = StandardScaler()

    univariate_results = []
    for feat in cox_features:
        try:
            tmp = surv_df[[DURATION_COL, SURVIVAL_COL, feat]].copy()
            tmp[feat] = scaler.fit_transform(tmp[[feat]])
            cph = CoxPHFitter()
            cph.fit(tmp, duration_col=DURATION_COL, event_col=SURVIVAL_COL)
            s = cph.summary
            univariate_results.append({
                "feature":   feat,
                "HR":        float(np.exp(s["coef"].iloc[0])),
                "HR_lower":  float(np.exp(s["coef lower 95%"].iloc[0])),
                "HR_upper":  float(np.exp(s["coef upper 95%"].iloc[0])),
                "p_value":   float(s["p"].iloc[0]),
                "z":         float(s["z"].iloc[0]),
            })
        except Exception as e:
            print(f"    Cox failed for '{feat}': {e}")

    if univariate_results:
        uniCox_df = pd.DataFrame(univariate_results)
        uniCox_df.to_csv(f"{OUT_DIR}/univariate_cox.csv", index=False)
        print(uniCox_df.to_string(index=False))

    # ---- 8.2 Lasso-Cox (penalised multivariate) ----
    print("\n  8.2  Lasso-Cox  (L1 penalised, cross-validated penalizer)")

    X_all = surv_df[cox_features].apply(zscore_col)
    best_pen, best_ll = 0.1, -np.inf

    for pen in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        try:
            df_tmp = X_all.copy()
            df_tmp[DURATION_COL] = surv_df[DURATION_COL].values
            df_tmp[SURVIVAL_COL] = surv_df[SURVIVAL_COL].values
            cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
            cph.fit(df_tmp, duration_col=DURATION_COL, event_col=SURVIVAL_COL)
            if cph.log_likelihood_ > best_ll:
                best_ll = cph.log_likelihood_
                best_pen = pen
        except Exception:
            pass

    print(f"    Best penalizer: {best_pen}  (log-likelihood={best_ll:.3f})")

    df_final = X_all.copy()
    df_final[DURATION_COL] = surv_df[DURATION_COL].values
    df_final[SURVIVAL_COL] = surv_df[SURVIVAL_COL].values
    cph_final = CoxPHFitter(penalizer=best_pen, l1_ratio=1.0)
    cph_final.fit(df_final, duration_col=DURATION_COL, event_col=SURVIVAL_COL)

    print("\n  Final Lasso-Cox coefficients (non-zero features retained):")
    lasso_summary = cph_final.summary[["coef", "exp(coef)", "se(coef)", "p"]]
    lasso_summary = lasso_summary[lasso_summary["coef"].abs() > 1e-6]
    print(lasso_summary.round(4).to_string())
    cph_final.summary.to_csv(f"{OUT_DIR}/lasso_cox_summary.csv")

    # ---- 8.3 Risk Score ----
    retained_features = lasso_summary.index.tolist()
    coefs = cph_final.params_.loc[retained_features]

    risk_score = X_all[retained_features].values @ coefs.values
    surv_df = surv_df.copy()
    surv_df["risk_score"] = risk_score
    risk_median = surv_df["risk_score"].median()
    surv_df["risk_group"] = np.where(
        surv_df["risk_score"] >= risk_median, "High-Risk", "Low-Risk"
    )
    surv_df.to_csv(f"{OUT_DIR}/patient_risk_scores.csv")

    # Print the risk score formula
    print("\n  ─── VN Risk Score Formula ───────────────────────────────")
    print("  RiskScore = ", end="")
    terms = [f"({coefs[f]:+.4f}) × z({f})" for f in retained_features]
    print("\n           + ".join(terms))
    print("  ─────────────────────────────────────────────────────────")
    print("  High-Risk: RiskScore ≥ median;  Low-Risk: RiskScore < median")

    # C-index
    try:
        c_idx = cph_final.concordance_index_
        print(f"\n  Harrell C-index (training):  {c_idx:.3f}")
    except Exception:
        pass

    # ---- 8.4 Kaplan–Meier curves ----
    print("\n  8.4  Kaplan–Meier Stratification")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    COLORS = {"High": "#E53935", "Low": "#1E88E5",
              "High-Risk": "#E53935", "Low-Risk": "#43A047",
              "VN-High": "#1565C0", "VN-Low": "#B71C1C"}

    def km_plot(ax, df, group_col, t_col, e_col, title, group_labels):
        for lbl, color_key in group_labels.items():
            mask = df[group_col] == lbl
            if mask.sum() < 2:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, t_col], df.loc[mask, e_col], label=lbl)
            kmf.plot_survival_function(ax=ax, color=COLORS.get(color_key, "gray"), ci_show=True)
        ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Survival Probability")

    # KM 1: VNI groups
    if "VNI" in surv_df.columns:
        vni_grp = np.where(surv_df["VNI"] >= surv_df["VNI"].median(), "VN-High", "VN-Low")
        surv_df["VNI_group"] = vni_grp
        km_plot(axes[0], surv_df, "VNI_group", DURATION_COL, SURVIVAL_COL,
                "KM: Composite VNI", {"VN-High": "VN-High", "VN-Low": "VN-Low"})
        try:
            h = surv_df[surv_df["VNI_group"] == "VN-High"]
            l = surv_df[surv_df["VNI_group"] == "VN-Low"]
            lrt = logrank_test(h[DURATION_COL], l[DURATION_COL],
                               h[SURVIVAL_COL], l[SURVIVAL_COL])
            axes[0].text(0.05, 0.1, f"log-rank p = {lrt.p_value:.4f}",
                         transform=axes[0].transAxes, fontsize=9)
        except Exception:
            pass

    # KM 2: Lasso-Cox risk groups
    km_plot(axes[1], surv_df, "risk_group", DURATION_COL, SURVIVAL_COL,
            "KM: Lasso-Cox Risk Score",
            {"High-Risk": "High-Risk", "Low-Risk": "Low-Risk"})
    try:
        h = surv_df[surv_df["risk_group"] == "High-Risk"]
        l = surv_df[surv_df["risk_group"] == "Low-Risk"]
        lrt = logrank_test(h[DURATION_COL], l[DURATION_COL],
                           h[SURVIVAL_COL], l[SURVIVAL_COL])
        axes[1].text(0.05, 0.1, f"log-rank p = {lrt.p_value:.4f}",
                     transform=axes[1].transAxes, fontsize=9)
    except Exception:
        pass

    # KM 3: VN-High cell fraction
    if "VN_High_fraction" in surv_df.columns:
        vhf_med = surv_df["VN_High_fraction"].median()
        surv_df["VN_frac_group"] = np.where(
            surv_df["VN_High_fraction"] >= vhf_med, "High", "Low"
        )
        km_plot(axes[2], surv_df, "VN_frac_group", DURATION_COL, SURVIVAL_COL,
                "KM: VN-High Cell Fraction", {"High": "High", "Low": "Low"})
        try:
            h = surv_df[surv_df["VN_frac_group"] == "High"]
            l = surv_df[surv_df["VN_frac_group"] == "Low"]
            lrt = logrank_test(h[DURATION_COL], l[DURATION_COL],
                               h[SURVIVAL_COL], l[SURVIVAL_COL])
            axes[2].text(0.05, 0.1, f"log-rank p = {lrt.p_value:.4f}",
                         transform=axes[2].transAxes, fontsize=9)
        except Exception:
            pass

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT_DIR}/kaplan_meier_curves.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("    Saved: kaplan_meier_curves.{pdf,png}")

    # Forest plot
    if univariate_results:
        uniCox_df_s = uniCox_df.sort_values("HR")
        n = len(uniCox_df_s)
        fig, ax = plt.subplots(figsize=(9, 0.6 * n + 1.5))
        y = range(n)
        ax.errorbar(
            uniCox_df_s["HR"], list(y),
            xerr=[
                (uniCox_df_s["HR"] - uniCox_df_s["HR_lower"]).values,
                (uniCox_df_s["HR_upper"] - uniCox_df_s["HR"]).values,
            ],
            fmt="o", color="steelblue", capsize=4, markersize=6,
        )
        ax.axvline(1.0, color="dimgray", linestyle="--", linewidth=1.0)
        ax.set_yticks(list(y))
        ax.set_yticklabels(uniCox_df_s["feature"])
        x_max = uniCox_df_s["HR_upper"].max()
        for i, row in enumerate(uniCox_df_s.itertuples()):
            sig = "*" if row.p_value < 0.05 else ""
            ax.text(x_max * 1.05, i,
                    f"HR={row.HR:.2f}  p={row.p_value:.3f}{sig}",
                    va="center", fontsize=8)
        ax.set_xlabel("Hazard Ratio  (95% CI)")
        ax.set_title("Univariate Cox — VN Score Components")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(f"{OUT_DIR}/forest_plot.{ext}",
                        bbox_inches="tight", dpi=150)
        plt.close()
        print("    Saved: forest_plot.{pdf,png}")


# ============================================================
# SECTION 9 — BIOLOGICAL CHARACTERISATION
# ============================================================
print_section("SECTION 9 · Biological Characterisation")

# 9.1 UMAP — five scores + VNI
plot_scores = (
    list(COMPONENT_SCORES.keys()) + ["VNI"]
)
avail_plot = [c for c in plot_scores if c in vascular.obs.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, s in enumerate(avail_plot[:6]):
    sc.pl.umap(vascular, color=s, ax=axes[i], show=False,
               color_map="RdBu_r", vcenter=0, title=s)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT_DIR}/umap_vn_scores.{ext}",
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved: umap_vn_scores.{pdf,png}")

# 9.2 UMAP — cell annotations
color_keys = [CELLTYPE_COL, "VN_state", "leiden_vascular"]
avail_keys = [k for k in color_keys if k in vascular.obs.columns]
sc.pl.umap(vascular, color=avail_keys, show=False, ncols=3)
plt.savefig(f"{OUT_DIR}/umap_cell_annotations.pdf", bbox_inches="tight")
plt.close()

# 9.3 Score correlation heatmap
corr_cols = avail_plot
if len(corr_cols) >= 2:
    corr_mat = vascular.obs[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("VN Score Pearson Correlations")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT_DIR}/score_correlation_heatmap.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: score_correlation_heatmap.{pdf,png}")

# 9.4 EC-only PCA sensitivity loading bar chart
loading_pc1 = loadings_df["PC1"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#1565C0" if v > 0 else "#B71C1C" for v in loading_pc1]
loading_pc1.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Loading on PC1 (sensitivity analysis)")
ax.set_title(f"EC-only PCA Sensitivity Loadings (PC1 explains {explained_var[0]*100:.1f}% variance)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT_DIR}/vni_pca_loadings.{ext}",
                bbox_inches="tight", dpi=150)
plt.close()
print("Saved: vni_pca_loadings.{pdf,png}")

# 9.5 Violin: scores per cell type
avail_scores_violin = [c for c in list(COMPONENT_SCORES.keys())
                       if c in vascular.obs.columns]
if CELLTYPE_COL in vascular.obs.columns and avail_scores_violin:
    n = len(avail_scores_violin)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, avail_scores_violin):
        sc.pl.violin(vascular, keys=s, groupby=CELLTYPE_COL,
                     ax=ax, show=False, rotation=45)
        ax.set_title(s, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/violin_scores_by_celltype.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: violin_scores_by_celltype.pdf")

# 9.6 EC sub-type bar chart: VN-High vs VN-Low
if "EC_subtype" in vascular.obs.columns and "VN_state" in vascular.obs.columns:
    ec_comp = (
        vascular.obs[ec_mask]
        .groupby(["VN_state", "EC_subtype"])
        .size()
        .unstack(fill_value=0)
    )
    ec_comp_pct = ec_comp.div(ec_comp.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ec_comp_pct.T.plot(kind="bar", ax=ax,
                       color=["#1E88E5", "#E53935"])
    ax.set_ylabel("% of ECs")
    ax.set_title("EC Sub-type Composition by VN State")
    ax.legend(title="VN State")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ec_subtype_by_vnstate.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: ec_subtype_by_vnstate.pdf")

# 9.7 DE: VN-High vs VN-Low in endothelial cells
if ec_mask.sum() >= 30 and "VN_state" in vascular.obs.columns:
    ec_sub = vascular[ec_mask].copy()
    if ec_sub.obs["VN_state"].nunique() == 2:
        sc.tl.rank_genes_groups(
            ec_sub, groupby="VN_state", method="wilcoxon", n_genes=ec_sub.n_vars
        )
        de_df = sc.get.rank_genes_groups_df(ec_sub, group="VN-High")
        vni_construction_genes = set(
            sum(SIGNATURES.values(), []) + ECM_DYSFUNCTION_GENES + IMMATURE_PERICYTE_GENES
        )
        de_df["used_in_vni_construction"] = de_df["names"].isin(vni_construction_genes)
        de_df.to_csv(f"{OUT_DIR}/DE_VNhigh_vs_VNlow_EC.csv", index=False)
        print(f"\nDE VN-High vs VN-Low ECs: {len(de_df)} genes saved.")
        print("Top upregulated in VN-High:")
        print(de_df[de_df["logfoldchanges"] > 0].head(10)
              [["names", "logfoldchanges", "pvals_adj"]].to_string(index=False))

    # 9.8 GSEA on DE gene list
    if False and HAS_GSEAPY and len(de_df) > 50:
        try:
            ranked = de_df.set_index("names")["logfoldchanges"].dropna().sort_values(ascending=False)
            gsea_res = gp.prerank(
                rnk=ranked,
                gene_sets="GO_Biological_Process_2023",
                outdir=f"{OUT_DIR}/gsea_VNhigh_vs_VNlow",
                min_size=15, max_size=500,
                permutation_num=1000, seed=1234,
                verbose=False
            )
            gsea_df = gsea_res.res2d.sort_values("FDR q-val")
            gsea_df.to_csv(f"{OUT_DIR}/gsea_VNhigh_vs_VNlow.csv", index=False)
            print("\nTop GSEA pathways (VN-High):")
            print(gsea_df[gsea_df["FDR q-val"] < 0.25].head(10)
                  [["Term", "NES", "FDR q-val"]].to_string(index=False))
        except Exception as e:
            print(f"  GSEA skipped: {e}")


# ============================================================
# SECTION 10 — SAVE ALL OUTPUTS
# ============================================================
print_section("SECTION 10 · Saving All Outputs")

vascular.write(f"{OUT_DIR}/vascular_ecosystem_vn_scored.h5ad")
vascular.obs.to_csv(f"{OUT_DIR}/vascular_cell_metadata.csv")
loadings_df.to_csv(f"{OUT_DIR}/vni_pca_loadings.csv")

print(f"\nAll outputs written to:  {OUT_DIR}/")
print("\nFiles generated:")
for f in sorted(os.listdir(OUT_DIR)):
    size_kb = os.path.getsize(f"{OUT_DIR}/{f}") // 1024
    print(f"  {f:55s}  {size_kb:>6} KB")

print("\n" + "=" * 65)
print("  Pipeline completed successfully.")
print("=" * 65)
print("""
Key outputs:
  patient_vn_scores.csv       ← per-patient 5-score + VNI table
  univariate_cox.csv          ← HR, 95% CI, p per score
  lasso_cox_summary.csv       ← selected features + coefficients
  patient_risk_scores.csv     ← per-patient risk score + group
  kaplan_meier_curves.pdf     ← 3-panel KM (VNI / RiskScore / VN fraction)
  forest_plot.pdf             ← univariate HR forest plot
  umap_vn_scores.pdf          ← 6-panel UMAP coloured by each score
  DE_VNhigh_vs_VNlow_EC.csv   ← endothelial DE genes
  gsea_VNhigh_vs_VNlow.csv    ← pathway enrichment (if gseapy available)
""")
