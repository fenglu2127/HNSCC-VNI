#!/usr/bin/env python3
"""Independent TCGA-HNSC survival analysis using the fixed five-axis VNI.

Results are written to outputs/tcga_five_axis_survival.

Primary VNI definition:
    mean[z(VM), z(PC - immature-PC), -z(HDA), z(IVI),
         z(EBI - ECM-dysfunction)]

The gene sets, directions, and equal weights are fixed before survival testing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "raw_data" / "tcga"
OUT = PKG / "outputs" / "tcga_five_axis_survival"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

EXPR_PATH = RAW / "tcga_hnsc_fpkm_matrix.csv.gz"
MAP_PATH = RAW / "ensembl_symbol_map.csv"
CLIN_PATH = RAW / "tcga_hnsc_clinical_full.csv"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


SIGNATURES = {
    "VM": [
        "PECAM1", "CDH5", "KDR", "TEK", "TIE1", "ENG", "NRP1",
        "CLDN5", "OCLN", "ESAM", "GJA4", "GJA5", "EFNB2", "HEY1",
        "HEY2", "NR2F2", "EMCN", "APLNR", "EPHB4", "DLL4", "JAG1",
        "NOTCH4", "FLT1",
    ],
    "PC_positive": [
        "PDGFRB", "CSPG4", "ANGPT1", "MCAM", "DES", "TAGLN", "MYH11",
        "NOTCH3", "ACTA2",
    ],
    "PC_immature": ["RGS5", "HEYL", "MCAM"],
    "HDA": [
        "HIF1A", "EPAS1", "VEGFA", "VEGFC", "ANGPT2", "CA9", "LDHA",
        "PDK1", "SLC2A1", "BNIP3", "BNIP3L", "NDRG1", "ESM1", "APLN",
        "CXCR4", "CXCL8", "S100A8", "S100A9", "MKI67",
    ],
    "IVI": [
        "ICAM1", "VCAM1", "SELE", "SELP", "CXCL9", "CXCL10", "CXCL11",
        "CCL2", "CCL5", "HLA-A", "HLA-B", "HLA-C", "TNFSF10", "CD274",
        "PDCD1LG2",
    ],
    "EBI_positive": [
        "COL4A1", "COL4A2", "COL4A3", "LAMA4", "LAMB1", "LAMB2", "LAMC1",
        "HSPG2", "NID1", "NID2", "TIMP1", "TIMP2",
    ],
    "EBI_dysfunction": ["MMP2", "MMP9", "MMP14", "MMP1", "FN1", "CTGF", "CCN2"],
}

AXES = ["VM_z", "PC_adj_z", "HDA_inv_z", "IVI_z", "EBI_adj_z"]
AXIS_LABELS = {
    "VM_z": "VM",
    "PC_adj_z": "PC-adjusted",
    "HDA_inv_z": "HDA-inverted",
    "IVI_z": "IVI",
    "EBI_adj_z": "EBI-adjusted",
    "VNI": "Five-axis VNI",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else s * np.nan


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p values, preserving the original index."""
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.notna()
    pv = p.loc[valid].to_numpy()
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out = pd.Series(np.nan, index=p.index, dtype=float)
    out.loc[valid] = restored
    return out


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    plt.close(fig)


def stage_number(value) -> float:
    if pd.isna(value):
        return np.nan
    x = str(value).upper().replace("STAGE", "").strip()
    if x.startswith("IV"):
        return 4.0
    if x.startswith("III"):
        return 3.0
    if x.startswith("II"):
        return 2.0
    if x.startswith("I"):
        return 1.0
    if x.startswith("0"):
        return 0.0
    return np.nan


def load_expression() -> tuple[pd.DataFrame, dict]:
    print(f"Loading expression: {EXPR_PATH}")
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr.index = expr.index.astype(str).str.replace(r"\.\d+$", "", regex=True)

    mapping = pd.read_csv(MAP_PATH, usecols=["query", "symbol"]).dropna()
    mapping["query"] = mapping["query"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    symbol_map = dict(zip(mapping["query"], mapping["symbol"].astype(str)))
    symbols = pd.Index([symbol_map.get(g, g) for g in expr.index], name="symbol")
    mapped_n = int(sum(g in symbol_map for g in expr.index))
    expr.index = symbols
    # TPM/FPKM-like gene-level values are additive when multiple Ensembl IDs map
    # to the same gene symbol; do not arbitrarily keep the first duplicate.
    expr = expr.groupby(level=0, sort=False).sum()
    expr = np.log2(expr + 1.0)
    info = {"mapped_ensembl_ids": mapped_n, "gene_symbols_after_aggregation": int(expr.shape[0])}
    print(f"Expression after symbol aggregation: {expr.shape[0]:,} genes x {expr.shape[1]:,} tumors")
    return expr, info


def run_ssgsea(expr: pd.DataFrame) -> pd.DataFrame:
    coverage = []
    genes = set(expr.index)
    for name, signature in SIGNATURES.items():
        present = sorted(set(signature) & genes)
        coverage.append({
            "signature": name,
            "n_defined": len(signature),
            "n_present": len(present),
            "genes_present": ";".join(present),
        })
    pd.DataFrame(coverage).to_csv(OUT / "signature_gene_coverage.csv", index=False)

    print("Running fresh ssGSEA for seven fixed modules...")
    result = gp.ssgsea(
        data=expr,
        gene_sets=SIGNATURES,
        outdir=None,
        sample_norm_method="rank",
        min_size=2,
        max_size=500,
        no_plot=True,
        threads=4,
        seed=1234,
        verbose=False,
    )
    res = result.res2d.copy()
    score_col = "ES" if "ES" in res.columns else "NES" if "NES" in res.columns else None
    if score_col is None:
        numeric = res.select_dtypes(include=[np.number]).columns
        if not len(numeric):
            raise RuntimeError("Could not identify ssGSEA score column")
        score_col = numeric[0]
    scores = res.pivot(index="Name", columns="Term", values=score_col)
    scores = scores.apply(pd.to_numeric, errors="coerce")
    scores.index.name = "patient_id"
    scores.to_csv(OUT / "ssgsea_seven_module_scores.csv")
    return scores


def build_vni(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["PC_adj_raw"] = out["PC_positive"] - out["PC_immature"]
    out["EBI_adj_raw"] = out["EBI_positive"] - out["EBI_dysfunction"]
    out["VM_z"] = zscore(out["VM"])
    out["PC_adj_z"] = zscore(out["PC_adj_raw"])
    out["HDA_inv_z"] = -zscore(out["HDA"])
    out["IVI_z"] = zscore(out["IVI"])
    out["EBI_adj_z"] = zscore(out["EBI_adj_raw"])
    out["VNI"] = out[AXES].mean(axis=1)
    out["VNI_group_median"] = np.where(out["VNI"] >= out["VNI"].median(), "High", "Low")
    q1, q3 = out["VNI"].quantile([0.25, 0.75])
    out["VNI_group_quartile"] = np.where(
        out["VNI"] <= q1, "Q1-Low", np.where(out["VNI"] >= q3, "Q4-High", "Middle")
    )
    out.to_csv(OUT / "five_axis_vni_scores.csv")
    pd.DataFrame({
        "component": AXES,
        "direction_in_vni": ["positive"] * 5,
        "weight": [0.2] * 5,
    }).to_csv(OUT / "vni_fixed_formula.csv", index=False)
    return out


def prepare_survival(vni: pd.DataFrame) -> pd.DataFrame:
    clinical = pd.read_csv(CLIN_PATH).drop_duplicates("patient_id").set_index("patient_id")
    d = vni.join(clinical, how="inner")
    d = d[(d["OS_time_months"] > 0) & d["OS_event"].isin([0, 1])].copy()
    d["age_z"] = zscore(d["age"])
    d["male"] = d["gender"].map({"male": 1.0, "female": 0.0})
    d["stage_ord"] = d["overall_stage"].map(stage_number)
    d.index.name = "patient_id"
    d.to_csv(OUT / "five_axis_vni_survival_data.csv")
    return d


def fit_cox_models(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CoxPHFitter, pd.DataFrame]:
    rows = []
    ph_rows = []
    for feature in AXES + ["VNI"]:
        x = d[["OS_time_months", "OS_event", feature]].dropna()
        model = CoxPHFitter()
        model.fit(x, "OS_time_months", "OS_event")
        s = model.summary.loc[feature]
        rows.append({
            "model": "univariate", "feature": feature, "n": len(x),
            "events": int(x.OS_event.sum()), "HR": float(np.exp(s["coef"])),
            "CI95_low": float(np.exp(s["coef lower 95%"])),
            "CI95_high": float(np.exp(s["coef upper 95%"])),
            "p": float(s["p"]), "training_c_index": float(model.concordance_index_),
        })
        ph = proportional_hazard_test(model, x, time_transform="rank").summary
        ph_rows.append({"model": f"univariate_{feature}", "feature": feature,
                        "ph_test_p": float(ph.loc[feature, "p"])})

    adjusted_features = ["VNI", "age_z", "male", "stage_ord"]
    adj = d[["OS_time_months", "OS_event"] + adjusted_features].dropna()
    adjusted = CoxPHFitter()
    adjusted.fit(adj, "OS_time_months", "OS_event")
    for feature, s in adjusted.summary.iterrows():
        rows.append({
            "model": "adjusted_VNI_age_gender_stage", "feature": feature,
            "n": len(adj), "events": int(adj.OS_event.sum()),
            "HR": float(np.exp(s["coef"])),
            "CI95_low": float(np.exp(s["coef lower 95%"])),
            "CI95_high": float(np.exp(s["coef upper 95%"])),
            "p": float(s["p"]), "training_c_index": float(adjusted.concordance_index_),
        })
    ph = proportional_hazard_test(adjusted, adj, time_transform="rank").summary
    for feature in adjusted_features:
        ph_rows.append({"model": "adjusted_VNI_age_gender_stage", "feature": feature,
                        "ph_test_p": float(ph.loc[feature, "p"])})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "cox_results.csv", index=False)
    ph_results = pd.DataFrame(ph_rows)
    ph_results.to_csv(OUT / "proportional_hazards_tests.csv", index=False)
    adjusted.summary.to_csv(OUT / "adjusted_cox_full_summary.csv")

    def add_multivariable(model_name: str, features: list[str]):
        x = d[["OS_time_months", "OS_event"] + features].dropna()
        model = CoxPHFitter()
        model.fit(x, "OS_time_months", "OS_event")
        for feature, s in model.summary.iterrows():
            rows.append({
                "model": model_name, "feature": feature, "n": len(x),
                "events": int(x.OS_event.sum()), "HR": float(np.exp(s["coef"])),
                "CI95_low": float(np.exp(s["coef lower 95%"])),
                "CI95_high": float(np.exp(s["coef upper 95%"])),
                "p": float(s["p"]), "training_c_index": float(model.concordance_index_),
            })
        ph_test = proportional_hazard_test(model, x, time_transform="rank").summary
        for feature in features:
            ph_rows.append({"model": model_name, "feature": feature,
                            "ph_test_p": float(ph_test.loc[feature, "p"])})
        model.summary.to_csv(OUT / f"{model_name}_full_summary.csv")

    add_multivariable("five_axes_joint", AXES)
    add_multivariable("clinical_only", ["age_z", "male", "stage_ord"])
    add_multivariable("five_axes_age_gender_stage", AXES + ["age_z", "male", "stage_ord"])

    results = pd.DataFrame(rows)
    results["BH_FDR_q"] = np.nan
    uni_mask = results["model"].eq("univariate")
    results.loc[uni_mask, "BH_FDR_q"] = bh_fdr(results.loc[uni_mask, "p"])
    results.to_csv(OUT / "cox_results.csv", index=False)
    ph_results = pd.DataFrame(ph_rows)
    ph_results.to_csv(OUT / "proportional_hazards_tests.csv", index=False)
    return results, ph_results, adjusted, adj


def repeated_cv(d: pd.DataFrame, features: list[str], model_name: str,
                repeats: int = 20, folds: int = 5) -> pd.DataFrame:
    x = d[["OS_time_months", "OS_event"] + features].dropna().copy()
    event = x["OS_event"].astype(int).to_numpy()
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234 + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, event), start=1):
            train = x.iloc[train_idx].copy()
            test = x.iloc[test_idx].copy()
            model = CoxPHFitter(penalizer=0.0)
            model.fit(train, "OS_time_months", "OS_event")
            risk = model.predict_partial_hazard(test).to_numpy().ravel()
            cidx = concordance_index(test["OS_time_months"], -risk, test["OS_event"])
            rows.append({"model": model_name, "repeat": repeat + 1, "fold": fold,
                         "n_train": len(train), "n_test": len(test), "c_index": cidx})
    return pd.DataFrame(rows)


def km_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = {
        "VNI_median": (d.VNI_group_median == "Low", d.VNI_group_median == "High"),
        "VNI_Q1_vs_Q4": (d.VNI_group_quartile == "Q1-Low", d.VNI_group_quartile == "Q4-High"),
    }
    for feature in AXES:
        med = d[feature].median()
        comparisons[f"{feature}_median"] = (d[feature] < med, d[feature] >= med)
    for name, (low_mask, high_mask) in comparisons.items():
        low = d.loc[low_mask]
        high = d.loc[high_mask]
        lr = logrank_test(low.OS_time_months, high.OS_time_months,
                          low.OS_event, high.OS_event)
        rows.append({
            "comparison": name, "n_low": len(low), "events_low": int(low.OS_event.sum()),
            "n_high": len(high), "events_high": int(high.OS_event.sum()),
            "logrank_p": float(lr.p_value),
        })
    out = pd.DataFrame(rows)
    out["BH_FDR_q"] = np.nan
    axes_mask = out["comparison"].isin([f"{x}_median" for x in AXES])
    out.loc[axes_mask, "BH_FDR_q"] = bh_fdr(out.loc[axes_mask, "logrank_p"])
    out.to_csv(OUT / "km_logrank_results.csv", index=False)
    return out


def draw_km(ax, low: pd.DataFrame, high: pd.DataFrame, label_low: str, label_high: str,
            xlim: tuple[int, int] = (0, 120)):
    from lifelines.plotting import add_at_risk_counts
    km_low = KaplanMeierFitter(label=f"{label_low} (n={len(low)})")
    km_high = KaplanMeierFitter(label=f"{label_high} (n={len(high)})")
    km_low.fit(low.OS_time_months, low.OS_event).plot_survival_function(
        ax=ax, ci_show=True, color=BLUE, linewidth=1.8)
    km_high.fit(high.OS_time_months, high.OS_event).plot_survival_function(
        ax=ax, ci_show=True, color=ORANGE, linewidth=1.8)
    lr = logrank_test(low.OS_time_months, high.OS_time_months,
                      low.OS_event, high.OS_event)
    ax.text(0.03, 0.08, f"Log-rank p={lr.p_value:.3g}", transform=ax.transAxes)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(frameon=False, loc="upper right")
    add_at_risk_counts(km_low, km_high, ax=ax, rows_to_show=["At risk"])


def make_figures(d: pd.DataFrame, cox: pd.DataFrame) -> None:
    low = d[d.VNI_group_median == "Low"]
    high = d[d.VNI_group_median == "High"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    draw_km(ax, low, high, "VNI-Low", "VNI-High")
    save_figure(fig, "km_vni_median")

    q1 = d[d.VNI_group_quartile == "Q1-Low"]
    q4 = d[d.VNI_group_quartile == "Q4-High"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    draw_km(ax, q1, q4, "VNI Q1", "VNI Q4")
    save_figure(fig, "km_vni_q1_q4")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2))
    for panel, (ax, feature) in enumerate(zip(axes.ravel(), AXES), start=1):
        med = d[feature].median()
        lo, hi = d[d[feature] < med], d[d[feature] >= med]
        draw_km(ax, lo, hi, f"{AXIS_LABELS[feature]} Low", f"{AXIS_LABELS[feature]} High",
                xlim=(0, 120))
        ax.text(-0.13, 1.05, chr(64 + panel), transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    save_figure(fig, "km_five_axes_median")

    uni = cox[(cox.model == "univariate")].copy().sort_values("HR")
    y = np.arange(len(uni))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(uni.HR, y,
                xerr=[uni.HR - uni.CI95_low, uni.CI95_high - uni.HR],
                fmt="o", color=GREEN, ecolor="#555555", capsize=3)
    ax.axvline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([AXIS_LABELS.get(x, x) for x in uni.feature])
    ax.set_xlabel("Hazard ratio per 1-SD increase (95% CI)")
    save_figure(fig, "cox_univariate_forest")


def write_latex() -> None:
    text = r"""% Fixed five-axis TCGA-HNSC survival figures (all-comer cohort; HPV status unavailable)
\begin{figure}[t]
\centering
\includegraphics[width=0.62\textwidth]{km_vni_median.pdf}
\caption{Overall survival in TCGA-HNSC stratified at the median of the fixed five-axis vascular normalization index (VNI). The VNI is the equal-weight mean of direction-aligned VM, PC-adjusted, HDA-inverted, IVI, and EBI-adjusted ssGSEA axes. Bands show 95\% confidence intervals; the table reports numbers at risk. HPV status was unavailable in the bundled clinical table.}
\label{fig:tcga_fixed_vni_median}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.62\textwidth]{km_vni_q1_q4.pdf}
\caption{Overall survival comparison between the lowest and highest quartiles of the fixed five-axis VNI in TCGA-HNSC. Middle-quartile tumors are excluded from this descriptive contrast.}
\label{fig:tcga_fixed_vni_quartiles}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{km_five_axes_median.pdf}
\caption{Median-split overall-survival curves for the five direction-aligned VNI component axes in TCGA-HNSC. These analyses identify which component, if any, drives an association of the composite score with survival.}
\label{fig:tcga_fixed_vni_axes}
\end{figure*}
"""
    (FIG / "latex_includes.tex").write_text(text, encoding="utf-8")


def main() -> None:
    expr, load_info = load_expression()
    scores = run_ssgsea(expr)
    vni = build_vni(scores)
    survival = prepare_survival(vni)
    cox, ph, _, adjusted_data = fit_cox_models(survival)

    cv_uni = repeated_cv(survival, ["VNI"], "VNI_univariate")
    cv_adj = repeated_cv(adjusted_data, ["VNI", "age_z", "male", "stage_ord"],
                         "VNI_age_gender_stage")
    cv_axes = repeated_cv(survival, AXES, "five_axes_joint")
    cv_clinical = repeated_cv(survival, ["age_z", "male", "stage_ord"], "clinical_only")
    cv_axes_adj = repeated_cv(
        survival, AXES + ["age_z", "male", "stage_ord"], "five_axes_age_gender_stage"
    )
    cv = pd.concat([cv_uni, cv_adj, cv_axes, cv_clinical, cv_axes_adj], ignore_index=True)
    cv.to_csv(OUT / "repeated_5fold_cv_cindex.csv", index=False)
    cv_summary = cv.groupby("model").c_index.agg(["count", "mean", "std", "median"])
    quantiles = cv.groupby("model").c_index.quantile([0.025, 0.975]).unstack()
    quantiles.columns = ["q025", "q975"]
    cv_summary = cv_summary.join(quantiles).reset_index()
    cv_summary.to_csv(OUT / "repeated_5fold_cv_summary.csv", index=False)

    km = km_stats(survival)
    make_figures(survival, cox)
    write_latex()

    manifest = {
        "analysis": "fixed five-axis VNI TCGA-HNSC survival",
        "hpv_filter": "not applied: HPV status absent from bundled clinical table",
        "inputs": {
            str(EXPR_PATH): sha256(EXPR_PATH),
            str(MAP_PATH): sha256(MAP_PATH),
            str(CLIN_PATH): sha256(CLIN_PATH),
        },
        "expression_loading": load_info,
        "n_expression_tumors": int(scores.shape[0]),
        "n_survival_tumors": int(len(survival)),
        "n_adjusted_complete_cases": int(len(adjusted_data)),
        "vni_formula": "mean(VM_z, PC_adj_z, HDA_inv_z, IVI_z, EBI_adj_z)",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCox results")
    print(cox.to_string(index=False))
    print("\nRepeated 5-fold CV")
    print(cv_summary.to_string(index=False))
    print("\nKM log-rank results")
    print(km.to_string(index=False))
    print(f"\nAll new outputs: {OUT}")


if __name__ == "__main__":
    main()
