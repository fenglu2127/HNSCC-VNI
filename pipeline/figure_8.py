from common import *
from common import _ec_module_bar
import common as figures
def Figure8A():
    mods = ["EC_maturation_phalanx_score", "EC_tip_sprouting_score", "EC_stalk_Notch_score",
            "pathological_TEC_embryonic_score", "abnormal_angiogenesis_score", "hypoxia_acidosis_score"]
    labels = ["EC maturation\n(Phalanx)", "EC tip\nsprouting", "Stalk/\nNotch", "Pathological TEC\nembryonic program",
              "Abnormal\nangiogenesis", "Hypoxia/\nacidosis"]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    _ec_module_bar(ax, mods, labels, "Figure 8A. Vascular Structure and Abnormal Growth Axes\n(VN-High vs VN-Low endothelial cells)")
    save(fig, "Figure8A.png")

def Figure8B():
    ec_mods = ["EC_antigen_presentation_score", "EC_leukocyte_adhesion_score",
               "EC_death_ligand_barrier_score", "CXCL9_10_Tcell_recruitment_score"]
    ec_lab = ["EC antigen presentation\n(HLA/B2M/TAP)", "Leukocyte adhesion\n(ICAM/VCAM/SELE)",
              "Death-ligand barrier\n(FASLG/PD-L1)", "CXCL9/10\nT-cell recruitment"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle("Figure 8B. Endothelial Immune Gate and T-Cell Quality", fontsize=13, fontweight="bold")
    _ec_module_bar(axes[0], ec_mods, ec_lab, "EC Immune-Gate Module Scores")
    # right: T-cell quality by neighborhood VN context (T cells split by nearest-EC VNI median)
    tc = pd.read_csv(D("tcell_scored.csv"), index_col=0, low_memory=False)
    tc["nbrVNI"] = knn_vni(tc)
    j = tc.dropna(subset=["nbrVNI"]).copy()
    med = j["nbrVNI"].median(); hi = j[j.nbrVNI > med]; lo = j[j.nbrVNI <= med]
    tmods = ["CD8_Cytotoxic", "Treg_Activity"]
    tlab = ["Cytotoxic T cells\n(CD8/GZMB/PRF1)", "Dysfunctional/Treg\n(PD1/FOXP3)"]
    ax = axes[1]; x = np.arange(2); w = 0.38
    mh = [hi[m].mean() for m in tmods]; sh = [hi[m].sem() for m in tmods]
    ml = [lo[m].mean() for m in tmods]; sl = [lo[m].sem() for m in tmods]
    ps = [stats.mannwhitneyu(hi[m].dropna(), lo[m].dropna()).pvalue for m in tmods]
    ax.bar(x - w/2, mh, w, yerr=sh, capsize=2.5, color=BLUE, label="T cells in high-VN context")
    ax.bar(x + w/2, ml, w, yerr=sl, capsize=2.5, color=RED, label="T cells in low-VN context")
    for i, p in enumerate(ps):
        ax.text(x[i], max(mh[i]+sh[i], ml[i]+sl[i]) + 0.01, stars(p), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(tlab, fontsize=9); ax.set_ylabel("Mean module score ± SEM")
    ax.set_title("T-Cell Quality by VN Context", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", ls="--", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "Figure8B.png")
MODULES = [
    "CAF_mechanical_compression_score",
    "ECM_remodeling_MMP_LOX_score",
    "YAP_TAZ_mechanotransduction_score",
    "glycolysis_lactate_score",
    "adenosine_immunosuppression_score",
    "SPP1_TAM_remodeling_score",
    "IL1B_inflammatory_myeloid_score",
]

LABELS = [
    "CAF mechanical\ncompression",
    "ECM remodeling\n(MMP/LOX)",
    "YAP/TAZ\nmechanotransduction",
    "Glycolysis /\nlactate",
    "Adenosine\nimmunosuppression",
    "SPP1+ TAM\nremodeling",
    "IL1B inflammatory\nmyeloid",
]

def main() -> None:
    figures.ec_review_split = ec_five_axis_split
    plt.rcParams["savefig.dpi"] = 300

    fig, ax = plt.subplots(figsize=(13, 5.6))
    figures._ec_module_bar(
        ax,
        MODULES,
        LABELS,
        "Figure 8C. Stromal Compression, Metabolic Immunosuppression, and Myeloid Inflammatory Axes\n"
        "(VNI-High vs VNI-Low endothelial cells)",
    )
    fig.tight_layout()

    png = OUT / "Figure8C.png"
    pdf = OUT / "Figure8C.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    df = ec_five_axis_split(MODULES)
    high = df[df["VN_split"] == "VN-High"]
    low = df[df["VN_split"] == "VN-Low"]
    rows = []
    for module, label in zip(MODULES, LABELS):
        h = pd.to_numeric(high[module], errors="coerce").dropna()
        l = pd.to_numeric(low[module], errors="coerce").dropna()
        test = stats.mannwhitneyu(h, l, alternative="two-sided")
        rows.append({
            "module": module,
            "display_label": label.replace("\n", " "),
            "grouping": "VNI-High >= EC median; VNI-Low < EC median",
            "high_n": len(h),
            "low_n": len(l),
            "high_mean": h.mean(),
            "low_mean": l.mean(),
            "high_sem": h.sem(),
            "low_sem": l.sem(),
            "mannwhitney_u": test.statistic,
            "p_two_sided": test.pvalue,
        })
    stats_path = OUT / "Figure8C_statistics.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {stats_path}")
