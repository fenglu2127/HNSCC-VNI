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

def figure8b_statistics():
    """Recompute cell-level summaries using EC and T-cell-context medians.

    Treg_Activity comprises FOXP3, IL2RA, IKZF2, IL10 and TGFB1;
    it is not the separate T_Exhaustion module and does not include PDCD1.
    """
    ec_mods = ["EC_antigen_presentation_score", "EC_leukocyte_adhesion_score",
               "EC_death_ligand_barrier_score", "CXCL9_10_Tcell_recruitment_score"]
    ec = ec_five_axis_split(ec_mods)
    tc = pd.read_csv(D("tcell_scored.csv"), index_col=0, low_memory=False)
    tc["nbrVNI"] = knn_vni(tc)
    j = tc.dropna(subset=["nbrVNI"]).copy()
    med = j["nbrVNI"].median()
    groups = [
        ("Fig8B_left", ec_mods, ec[ec.VN_split == "VN-High"],
         ec[ec.VN_split == "VN-Low"], "five_axis_VNI",
         "High >= EC median; Low < EC median"),
        ("Fig8B_right", ["CD8_Cytotoxic", "Treg_Activity"],
         j[j.nbrVNI >= med], j[j.nbrVNI < med],
         "mean_VNI_of_10_nearest_ECs_in_atlas_UMAP",
         "High >= T-cell neighbor-VNI median; Low < median"),
    ]
    rows = []
    for panel, modules, high, low, variable, rule in groups:
        for module in modules:
            h = pd.to_numeric(high[module], errors="coerce").dropna()
            l = pd.to_numeric(low[module], errors="coerce").dropna()
            test = stats.mannwhitneyu(h, l, alternative="two-sided")
            rows.append(dict(panel=panel, module=module,
                grouping_variable=variable, threshold_rule=rule,
                high_n=len(h), low_n=len(l), high_mean=h.mean(), low_mean=l.mean(),
                high_sem=h.sem(), low_sem=l.sem(), high_minus_low=h.mean()-l.mean(),
                mannwhitney_u=test.statistic, p_two_sided=test.pvalue))
    return pd.DataFrame(rows)

def Figure8B(validate_layout=None):
    """Render EC immune-interface and T-cell modules from recomputed summaries.

    Optional validate_layout(fig) supports external QA without requiring a
    machine-specific plotting toolkit for normal repository execution.
    """
    d = figure8b_statistics()
    style = {'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7, 'axes.titlesize':9, 'axes.labelsize':8, 'xtick.labelsize':7,
        'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42,
        'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.2,4.03),
                                 gridspec_kw={'width_ratios':[1.65,1]})
        labels = [
            ['EC antigen\npresentation\n(HLA/B2M/TAP)',
             'Leukocyte\nadhesion\n(ICAM/VCAM/SELE)',
             'Death-ligand\nbarrier\n(FASLG/PD-L1)', 'CXCL9/10\nT-cell\nrecruitment'],
            ['Cytotoxic T cells\n(CD8/GZMB/PRF1)', 'Treg-associated\nmodule']]
        for ax, panel, labs, title in zip(axes, ['Fig8B_left','Fig8B_right'], labels,
                                         ['EC immune-gate modules','T-cell context']):
            s = d[d.panel == panel]
            x = np.arange(len(s)); w = .38
            for off, group, color, lab in [
                (-w/2,'high',BLUE,'VNI-high EC' if panel.endswith('left') else 'High VNI'),
                (w/2,'low',RED,'VNI-low EC' if panel.endswith('left') else 'Low VNI')]:
                ax.bar(x+off,s[group+'_mean'],w,yerr=s[group+'_sem'],capsize=1.8,
                       color=color,label=lab,error_kw={'elinewidth':.8})
            for i, row in enumerate(s.itertuples()):
                y = max(row.high_mean+row.high_sem,row.low_mean+row.low_sem)+.035
                negative = y < 0
                if negative:
                    y = min(row.high_mean-row.high_sem,row.low_mean-row.low_sem)-.035
                ax.text(i,y,stars(row.p_two_sided),ha='center',
                        va='top' if negative else 'bottom',fontsize=7,fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(labs,rotation=0,ha='center')
            ax.set_title(title,fontweight='bold',pad=8)
            ax.set_ylabel('Mean module score ± SEM')
            ax.axhline(0,color='#555555',lw=.7)
            ax.grid(False); ax.set_axisbelow(True)
            ax.legend(loc='upper right' if panel.endswith('left') else 'center right')
        axes[0].set_ylim(-.36,1.93); axes[1].set_ylim(-.4,.95)
        fig.subplots_adjust(left=.115,right=.985,bottom=.32,top=.90,wspace=.34)
        fig.canvas.draw()
        if validate_layout is not None:
            validate_layout(fig)
        out = Path(figures.FIG)
        fig.savefig(out/'Figure8B.png',dpi=600,facecolor='white')
        fig.savefig(out/'Figure8B.pdf',facecolor='white')
        fig.savefig(out/'Figure8B.svg',facecolor='white')
        plt.close(fig)
    d.to_csv(out/'Figure8B_statistics.csv',index=False)
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
