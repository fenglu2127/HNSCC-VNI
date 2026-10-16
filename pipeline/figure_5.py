from common import *
import common as figures
def Figure5A():
    frac = np.load(D("knn_tcell_fracs.npy"))
    m = meta(); ec = m[m["EC_subtype"].notna()]  # VN_state column split (matches manuscript Fig 5A)
    assert len(frac) == len(ec), f"{len(frac)} vs {len(ec)}"
    hi = frac[(ec["VN_state"] == "VN-High").values]
    lo = frac[(ec["VN_state"] == "VN-Low").values]
    # Nearly all ECs have zero T-cell neighbors in transcriptomic UMAP.  Plot the
    # prevalence of any T-cell adjacency instead of an unreadable zero-inflated violin.
    table = np.array([[(hi > 0).sum(), (hi == 0).sum()],
                      [(lo > 0).sum(), (lo == 0).sum()]])
    _, p = stats.fisher_exact(table)
    any_pct = 100 * np.array([(hi > 0).mean(), (lo > 0).mean()])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bars = ax.bar([0, 1], any_pct, color=[BLUE, RED], width=0.62)
    for i, (bar, mean_frac) in enumerate(zip(bars, [hi.mean(), lo.mean()])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{bar.get_height():.2f}%\nmean fraction={mean_frac:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["VN-High", "VN-Low"])
    ax.set_ylabel("ECs with ≥1 T-cell UMAP neighbor (%)")
    ax.set_ylim(0, max(any_pct) * 1.35)
    ax.set_title(f"T-cell adjacency in transcriptomic UMAP\n(Fisher exact p = {p:.2e})", fontweight="bold")
    save(fig, "Figure5A.png")
    print(f"    [check5A] Fisher p={p:.2e} any-neighbor hi/lo={any_pct[0]:.2f}%/{any_pct[1]:.2f}% "
          f"mean-fraction hi/lo={hi.mean():.3f}/{lo.mean():.3f}")

def _order_q1q4(prop):
    """Return prop with rows ordered [Q1, Q4], matched by 'Q1'/'Q4' in the index label
    (robust to row order and to Chinese or English index text)."""
    idx = [str(i) for i in prop.index]
    q1 = next((i for i in idx if "Q1" in i), idx[0])
    q4 = next((i for i in idx if "Q4" in i), idx[-1])
    prop = prop.loc[[q1, q4]].copy()
    prop.index = ["Q1 (low VNI)", "Q4 (high VNI)"]
    return prop

def Figure5B():
    # T-cell and macrophage/myeloid SUBTYPE composition by neighborhood VNI quartile (Q1 vs Q4)
    pt = _order_q1q4(pd.read_csv(D("tcell_subtype_proportion.csv"), index_col=0))
    pm = _order_q1q4(pd.read_csv(D("myeloid_subtype_proportion.csv"), index_col=0))
    t_cols = ["CD8T", "CytotoxicT/NK", "Dysfunctional", "Naive", "Tcm", "Tfh", "Treg", "Proliferating"]
    m_cols = ["Macrophages", "CD14+ Mono", "Mac/Mono/Neut", "cDC", "CD16+ Mono", "Mast", "pDC"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, prop, cols, lab in [(axes[0], pt, t_cols, "T-cell"), (axes[1], pm, m_cols, "Myeloid")]:
        cols = [c for c in cols if c in prop.columns]; xx = np.arange(len(cols)); w = 0.38
        ax.bar(xx - w/2, prop.loc["Q4 (high VNI)", cols], w, color=BLUE, label="Q4 (high VNI)")
        ax.bar(xx + w/2, prop.loc["Q1 (low VNI)", cols], w, color=RED, label="Q1 (low VNI)")
        ax.set_xticks(xx); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Proportion (%)"); ax.legend(frameon=False)
        ax.set_title(f"{lab} subtype composition by atlas-UMAP VNI quartile", fontsize=11, fontweight="bold")
    fig.tight_layout(); save(fig, "Figure5B.png")
PANELS = [
    ("T cell", "CD8_Cytotoxic", "CD8 Cytotoxic"),
    ("T cell", "T_Exhaustion", "T Exhaustion"),
    ("T cell", "Treg_Activity", "Treg Activity"),
    ("T cell", "T_Effector", "T Effector"),
    ("Myeloid", "M1_like", "M1-like Macrophage"),
    ("Myeloid", "M2_like", "M2-like Macrophage"),
    ("Myeloid", "SPP1_TAM", "SPP1+ TAM"),
]

def quartiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    v = pd.to_numeric(df["VNI"], errors="coerce")
    q1 = float(v.quantile(0.25))
    q4 = float(v.quantile(0.75))
    return df[v <= q1], df[v >= q4], q1, q4

def main() -> None:
    tc = pd.read_csv(figures.D("tcell_scored.csv"), index_col=0, low_memory=False)
    my = pd.read_csv(figures.D("myeloid_scored.csv"), index_col=0, low_memory=False)
    tc["VNI"] = figures.knn_vni(tc)
    my["VNI"] = figures.knn_vni(my)
    frames = {"T cell": tc, "Myeloid": my}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(16, 8.2))
    grid = fig.add_gridspec(2, 8)
    axes = [
        fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[0, 6:8]),
        fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3:5]),
        fig.add_subplot(grid[1, 5:7]),
    ]
    rows = []
    for ax, (lineage, column, label) in zip(axes, PANELS):
        q1_cells, q4_cells, q1_cut, q4_cut = quartiles(frames[lineage])
        low = pd.to_numeric(q1_cells[column], errors="coerce").dropna().to_numpy()
        high = pd.to_numeric(q4_cells[column], errors="coerce").dropna().to_numpy()
        test = stats.mannwhitneyu(low, high, alternative="two-sided")

        parts = ax.violinplot([low, high], showextrema=False, showmedians=True)
        for body, color in zip(parts["bodies"], [figures.RED, figures.BLUE]):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.85)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Q1 (low)", "Q4 (high)"])
        ax.set_ylabel("Module score")
        ax.set_title(f"{label}  {figures.stars(test.pvalue)}", fontweight="bold")

        rows.append({
            "lineage": lineage,
            "module": column,
            "display_label": label,
            "q1_vni_cutoff": q1_cut,
            "q4_vni_cutoff": q4_cut,
            "q1_n": len(low),
            "q4_n": len(high),
            "q1_mean": low.mean(),
            "q4_mean": high.mean(),
            "mannwhitney_u": test.statistic,
            "p_two_sided": test.pvalue,
        })

    fig.tight_layout(w_pad=2.2, h_pad=2.0)

    png = OUT / "Figure5C.png"
    pdf = OUT / "Figure5C.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats_path = OUT / "Figure5C_statistics.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {stats_path}")
