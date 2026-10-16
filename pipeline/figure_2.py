from common import *
def figure2A_mean_scores():
    cols = ["VM_Score", "PC_Score", "HDA_Score", "IVI_Score", "EBI_Score"]
    labels = ["VM", "PC", "HDA", "IVI", "EBI"]
    ec = ec_vni_split()  # EC cells split by VNI median (matches manuscript Fig 2A)
    g = ec.groupby("VN_split")[cols].apply(lambda d: d.apply(pd.to_numeric, errors="coerce").mean())
    hi, lo = g.loc["VN-High"], g.loc["VN-Low"]
    x = np.arange(len(cols)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, hi, w, label="VN-High", color=BLUE)
    ax.bar(x + w/2, lo, w, label="VN-Low", color=RED)
    ax.axhline(0, color="k", lw=0.8); ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean module score"); ax.legend()
    ax.set_title("Mean Five-Axis VN Scores: VN-High vs VN-Low", fontweight="bold")
    save(fig, "Figure2A.png")
    print("    [check2A] VM hi/lo=%.2f/%.2f  IVI=%.2f/%.2f  EBI=%.2f/%.2f" %
          (hi.VM_Score, lo.VM_Score, hi.IVI_Score, lo.IVI_Score, hi.EBI_Score, lo.EBI_Score))

def figure2B_loadings():
    # The primary VNI is an equal-weight mean of five direction-aligned axes.
    pc1 = pd.Series(0.2, index=["VM", "PC", "HDA (inverted)", "IVI", "EBI"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(pc1)), pc1.values, color=BLUE)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(pc1))); ax.set_xticklabels(pc1.index, rotation=30, ha="right")
    ax.set_ylabel("Pre-specified VNI weight")
    ax.set_title("Primary VNI Composition (Equal-Weight, Direction-Aligned)", fontweight="bold")
    save(fig, "Figure2B.png")
    print("    [check2B] five fixed weights = 0.2")

def figure2C_correlation():
    m = meta()
    cols = ["VM_Score", "PC_Score_adj", "HDA_Score", "IVI_Score", "EBI_Score_adj", "VNI"]
    C = m[cols].apply(pd.to_numeric, errors="coerce").corr().values
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(C[i, j]) > 0.55 else "black", fontsize=11)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title("VN Score Pearson Correlations", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, "Figure2C.png")
