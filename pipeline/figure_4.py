from common import *
def figure4_ec_subtype():
    m = meta()
    ct = pd.crosstab(m["EC_subtype"], m["VN_state"], normalize="columns") * 100
    order = ["Arterial", "Capillary", "Lymphatic", "Stalk", "Tip", "Venous"]
    ct = ct.reindex(order); hi, lo = ct["VN-High"], ct["VN-Low"]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - w/2, hi, w, label="VN-High", color=BLUE)
    ax.bar(x + w/2, lo, w, label="VN-Low", color=RED)
    ax.grid(axis="y", ls="--", alpha=0.35)
    for st in ["Arterial", "Lymphatic", "Tip", "Venous"]:
        i = order.index(st); d = hi[st] - lo[st]; top = max(hi[st], lo[st])
        ax.text(x[i], top + 0.8, f"{d:+.1f}%", ha="center", fontsize=10.5, fontweight="bold",
                color=BLUE if d > 0 else RED)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Proportion of endothelial cells (%)")
    ax.set_title("Endothelial Subtype Composition: VN-High vs VN-Low", fontweight="bold")
    ax.legend()
    save(fig, "Figure4.png")
