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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from common import save
def save_figure(fig,name):
    save(fig,name)
BLUE = "#0072B2"

ORANGE = "#D55E00"

GREEN = "#009E73"

AXES = ["VM_z", "PC_adj_z", "HDA_inv_z", "IVI_z", "EBI_adj_z"]

AXIS_LABELS = {
    "VM_z": "VM",
    "PC_adj_z": "PC-adjusted",
    "HDA_inv_z": "HDA-inverted",
    "IVI_z": "IVI",
    "EBI_adj_z": "EBI-adjusted",
    "VNI": "Five-axis VNI",
}

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
    save_figure(fig, "Figure6A")


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
    save_figure(fig, "Figure6C")

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
    save_figure(fig, "Figure6B")
