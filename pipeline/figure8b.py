"""Figure 8B: existing summary statistics; corrected Treg module label.

Contract: EC immune-interface and T-cell-associated expression patterns.
Quantitative grid; no observations excluded or statistics recalculated.
Each row maps module -> group mean, SEM and two-sided Mann–Whitney p.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'submission_release_2026-10-15/投稿版 /outputs/source_data/Figure8_statistics.csv'
sys.path.insert(0, '/Users/leo/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'], 'font.size':7,
    'axes.titlesize':9, 'axes.labelsize':8, 'xtick.labelsize':7,
    'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
d = pd.read_csv(SOURCE)
fig, axes = plt.subplots(1, 2, figsize=(7.2,4.03), gridspec_kw={'width_ratios':[1.65,1]})
labels = [
    ['EC antigen\npresentation\n(HLA/B2M/TAP)', 'Leukocyte\nadhesion\n(ICAM/VCAM/SELE)',
     'Death-ligand\nbarrier\n(FASLG/PD-L1)', 'CXCL9/10\nT-cell\nrecruitment'],
    ['Cytotoxic T cells\n(CD8/GZMB/PRF1)', 'Treg-associated\nmodule']]
for ax, panel, labs, title in zip(axes, ['Fig8B_left','Fig8B_right'], labels,
                                 ['EC immune-gate modules','T-cell context']):
    s=d[d.panel==panel].copy()
    assert len(s)==len(labs)
    x=np.arange(len(s)); w=.38
    for off,group,color,lab in [(-w/2,'high','#3b78c3','VNI-high EC' if panel.endswith('left') else 'High VNI'),
                              (w/2,'low','#e24a4a','VNI-low EC' if panel.endswith('left') else 'Low VNI')]:
        ax.bar(x+off,s[group+'_mean'],w,yerr=s[group+'_sem'],capsize=1.8,
               color=color,label=lab,error_kw={'elinewidth':.8})
    for i,row in enumerate(s.itertuples()):
        p=row.p_two_sided
        star='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
        y=max(row.high_mean+row.high_sem,row.low_mean+row.low_sem)+.035
        negative=y<0
        if negative: y=min(row.high_mean-row.high_sem,row.low_mean-row.low_sem)-.035
        ax.text(i,y,star,ha='center',va='top' if negative else 'bottom',fontsize=7,fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labs,rotation=0,ha='center')
    ax.set_title(title,fontweight='bold',pad=8)
    ax.set_ylabel('Mean module score ± SEM')
    ax.axhline(0,color='#555555',lw=.7)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right' if panel.endswith('left') else 'center right')
axes[0].set_ylim(-.36,1.93)
axes[1].set_ylim(-.4,.95)
fig.subplots_adjust(left=.115,right=.985,bottom=.32,top=.90,wspace=.34)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(HERE/'Figure8B.alignment.json'),strict=True)
fig.savefig(HERE/'Figure8B.png',dpi=600,facecolor='white')
fig.savefig(HERE/'Figure8B.pdf',facecolor='white')
fig.savefig(HERE/'Figure8B.svg',facecolor='white')
d[d.panel.isin(['Fig8B_left','Fig8B_right'])].to_csv(HERE/'Figure8B_source_data.csv',index=False)
