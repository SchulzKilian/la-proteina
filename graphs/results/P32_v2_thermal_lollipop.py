# P32_v2: Same data as v1 but as a lollipop chart, sorted by magnitude — clearer asymmetry.
# Visualizes: F8 (b.i) / E020 / E026
# DATA: inline from F8 (b.i) prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "log(acidic/basic)",
    "aliphatic index",
    "IVYWREL",
    "charged fraction",
    "GRAVY",
    "aromatic fraction",
]
d_pdb  = [+2.20, +1.10, +0.95, +0.85, +0.65, -1.30]

# Sort by magnitude descending
order = np.argsort(-np.abs(d_pdb))
metrics_s = [metrics[i] for i in order]
d_s = [d_pdb[i] for i in order]
colors = ["#1f77b4" if d > 0 else "#d62728" for d in d_s]

y = np.arange(len(metrics_s))
fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.6))
ax.hlines(y, 0, d_s, color=colors, lw=1.4)
ax.scatter(d_s, y, color=colors, s=60, zorder=3, edgecolor="black", lw=0.4)
for yi, v, m in zip(y, d_s, metrics_s):
    offset = 6 if v > 0 else -6
    ha = "left" if v > 0 else "right"
    ax.text(v, yi, f" {v:+.2f}", ha=ha, va="center", fontsize=8)

ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(metrics_s, fontsize=9)
ax.set_xlabel(r"Cohen's $d$ (gen vs PDB)")
ax.set_title(r"Gen Composition Looks Thermostable, Except Aromatics")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")
ax.invert_yaxis()

fig.savefig(Path(__file__).with_suffix(".pdf"))
