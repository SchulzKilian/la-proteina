# P12_v2: same data but as a horizontal dotplot of rates per (L, arm)
# Differs from v1: emphasizes within-L comparison across arms more cleanly
# Visualizes: F5 / F7 / E014 / E019
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

Ls = [50, 100, 200]
arms = ["canonical (wd=0.05)", "v2 (wd=0.1)", "wd=0", "sparse K40"]
rates = {
    "canonical (wd=0.05)": [0.87, 0.83, 0.57],
    "v2 (wd=0.1)":         [0.00, 0.00, 0.00],
    "wd=0":                [0.30, 0.13, 0.07],
    "sparse K40":          [0.40, 0.33, 0.17],
}
COLORS = ["#1f77b4", "#d62728", "#7f7f7f", "#9467bd"]
MARKERS = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.5))
y = np.arange(len(Ls))
for i, arm in enumerate(arms):
    ax.scatter(rates[arm], y + (i-1.5)*0.10,
               s=70, color=COLORS[i], marker=MARKERS[i],
               label=arm, edgecolor="black", lw=0.4, zorder=3)

ax.set_yticks(y); ax.set_yticklabels([f"$L{{=}}{L}$" for L in Ls])
ax.set_xlim(-0.02, 1.0)
ax.set_xlabel("Designability rate (N=30)")
ax.axvline(0.5, color="#555", ls=":", lw=0.6)
ax.set_title("Per-length Designability across Recipes")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")

fig.savefig(Path(__file__).with_suffix(".pdf"))
