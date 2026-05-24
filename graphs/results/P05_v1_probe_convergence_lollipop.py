# P05_v1: probe convergence-epoch hierarchy, lollipop chart colored by regime
# Visualizes: F1 additional observation
# DATA: inline from content_masterarbeit.md F1 epoch-to-90% table
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# (property, epoch-to-90%, regime)
data = [
    ("Aromatic frac.", 1, "fast"),
    ("Net charge", 1, "fast"),
    ("pI", 2, "fast"),
    ("Hydrophobic patch", 2, "fast"),
    ("SAP", 3, "medium"),
    ("Shannon entropy", 4, "medium"),
    ("TANGO", 5, "medium"),
    ("CamSol", 6, "medium"),
    ("IUPred3", 7, "medium"),
    ("SCM", 9, "slow"),
    ("Hydropathy", 11, "slow"),
    ("Rg", 14, "slow"),
    ("SWI", 18, "slow"),  # outlier (fast on folds 2-4 only)
]
data = sorted(data, key=lambda r: r[1])
COLORS = {"fast": "#2ca02c", "medium": "#1f77b4", "slow": "#d62728"}

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.8))
for i, (name, ep, reg) in enumerate(data):
    ax.plot([0, ep], [i, i], color=COLORS[reg], lw=1.2, zorder=3)
    ax.scatter([ep], [i], s=46, color=COLORS[reg], zorder=4, edgecolor="black", lw=0.4)
    ax.text(ep + 0.5, i, str(ep), va="center", fontsize=8)

# annotate SWI outlier
ax.annotate("fast on folds 2-4\n(fold-0 outlier)", xy=(18, 12), xytext=(11, 10),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"))

ax.set_yticks(range(len(data)))
ax.set_yticklabels([d[0] for d in data])
ax.set_xlabel(r"Epoch to 90\% of final $R^2$")
ax.set_title(r"Convergence Speed Hierarchy")

# legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", color=COLORS[r], linestyle="",
                  markersize=6, label=r.title()) for r in ["fast","medium","slow"]]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
ax.set_xlim(0, 22)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="x")
fig.savefig(Path(__file__).with_suffix(".pdf"))
