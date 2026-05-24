# P33_v2: Focus on just the F12 headline — per-query-pair Jaccard at L=50/100/200,
# annotated with the "0.06 at L=200" collapse number.
# Visualizes: F12 / E061
# DATA: inline from F12 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L = np.array([50, 100, 200])
mean_jaccard = np.array([0.28, 0.14, 0.06])
p25 = mean_jaccard - 0.05
p75 = mean_jaccard + 0.06

fig, ax = plt.subplots(figsize=figsize(0.7))
ax.errorbar(L, mean_jaccard, yerr=[mean_jaccard - p25, p75 - mean_jaccard],
            fmt="o", color="#1f77b4", lw=1.4, capsize=4, markersize=7,
            ecolor="#1f77b4", markeredgecolor="black", markeredgewidth=0.4)

for x_, y_ in zip(L, mean_jaccard):
    ax.annotate(f"{y_:.2f}", (x_, y_), xytext=(0, 10), textcoords="offset points",
                fontsize=9, ha="center")

# Reference shared-K-set sparse line
ax.axhline(0.8, color="#7f7f7f", lw=0.6, ls=":")
ax.text(205, 0.82, "sparse-K-set shared\nstructure regime", fontsize=7,
        color="#7f7f7f", ha="right", va="bottom")

ax.set_xticks(L)
ax.set_xlabel("Protein Length $L$ (residues)")
ax.set_ylabel("Mean Per-Query-Pair Jaccard (top-64)")
ax.set_title("Dense Routing Collapses to $\\sim0.06$ at $L{=}200$")
ax.set_ylim(0, 0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
