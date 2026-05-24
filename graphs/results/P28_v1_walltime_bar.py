# P28_v1: Steering walltime overhead — simple grouped bar (unsteered / 1-fold steered / 5-fold ens steered).
# Visualizes: E069
# DATA: inline from E069 prose (approximate)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

cats = ["unsteered", "steered\n(1-fold pred.)", "steered\n(5-fold ens.)"]
# Per-protein wall (s) on an L4 GPU at L=500, nsteps=400, batch=4
walltime = [320, 380, 580]   # approximate
ratio    = [1.0, 1.19, 1.81]

x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=figsize(0.7))
colors = ["#7f7f7f", "#1f77b4", "#d62728"]
bars = ax.bar(x, walltime, width=0.6, color=colors, edgecolor="black", lw=0.4, zorder=3)
for b, v, r in zip(bars, walltime, ratio):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}s\n($\\times{r:.2f}$)",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("Per-Protein Wall (s, L4 GPU)")
ax.set_title("Steering Walltime Overhead")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")
ax.set_ylim(0, max(walltime) * 1.18)

fig.savefig(Path(__file__).with_suffix(".pdf"))
