# P02_v2: nsteps cliff — before/after bars (200 vs 400) at L=300, headline-format
# Differs from v1 by collapsing to the two operating-points that matter
# Visualizes: hard-rule nsteps=400 (CLAUDE.md / E019)
# DATA: inline from CLAUDE.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=figsize(0.55))
labels = [r"$n_{\mathrm{steps}}{=}200$", r"$n_{\mathrm{steps}}{=}400$"]
vals = [22.5, 0.80]
colors = ["#d62728", "#1f77b4"]

bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", lw=0.5, zorder=3)
ax.axhline(2.0, color="#555", ls=":", lw=0.8, zorder=2)
ax.text(1.45, 2.0, r"2 \AA\ bar", color="#555", fontsize=8.5, ha="right", va="bottom")

for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)

ax.set_ylabel(r"scRMSD (\AA)")
ax.set_yscale("log")
ax.set_ylim(0.3, 60)
ax.set_title(r"L=300, identical seed / model")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
