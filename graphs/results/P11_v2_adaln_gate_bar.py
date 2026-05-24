# P11_v2: same data as table, but as a horizontal bar chart for visual hierarchy
# Differs from v1 by chart form (table vs bar)
# Visualizes: F5 / E009 / E015
# DATA: inline from content_masterarbeit.md F5
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

names = [
    "L7 transition", "L7 mhba", "L8 transition", "L8 mhba",
    "L9 transition", "L9 mhba", "L10 transition",
    "L11 transition", "L12 transition", "L13 transition",
]
ratios = [0.26, 0.31, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.52, 0.60]

# sort ascending for clarity (worst at top)
order = np.argsort(ratios)
names = [names[i] for i in order][::-1]
ratios = [ratios[i] for i in order][::-1]

fig, ax = plt.subplots(figsize=figsize(0.8, ratio=0.75))
y = np.arange(len(names))
colors = ["#d62728" if r < 0.40 else "#ff7f0e" if r < 0.55 else "#1f77b4" for r in ratios]
ax.barh(y, ratios, color=colors, edgecolor="black", lw=0.4, zorder=3)
for yi, r in zip(y, ratios):
    ax.text(r + 0.01, yi, f"{r:.2f}", va="center", fontsize=8)

ax.axvline(1.0, color="#555", ls=":", lw=0.7)
ax.text(1.0, -0.5, "  unchanged", color="#555", fontsize=8)

ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0, 1.1)
ax.set_xlabel(r"$\| w_{\mathrm{v2}} \| / \| w_{\mathrm{old}} \|$")
ax.set_title("AdaLN-Zero Gate Norm Ratio (v2 / canonical), 10 Worst Layers")
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")

fig.savefig(Path(__file__).with_suffix(".pdf"))
