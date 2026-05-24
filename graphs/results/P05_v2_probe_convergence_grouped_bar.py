# P05_v2: same data as v1, but a grouped-bar arrangement by regime
# Differs from v1: groups by fast/medium/slow rather than sorting numerically
# Visualizes: F1 additional observation
# DATA: inline from content_masterarbeit.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

data = {
    "fast": [("Aromatic frac.", 1), ("Net charge", 1), ("pI", 2), ("Hydrophobic patch", 2)],
    "medium": [("SAP", 3), ("Shannon entropy", 4), ("TANGO", 5), ("CamSol", 6), ("IUPred3", 7)],
    "slow": [("SCM", 9), ("Hydropathy", 11), ("Rg", 14), ("SWI", 18)],
}
COLORS = {"fast": "#2ca02c", "medium": "#1f77b4", "slow": "#d62728"}

fig, ax = plt.subplots(figsize=figsize(0.9, ratio=0.55))
x_off = 0
xticks = []
labels = []
group_xc = []
for reg, items in data.items():
    xs = np.arange(len(items)) + x_off
    vals = [v for _, v in items]
    ax.bar(xs, vals, color=COLORS[reg], edgecolor="black", lw=0.4, zorder=3, width=0.7)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, str(v), ha="center", fontsize=8)
    xticks.extend(xs.tolist())
    labels.extend([n for n, _ in items])
    group_xc.append(xs.mean())
    x_off += len(items) + 0.8

ax.set_xticks(xticks)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Epoch to $90\\%$ of final $R^2$")
ax.set_title("Convergence Speed by Regime")
ax.set_ylim(0, 22)

# regime headers
for x, name in zip(group_xc, list(data.keys())):
    ax.text(x, 21.5, name.title(), ha="center", fontsize=10, color=COLORS[name])

ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="y")
fig.savefig(Path(__file__).with_suffix(".pdf"))
