# P04_v1: Multi-task property predictor 5-fold val R^2 — horizontal bar sorted by mean
# Visualizes: F1 / E001
# DATA: inline from content_masterarbeit.md F1 table
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# 13 properties: (name, mean R^2, fold spread [min, max])
props = [
    ("Aromatic frac.", 0.99, [0.99, 0.99]),
    ("Net charge",     0.99, [0.99, 0.99]),
    ("pI",             0.98, [0.97, 0.99]),
    ("Hydrophobic patch", 0.98, [0.97, 0.98]),
    ("SAP",            0.96, [0.95, 0.97]),
    ("TANGO",          0.95, [0.94, 0.96]),
    ("Shannon entropy", 0.95, [0.94, 0.96]),
    ("IUPred3",        0.93, [0.92, 0.94]),
    ("CamSol",         0.92, [0.91, 0.93]),
    ("SCM",            0.91, [0.89, 0.92]),
    ("Rg",             0.89, [0.86, 0.91]),
    ("Hydropathy",     0.86, [0.83, 0.88]),
    ("SWI",            0.77, [0.38, 0.98]),  # SWI has the wide spread
]
props = sorted(props, key=lambda r: r[1])

names = [p[0] for p in props]
means = [p[1] for p in props]
lows  = [p[2][0] for p in props]
highs = [p[2][1] for p in props]

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.85))
y = np.arange(len(props))
bars = ax.barh(y, means, color="#1f77b4", height=0.55, edgecolor="black",
               lw=0.4, zorder=3)
# Per-fold spread (low-high horizontal error)
for yi, lo, hi, m in zip(y, lows, highs, means):
    ax.plot([lo, hi], [yi, yi], color="#333", lw=1.0, zorder=4)
    ax.scatter([lo, hi], [yi, yi], color="#333", s=8, zorder=4)

ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlim(0.3, 1.02)
ax.set_xlabel(r"5-fold CV $R^2$")
ax.set_title(r"Multi-task Predictor: 13 Developability Properties")
ax.axvline(0.8, color="#d62728", ls=":", lw=0.7)
ax.text(0.8, len(y) - 0.5, " $R^2{=}0.8$", color="#d62728", fontsize=8, va="top")

# annotate SWI's wide spread
ax.annotate("wide fold spread\n(std=0.01 target)", xy=(0.38, 0), xytext=(0.42, 1.6),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"))

ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="x")
fig.savefig(Path(__file__).with_suffix(".pdf"))
