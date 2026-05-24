# P04_v2: same data as v1, but as a strip-style scatter (one row per property,
# per-fold dot, mean tick). Differs from v1: no bar, emphasizes spread over rank.
# Visualizes: F1 / E001
# DATA: inline from content_masterarbeit.md F1 table
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

props = [
    ("Aromatic frac.", [0.99, 0.99, 0.99, 0.99, 0.99]),
    ("Net charge",     [0.99, 0.99, 0.99, 0.99, 0.99]),
    ("pI",             [0.98, 0.97, 0.98, 0.99, 0.98]),
    ("Hydrophobic patch", [0.98, 0.97, 0.98, 0.98, 0.98]),
    ("SAP",            [0.96, 0.95, 0.97, 0.96, 0.96]),
    ("TANGO",          [0.95, 0.94, 0.96, 0.95, 0.95]),
    ("Shannon entropy", [0.95, 0.94, 0.96, 0.95, 0.95]),
    ("IUPred3",        [0.93, 0.92, 0.94, 0.93, 0.93]),
    ("CamSol",         [0.92, 0.91, 0.93, 0.92, 0.92]),
    ("SCM",            [0.91, 0.89, 0.92, 0.91, 0.91]),
    ("Rg",             [0.89, 0.86, 0.91, 0.89, 0.89]),
    ("Hydropathy",     [0.86, 0.83, 0.88, 0.86, 0.86]),
    ("SWI",            [0.38, 0.98, 0.85, 0.92, 0.72]),  # wide
]
props = sorted(props, key=lambda r: np.mean(r[1]))

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.85))
for i, (name, folds) in enumerate(props):
    ax.scatter(folds, [i]*len(folds), color="#7f7f7f", s=20, alpha=0.7, zorder=3)
    m = np.mean(folds)
    ax.scatter([m], [i], marker="|", color="#d62728", s=140, lw=2, zorder=4)

ax.set_yticks(range(len(props)))
ax.set_yticklabels([p[0] for p in props])
ax.set_xlim(0.3, 1.02)
ax.set_xlabel(r"Per-fold $R^2$")
ax.set_title(r"Per-fold $R^2$ Spread (mean tick in red)")
ax.axvline(0.8, color="#d62728", ls=":", lw=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="x")
fig.savefig(Path(__file__).with_suffix(".pdf"))
