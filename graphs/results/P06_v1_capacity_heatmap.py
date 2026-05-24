# P06_v1: capacity probing R^2 matrix as heatmap, rows grouped by Class A/B
# Visualizes: F4 / E002
# DATA: inline from content_masterarbeit.md F4 table
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# 7 probes (low to high capacity): linear, MLP-1, MLP-2, MLP-3, BiLSTM, Tx-small, Tx-large
probes = ["Lin", "MLP-1", "MLP-2", "MLP-3", "BiLSTM", "Tx-S", "Tx-L"]

# Properties grouped by class (Class A: per-residue MLP suffices; Class B: needs attn)
class_A = [
    ("Aromatic frac.", [0.93, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]),
    ("Net charge",     [0.94, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]),
    ("Hydrophobic patch", [0.88, 0.93, 0.95, 0.96, 0.97, 0.98, 0.98]),
    ("Shannon entropy", [0.78, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95]),
    ("CamSol",         [0.75, 0.83, 0.86, 0.88, 0.90, 0.92, 0.92]),
    ("pI",             [0.84, 0.91, 0.94, 0.95, 0.96, 0.97, 0.98]),
]
class_B = [
    ("TANGO",   [0.41, 0.55, 0.62, 0.66, 0.78, 0.91, 0.95]),
    ("SAP",     [0.45, 0.58, 0.65, 0.70, 0.81, 0.92, 0.96]),
    ("IUPred3", [0.50, 0.62, 0.68, 0.72, 0.80, 0.90, 0.93]),
    ("SCM",     [0.39, 0.51, 0.58, 0.63, 0.74, 0.86, 0.91]),
    ("Rg",      [0.30, 0.42, 0.48, 0.55, 0.69, 0.83, 0.89]),
    ("Hydropathy", [0.34, 0.45, 0.51, 0.56, 0.71, 0.82, 0.86]),
    ("SWI",     [0.10, 0.18, 0.22, 0.26, 0.32, 0.36, 0.38]),  # F0
]

rows = class_A + class_B
names = [r[0] for r in rows]
M = np.array([r[1] for r in rows])

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.85))
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, zorder=3)

# Bold cells where Tx-L clears 0.80 (last col)
for i, row in enumerate(M):
    for j, v in enumerate(row):
        if v >= 0.80:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white", weight="bold")
        else:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white")

ax.set_xticks(range(len(probes)))
ax.set_xticklabels(probes)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)

# Class divider
ax.axhline(len(class_A) - 0.5, color="white", lw=1.2)
ax.text(-1.6, (len(class_A) - 1) / 2, "Class A", ha="right", va="center",
        fontsize=9, rotation=90, color="#2ca02c")
ax.text(-1.6, len(class_A) + (len(class_B) - 1) / 2, "Class B", ha="right", va="center",
        fontsize=9, rotation=90, color="#d62728")

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label(r"$R^2$")

# SWI Fold-0 asterisk
ax.text(7.1, len(rows)-1, "*", fontsize=14, color="#d62728")
ax.text(8.0, len(rows)-1, "5-fold mean = 0.77", fontsize=7.5, color="#d62728",
        va="center", clip_on=False)

ax.set_title(r"Capacity Probing: 7 Probes $\times$ 13 Properties")
fig.savefig(Path(__file__).with_suffix(".pdf"))
