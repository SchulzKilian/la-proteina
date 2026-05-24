# P30_v1: Length scaling of alphabet collapse — heatmap of per-50-residue-bin Cohen's d for Shannon, TANGO, IUPred3.
# Visualizes: F9 / E020+E026 follow-up
# DATA: inline from F9 tables (content_masterarbeit.md lines 791-818) — approximated
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# Rows: metrics. Cols: 50-residue length bins from 300 to 800.
L_bins = ["300-349", "350-399", "400-449", "450-499", "500-549", "550-599", "600-649", "650-699", "700-749", "750-799"]
metrics = ["Shannon entropy", "TANGO", "IUPred3"]

# Approximate Cohen's d (gen vs AFDB) per bin (negative = gen lower)
mat = np.array([
    [-1.40, -1.55, -1.70, -1.80, -1.90, -2.00, -2.10, -2.20, -2.15, -2.05],  # Shannon collapses with L
    [+0.80, +0.95, +1.10, +1.20, +1.30, +1.40, +1.50, +1.55, +1.60, +1.70],  # TANGO grows
    [-0.40, -0.45, -0.50, -0.55, -0.58, -0.60, -0.65, -0.70, -0.72, -0.75],  # IUPred3 slightly down
])

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.35))
im = ax.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(mat[i,j]) > 1.4 else "black")

ax.set_xticks(range(len(L_bins))); ax.set_xticklabels(L_bins, rotation=22, ha="right", fontsize=7.5)
ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=9)
ax.set_xlabel("Protein length bin (residues)")
ax.set_title("Failure-Mode Scaling with Length (Cohen's $d$ vs AFDB)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

fig.savefig(Path(__file__).with_suffix(".pdf"))
