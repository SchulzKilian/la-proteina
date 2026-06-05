# P46_v1: Layer-arrangement map for budget-matched 7-sparse/7-dense hybrid
# sampling at L=100. One horizontal strip of 14 cells (layers 0..13) per
# configuration, colored dense(blue)/sparse(amber); all rows are 7 sparse +
# 7 dense with the sparse K-budget matched (sum K=280). The number at right
# is L=100 designability (scRMSD < 2 A) as k/12. Rows ordered best -> worst.
#
# DATA: hardcoded from experiments.md E096 (budget-matched 7-sparse/7-dense
#   rearrangement sweep, N=12, L=100, nsteps=400, dense-trained ckpt
#   best_val_..._2646, inference-only sparse substitution).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

# --- Palette (consistent with the P38 sparse-attention family) ---
C_DENSE = "#4f7cb0"   # dense  = blue
C_SPARSE = "#e08214"  # sparse = amber

# (name, 14-char mask, designable_at_L100, total). T = sparse, F = dense,
# 14 chars layers 0->13. Ordered best -> worst (top -> bottom).
rows_raw = [
    ("lower_half_sparse", "TTTTTTTFFFFFFF", 12, 12),  # sparse 0-6, dense 7-13  [winner]
    ("dense_split_3_4",   "TTTTFFFTTTFFFF",  8, 12),  # dense at 4-6 and 10-13
    ("alternating_strict","TFTFTFTFTFTFTF",  6, 12),  # dense every odd layer
    ("dense_split_4_3",   "TTTFFFFTTTTFFF",  1, 12),  # dense 3-6 and 11-13
    ("dense_middle",      "TTTTFFFFFFFTTT",  0, 12),  # dense 4-10 (sparse ends)
    ("dense_bookends",    "FFFFTTTTTTTFFF",  0, 12),  # dense 0-3 and 11-13
]
# Self-check the masks: each exactly 14 chars, 7 T and 7 F.
for name, mask, _, _ in rows_raw:
    assert len(mask) == 14, f"{name}: mask len {len(mask)} != 14"
    assert mask.count("T") == 7 and mask.count("F") == 7, f"{name}: not 7T/7F"

nrows = len(rows_raw)
# mask matrix: 1 = sparse (amber), 0 = dense (blue)
mat = np.array([[1 if ch == "T" else 0 for ch in mask] for _, mask, _, _ in rows_raw])

# ============================================================================
fig, ax = plt.subplots(figsize=figsize(0.72, 0.5))

cmap = mcolors.ListedColormap([C_DENSE, C_SPARSE])
ax.imshow(mat, cmap=cmap, aspect="auto", origin="upper",
          interpolation="nearest", extent=[-0.5, 13.5, nrows - 0.5, -0.5])

# thin grid between cells
for j in range(15):
    ax.axvline(j - 0.5, color="white", lw=0.6)
for i in range(nrows + 1):
    ax.axhline(i - 0.5, color="white", lw=0.6)

ax.set_yticks([])
ax.set_xticks([0, 6, 7, 13])
ax.set_xticklabels(["0", "6", "7", "13"], fontsize=7)
ax.set_xlabel("transformer layer (0 = input, 13 = output)", fontsize=8)
ax.set_xlim(-0.5, 13.5)

# designability k/12 to the right of each strip (winner in bold)
for i, (_, _, dgn, tot) in enumerate(rows_raw):
    bold = dgn == tot
    ax.text(13.9, i, rf"\textbf{{{dgn}/{tot}}}" if bold else rf"{dgn}/{tot}",
            va="center", ha="left", fontsize=7.5, color="black")

# cell-color legend on the left, where the row labels used to be
handles = [Patch(facecolor=C_DENSE, label="dense"),
           Patch(facecolor=C_SPARSE, label="sparse")]
ax.legend(handles=handles, loc="center right", bbox_to_anchor=(-0.03, 0.5),
          ncol=1, frameon=False, fontsize=8, handlelength=1.0,
          handletextpad=0.5, labelspacing=0.6)

fig.tight_layout()
fig.subplots_adjust(left=0.16, right=0.9)

fig.savefig(Path(__file__).with_suffix(".pdf"))
