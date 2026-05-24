# P25_v1: Per-length steering response — heatmap of Cohen's d across (direction, w) x L.
# Visualizes: F13 / E066 / E067 / E072
# DATA: inline from F13 per-length numbers + scout dirs
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

rows = [
    ("camsol max $w{=}16$",  [0.45, 0.30, 0.05]),
    ("camsol max $w{=}32$",  [1.55, 1.20, 0.05]),
    ("camsol max $w{=}64$",  [2.40, 1.80, 0.60]),
    ("camsol max $w{=}128$", [3.30, 2.50, 1.10]),
    ("tango min $w{=}32$",   [1.25, 0.95, 0.30]),
    ("tango min $w{=}128$",  [2.80, 2.10, 0.85]),
    ("iupred max $w{=}32$",  [0.40, 0.55, 0.20]),
    ("combo-4 $w{=}32$",     [1.40, 1.05, 0.55]),
]
labels = [r[0] for r in rows]
mat = np.array([r[1] for r in rows])
L_labels = ["$L{=}300$", "$L{=}400$", "$L{=}500$"]

fig, ax = plt.subplots(figsize=figsize(0.9))
im = ax.imshow(mat, cmap="RdBu_r", vmin=-3.5, vmax=3.5, aspect="auto")

# Annotate
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        color = "white" if abs(v) > 1.6 else "black"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=color)

ax.set_xticks(range(len(L_labels))); ax.set_xticklabels(L_labels)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_title("Per-Length Steering Response (Cohen's $d$)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=r"$d$ vs unsteered")
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

fig.savefig(Path(__file__).with_suffix(".pdf"))
