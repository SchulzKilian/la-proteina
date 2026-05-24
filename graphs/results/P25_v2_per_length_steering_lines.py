# P25_v2: Alternative — lines (one per recipe) showing how Cohen's d falls off as L grows.
# Emphasizes the "under-steering at long L" pattern.
# Visualizes: F13 / E066 / E067 / E072
# DATA: inline from F13 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L = np.array([300, 400, 500])

curves = [
    ("camsol max $w{=}32$",  [1.55, 1.20, 0.05], "#1f77b4", "o"),
    ("camsol max $w{=}128$", [3.30, 2.50, 1.10], "#d62728", "s"),
    ("tango min $w{=}32$",   [1.25, 0.95, 0.30], "#2ca02c", "^"),
    ("combo-4 $w{=}32$",     [1.40, 1.05, 0.55], "#9467bd", "D"),
]

fig, ax = plt.subplots(figsize=figsize(0.9))
for label, vals, c, m in curves:
    ax.plot(L, vals, "-", color=c, marker=m, lw=1.4, markersize=5, label=label)

ax.axhline(0.8, color="#7f7f7f", ls=":", lw=0.6)
ax.text(495, 0.85, "$d{=}0.8$ (large)", ha="right", fontsize=7.5, color="#7f7f7f")

ax.set_xlabel("Protein Length $L$ (residues)")
ax.set_ylabel(r"Cohen's $d$ vs Unsteered")
ax.set_title("Steering Effect Decays with Length")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.set_xticks(L); ax.set_ylim(0, 3.6)

fig.savefig(Path(__file__).with_suffix(".pdf"))
