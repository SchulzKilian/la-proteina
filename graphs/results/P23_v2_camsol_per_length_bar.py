# P23_v2: Grouped-bar Cohen's d table per (L, w) instead of distribution strip.
# Visualizes: F13 / E076
# DATA: inline from F13 numbers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L = ["300", "400", "500"]
d32  = [1.55, 1.20, 0.05]
d128 = [3.30, 2.50, 1.10]

x = np.arange(len(L))
w = 0.35
fig, ax = plt.subplots(figsize=figsize(0.85))
b1 = ax.bar(x - w/2, d32,  width=w, color="#1f77b4", edgecolor="black", lw=0.4, label="$w{=}32$", zorder=3)
b2 = ax.bar(x + w/2, d128, width=w, color="#d62728", edgecolor="black", lw=0.4, label="$w{=}128$", zorder=3)

for bars, vals in [(b1, d32), (b2, d128)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:+.2f}",
                ha="center", va="bottom", fontsize=7.5)

ax.axhline(0.8, color="#7f7f7f", lw=0.6, ls=":")
ax.text(2.4, 0.85, "$d{=}0.8$ (large)", fontsize=7, color="#7f7f7f", ha="right")
ax.set_xticks(x); ax.set_xticklabels([f"$L{{=}}{l}$" for l in L])
ax.set_ylabel(r"Cohen's $d$ vs Unsteered")
ax.set_title("Real CamSol Effect Size by Protein Length")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")
ax.set_ylim(0, 4.0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
