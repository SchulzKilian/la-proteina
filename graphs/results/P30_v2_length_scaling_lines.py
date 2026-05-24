# P30_v2: Length scaling — multi-line plot, one line per metric, x = L bin.
# Visualizes: F9 / E020+E026 follow-up
# DATA: inline from F9 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L_mid = np.array([325, 375, 425, 475, 525, 575, 625, 675, 725, 775])

curves = [
    ("Shannon entropy", [-1.40, -1.55, -1.70, -1.80, -1.90, -2.00, -2.10, -2.20, -2.15, -2.05], "#1f77b4", "o"),
    ("TANGO",           [ 0.80,  0.95,  1.10,  1.20,  1.30,  1.40,  1.50,  1.55,  1.60,  1.70], "#d62728", "s"),
    ("IUPred3",         [-0.40, -0.45, -0.50, -0.55, -0.58, -0.60, -0.65, -0.70, -0.72, -0.75], "#2ca02c", "^"),
]

fig, ax = plt.subplots(figsize=figsize(0.9))
for label, vals, c, m in curves:
    ax.plot(L_mid, vals, "-", color=c, marker=m, lw=1.4, markersize=4.5, label=label)
ax.axhline(0, color="black", lw=0.5)
ax.axhline(0.8, color="#7f7f7f", lw=0.5, ls=":")
ax.axhline(-0.8, color="#7f7f7f", lw=0.5, ls=":")
ax.fill_between(L_mid, -0.5, 0.5, color="#7f7f7f", alpha=0.05)

ax.set_xlabel("Protein Length (residues, bin midpoint)")
ax.set_ylabel(r"Cohen's $d$ (gen vs AFDB)")
ax.set_title("Failure Mode Grows with the Autoregressive Horizon")
ax.legend(frameon=False, fontsize=8.5, loc="center right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
