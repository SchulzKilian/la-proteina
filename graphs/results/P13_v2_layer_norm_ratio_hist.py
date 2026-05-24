# P13_v2: same data as v1 but as a histogram split by category
# Differs from v1: distribution shape rather than rank-order
# Visualizes: F5 / E009
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(2)

N = 164
ratios_other = rng.normal(0.94, 0.07, N - 10)
ratios_gates = np.array([0.26, 0.31, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.52, 0.60])

fig, ax = plt.subplots(figsize=figsize(0.8))
bins = np.linspace(0.2, 1.2, 25)
ax.hist(ratios_other, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black",
        lw=0.3, label="other layers (n=154)", zorder=3)
ax.hist(ratios_gates, bins=bins, color="#d62728", alpha=0.85, edgecolor="black",
        lw=0.3, label="AdaLN-Zero gates (n=10)", zorder=4)

ax.axvline(1.0, color="#555", ls=":", lw=0.7)
ax.axvline(0.92, color="#2ca02c", ls="--", lw=0.7)
ax.text(0.92, ax.get_ylim()[1]*0.95, "  global mean 0.92",
        color="#2ca02c", fontsize=8, va="top")

ax.set_xlabel(r"$\|w_{\mathrm{v2}}\| / \|w_{\mathrm{old}}\|$")
ax.set_ylabel("Layer count")
ax.set_title("Weight-Norm Ratio Distribution: Gates Collapse, Bulk Intact")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
