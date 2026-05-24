# P20_v2: Alternative framing — show only the (predictor - real) gap on log-y, all 5 fix attempts on one chart.
# This emphasizes the gap-closure rather than the raw curves.
# Visualizes: F10 / E028 / E032 / E050
# DATA: inline from F10 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

w = np.array([1, 2, 4, 8, 16])

variants = [
    ("clean (1-fold)",    [-11, -41, -82, -130, -203], "#d62728", "o"),
    ("clean ensemble",    [-9,  -32, -65, -110, -180], "#ff7f0e", "s"),
    ("NA-v1 (1-fold)",    [-7,  -18, -33,  -52,  -71], "#9467bd", "^"),
    ("NA-v1 5-fold ens.", [-2,  -3,  -2,   +1,   +4], "#1f77b4", "D"),
]

fig, ax = plt.subplots(figsize=figsize(0.9))
for label, vals, c, m in variants:
    vals = np.array(vals)
    # Plot |gap| with sign indicated by color (negative = red shades)
    ax.plot(w, vals, "-", color=c, lw=1.3, marker=m, label=label, markersize=4.5)

ax.axhline(0, color="black", lw=0.6, ls="-")
ax.set_xscale("log", base=2)
ax.set_xticks(w); ax.set_xticklabels([str(v) for v in w])
ax.set_xlabel("Steering Weight $w$")
ax.set_ylabel("Predictor $-$ Real Gap (TANGO units)")
ax.set_title("Gradient-Hacking Gap: Closed by NA-v1 + Ensemble")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.text(16.5, 0, "  zero gap\n  (target)", fontsize=7, va="center", color="#333")

fig.savefig(Path(__file__).with_suffix(".pdf"))
