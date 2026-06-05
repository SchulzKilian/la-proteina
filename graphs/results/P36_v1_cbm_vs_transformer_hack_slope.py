# P36_v1: gradient hacking, minimal slope — unsteered vs. steered real disorder.
# Two lines (CBM, noise-aware ensemble) from a common unsteered prior; the ensemble's
# real disorder goes UP under "minimize" steering (Goodhart hack), the CBM's goes DOWN.
# Deliberately text-light — the numbers / setup belong in the caption.
# Visualizes: F17 / E109 (noise-aware ensemble) vs E110 (CBM), iupred3 down-reg, w=32, n=48.
# DATA: inline from content_masterarbeit.md F17.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

PRIOR = 0.158  # unsteered generated-set disorder (common origin for both methods)
# (label, steered real disorder @ w=32, colour)
methods = [
    ("CBM",                  0.128, "#1f77b4"),
    ("Noise-aware ensemble", 0.180, "#d62728"),
]

fig, ax = plt.subplots(figsize=figsize(0.55, ratio=0.9))
x = [0, 1]
for name, steered, c in methods:
    ax.plot(x, [PRIOR, steered], "-o", color=c, lw=2.0, ms=6, zorder=3, label=name)

ax.set_xticks(x)
ax.set_xticklabels([r"unsteered", r"steered"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(0.10, 0.20)
ax.set_ylabel(r"IUPred3 fraction disordered")
ax.set_title(r"Minimise disorder")
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="y")
fig.savefig(Path(__file__).with_suffix(".pdf"))
