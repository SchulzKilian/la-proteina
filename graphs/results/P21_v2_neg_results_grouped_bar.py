# P21_v2: Alternative — vertical grouped bar showing gap at w=4, 8, 16 for each variant.
# Emphasizes that the gap grows with w (i.e. the failure is dose-dependent).
# Visualizes: F10 / E028 / E029 / E030 / E031 / E032
# DATA: inline from F10 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

variants = [
    "clean\nbaseline",
    "smaller\npred.",
    "longer\ntrain",
    "L2 on\ngrad",
    "freeze\nlayer 1",
    "clean\nensemble",
    "NA-v1\n(1-fold)",
    "NA-v1\nens.",
]
# (w=4, w=8, w=16)
gaps = np.array([
    [-82, -130, -203],
    [-71, -118, -187],
    [-65, -110, -171],
    [-54, -98,  -154],
    [-43, -82,  -132],
    [-38, -68,  -118],
    [-32, -52,  -71],
    [-2,   1,   +4],
])

x = np.arange(len(variants))
w = 0.27
fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.55))
COLORS = ["#fdae61", "#d7191c", "#7b3294"]
labels = ["$w{=}4$", "$w{=}8$", "$w{=}16$"]
for i in range(3):
    ax.bar(x + (i-1)*w, gaps[:, i], width=w, color=COLORS[i], edgecolor="black",
           lw=0.3, label=labels[i], zorder=3)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=7.5)
ax.set_ylabel("Predictor $-$ Real Gap (TANGO)")
ax.set_title("Gap Growth with $w$ Across Predictor Variants")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
