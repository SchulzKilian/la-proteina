# P14_v2: same data as a side-by-side "rank by val" vs "rank by design" pair plot
# Differs from v1 (scatter) by showing rank inversion explicitly
# Visualizes: F5 / F7 / F11 / E054
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

pts = [
    ("canonical_2646", 4.71, 0.76),
    ("v2_2078",        4.44, 0.00),
    ("wd0_1638",       4.55, 0.17),
    ("sparse_K40_1259",4.50, 0.30),
    ("lastv2_1952",    4.50, 0.10),
    ("scnbr_t04_1133", 4.66, 0.17),
    ("conv_2331",      4.74, 0.00),
    ("K64_944",        4.85, 0.05),
]
# Rank: lower val = better
val_rank   = sorted(pts, key=lambda r: r[1])    # ascending val (best first)
design_rank = sorted(pts, key=lambda r: -r[2])   # descending design

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.7))
labels = [p[0] for p in pts]
# Map label -> position on each axis
val_pos    = {l: i for i, (l, _, _) in enumerate(val_rank)}
design_pos = {l: i for i, (l, _, _) in enumerate(design_rank)}

for label in labels:
    y0 = val_pos[label]
    y1 = design_pos[label]
    color = "#1f77b4" if "canonical" in label else (
            "#d62728" if "v2" in label or "conv" in label else "#7f7f7f")
    ax.plot([0, 1], [-y0, -y1], color=color, lw=1.0, marker="o", markersize=5)
    ax.text(-0.04, -y0, label, ha="right", va="center", fontsize=7.5)
    ax.text(1.04, -y1, label, ha="left", va="center", fontsize=7.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Rank by val\\_loss\n(best = top)", "Rank by designability\n(best = top)"])
ax.set_xlim(-0.5, 1.5)
ax.set_yticks([])
ax.set_title("Val-Loss Rank vs Designability Rank")
ax.spines[["top","right","left","bottom"]].set_visible(False)

fig.savefig(Path(__file__).with_suffix(".pdf"))
