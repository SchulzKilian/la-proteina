# P14_v1: val loss vs designability rate scatter, one labeled point per ckpt
# Visualizes: F5 / F7 / F11 / E054
# DATA: inline (assembled from F5/F7/F11/E054 + variant entries; numbers approximate)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# (ckpt label, wandb val_loss, pooled designability rate)
pts = [
    ("canonical_2646", 4.71, 0.76),
    ("v2_2078",        4.44, 0.00),  # better val, zero design
    ("wd0_1638",       4.55, 0.17),
    ("sparse_K40_1259",4.50, 0.30),
    ("lastv2_1952",    4.50, 0.10),
    ("scnbr_t04_1133", 4.66, 0.17),  # converged variant
    ("conv_2331",      4.74, 0.00),
    ("conv_2961",      4.72, 0.00),
    ("conv_3716",      4.71, 0.00),
    ("K64_944",        4.85, 0.05),
    ("K64_1800",       4.78, 0.10),
]
fig, ax = plt.subplots(figsize=figsize(0.95))
xs = [p[2] for p in pts]
ys = [p[1] for p in pts]
labels = [p[0] for p in pts]

# Color canonical, dead arms differently
colors = ["#1f77b4" if "canonical" in l else
          ("#d62728" if "v2" in l or "conv" in l else
           ("#7f7f7f")) for l in labels]
ax.scatter(xs, ys, c=colors, s=46, edgecolor="black", lw=0.4, zorder=3)
for x, y, l in zip(xs, ys, labels):
    ax.annotate(l, (x, y), xytext=(4, 3), textcoords="offset points",
                fontsize=7, color="#333")

# Trend annotation: lower val + higher design = top-right
ax.set_xlabel("Pooled Designability Rate (N=30)")
ax.set_ylabel(r"Wandb best val\_loss (lower is `better')")
ax.set_title("Val Loss vs Designability: They Don't Track")
ax.invert_yaxis()  # so lower val_loss is up
ax.axhline(4.71, color="#1f77b4", ls=":", lw=0.6)
ax.axvline(0.5, color="#555", ls=":", lw=0.6)

# Annotate decoupling
ax.annotate("v2: best val, ZERO design", xy=(0.0, 4.44), xytext=(0.1, 4.42),
            fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.6))

ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.text(0.02, 0.04, "Caveat: wandb val\\_loss is NOT cross-run comparable (E054).",
        transform=ax.transAxes, fontsize=7, color="#555", style="italic")

fig.savefig(Path(__file__).with_suffix(".pdf"))
