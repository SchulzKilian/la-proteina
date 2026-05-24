# P01_v1: Two-route thesis overview as a block diagram (architectural vs steering),
# with Findings F1-F13 grouped by route and the L>=300 designability bar marker.
# Plot type chosen: schematic. Boxes-and-arrows form is most readable for an overview map.
# Visualizes: thesis intent / Findings F1-F13 grouping
# Source data: editorial synthesis from content_masterarbeit.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.55))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")

C_BASE = "#7f7f7f"
C_ARCH = "#1f77b4"
C_STEER = "#d62728"
C_SCAFF = "#9467bd"
C_DIAG = "#2ca02c"

def box(x, y, w, h, text, fc, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, linewidth=0.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9,
            multialignment="center")

def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=0.8, color="black", linestyle=ls))

# Title bar
ax.text(5, 5.9, r"\textbf{La Proteina: Two Routes Past the L $\geq$ 300 Designability Bar}",
        ha="center", va="center", fontsize=11)

# Baseline node
box(4.0, 4.7, 2.0, 0.6, r"CA-only baseline (F5, F7)", "#e7e7e7")

# Two arrows
arrow(4.6, 4.7, 2.5, 3.95)
arrow(5.4, 4.7, 7.5, 3.95)

# Architectural route column
box(0.7, 3.4, 3.6, 0.55, r"Architectural route", C_ARCH)
box(0.4, 2.55, 1.95, 0.6, "F11: per-$t$ val loss\n(parallel curves)", C_DIAG)
box(2.55, 2.55, 1.85, 0.6, "F12: per-query dense\nrouting audit", C_DIAG)
box(0.7, 1.75, 3.6, 0.6, "Variants: sparse, BigBird,\ndownsampled, hybrid", C_ARCH)
box(0.7, 0.95, 3.6, 0.5, r"\textbf{Outcome:} no variant beats canonical", "#ffd9d9")

# Steering route column
box(5.7, 3.4, 3.6, 0.55, "Steering route (latent guidance)", C_STEER)
box(5.4, 2.55, 1.85, 0.6, "F1: 13-probe $R^2$\n(latent rich)", C_SCAFF)
box(7.45, 2.55, 1.85, 0.6, "F3: latent geometry\ndisentangled", C_SCAFF)
box(5.4, 1.75, 1.85, 0.6, "F2: flow curvature\n$R{=}0.51/0.94$", C_SCAFF)
box(7.45, 1.75, 1.85, 0.6, "F4: class A/B\nprobe capacity", C_SCAFF)
box(5.7, 0.95, 3.6, 0.5, r"\textbf{F10/F13: $+91\%$ P(soluble) at $41.7\%$ codesign}", "#d4edda")

# Legend (use small color squares instead of \blacksquare)
from matplotlib.patches import Rectangle
def legend_item(x, y, color, label):
    ax.add_patch(Rectangle((x, y), 0.18, 0.18, color=color))
    ax.text(x + 0.25, y + 0.09, label, color="black", fontsize=8, va="center")

legend_item(0.4, 0.35, C_SCAFF, "scaffold")
legend_item(2.0, 0.35, C_DIAG, "diagnostic")
legend_item(3.8, 0.35, C_ARCH, "architectural")
legend_item(5.8, 0.35, C_STEER, "steering")
ax.text(7.6, 0.45, "F8/F9: AA collapse (joint head)", color="black", fontsize=8)

fig.savefig(Path(__file__).with_suffix(".pdf"))
