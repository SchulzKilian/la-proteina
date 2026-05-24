# P01_v2: Two-route overview as a left-to-right flowchart from "Baseline" to
# "L>=300 deliverable". Differs from v1 in being a linear flow (not column-split),
# showing temporal/logical progression rather than route taxonomy.
# Visualizes: thesis intent
# Source data: editorial synthesis from content_masterarbeit.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.42))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

def box(x, y, w, h, text, fc, ec="black", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, linewidth=0.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(x0, y0, x1, y1, color="black"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=0.8, color=color))

# Stage 1: Baseline
box(0.1, 2.2, 1.8, 0.7, "CA-only\nbaseline\n(F5, F7)", "#e7e7e7")
ax.text(1.0, 1.95, "E008, E009, E019", ha="center", va="top", fontsize=7, color="#555")

# Stage 2: split into two routes
arrow(1.9, 2.6, 2.6, 4.0, "#1f77b4")
arrow(1.9, 2.4, 2.6, 1.0, "#d62728")

# Top: architectural route
box(2.6, 3.6, 2.2, 0.7, "Architectural\nvariants", "#1f77b4")
arrow(4.8, 3.95, 5.5, 3.95)
box(5.5, 3.6, 2.4, 0.7, "F11 / F12\ndiagnostics", "#2ca02c")
arrow(7.9, 3.95, 8.6, 3.95)
box(8.6, 3.6, 3.2, 0.7, "Dead-arm gallery (negative)", "#ffd9d9")

# Bottom: steering route
box(2.6, 0.6, 2.2, 0.7, "Latent scaffold\n(F1-F4, F6)", "#9467bd")
arrow(4.8, 0.95, 5.5, 0.95)
box(5.5, 0.6, 2.4, 0.7, "Predictor + steering\n(F10)", "#d62728")
arrow(7.9, 0.95, 8.6, 0.95)
box(8.6, 0.6, 3.2, 0.7, r"$+91\%$ P(soluble), $41.7\%$ codesign" "\n" "(F13)", "#d4edda")

# Vertical bar between the two outcomes
ax.text(11.2, 2.4, "$L\\geq 300$\nbar", ha="center", va="center", fontsize=8.5, color="#333")
ax.plot([11.2, 11.2], [1.4, 3.5], color="#333", lw=0.8, ls=":")

# Caption block
ax.text(6, 0.05, r"Architectural route: no variant strictly beats canonical. Steering route clears the bar.",
        ha="center", va="bottom", fontsize=8.5, color="#333")

fig.savefig(Path(__file__).with_suffix(".pdf"))
