# P22_v2: Dual-axis line — codesign on left axis, Delta-real-CamSol on right axis, both vs w.
# Makes the trade-off explicit as w increases.
# Visualizes: F10 / F13 / E066 / E076
# DATA: inline from F13 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

w = np.array([0, 16, 32, 48, 64, 128])
codesign = np.array([47.9, 44.2, 41.7, 29.2, 18.8, 2.1])
camsol = np.array([0.0, 0.18, 0.65, 1.40, 2.30, 5.00])

fig, ax = plt.subplots(figsize=figsize(0.95))
ax2 = ax.twinx()

l1, = ax.plot(w, codesign, "-o", color="#1f77b4", lw=1.4, label="Codesign (\\%)", markersize=5)
l2, = ax2.plot(w, camsol, "--s", color="#d62728", lw=1.4, label="$\\Delta$ CamSol ($\\sigma$)",
               markersize=5, mfc="white")

# Mark knee
ax.axvline(32, color="#7f7f7f", ls=":", lw=0.8)
ax.annotate("knee $w{=}32$", xy=(32, 41.7), xytext=(40, 50),
            fontsize=8, color="#333",
            arrowprops=dict(arrowstyle="->", lw=0.5, color="#333"))

ax.set_xlabel("Steering Weight $w$")
ax.set_ylabel("Codesign Rate (\\%)", color="#1f77b4")
ax2.set_ylabel(r"$\Delta$ CamSol vs Unsteered ($\sigma$)", color="#d62728")
ax.tick_params(axis="y", colors="#1f77b4")
ax2.tick_params(axis="y", colors="#d62728")
ax.set_title("Codesign vs $\\Delta$ CamSol: Both as $w$ Increases")
ax.legend(handles=[l1, l2], frameon=False, fontsize=8.5, loc="center right")
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
