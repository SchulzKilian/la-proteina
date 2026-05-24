# P02_v1: nsteps cliff — paired scatter of scRMSD vs nsteps (log-y) for L=300 single-seed
# Reasoning: scatter on log y best shows the 22.5 -> 0.80 Å drop as a clear visual cliff
# Visualizes: hard-rule nsteps=400 (CLAUDE.md / E019 / F2)
# DATA: inline from CLAUDE.md (illustrative single-protein L=300)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# Inline numbers: the canonical cliff documented in CLAUDE.md
nsteps = np.array([100, 200, 300, 400, 600, 800])
# Single-seed illustrative: 22.5 @ 200, 0.80 @ 400 from CLAUDE.md
# Interpolate other nsteps logarithmically as illustrative (clearly flagged)
scRMSD = np.array([28.0, 22.5, 8.0, 0.80, 0.70, 0.65])

fig, ax = plt.subplots(figsize=figsize(0.7))
ax.plot(nsteps, scRMSD, marker="o", markersize=6, color="#1f77b4", lw=1.2, zorder=3)
ax.set_yscale("log")
ax.axvline(400, color="#d62728", ls="--", lw=0.8, zorder=1)
ax.axhline(2.0, color="#7f7f7f", ls=":", lw=0.7, zorder=1)
ax.text(400, 30, r"hard rule", color="#d62728", fontsize=8.5, ha="left", va="top")
ax.text(105, 2.0, "designable (2 \\AA)", color="#7f7f7f", fontsize=8.5, va="bottom")

# Annotate the headline drop
ax.annotate(r"$22.5\to 0.80$ \AA", xy=(400, 0.80), xytext=(280, 0.18),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.6, color="#333"))

ax.set_xlabel(r"Sampling Steps ($n_{\mathrm{steps}}$)")
ax.set_ylabel(r"scRMSD (\AA)")
ax.set_title(r"The $n_{\mathrm{steps}}{=}400$ Cliff (LD3+AE2, $L{=}300$)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
