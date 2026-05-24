# P26_v1: Multi-objective cocktail — grouped bar of SWI Delta-sigma + codesign rate at w=32.
# Single vs 2-obj vs 4-obj recipes.
# Visualizes: E068 / E072
# DATA: inline from E068/E072 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

recipes = ["camsol max\n(1-obj)", "camsol+tango\n(2-obj)", "camsol+tango+\nsap+scm$^+$ (4-obj)"]
swi_delta = [0.65, 0.78, 0.95]    # SWI delta-sigma at w=32
codesign  = [41.7, 38.9, 35.4]    # codesign rate %

x = np.arange(len(recipes))
w = 0.38

fig, ax = plt.subplots(figsize=figsize(0.9))
ax2 = ax.twinx()

b1 = ax.bar(x - w/2, swi_delta, width=w, color="#1f77b4", edgecolor="black", lw=0.4,
            label=r"$\Delta$ SWI ($\sigma$)", zorder=3)
b2 = ax2.bar(x + w/2, codesign, width=w, color="#d62728", edgecolor="black", lw=0.4,
            alpha=0.85, label="Codesign (\\%)", zorder=3)

for b, v in zip(b1, swi_delta):
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:+.2f}", ha="center",
            va="bottom", fontsize=8, color="#1f77b4")
for b, v in zip(b2, codesign):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}", ha="center",
             va="bottom", fontsize=8, color="#d62728")

ax.set_xticks(x); ax.set_xticklabels(recipes, fontsize=8.5)
ax.set_ylabel(r"$\Delta$ SWI ($\sigma$, vs unsteered)", color="#1f77b4")
ax2.set_ylabel(r"Codesign (\%)", color="#d62728")
ax.tick_params(axis="y", colors="#1f77b4"); ax2.tick_params(axis="y", colors="#d62728")
ax.set_title("Multi-Objective Cocktails at $w{=}32$")
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
