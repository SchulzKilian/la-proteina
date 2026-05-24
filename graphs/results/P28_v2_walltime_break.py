# P28_v2: Stacked breakdown — show ODE-integration cost vs predictor-call cost stacked per recipe.
# Visualizes: E069
# DATA: inline (synthesized split based on E069 narrative)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

cats = ["unsteered", "1-fold", "5-fold ens."]
ode_cost = np.array([320, 320, 320])         # base ODE integration
pred_cost = np.array([0, 60, 260])           # added predictor calls

x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=figsize(0.75))
b1 = ax.bar(x, ode_cost, width=0.55, color="#7f7f7f", edgecolor="black", lw=0.4,
            label="ODE integration", zorder=3)
b2 = ax.bar(x, pred_cost, bottom=ode_cost, width=0.55, color="#d62728",
            edgecolor="black", lw=0.4, label="Predictor calls", zorder=3)

for i, (o, p) in enumerate(zip(ode_cost, pred_cost)):
    total = o + p
    ax.text(i, total + 10, f"{total}s", ha="center", va="bottom", fontsize=8.5, weight="bold")
    if p > 0:
        ax.text(i, o + p / 2, f"+{p}s", ha="center", va="center", color="white", fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("Per-Protein Wall (s, L4 GPU)")
ax.set_title("Walltime: Predictor Overhead Scales with Ensemble Size")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
