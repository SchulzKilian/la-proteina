# P27_v2: Mean pairwise TM vs w, one line per L; emphasizes that mean diversity is almost flat across w.
# Visualizes: F10 / E036
# Source: results/noise_aware_ensemble_sweep/diversity_pairwise_tm.csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
import csv

CSV = Path("/home/ks2218/la-proteina/results/noise_aware_ensemble_sweep/diversity_pairwise_tm.csv")
rows = list(csv.DictReader(CSV.open()))

def get(direction, w, L):
    for r in rows:
        if r["direction"] == direction and int(r["w"]) == w and int(r["L"]) == L:
            return float(r["mean"])
    return np.nan

ws = [1, 2, 4, 8, 16]
L_list = [300, 400, 500]
colors = ["#1f77b4", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=figsize(0.85))
for L, c in zip(L_list, colors):
    ys = [get("camsol_max", w, L) for w in ws]
    ax.plot(ws, ys, "-o", color=c, lw=1.3, markersize=5, label=f"$L{{=}}{L}$")
ax.axhline(0.413, color="#7f7f7f", ls="--", lw=0.8, label="unsteered (0.413)")

ax.set_xscale("log", base=2)
ax.set_xticks(ws); ax.set_xticklabels([str(w) for w in ws])
ax.set_xlabel("Steering Weight $w$ (camsol max)")
ax.set_ylabel("Mean Pairwise TM-score")
ax.set_title("Mean TM Stays Flat Across $w$ — Ensemble Not Collapsed")
ax.legend(frameon=False, fontsize=8, loc="best")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.set_ylim(0.25, 0.65)

fig.savefig(Path(__file__).with_suffix(".pdf"))
