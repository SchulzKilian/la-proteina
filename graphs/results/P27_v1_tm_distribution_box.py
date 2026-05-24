# P27_v1: Ensemble pairwise TM-score per (direction, w) cell — boxplot with unsteered baseline.
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

# Pick a representative L=300 only, camsol_max, ws in {1, 4, 16}
def pick(direction, w, L):
    for r in rows:
        if r["direction"] == direction and r["w"] == str(w) and r["L"] == str(L):
            return float(r["mean"]), float(r["p10"]), float(r["p90"])
    return None

ws = [1, 4, 16]
L_list = [300, 400, 500]
cells = []
labels = []
for L in L_list:
    for w in ws:
        m, p10, p90 = pick("camsol_max", w, L)
        cells.append((m, p10, p90))
        labels.append(f"$w{{=}}{w}$\n$L{{=}}{L}$")

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.5))
x = np.arange(len(cells))
means = [c[0] for c in cells]; p10 = [c[1] for c in cells]; p90 = [c[2] for c in cells]

# Make box-like artifacts: rectangle from p10 to p90, mean as horizontal line
for i, (m, lo, hi) in enumerate(cells):
    ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color="#1f77b4", alpha=0.2, lw=0))
    ax.hlines(m, i - 0.3, i + 0.3, color="#1f77b4", lw=1.4)
    ax.scatter([i], [m], color="#1f77b4", s=24, zorder=3, edgecolor="black", lw=0.4)

ax.axhline(0.413, color="#d62728", ls="--", lw=0.8, label="unsteered baseline (0.413)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Mean Pairwise TM-score")
ax.set_title("Ensemble Diversity Across (w, L) — Steering Does Not Collapse")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")
ax.set_ylim(0.2, 0.9)

fig.savefig(Path(__file__).with_suffix(".pdf"))
