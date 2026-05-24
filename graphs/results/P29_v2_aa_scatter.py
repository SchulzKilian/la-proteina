# P29_v2: Scatter — gen_freq vs ref_freq with x=y line; AAs labeled, off-diagonal = collapse.
# Visualizes: F8 / E020 / E026
# Source: results/aa_composition_nsteps400/stratified_vs_afdb/aa_composition.csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
import csv

CSV = Path("/home/ks2218/la-proteina/results/aa_composition_nsteps400/stratified_vs_afdb/aa_composition.csv")
rows = list(csv.DictReader(CSV.open()))

aa = [r["aa"] for r in rows]
ref = np.array([float(r["ref_mean"]) for r in rows])
gen = np.array([float(r["gen_mean"]) for r in rows])
rel = np.array([float(r["rel_diff_pct"]) for r in rows])

fig, ax = plt.subplots(figsize=figsize(0.85))

# Identity line
lim = max(ref.max(), gen.max()) * 1.1
ax.plot([0, lim], [0, lim], color="#7f7f7f", ls="--", lw=0.8)

# Color by deviation magnitude
colors = ["#d62728" if abs(r) > 60 else ("#1f77b4" if abs(r) > 25 else "#7f7f7f") for r in rel]
ax.scatter(ref, gen, c=colors, s=42, edgecolor="black", lw=0.3, zorder=3)

# Label outliers
for a, x_, y_, r in zip(aa, ref, gen, rel):
    if abs(r) > 25 or a in ("E", "N", "M", "W"):
        offset = (5, 5) if y_ > x_ else (-12, -8)
        ax.annotate(a, (x_, y_), xytext=offset, textcoords="offset points", fontsize=9, weight="bold")

ax.set_xlabel(r"AFDB Reference Mole Fraction")
ax.set_ylabel(r"Generated Mole Fraction")
ax.set_title("Gen vs AFDB: $\\mathrm{E, N}$ Over, $\\mathrm{M, W, F, H}$ Under")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

fig.savefig(Path(__file__).with_suffix(".pdf"))
