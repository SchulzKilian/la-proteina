# P31_v1: Aromatic burial-targeting ratio (P(aromatic|buried) / P(aromatic|exposed)) — slope chart.
# Lines: AFDB -> gen for W, F, Y, H, Aromatic (any).
# Visualizes: F8 (c) / E023 / E026
# Source: results/aromatic_burial_afdb/aromatic_frequencies.csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
import csv

CSV = Path("/home/ks2218/la-proteina/results/aromatic_burial_afdb/aromatic_frequencies.csv")
rows = list(csv.DictReader(CSV.open()))

# index by (set, residue) -> ratio
ratios = {}
for r in rows:
    ratios[(r["set"], r["residue"])] = (float(r["ratio_bur_exp"]),
                                        float(r["ratio_bur_exp_lo"]),
                                        float(r["ratio_bur_exp_hi"]))

residues = ["W", "F", "Y", "H", "Aromatic"]
colors   = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]

fig, ax = plt.subplots(figsize=figsize(0.85))
x_ref, x_gen = 0, 1
for res, c in zip(residues, colors):
    if ("ref", res) not in ratios or ("gen", res) not in ratios:
        continue
    yr, yr_lo, yr_hi = ratios[("ref", res)]
    yg, yg_lo, yg_hi = ratios[("gen", res)]
    ax.plot([x_ref, x_gen], [yr, yg], "-o", color=c, lw=1.4, markersize=5)
    # Error bars
    ax.errorbar([x_ref], [yr], yerr=[[yr - yr_lo], [yr_hi - yr]], color=c, capsize=2, lw=0.6)
    ax.errorbar([x_gen], [yg], yerr=[[yg - yg_lo], [yg_hi - yg]], color=c, capsize=2, lw=0.6)
    # Label at gen side
    ax.text(x_gen + 0.04, yg, res, color=c, fontsize=9, va="center", weight="bold")

ax.axhline(1.0, color="#7f7f7f", lw=0.6, ls=":")
ax.text(0.5, 1.05, "no bias", ha="center", fontsize=7.5, color="#7f7f7f")
ax.set_xticks([x_ref, x_gen]); ax.set_xticklabels(["AFDB ref", "Gen (nsteps=400)"])
ax.set_ylabel(r"P(aromatic $|$ buried) / P(aromatic $|$ exposed)")
ax.set_title("Aromatic Burial Targeting: Gen Sharper Than AFDB")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")
ax.set_xlim(-0.15, 1.3)

fig.savefig(Path(__file__).with_suffix(".pdf"))
