# P31_v2: Aromatic burial ratio — grouped bar (gen vs ref) for each aromatic, with bootstrap CIs.
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

residues = ["W", "F", "Y", "H", "Aromatic"]
def pick(set_, res):
    for r in rows:
        if r["set"] == set_ and r["residue"] == res:
            return float(r["ratio_bur_exp"]), float(r["ratio_bur_exp_lo"]), float(r["ratio_bur_exp_hi"])
    return np.nan, np.nan, np.nan

x = np.arange(len(residues))
w = 0.4
fig, ax = plt.subplots(figsize=figsize(0.9))
gen_vals = []; ref_vals = []
gen_err = [[], []]; ref_err = [[], []]
for res in residues:
    g, gl, gh = pick("gen", res)
    r, rl, rh = pick("ref", res)
    gen_vals.append(g); ref_vals.append(r)
    gen_err[0].append(g - gl); gen_err[1].append(gh - g)
    ref_err[0].append(r - rl); ref_err[1].append(rh - r)

ax.bar(x - w/2, ref_vals, width=w, color="#1f77b4", edgecolor="black", lw=0.4,
       yerr=ref_err, capsize=2, error_kw=dict(lw=0.5), label="AFDB ref", zorder=3)
ax.bar(x + w/2, gen_vals, width=w, color="#d62728", edgecolor="black", lw=0.4,
       yerr=gen_err, capsize=2, error_kw=dict(lw=0.5), label="Gen (nsteps=400)", zorder=3)

ax.axhline(1.0, color="#7f7f7f", lw=0.6, ls=":")
ax.set_xticks(x); ax.set_xticklabels(residues)
ax.set_ylabel(r"Burial Ratio P(AA$|$buried)/P(AA$|$exposed)")
ax.set_title("Aromatic Burial: Gen Over-Targets W and Y")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
