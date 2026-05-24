# P29_v1: Alphabet collapse — diverging bar of (gen - ref) / ref per AA, AFDB vs PDB.
# Visualizes: F8 / E020 / E026
# Source: results/aa_composition_nsteps400/stratified_vs_afdb/aa_composition.csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
import csv

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

AFDB_CSV = Path("/home/ks2218/la-proteina/results/aa_composition_nsteps400/stratified_vs_afdb/aa_composition.csv")
PDB_CSV  = Path("/home/ks2218/la-proteina/results/aa_composition_nsteps400/stratified_vs_pdb/aa_composition.csv")

def load(p):
    rows = {r["aa"]: float(r["rel_diff_pct"]) for r in csv.DictReader(p.open())}
    return rows
afdb = load(AFDB_CSV)
pdb  = load(PDB_CSV) if PDB_CSV.exists() else afdb  # fallback if missing

# Sort AAs by AFDB rel_diff descending
sorted_aas = sorted(AA_ORDER, key=lambda a: afdb.get(a, 0))
afdb_vals = [afdb.get(a, np.nan) for a in sorted_aas]
pdb_vals  = [pdb.get(a, np.nan) for a in sorted_aas]

y = np.arange(len(sorted_aas))
w = 0.4
fig, ax = plt.subplots(figsize=figsize(0.9, ratio=0.95))
ax.barh(y - w/2, afdb_vals, height=w, color="#1f77b4", edgecolor="black", lw=0.3, label="vs AFDB", zorder=3)
ax.barh(y + w/2, pdb_vals,  height=w, color="#ff7f0e", edgecolor="black", lw=0.3, label="vs PDB", zorder=3)

ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(sorted_aas, fontsize=9)
ax.set_xlabel(r"Relative Difference (\%)")
ax.set_title("Alphabet Collapse: Gen vs Natural")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")

fig.savefig(Path(__file__).with_suffix(".pdf"))
