"""Render top-5 over / top-5 underrepresented AAs as a PNG table figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AFDB = pd.read_csv(ROOT / "results/aa_composition_nsteps400/stratified_vs_afdb/aa_composition.csv")
PDB = pd.read_csv(ROOT / "results/aa_composition_nsteps400/stratified_vs_pdb/aa_composition.csv")
OUT = ROOT / "figures/aa_composition_top10.png"

THREE = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

m = AFDB[["aa", "ref_mean", "gen_mean"]].rename(columns={"ref_mean": "afdb", "gen_mean": "gen"})
m = m.merge(PDB[["aa", "ref_mean"]].rename(columns={"ref_mean": "pdb"}), on="aa")
m["rel_afdb"] = (m["gen"] - m["afdb"]) / m["afdb"] * 100
m["rel_pdb"] = (m["gen"] - m["pdb"]) / m["pdb"] * 100
m = m.sort_values("rel_afdb", ascending=False).reset_index(drop=True)

top5 = m.head(5)
bot5 = m.tail(5).iloc[::-1]  # most-underrepresented first

OVER_BG = "#e8f5e9"   # pale green
UNDER_BG = "#ffebee"  # pale red
HEADER_BG = "#37474f"

def signed(x: float) -> str:
    return f"+{x:.1f}" if x >= 0 else f"−{abs(x):.1f}"

def build_rows(df, kind):
    return [[
        f"{THREE[r['aa']]} ({r['aa']})",
        f"{r['afdb']*100:.2f}",
        f"{r['pdb']*100:.2f}",
        f"{r['gen']*100:.2f}",
        signed(r['rel_afdb']),
        signed(r['rel_pdb']),
    ] for _, r in df.iterrows()]

cols = ["Residue", "AFDB (%)", "PDB (%)", "Generated (%)", "Δ vs AFDB (%)", "Δ vs PDB (%)"]

fig, (ax_over, ax_under) = plt.subplots(2, 1, figsize=(8.4, 5.2),
                                        gridspec_kw={"hspace": 0.55})

for ax, title, rows, bg in [
    (ax_over, "Five most over-represented residues", build_rows(top5, "over"), OVER_BG),
    (ax_under, "Five most under-represented residues", build_rows(bot5, "under"), UNDER_BG),
]:
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=8)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor(HEADER_BG)
        c.set_text_props(color="white", weight="bold")
        c.set_edgecolor("white")
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            cell = tbl[i, j]
            cell.set_facecolor(bg if (i % 2 == 1) else "white")
            cell.set_edgecolor("#cfd8dc")
            if j == 0:
                cell.set_text_props(weight="bold")
            elif j >= 4:
                cell.set_text_props(weight="bold")

fig.suptitle(
    "La-Proteina jointly generated sequences vs natural reference proteomes",
    fontsize=11, y=0.985,
)
fig.text(0.5, 0.005,
         "n = 1000 generated proteins, lengths 300–800, nsteps = 400. "
         "Δ columns: relative deviation of generated frequency from reference.",
         ha="center", fontsize=8.5, color="#455a64")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
