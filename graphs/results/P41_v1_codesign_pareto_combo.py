# P41_v1: Codesign rate vs real CamSol solubility Pareto frontier — 2-obj combo.
# Visualizes: E068 / E116 steering Pareto for the camsol_max + tango_min cocktail
#   (both objectives at weight 1.0), on the real CamSol solubility axis.
# CAVEAT: this is a TWO-objective cell (solubility AND aggregation). The x-axis
#   shows only the SOLUBILITY (CamSol) delivery; the simultaneous TANGO reduction
#   is not on this axis. Read alongside P22_v1 (camsol-alone): at matched w the
#   combo delivers slightly less CamSol than camsol-alone because it splits effort
#   into the TANGO objective (not shown here) — that trade is the point of the combo.
# DATA:
#   x = real CamSol pH 7, mean per cell, from results/camsol_ph7_full_2026_05_27.csv
#       (E116, `combo_camsol_tango_w*` subruns). Unsteered anchor = seed-paired
#       unguided control (`sanity_*_u`, seeds 42-57, mean ~1.63).
#   y = codesign rate per cell, from results/combo_camsol_tango_scout/
#       steering_cost_audit.csv (NA-v1); unsteered codesign anchor = n=48 paired
#       baseline (47.9 %, E070).
# NOTE: no in-axes title (caption goes underneath in LaTeX); no knee annotation.
import sys
import csv
import collections
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
CAMSOL_CSV = ROOT / "results" / "camsol_ph7_full_2026_05_27.csv"
AUDIT_CSV = ROOT / "results" / "combo_camsol_tango_scout" / "steering_cost_audit.csv"
DIRECTION = "combo_camsol_tango"
CELL_PREFIX = "combo_camsol_tango_w"     # subrun pattern in the CamSol CSV
WSET = [32, 48, 64, 128]
XLABEL = r"CamSol intrinsic solubility (pH 7)"
CODESIGN_UNSTEERED = 47.9                # n=48 paired baseline (E070)
COLOR = "#3b6fb6"

# --- real CamSol mean per cell + seed-paired unguided anchor ---
vals = collections.defaultdict(list)
unguided = []
with open(CAMSOL_CSV, newline="") as f:
    for r in csv.DictReader(f):
        try:
            v = float(r["camsol_ph7"])
        except ValueError:
            continue
        sr = r["subrun"]
        if sr.startswith("sanity_") and sr.endswith("_u"):
            unguided.append(v)
        else:
            vals[sr].append(v)
camsol_anchor = statistics.mean(unguided)
camsol_by_w = {w: statistics.mean(vals[f"{CELL_PREFIX}{w}"]) for w in WSET}

# --- codesign per cell ---
codesign_by_w = {}
with open(AUDIT_CSV, newline="") as f:
    for r in csv.DictReader(f):
        if r["direction"] == DIRECTION and int(r["w"]) in WSET:
            codesign_by_w[int(r["w"])] = float(r["codesign_rate"]) * 100.0

# Point lists: unsteered anchor (w=0) first, then steered cells by w.
ws = [0] + WSET
val = [camsol_anchor] + [camsol_by_w[w] for w in WSET]
cod = [CODESIGN_UNSTEERED] + [codesign_by_w[w] for w in WSET]

fig, ax = plt.subplots(figsize=figsize(0.68, 0.72))

# Pareto trade-off curve (codesign falls as solubility is pushed).
ax.plot(val, cod, "-", color="#9aa0a6", lw=0.8, zorder=1)
ax.scatter(val[0], cod[0], marker="*", s=170, color="#d62728",
           edgecolor="black", lw=0.5, zorder=4, label=r"unsteered ($w{=}0$)")
ax.scatter(val[1:], cod[1:], marker="o", s=70, color=COLOR,
           edgecolor="black", lw=0.5, zorder=3, label="steered")
for w, x, y in zip(ws[1:], val[1:], cod[1:]):
    ax.annotate(rf"$w{{=}}{w}$", (x, y), xytext=(6, 6),
                textcoords="offset points", fontsize=8)

ax.set_xlabel(XLABEL)
ax.set_ylabel(r"Codesign rate (\%)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
