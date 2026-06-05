# P40_v1: Codesign rate vs RAW property score Pareto frontier — iupred_max direction.
# Visualizes: E067 / E070 steering Pareto. As steering weight w rises, the raw
#   IUPred disordered fraction grows while codesign falls. Sharper knee than
#   camsol/tango: by w=64 the raw fraction is ~0.91 (fully-disordered IDP regime),
#   where codesign is mechanistically ill-posed; w=128 saturates at 1.0.
# DATA: results/iupred_max_scout/steering_cost_audit.csv (NA-v1 predictor). Raw
#   per-cell property mean = `prop_value` (prop_target=iupred3_fraction_disordered);
#   unsteered raw anchor = `prop_anchor` (0.158). Codesign = per-cell `codesign_rate`;
#   unsteered codesign anchor = n=48 paired baseline (47.9 %, E070). iupred_max was
#   swept at w in {32, 64, 128} for the E070 Pareto (no w=48 cell).
# NOTE: raw scores on x (not sigma). No title (caption underneath in LaTeX); no
#   knee annotation.
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

CSV = (Path(__file__).parent.parent.parent / "results"
       / "iupred_max_scout" / "steering_cost_audit.csv")
DIRECTION = "iupred_max"
WSET = {32, 64, 128}
XLABEL = r"IUPred disordered fraction"
INVERT_X = False                       # higher disorder is the steering target -> rightward
CODESIGN_UNSTEERED = 47.9              # n=48 paired baseline (E070)
COLOR = "#3b6fb6"

# Read raw per-cell property value + codesign for this direction.
steered, anchor_raw = [], None
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        if r["direction"] != DIRECTION or int(r["w"]) not in WSET:
            continue
        anchor_raw = float(r["prop_anchor"])
        steered.append((int(r["w"]), float(r["prop_value"]),
                        float(r["codesign_rate"]) * 100.0))
steered.sort()                                   # by w

# Point lists: unsteered anchor (w=0) first, then steered cells by w.
ws = [0] + [s[0] for s in steered]
val = [anchor_raw] + [s[1] for s in steered]
cod = [CODESIGN_UNSTEERED] + [s[2] for s in steered]

fig, ax = plt.subplots(figsize=figsize(0.68, 0.72))

# Pareto trade-off curve (codesign falls as the property is pushed).
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
if INVERT_X:
    ax.invert_xaxis()
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
