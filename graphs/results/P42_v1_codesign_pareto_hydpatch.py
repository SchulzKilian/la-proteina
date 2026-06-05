# P42_v1: Codesign rate vs property-score Pareto frontier — hydpatch_min direction.
# Visualizes: E102 steering Pareto. As steering weight w rises, the hydrophobic
#   patch total area falls (less aggregation-prone surface) while codesign falls.
#   This is the GENTLEST knee of any steered property: at w=32 codesign is 50.0 %
#   (above the 47.9 % baseline), and at w=64 it still holds 35.4 % vs camsol 18.8 /
#   tango 22.9 / iupred 10.4 / combo 20.8 % — the most free-lunch axis.
# DATA: results/hydpatch_min_sweep/steering_cost_audit.csv (NA-v1 predictor).
#   `hydrophobic_patch_total_area` is measured directly in the developability
#   panel, so `prop_value` is the real property score (no proxy). Unsteered raw
#   anchor = `prop_anchor`; unsteered codesign anchor = n=48 paired baseline
#   (47.9 %, this CSV's codesign_anchor / E070).
# NOTE: scores in property units (not sigma). Lower hydrophobic-patch area is
#   better, so the x-axis is inverted (improvement reads rightward, matching the
#   other panels). No title (caption underneath in LaTeX); no knee annotation.
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

CSV = (Path(__file__).parent.parent.parent / "results"
       / "hydpatch_min_sweep" / "steering_cost_audit.csv")
DIRECTION = "hydpatch_min"
WSET = {32, 48, 64, 128}
XLABEL = r"Hydrophobic patch area (\AA$^2$)"
INVERT_X = True                        # lower area is better -> improvement rightward
CODESIGN_UNSTEERED = 47.9              # n=48 paired baseline (E070)
COLOR = "#3b6fb6"

# Read property value + codesign for this direction.
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
