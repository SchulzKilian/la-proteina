# P44_v2: Steering codesign-vs-property Pareto frontiers, three properties in one
#   row (hydrophobic patch | CamSol | TANGO), shared codesign axis + one legend.
#   VERSION v2 = WITH +/-1 SEM horizontal error bars. See P44_v1 for the clean version.
# Each point is a steering cell: x = mean real property score over the cell's
#   proteins, y = codesign rate. As w rises the property is pushed harder and
#   codesign falls. hydpatch & TANGO are "lower = better", so those x-axes are
#   inverted (improvement reads rightward, matching CamSol).
# w-grid: full dose-response {1,2,4,8,16,32,48,64,128} per property (low-w foot
#   + high-w head). Points carry NO per-point w labels and a single fixed marker
#   size (w is read off the monotone frontier order, not annotated).
# DATA (all real measured values, never predictor output):
#   - hydpatch: results/hydpatch_min_sweep/hydpatch_min_w*/properties_guided.csv
#       (`hydrophobic_patch_total_area`); codesign from that sweep's audit.
#   - CamSol:   results/camsol_ph7_full_2026_05_27.csv (E116, real CamSol pH 7,
#       `camsol_max_w*` subruns); codesign from the NA-v1 audit (camsol_max).
#   - TANGO:    results/noise_aware_high_w_scout/tango_min_w*/properties_guided.csv
#       (`tango_total`); codesign from the same NA-v1 audit (tango_min).
#   Unsteered anchor: seed-matched unguided control
#       results/sanity_unsteered_seed42_45/unguided/properties_unguided.csv (n=48)
#       for hydpatch/TANGO, and E116 `sanity_*_u` for CamSol; codesign anchor =
#       n=48 paired baseline 47.9 % (E070).
import sys
import csv
import collections
import statistics
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SHOW_SEM = True                           # <-- the only difference vs P44_v2

ROOT = Path(__file__).parent.parent.parent / "results"
UNGUIDED = ROOT / "sanity_unsteered_seed42_45" / "unguided" / "properties_unguided.csv"
NA_AUDIT_HIGH = ROOT / "noise_aware_high_w_scout" / "steering_cost_audit.csv"
NA_AUDIT_LOW = ROOT / "noise_aware_ensemble_sweep" / "steering_cost_audit.csv"
HYD_AUDIT = ROOT / "hydpatch_min_sweep" / "steering_cost_audit.csv"
E116 = ROOT / "camsol_ph7_full_2026_05_27.csv"
WSET = [1, 2, 4, 8, 16, 32, 48, 64, 128]


def tango_prop_path(w):
    # low-w TANGO real props live in the ensemble sweep; high-w in the scout.
    tree = "noise_aware_ensemble_sweep" if w <= 16 else "noise_aware_high_w_scout"
    return ROOT / f"{tree}/tango_min_w{w}/properties_guided.csv"
CODESIGN_UNSTEERED = 47.9                 # n=48 paired baseline (E070)
C_STEER, C_UNS, C_BAND = "#3b6fb6", "#d62728", "#9aa0a6"


def col_floats(path, prop):
    xs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                xs.append(float(r[prop]))
            except (ValueError, KeyError):
                pass
    return xs


def msem(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, s


def codesign(direction, *audits):
    # merge codesign rates across audits (low-w ensemble + high-w scout).
    d = {}
    for audit in audits:
        with open(audit, newline="") as f:
            for r in csv.DictReader(f):
                if r["direction"] == direction and int(r["w"]) in WSET:
                    d[int(r["w"])] = float(r["codesign_rate"]) * 100.0
    return d


# --- CamSol (E116): group per-protein real CamSol by cell + seed-paired anchor ---
e116 = collections.defaultdict(list)
camsol_base = []
with open(E116, newline="") as f:
    for r in csv.DictReader(f):
        try:
            v = float(r["camsol_ph7"])
        except ValueError:
            continue
        sr = r["subrun"]
        if sr.startswith("sanity_") and sr.endswith("_u"):
            camsol_base.append(v)
        else:
            e116[sr].append(v)

# Per-panel spec: (anchor values, {w: values}, codesign dict, xlabel, invert)
PANELS = [
    (col_floats(UNGUIDED, "hydrophobic_patch_total_area"),
     {w: col_floats(ROOT / f"hydpatch_min_sweep/hydpatch_min_w{w}/properties_guided.csv",
                    "hydrophobic_patch_total_area") for w in WSET},
     codesign("hydpatch_min", HYD_AUDIT),
     r"Hyd.\ patch area (\AA$^2$)", True),
    (camsol_base,
     {w: e116[f"camsol_max_w{w}"] for w in WSET},
     codesign("camsol_max", NA_AUDIT_LOW, NA_AUDIT_HIGH),
     r"CamSol score (pH 7)", False),
    (col_floats(UNGUIDED, "tango_total"),
     {w: col_floats(tango_prop_path(w), "tango_total") for w in WSET},
     codesign("tango_min", NA_AUDIT_LOW, NA_AUDIT_HIGH),
     r"TANGO score", True),
]

fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, 0.40), sharey=True)

for ax, (anchor, cells, cod, xlabel, invert) in zip(axes, PANELS):
    means = [msem(anchor)[0]] + [msem(cells[w])[0] for w in WSET]
    sems = [msem(anchor)[1]] + [msem(cells[w])[1] for w in WSET]
    cods = [CODESIGN_UNSTEERED] + [cod[w] for w in WSET]

    ax.plot(means, cods, "-", color=C_BAND, lw=0.8, zorder=1)
    if SHOW_SEM:
        ax.errorbar(means, cods, xerr=sems, fmt="none", ecolor=C_BAND,
                    elinewidth=1.0, capsize=2, zorder=2)
    ax.scatter(means[0], cods[0], marker="*", s=150, color=C_UNS,
               edgecolor="black", lw=0.5, zorder=4)
    ax.scatter(means[1:], cods[1:], marker="o", s=55, color=C_STEER,
               edgecolor="black", lw=0.5, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 56)
    ax.margins(x=0.16)
    if invert:
        ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

axes[0].set_ylabel(r"Codesign rate (\%)")

# One shared legend at the top.
handles = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=C_UNS,
           markeredgecolor="black", markersize=12, label=r"unsteered ($w{=}0$)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_STEER,
           markeredgecolor="black", markersize=7, label="steered"),
]
if SHOW_SEM:
    handles.append(Line2D([0], [0], color=C_BAND, lw=1.0, label=r"$\pm 1$ SEM"))
fig.legend(handles=handles, loc="upper center", ncol=len(handles),
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))

fig.tight_layout(w_pad=2.2)
fig.subplots_adjust(top=0.86)
fig.savefig(Path(__file__).with_suffix(".pdf"))
