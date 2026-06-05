# P43_v1: Gradient-hacking vs CBM fix on iupred down-regulation (target=0.123).
# Visualizes: E109 (normal NA-v1 steering hacks) vs E110 (CBM fixes it). For the
#   `iupred3_fraction_disordered` target=0.123 DOWN-regulation, the normal
#   predictor's CLAIM slides down toward the setpoint while the REAL measured
#   disorder climbs UP, away from the target and above the unsteered baseline
#   (0.158) -- the Goodhart signature. Swapping in the concept-bottleneck (CBM)
#   predictor makes predicted AND real both land near the target (honest control).
# DATA (all read live from source; nothing is a predictor value masquerading as
#   real -- the two are plotted as SEPARATE series):
#   - REAL  = mean `iupred3_fraction_disordered` measured by IUPred3 on the decoded
#     sequences, from results/iupred_target_sweep/iupred_target_w*/properties_guided.csv
#     (normal NA-v1) and results/iupred_target_cbm/iupred_target_cbm_w32/... (CBM).
#   - PREDICTED = mean of the predictor's final-step claim, from the per-protein
#     diagnostics JSONs (`predicted_properties.iupred3_fraction_disordered`, last
#     ODE step; batch elem 0 per (seed,length) file).
#   NOTE: the sweep's steering_cost_audit.csv is mislabeled (prop_target=swi) per
#   E109 caveat -- we deliberately do NOT use it; real comes from properties_guided.csv.
# NOTE: no in-axes title (caption goes underneath in LaTeX).
import sys
import csv
import json
import glob
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent / "results"
TARGET = 0.123        # natural setpoint the steering aims for (down-regulation)
BASELINE = 0.158      # unsteered La-Proteina disorder (E067/E099)
PROP = "iupred3_fraction_disordered"
C_HACK = "#d62728"    # normal NA-v1 (hacked)
C_CBM = "#1b9e77"     # CBM (honest)


def real_mean(props_csv):
    vs = []
    with open(props_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                vs.append(float(r[PROP]))
            except (ValueError, KeyError):
                pass
    return statistics.mean(vs)


def pred_final_mean(diag_dir):
    vs = []
    for fp in glob.glob(f"{diag_dir}/*_diagnostics.json"):
        steps = json.load(open(fp))
        pp = steps[-1].get("predicted_properties", {})
        if PROP in pp:
            vs.append(pp[PROP])
    return statistics.mean(vs)


# --- Normal NA-v1 dose-response (real + predicted), w in {8,16,24,32} ---
NA_W = [8, 16, 24, 32]
na_real = [real_mean(ROOT / f"iupred_target_sweep/iupred_target_w{w}/properties_guided.csv")
           for w in NA_W]
na_pred = [pred_final_mean(ROOT / f"iupred_target_sweep/iupred_target_w{w}/diagnostics")
           for w in NA_W]

# --- CBM single cell (w=32) ---
CBM_W = 32
cbm_real = real_mean(ROOT / "iupred_target_cbm/iupred_target_cbm_w32/properties_guided.csv")
cbm_pred = pred_final_mean(ROOT / "iupred_target_cbm/iupred_target_cbm_w32/diagnostics")

fig, ax = plt.subplots(figsize=figsize(0.82, 0.66))

# Reference levels.
ax.axhline(BASELINE, color="#9aa0a6", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.axhline(TARGET, color="black", lw=0.8, ls=":", zorder=1)
ax.text(NA_W[0] - 0.3, BASELINE + 0.002, r"unsteered baseline ($0.158$)",
        fontsize=7.5, color="#6b6b6b", va="bottom")
ax.text(NA_W[0] - 0.3, TARGET - 0.004, r"target ($0.123$)",
        fontsize=7.5, color="black", va="top")

# Normal NA-v1: predicted dives toward target, real climbs away -> shade the gap.
ax.fill_between(NA_W, na_pred, na_real, color=C_HACK, alpha=0.10, zorder=1.5)
ax.plot(NA_W, na_real, "-o", color=C_HACK, lw=1.4, ms=5, zorder=3,
        label="normal (NA-v1): real")
ax.plot(NA_W, na_pred, "--o", color=C_HACK, lw=1.2, ms=5, zorder=3,
        markerfacecolor="white", label="normal (NA-v1): predicted")

# CBM: predicted and real both near target (honest). Connector shows tiny gap.
ax.plot([CBM_W, CBM_W], [cbm_pred, cbm_real], "-", color=C_CBM, lw=1.0, zorder=3)
ax.plot(CBM_W, cbm_real, "*", color=C_CBM, ms=15, zorder=5,
        markeredgecolor="black", markeredgewidth=0.4, label="CBM: real")
ax.plot(CBM_W, cbm_pred, "s", color="white", ms=7, zorder=5,
        markeredgecolor=C_CBM, markeredgewidth=1.3, label="CBM: predicted")

# Call out the divergence.
ax.annotate("predictor says $\\downarrow$,\nreality goes $\\uparrow$\n(Goodhart hack)",
            xy=(24, (na_pred[2] + na_real[2]) / 2), xytext=(13.5, 0.142),
            fontsize=7.5, color=C_HACK, ha="left", va="center")

ax.set_xlabel(r"Steering weight $w$")
ax.set_ylabel(r"IUPred disordered fraction")
ax.set_xticks(NA_W)
ax.set_xlim(NA_W[0] - 1.5, NA_W[-1] + 2)
ax.set_ylim(0.08, 0.195)
ax.legend(loc="center right", frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
