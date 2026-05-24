# P12_v1: wd x design quality — grouped bar (one group per L, four bars per group)
# Visualizes: F5 / F7 / E014 / E019 (headline figure for the wd-vs-designability story)
# DATA: inline from F7 N=30 table (content_masterarbeit.md lines 645-650)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

Ls = [50, 100, 200]
arms = ["canonical (wd=0.05)", "v2 (wd=0.1)", "wd=0", "sparse K40"]
# N=30 designability rates (fraction designable) per L
rates = {
    "canonical (wd=0.05)": [0.87, 0.83, 0.57],   # step 2646
    "v2 (wd=0.1)":         [0.00, 0.00, 0.00],   # step 2078
    "wd=0":                [0.30, 0.13, 0.07],   # step 1638
    "sparse K40":          [0.40, 0.33, 0.17],   # step 1259
}
COLORS = ["#1f77b4", "#d62728", "#7f7f7f", "#9467bd"]

fig, ax = plt.subplots(figsize=figsize(0.95))
x = np.arange(len(Ls))
w = 0.18
for i, (arm, c) in enumerate(zip(arms, COLORS)):
    vals = rates[arm]
    xs = x + (i - 1.5) * w
    bars = ax.bar(xs, vals, width=w, color=c, label=arm, edgecolor="black", lw=0.4, zorder=3)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.02, f"{int(v*30)}/30" if v > 0 else "0", ha="center", fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels([f"$L{{=}}{L}$" for L in Ls])
ax.set_ylabel("Designability rate (N=30)")
ax.set_ylim(0, 1.0)
ax.set_title("Designability vs Training Recipe / Architecture")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

# pooled total
totals = {arm: int(np.array(rates[arm]).sum() * 30) for arm in arms}
ax.text(0.02, 0.02, "Pooled /90: " + ", ".join(f"{a.split()[0]}={totals[a]}" for a in arms),
        transform=ax.transAxes, fontsize=7, color="#555")

fig.savefig(Path(__file__).with_suffix(".pdf"))
