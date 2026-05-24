# P15_v1: variant designability ladder — grouped bar by L (50/100/200), one cluster per variant
# Visualizes: E019/E021/E034/E039/E053/E055/E056/E077
# DATA: inline from experiments.md entries (rates per L per variant)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# Variant: (label, step/N, [r50, r100, r200])
variants = [
    ("canonical (N=30)",          "2646",  [0.87, 0.83, 0.57]),
    ("sparse K40 (N=6)",          "1259",  [0.83, 0.50, 0.17]),
    ("sparse K40+PU (N=6)",       "1133",  [0.33, 0.17, 0.00]),
    ("scnbr_t04+FixC2 (N=6)",     "1133",  [0.50, 0.33, 0.00]),
    ("downsampled (N=6)",         "2331",  [0.17, 0.00, 0.00]),
    ("K64-curric (N=6)",          "944",   [0.33, 0.17, 0.00]),
    ("K64-curric+BigBird (N=6)",  "1100",  [0.17, 0.00, 0.00]),
    ("K40+curric+self (N=6)",     "1133",  [0.50, 0.33, 0.17]),
]

labels = [v[0] for v in variants]
rates  = np.array([v[2] for v in variants])

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.55))
x = np.arange(len(variants))
w = 0.25
COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
for i, (L, c) in enumerate(zip([50, 100, 200], COLORS)):
    bars = ax.bar(x + (i-1)*w, rates[:, i], width=w, color=c,
                  edgecolor="black", lw=0.4, label=f"$L{{=}}{L}$", zorder=3)

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.5)
ax.set_ylabel("Designability rate")
ax.set_title("Architectural Variants: No Variant Strictly Beats Canonical")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5)
ax.axhline(rates[0].mean(), color="#1f77b4", ls=":", lw=0.6)
ax.text(len(variants)-0.5, rates[0].mean(), f"canon mean={rates[0].mean():.2f}",
        ha="right", va="bottom", color="#1f77b4", fontsize=7)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
