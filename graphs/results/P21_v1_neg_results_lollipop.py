# P21_v1: Negative-results ladder — horizontal lollipop of predictor-fix attempts ordered by gap @ w=16.
# Visualizes: F10 / E028 / E029 / E030 / E031 / E032
# DATA: inline from F10 negative-results table (content_masterarbeit.md lines 981-989)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# (approach, gap_at_w16). Negative gap = predictor under-reports vs real (gradient hacking).
fixes = [
    ("clean predictor (baseline)",          -203),
    ("smaller predictor (1L)",              -187),
    ("longer training",                      -171),
    ("L2 weight on grad",                    -154),
    ("freeze first layer",                  -132),
    ("clean ensemble (5-fold)",             -118),
    ("noise-aware (1-fold)",                 -71),
    ("NA-v1 ensemble (5-fold) [n=48]",        +4),
    ("NA-v1 ensemble (5-fold) [n=4 pilot]",   +9),
]

# Sort so worst at top, best at bottom
fixes_sorted = sorted(fixes, key=lambda x: x[1])
labels = [a for a, _ in fixes_sorted]
vals   = [v for _, v in fixes_sorted]
colors = ["#d62728" if v < -50 else ("#ff7f0e" if v < 0 else "#1f77b4") for v in vals]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.55))
y = np.arange(len(labels))
ax.hlines(y, 0, vals, colors=colors, lw=1.4)
ax.scatter(vals, y, color=colors, s=40, zorder=3, edgecolor="black", lw=0.4)

# Annotate each
for yi, v in zip(y, vals):
    offset = 8 if v < 0 else -8
    ha = "left" if v < 0 else "right"
    ax.annotate(f"{v:+d}", (v, yi), xytext=(offset, 0), textcoords="offset points",
                ha=ha, va="center", fontsize=8)

ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Predictor $-$ Real Gap at $w{=}16$ (TANGO units)")
ax.axvline(0, color="black", lw=0.6)
ax.set_title("Five Plausible Fixes Failed Before NA-v1 + Ensemble Worked")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")

fig.savefig(Path(__file__).with_suffix(".pdf"))
