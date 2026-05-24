# P18_v2: sparse-vs-dense throughput as side-by-side bars over a few L points + memory annotation
# (Alternative framing: discrete bars per L instead of continuous lines, easier to read magnitudes.)
# Visualizes: E073 / E074
# DATA: inline (synthesized from E073/E074 prose; sparse > dense at L=512, dense OOM at L>800)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L = ["128", "256", "512", "768", "1024", "1536"]
dense = [0.16, 0.28, 0.69, 1.40, np.nan, np.nan]   # OOM beyond ~800
sparse = [0.23, 0.27, 0.31, 0.36, 0.43, 0.51]
oom_idx = [i for i, v in enumerate(dense) if np.isnan(v)]

x = np.arange(len(L))
w = 0.4
fig, ax = plt.subplots(figsize=figsize(0.85))
b1 = ax.bar(x - w/2, [d if not np.isnan(d) else 0 for d in dense], width=w,
            color="#1f77b4", edgecolor="black", lw=0.4, label="dense canonical", zorder=3)
b2 = ax.bar(x + w/2, sparse, width=w, color="#d62728", edgecolor="black",
            lw=0.4, label="sparse K=40", zorder=3)

# OOM hatched markers
for i in oom_idx:
    ax.bar(i - w/2, max(sparse)*1.1, width=w, color="white", hatch="///",
           edgecolor="#1f77b4", lw=0.4, zorder=2)
    ax.text(i - w/2, max(sparse)*1.1, "OOM", ha="center", va="bottom",
            fontsize=7, color="#1f77b4")

# Annotate each bar
for i, v in enumerate(sparse):
    ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
for i, v in enumerate(dense):
    if not np.isnan(v):
        ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(L)
ax.set_xlabel("Sequence Length $L$")
ax.set_ylabel("Time / Opt Step (s)")
ax.set_title("Per-Step Wall: Sparse Loses at $L{=}512$, Wins at $L{\\geq}1024$")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
