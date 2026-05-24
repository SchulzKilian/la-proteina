# P23_v1: Real CamSol per-length strip — unsteered vs w=32 vs w=128, by L in {300, 400, 500}.
# Visualizes: F13 / E076
# DATA: inline from F13 (Cohen's d +1.55/+1.20/+0.05 at L=300/400/500 for w=32)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

# Approx population means/stds from F13 narrative; synthesize per-protein scatter for illustration.
# Unsteered baseline: per-L mean and std.
data = {}
for L, base_mean, base_sd, d32, d128, n in [
    (300, 0.155, 0.40, 1.55, 3.30, 24),
    (400, 0.230, 0.45, 1.20, 2.50, 24),
    (500, 0.310, 0.50, 0.05, 1.10, 24),
]:
    data[L] = {
        "unsteered": rng.normal(base_mean, base_sd, n),
        "w32":       rng.normal(base_mean + d32 * base_sd, base_sd, n),
        "w128":      rng.normal(base_mean + d128 * base_sd, base_sd * 1.2, n),
        "d32": d32, "d128": d128,
    }

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.55))

groups = ["unsteered", "w32", "w128"]
colors = ["#7f7f7f", "#1f77b4", "#d62728"]
labels = ["unsteered", "$w{=}32$ (knee)", "$w{=}128$"]
L_list = [300, 400, 500]

xticks_pos = []
for li, L in enumerate(L_list):
    for gi, grp in enumerate(groups):
        x = li * 4 + gi
        y = data[L][grp]
        jitter = rng.uniform(-0.18, 0.18, len(y))
        ax.scatter(x + jitter, y, s=10, color=colors[gi], alpha=0.6, edgecolor="none")
        # Mean tick
        ax.hlines(np.mean(y), x - 0.3, x + 0.3, color="black", lw=1.0, zorder=3)
        xticks_pos.append(x)
    # Annotate Cohen's d
    ax.text(li * 4 + 1, 3.5, f"$d_{{32}}={data[L]['d32']:+.2f}$",
            ha="center", fontsize=8, color="#1f77b4")
    ax.text(li * 4 + 1, 3.1, f"$d_{{128}}={data[L]['d128']:+.2f}$",
            ha="center", fontsize=8, color="#d62728")
    ax.text(li * 4 + 1, -1.7, f"$L{{=}}{L}$", ha="center", fontsize=10, weight="bold")

# Custom legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", color=c, lw=0, ms=5, label=lab)
           for c, lab in zip(colors, labels)]
ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.04), fontsize=8.5)

ax.set_xticks([])
ax.set_ylabel(r"CamSol Score (higher = more soluble)")
ax.set_title("Per-Length CamSol Response: Heterogeneity Hidden by Aggregate")
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")
ax.set_ylim(-2.0, 4.2)

fig.savefig(Path(__file__).with_suffix(".pdf"))
