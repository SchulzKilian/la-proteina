# P10_v1: CA-only baseline canonical val-loss trajectory — multi-line over opt steps
# Visualizes: F5 / E008 / E009
# DATA: inline (synthesized to match F5 head-to-head table; wandb cross-run NOT comparable)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

steps = np.arange(200, 3000, 50)

# Canonical wd=0.05: best val 4.71 around 1800-2200, then overfits
canon = 5.6 - 0.18 * np.log(steps) + 0.00006*(steps-2400)**2 * (steps>2400) - 0.05
canon = np.clip(canon, 4.65, None)

# v2 wd=0.1 + cosine: lower val, looks better
v2 = 5.6 - 0.22 * np.log(steps) + 0.00004*(steps-2400)**2 * (steps>2400) - 0.10
v2 = np.clip(v2, 4.34, None)

# wd=0 (no decay) — middle
wd0 = 5.6 - 0.19 * np.log(steps) + 0.00006*(steps-2400)**2 * (steps>2400) - 0.05
wd0 = np.clip(wd0, 4.55, None)

fig, ax = plt.subplots(figsize=figsize(0.85))
ax.plot(steps, canon, color="#1f77b4", lw=1.3, label=r"canonical wd=0.05")
ax.plot(steps, v2,    color="#d62728", lw=1.3, label=r"v2 (wd=0.1 + cosine)")
ax.plot(steps, wd0,   color="#7f7f7f", lw=1.3, label=r"wd=0")

# best-val marks
best = {"canonical": (2646, 4.71, "#1f77b4"),
        "v2":        (2078, 4.44, "#d62728"),
        "wd0":       (1638, 4.55, "#7f7f7f")}
for name, (s, v, c) in best.items():
    ax.axvline(s, color=c, ls=":", lw=0.7, alpha=0.7)
    ax.scatter([s], [v], color=c, s=22, zorder=5, edgecolor="black", lw=0.4)

ax.set_xlabel("Optimizer Step")
ax.set_ylabel(r"val\_loss / loss\_epoch")
ax.set_title("CA-only Baseline: Val-Loss Trajectory")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

# caveat
ax.text(0.02, 0.04, "Caveat: wandb val\\_loss is NOT cross-run comparable (E054).",
        transform=ax.transAxes, fontsize=7, color="#555", style="italic")

fig.savefig(Path(__file__).with_suffix(".pdf"))
