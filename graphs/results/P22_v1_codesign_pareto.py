# P22_v1: Codesign rate vs Delta-real (CamSol) Pareto frontier with production-knee annotation.
# Visualizes: F10 / F13 / E066 / E076
# DATA: inline from F13 cross-reference table (content_masterarbeit.md lines 1518-1521)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# (w, real_delta_camsol_sigma, codesign_rate_pct, marker)
pts = [
    (0,     0.00,  47.9, "o", "unsteered ($w{=}0$, $n{=}200$)"),
    (16,    0.18,  44.2, "s", "camsol max $w{=}16$"),
    (32,    0.65,  41.7, "D", "camsol max $w{=}32$ (knee)"),
    (48,    1.40,  29.2, "^", "camsol max $w{=}48$"),
    (64,    2.30,  18.8, "v", "camsol max $w{=}64$"),
    (128,   5.00,   2.1, "X", "camsol max $w{=}128$"),
]

fig, ax = plt.subplots(figsize=figsize(0.95))

# Pareto frontier connectors (in order of x ascending)
xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
order = np.argsort(xs)
xs_s = np.array(xs)[order]; ys_s = np.array(ys)[order]
ax.plot(xs_s, ys_s, "-", color="#7f7f7f", lw=0.6, alpha=0.5, zorder=1)

# Color by efficiency (codesign per delta)
cmap = plt.get_cmap("viridis")
for w, dx, dy, m, label in pts:
    eff = dy / max(dx, 0.1)
    color = cmap(min(eff / 60.0, 1.0))
    ax.scatter(dx, dy, marker=m, s=80, c=[color], edgecolor="black", lw=0.5, zorder=3)
    ax.annotate(f"$w{{=}}{w}$", (dx, dy), xytext=(6, 6), textcoords="offset points", fontsize=7.5)

# Highlight knee
knee = pts[2]  # w=32
ax.add_patch(plt.Circle((knee[1], knee[2]), 0.25, fill=False, edgecolor="#d62728", lw=1.4, zorder=4))
ax.annotate("production knee:\n$+0.65\\sigma$ CamSol, 41.7\\% codesign",
            xy=(knee[1], knee[2]), xytext=(1.5, 30), fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", lw=0.5, color="#d62728"))

ax.set_xlabel(r"$\Delta$ CamSol vs Unsteered ($\sigma$ units)")
ax.set_ylabel(r"Codesign Rate (\%)")
ax.set_title("Steering Pareto: Real-CamSol Gain vs Codesign Retention")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
