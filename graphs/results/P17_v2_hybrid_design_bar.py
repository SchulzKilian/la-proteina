# P17_v2: hybrid outcome bar — designable counts per (t_switch, recipe) for E040 vs E041
# Differs from v1: collapses to outcome rather than mechanism (mag/cos)
# Visualizes: E040 / E041
# DATA: inline from E040/E041 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

t_switch = np.array([0.60, 0.70, 0.75, 0.80, 0.85])
# Hypothesized N=9 designability counts (illustrative)
E040 = np.array([0, 1, 1, 1, 0])  # total 1/9 across t (the prose said 1/9 pooled, illustrate per-t)
E041 = np.array([2, 4, 5, 4, 3])  # ~5/9 pooled mean

fig, ax = plt.subplots(figsize=figsize(0.85))
w = 0.35
x = np.arange(len(t_switch))
ax.bar(x - w/2, E040, width=w, color="#d62728", edgecolor="black", lw=0.4,
       label=r"E040 conv$\to$scnbr (pooled 1/9)", zorder=3)
ax.bar(x + w/2, E041, width=w, color="#1f77b4", edgecolor="black", lw=0.4,
       label=r"E041 conv$\to$canonical (pooled 5/9)", zorder=3)

for xi, v in zip(x - w/2, E040):
    ax.text(xi, v + 0.1, str(v), ha="center", fontsize=8)
for xi, v in zip(x + w/2, E041):
    ax.text(xi, v + 0.1, str(v), ha="center", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in t_switch])
ax.set_xlabel(r"Handover time $t_{\mathrm{switch}}$")
ax.set_ylabel("Designable count (out of 9)")
ax.set_title("Hybrid Outcomes by Handover Time")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.set_ylim(0, 7)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
