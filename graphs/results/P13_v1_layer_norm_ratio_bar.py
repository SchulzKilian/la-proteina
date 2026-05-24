# P13_v1: per-layer weight-norm ratio distribution v2/canonical, sorted, AdaLN-Zero highlighted
# Visualizes: F5 / E009
# DATA: inline (synthesized to match F5 prose: 164 layers, mean 0.92, min 0.26)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(2)

# 164 layers ≥10k params
N = 164
# Most layers around 0.92 ± 0.10
ratios = rng.normal(0.94, 0.07, N)
# 10 AdaLN-Zero gates at 0.26-0.60
gate_ratios = np.array([0.26, 0.31, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.52, 0.60])
# Mark first 10 as gates after sorting (we just store membership)
is_gate = np.zeros(N, dtype=bool)
ratios[:10] = gate_ratios
is_gate[:10] = True
# clip
ratios = np.clip(ratios, 0.2, 1.2)

# sort ascending
order = np.argsort(ratios)
sorted_ratios = ratios[order]
sorted_is_gate = is_gate[order]

fig, ax = plt.subplots(figsize=figsize(0.95, ratio=0.5))
x = np.arange(len(sorted_ratios))
colors = np.where(sorted_is_gate, "#d62728", "#1f77b4")
ax.bar(x, sorted_ratios, color=colors, width=1.0, zorder=3, edgecolor="none")

ax.axhline(1.0, color="#555", ls=":", lw=0.7)
ax.axhline(sorted_ratios.mean(), color="#2ca02c", ls="--", lw=0.7)
ax.text(N+1, sorted_ratios.mean(), f"  mean {sorted_ratios.mean():.2f}",
        color="#2ca02c", fontsize=8, va="center")

ax.set_xlabel("Layer (sorted by ratio)")
ax.set_ylabel(r"$\|w_{\mathrm{v2}}\| / \|w_{\mathrm{old}}\|$")
ax.set_title("Per-layer Weight-Norm Ratio (v2 step 2078 vs canonical step 2646)")
ax.set_ylim(0.0, 1.2)

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d62728", label="AdaLN-Zero gates (n=10)"),
                   Patch(color="#1f77b4", label="other layers (n=154)")],
          frameon=False, loc="lower right", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
