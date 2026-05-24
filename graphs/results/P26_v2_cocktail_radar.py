# P26_v2: Radar chart — per-property delivery for each of the three recipes (camsol, tango, sap, scm, codesign).
# (Alternative framing: see all objectives at once rather than only the SWI summary.)
# Visualizes: E068 / E072
# DATA: inline (approximated from E072 prose)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

axes_labels = ["CamSol", "Tango", "SAP", "SCM$^+$", "Codesign"]
N = len(axes_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Normalized magnitude (0 to 1) along each axis. Codesign axis: 1 = highest retention.
recipes = {
    "camsol max (1-obj)":             [0.80, 0.10, 0.05, 0.05, 0.85],
    "camsol + tango (2-obj)":         [0.75, 0.65, 0.10, 0.05, 0.78],
    "camsol+tango+sap+scm$^+$ (4)":   [0.70, 0.60, 0.55, 0.60, 0.70],
}
colors = ["#1f77b4", "#d62728", "#2ca02c"]

fig, ax = plt.subplots(figsize=figsize(0.85), subplot_kw=dict(polar=True))
for (label, vals), c in zip(recipes.items(), colors):
    vals = vals + vals[:1]
    ax.plot(angles, vals, "-", lw=1.4, color=c, label=label)
    ax.fill(angles, vals, alpha=0.10, color=c)

ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=8.5)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
ax.set_ylim(0, 1.0)
ax.set_title("Cocktail Delivery (Normalized, $w{=}32$)", pad=20)
ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.35, 1.1))

fig.savefig(Path(__file__).with_suffix(".pdf"))
