# P32_v1: Thermal-stability proxy contradiction — diverging bar of Cohen's d, AFDB and PDB series.
# Visualizes: F8 (b.i) / E020 / E026
# DATA: inline from F8 (b.i) prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "aliphatic index",
    "IVYWREL",
    "GRAVY",
    "charged fraction",
    "log(acidic/basic)",
    "aromatic fraction",
]
# Cohen's d gen vs AFDB and gen vs PDB (approx from F8(b.i))
d_afdb = [+0.55, +0.45, +0.30, +0.40, +1.10, -0.65]
d_pdb  = [+1.10, +0.95, +0.65, +0.85, +2.20, -1.30]

y = np.arange(len(metrics))
w = 0.4
fig, ax = plt.subplots(figsize=figsize(0.9, ratio=0.75))
ax.barh(y - w/2, d_afdb, height=w, color="#1f77b4", edgecolor="black", lw=0.4, label="vs AFDB", zorder=3)
ax.barh(y + w/2, d_pdb,  height=w, color="#ff7f0e", edgecolor="black", lw=0.4, label="vs PDB", zorder=3)
ax.axvline(0, color="black", lw=0.6)
ax.axvline(0.8, color="#7f7f7f", lw=0.4, ls=":"); ax.axvline(-0.8, color="#7f7f7f", lw=0.4, ls=":")

# Highlight aromatic-fraction
ax.annotate("only metric pointing\nthe other way",
            xy=(d_pdb[-1], len(metrics) - 1 + w/2),
            xytext=(-1.4, len(metrics) - 0.5), fontsize=7.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", lw=0.4, color="#d62728"))

ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=9)
ax.set_xlabel(r"Cohen's $d$ (gen vs reference)")
ax.set_title(r"Thermal-Stability Proxies: Gen Looks Stable Except in Aromatics")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="x")

fig.savefig(Path(__file__).with_suffix(".pdf"))
