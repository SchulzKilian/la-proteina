# P08_v2: flow curvature as a single bar comparison of R values, with annotations
# Differs from v1: a pure summary bar — no time curve
# Visualizes: F2 / E004
# DATA: inline from F2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=figsize(0.55))
chans = [r"$\mathtt{bb\_ca}$", r"$\mathtt{local\_latents}$"]
R = [0.94, 0.51]
bars = ax.bar(chans, R, color=["#1f77b4", "#d62728"], edgecolor="black", lw=0.5,
              width=0.5, zorder=3)
for b, v in zip(bars, R):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=10)

ax.axhline(1.0, color="#555", ls=":", lw=0.7)
ax.text(1.45, 1.0, " straight-line\n flow", color="#555", fontsize=8, va="center")

ax.set_ylim(0, 1.1)
ax.set_ylabel(r"Straightness Ratio $R$")
ax.set_title(r"Flow Curvature (lower $R$ = more curved)")
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
