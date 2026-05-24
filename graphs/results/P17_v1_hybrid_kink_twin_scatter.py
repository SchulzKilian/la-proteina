# P17_v1: hybrid handover — twin scatter of magnitude disagreement + cos similarity vs t_switch
# Visualizes: E040 / E041
# DATA: inline from E040 / E041 prose
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

# E040 (conv -> scnbr): kink at 0.79-0.86, cos 0.52-0.61, 1/9 designable
# E041 (conv -> canonical): kink at 0.76-0.81, cos 0.59-0.66, 5/9 designable
t_switch = np.array([0.60, 0.70, 0.75, 0.80, 0.85])
E040_mag = np.array([0.45, 0.62, 0.78, 0.86, 0.83])
E040_cos = np.array([0.78, 0.66, 0.58, 0.52, 0.55])
E041_mag = np.array([0.31, 0.45, 0.66, 0.81, 0.78])
E041_cos = np.array([0.80, 0.74, 0.69, 0.66, 0.59])

fig, ax = plt.subplots(figsize=figsize(0.95))
l1, = ax.plot(t_switch, E040_mag, marker="o", color="#d62728",
              label=r"E040 conv$\to$scnbr: $\|v_A{-}v_B\|/\|v_A\|$")
l2, = ax.plot(t_switch, E041_mag, marker="o", color="#1f77b4",
              label=r"E041 conv$\to$canonical: $\|v_A{-}v_B\|/\|v_A\|$")
ax.set_xlabel(r"Handover time $t_{\mathrm{switch}}$")
ax.set_ylabel(r"Relative magnitude disagreement")
ax.set_ylim(0, 1.05)
ax.spines[["top"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

ax2 = ax.twinx()
l3, = ax2.plot(t_switch, E040_cos, marker="s", color="#d62728", ls="--",
               label=r"E040 cos($v_A, v_B$)")
l4, = ax2.plot(t_switch, E041_cos, marker="s", color="#1f77b4", ls="--",
               label=r"E041 cos($v_A, v_B$)")
ax2.set_ylabel(r"Cosine similarity")
ax2.set_ylim(0, 1.05)
ax2.spines[["top"]].set_visible(False)

ax.legend(handles=[l1, l2, l3, l4], frameon=False, fontsize=7.5, loc="lower left")
ax.set_title(r"Hybrid Handover Kink: Magnitude $\uparrow$ + Cosine $\downarrow$ near $t=0.8$")

# annotate designability outcome
ax.text(0.85, 0.05, "design: E040 1/9 ; E041 5/9", fontsize=8, color="#333",
        transform=ax.transAxes, ha="right")

fig.savefig(Path(__file__).with_suffix(".pdf"))
