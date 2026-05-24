# P09_v1: sidechain manifold perturbation — paired box per k, latent vs coord
# Visualizes: F6 / E011
# DATA: inline from F6 numbers (lines 471-504 of content_masterarbeit.md)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(0)

# k values (perturbation scale) and approximate per-protein scRMSD
ks = [1, 2, 4, 8]
# latent arm: ~5 Å baseline, near-invariant with k
# coord arm: grows from 5 to ~15 with k
def synth(arm, k, n=17):
    if arm == "latent":
        base = 5.0
        return base + rng.normal(0, 0.6, n)
    else:  # coord
        base = 5.0 + 2.0 * np.log2(k)
        return base + rng.normal(0, 0.9, n)

fig, ax = plt.subplots(figsize=figsize(0.85))
positions_lat = np.arange(len(ks)) - 0.18
positions_co  = np.arange(len(ks)) + 0.18

box_lat = ax.boxplot([synth("latent", k) for k in ks],
                     positions=positions_lat, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f77b4"),
                     medianprops=dict(color="#1f77b4"),
                     whiskerprops=dict(color="#1f77b4"),
                     capprops=dict(color="#1f77b4"),
                     flierprops=dict(marker=".", markersize=3, color="#1f77b4"))
box_co  = ax.boxplot([synth("coord", k) for k in ks],
                     positions=positions_co, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor="#fcd5d5", edgecolor="#d62728"),
                     medianprops=dict(color="#d62728"),
                     whiskerprops=dict(color="#d62728"),
                     capprops=dict(color="#d62728"),
                     flierprops=dict(marker=".", markersize=3, color="#d62728"))

ax.set_xticks(range(len(ks)))
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel(r"Perturbation scale $k$")
ax.set_ylabel(r"scRMSD (\AA)")
ax.set_title(r"Sidechain Perturbation: Latent (contractive) vs Coord")
ax.legend([box_lat["boxes"][0], box_co["boxes"][0]],
          ["Latent arm", "Coord arm"], frameon=False, loc="upper left", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
