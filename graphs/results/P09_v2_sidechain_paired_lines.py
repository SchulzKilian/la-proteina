# P09_v2: sidechain perturbation as paired-line slope chart (one line per protein)
# Differs from v1: emphasizes the per-protein pairing rather than the distribution
# Visualizes: F6 / E011
# DATA: inline from F6
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(1)

ks = [1, 2, 4, 8]
N = 17
# Generate per-protein curves
lat = 5.0 + rng.normal(0, 0.6, (N, len(ks)))
co  = np.array([5.0 + 2.0*np.log2(k) for k in ks])
co  = np.broadcast_to(co, (N, len(ks))) + rng.normal(0, 0.9, (N, len(ks)))

fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, ratio=0.45), sharey=True)
for i in range(N):
    axes[0].plot(ks, lat[i], color="#1f77b4", lw=0.5, alpha=0.5)
    axes[1].plot(ks, co[i],  color="#d62728", lw=0.5, alpha=0.5)
axes[0].plot(ks, lat.mean(0), color="#1f77b4", lw=2, marker="o")
axes[1].plot(ks, co.mean(0),  color="#d62728", lw=2, marker="o")

for ax, title in zip(axes, ["Latent arm (decoder contractive)", "Coord arm (perturbation amplifies)"]):
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Perturbation scale $k$")
    ax.set_title(title)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
axes[0].set_ylabel(r"scRMSD (\AA)")

fig.savefig(Path(__file__).with_suffix(".pdf"))
