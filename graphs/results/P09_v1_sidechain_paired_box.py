# P09_v1: sidechain manifold perturbation — paired box per k, latent vs coord
# Visualizes: F6 / E011
# DATA: REAL per-protein values from the manifold perturbation experiment
#   (E011, AE1_ucond_512, 17 proteins L=65-217). Loaded from
#   inference/manifold_tidy_eval_manifold_perturbation.csv.
#   Metric = all-atom RMSD of the perturbed structure to its ESMFold
#   re-prediction (lower = more on-manifold). k = sidechain perturbation scale.
#   Latent arm: perturb in AE1 latent space then decode; near-invariant to k
#   (decoder is contractive). Coord arm: perturb sidechain coords directly;
#   RMSD grows with k.
import sys
import csv
import collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "inference" / "manifold_tidy_eval_manifold_perturbation.csv"

# group all-atom RMSD by (space, k)
vals = collections.defaultdict(list)
with open(DATA, newline="") as f:
    for r in csv.DictReader(f):
        vals[(r["space"], float(r["k"]))].append(float(r["all_atom_rmsd"]))

ks = sorted({k for (_, k) in vals})            # [0.1, 0.3, 0.5, 1.0, 2.0]
lat = [vals[("latent", k)] for k in ks]
co  = [vals[("coord",  k)] for k in ks]

fig, ax = plt.subplots(figsize=figsize(0.85))
positions_lat = np.arange(len(ks)) - 0.18
positions_co  = np.arange(len(ks)) + 0.18

box_lat = ax.boxplot(lat,
                     positions=positions_lat, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f77b4"),
                     medianprops=dict(color="#1f77b4"),
                     whiskerprops=dict(color="#1f77b4"),
                     capprops=dict(color="#1f77b4"),
                     flierprops=dict(marker=".", markersize=3, markeredgecolor="#1f77b4"))
box_co  = ax.boxplot(co,
                     positions=positions_co, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor="#fcd5d5", edgecolor="#d62728"),
                     medianprops=dict(color="#d62728"),
                     whiskerprops=dict(color="#d62728"),
                     capprops=dict(color="#d62728"),
                     flierprops=dict(marker=".", markersize=3, markeredgecolor="#d62728"))

YCAP = 25.0
n_off = sum(1 for v in list(np.concatenate(lat)) + list(np.concatenate(co)) if v > YCAP)
ax.set_ylim(0, YCAP)
if n_off:
    ax.text(0.985, 0.97, rf"{n_off} pt off-scale ($>${YCAP:g}\,\AA)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.4")

ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"{k:g}" for k in ks])
ax.set_xlabel(r"Sidechain perturbation scale $k$")
ax.set_ylabel(r"All-atom RMSD to ESMFold (\AA)")
ax.set_title(r"Sidechain Perturbation: Latent (contractive) vs Coord")
ax.legend([box_lat["boxes"][0], box_co["boxes"][0]],
          ["Latent arm", "Coord arm"], frameon=False, loc="upper left", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
