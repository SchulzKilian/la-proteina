# P09_v2: sidechain perturbation as paired-line slope chart (one line per protein)
# Differs from v1: emphasizes the per-protein trajectory rather than the distribution
# Visualizes: F6 / E011
# DATA: REAL per-protein values from the manifold perturbation experiment
#   (E011, AE1_ucond_512, 17 proteins L=65-217). Loaded from
#   inference/manifold_tidy_eval_manifold_perturbation.csv.
#   Metric = all-atom RMSD of the perturbed structure to its ESMFold
#   re-prediction. One faint line per protein; bold line = mean across proteins.
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

# per-protein RMSD trajectory across k, for each space
traj = {"latent": collections.defaultdict(dict), "coord": collections.defaultdict(dict)}
kset = set()
with open(DATA, newline="") as f:
    for r in csv.DictReader(f):
        k = float(r["k"]); kset.add(k)
        traj[r["space"]][r["protein_id"]][k] = float(r["all_atom_rmsd"])

ks = sorted(kset)                              # [0.1, 0.3, 0.5, 1.0, 2.0]

def matrix(space):
    pids = sorted(traj[space])
    return np.array([[traj[space][p][k] for k in ks] for p in pids])

lat = matrix("latent")
co  = matrix("coord")

YCAP = 25.0
fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, ratio=0.45), sharey=True)
for i in range(lat.shape[0]):
    axes[0].plot(ks, lat[i], color="#1f77b4", lw=0.5, alpha=0.5)
for i in range(co.shape[0]):
    axes[1].plot(ks, co[i],  color="#d62728", lw=0.5, alpha=0.5)
# bold summary line = MEDIAN (robust to the off-scale outliers)
axes[0].plot(ks, np.median(lat, 0), color="#1f77b4", lw=2, marker="o", label="median")
axes[1].plot(ks, np.median(co, 0),  color="#d62728", lw=2, marker="o", label="median")

specs = [(axes[0], "Latent arm (decoder contractive)", lat),
         (axes[1], "Coord arm (perturbation amplifies)", co)]
for ax, title, mat in specs:
    ax.set_xscale("log")
    ax.set_ylim(0, YCAP)
    ax.set_xlabel(r"Sidechain perturbation scale $k$")
    ax.set_title(title)
    ax.set_xticks(ks); ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.minorticks_off()
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    n_off = int((mat > YCAP).any(1).sum())
    if n_off:
        ax.text(0.985, 0.97, rf"{n_off} protein off-scale",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.4")
axes[0].set_ylabel(r"All-atom RMSD to ESMFold (\AA)")

fig.savefig(Path(__file__).with_suffix(".pdf"))
