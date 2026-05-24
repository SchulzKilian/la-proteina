# P06_v2: same data as heatmap, but as 2 line panels (A vs B) showing R^2 vs probe capacity
# Differs from v1: line/scaling view rather than matrix view; emphasizes the curve shape
# Visualizes: F4 / E002
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

probes = ["Lin", "MLP-1", "MLP-2", "MLP-3", "BiLSTM", "Tx-S", "Tx-L"]
x = np.arange(len(probes))

class_A = [
    ("Aromatic frac.", [0.93, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]),
    ("Net charge",     [0.94, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]),
    ("Hydrophobic patch", [0.88, 0.93, 0.95, 0.96, 0.97, 0.98, 0.98]),
    ("Shannon entropy", [0.78, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95]),
    ("CamSol",         [0.75, 0.83, 0.86, 0.88, 0.90, 0.92, 0.92]),
    ("pI",             [0.84, 0.91, 0.94, 0.95, 0.96, 0.97, 0.98]),
]
class_B = [
    ("TANGO",   [0.41, 0.55, 0.62, 0.66, 0.78, 0.91, 0.95]),
    ("SAP",     [0.45, 0.58, 0.65, 0.70, 0.81, 0.92, 0.96]),
    ("IUPred3", [0.50, 0.62, 0.68, 0.72, 0.80, 0.90, 0.93]),
    ("SCM",     [0.39, 0.51, 0.58, 0.63, 0.74, 0.86, 0.91]),
    ("Rg",      [0.30, 0.42, 0.48, 0.55, 0.69, 0.83, 0.89]),
    ("Hydropathy", [0.34, 0.45, 0.51, 0.56, 0.71, 0.82, 0.86]),
    ("SWI",     [0.10, 0.18, 0.22, 0.26, 0.32, 0.36, 0.38]),
]

fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize(1.0, ratio=0.45), sharey=True)
for name, vals in class_A:
    axA.plot(x, vals, marker="o", markersize=3, lw=1.0, label=name)
for name, vals in class_B:
    axB.plot(x, vals, marker="o", markersize=3, lw=1.0, label=name)

for ax, title, color in [(axA, "Class A (per-residue MLP suffices)", "#2ca02c"),
                          (axB, "Class B (needs attention)", "#d62728")]:
    ax.set_xticks(x); ax.set_xticklabels(probes, fontsize=8, rotation=30)
    ax.axhline(0.8, color="#555", ls=":", lw=0.6)
    ax.set_xlabel("Probe Capacity")
    ax.set_title(title, color=color, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_ylim(0, 1.05)
axA.set_ylabel(r"$R^2$")

fig.savefig(Path(__file__).with_suffix(".pdf"))
