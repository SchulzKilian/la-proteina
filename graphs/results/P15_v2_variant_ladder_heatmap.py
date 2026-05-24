# P15_v2: same data, variant x L heatmap (diverging colormap centered at canonical at L)
# Differs from v1: shows relative deficit vs canonical per cell
# Visualizes: variants vs canonical
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

variants = [
    ("canonical",              [0.87, 0.83, 0.57]),
    ("sparse K40",             [0.83, 0.50, 0.17]),
    ("sparse K40+PU",          [0.33, 0.17, 0.00]),
    ("scnbr\\_t04+FixC2",      [0.50, 0.33, 0.00]),
    ("downsampled",            [0.17, 0.00, 0.00]),
    ("K64-curric",             [0.33, 0.17, 0.00]),
    ("K64-curric+BigBird",     [0.17, 0.00, 0.00]),
    ("K40+curric+self",        [0.50, 0.33, 0.17]),
]
labels = [v[0] for v in variants]
rates = np.array([v[1] for v in variants])
canon = rates[0]
delta = rates - canon  # negative => worse than canon

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.7))
im = ax.imshow(delta, aspect="auto", cmap="RdBu", vmin=-0.9, vmax=0.9)
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        ax.text(j, i, f"{rates[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(delta[i,j]) > 0.4 else "black")

ax.set_xticks(range(3)); ax.set_xticklabels([r"$L{=}50$", r"$L{=}100$", r"$L{=}200$"])
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
cbar.set_label(r"$\Delta$ rate vs canonical")

ax.set_title("Variant Designability vs Canonical (cell value = rate)")
fig.savefig(Path(__file__).with_suffix(".pdf"))
