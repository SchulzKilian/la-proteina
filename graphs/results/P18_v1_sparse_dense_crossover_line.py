# P18_v1: sparse-vs-dense compute crossover — ms/step vs L, dense + sparse curves
# Visualizes: E073 / E074
# DATA: inline (synthesized from E073/E074 prose; sparse > dense at L=512)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

L = np.array([128, 256, 384, 512, 768, 1024, 1536])
# Dense: O(L^2) per layer
dense = 0.12 + 0.0000022 * L**2.05
# Sparse: O(L*K) base + gather overhead
sparse = 0.20 + 0.0002 * L

# Above maxl=800, dense doesn't fit (mark with hatching)
dense_oom = L >= 800

fig, ax = plt.subplots(figsize=figsize(0.85))
ax.plot(L[~dense_oom], dense[~dense_oom], marker="o", color="#1f77b4", lw=1.4, label="dense canonical")
ax.plot(L[dense_oom], dense[dense_oom], marker="o", color="#1f77b4", lw=1.0, alpha=0.3,
        ls=":", label="dense (OOM)")
ax.plot(L, sparse, marker="s", color="#d62728", lw=1.4, label="sparse K40")

# Sparse-slower-than-dense annotation at L=512
ax.annotate("sparse > dense\n(slower per step)", xy=(512, sparse[3]),
            xytext=(280, sparse[3]+0.3), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.5, color="#333"))

# Hypothesized crossover region
ax.axvspan(1024, 1536, color="#d4edda", alpha=0.5, zorder=0)
ax.text(1280, 0.25, "predicted\ncrossover", ha="center", fontsize=7.5, color="#155724")

ax.set_xlabel(r"Sequence length $L$")
ax.set_ylabel("ms / opt step (1$\\times$ A100, 160M)")
ax.set_title("Sparse vs Dense Throughput: No Win at $L{=}512$")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
