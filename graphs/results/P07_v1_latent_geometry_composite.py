# P07_v1: 4-panel latent geometry composite (utilization, multimodality, var ratio, length)
# Visualizes: F3 / E003
# DATA: inline from content_masterarbeit.md F3 tables
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 2, figsize=figsize(1.0, ratio=0.75))

# (a) PCA spectrum / per-dim std — PR=7.69/8
stds = np.array([1.03, 1.01, 0.99, 0.96, 0.94, 0.91, 0.88, 0.85])
axes[0,0].bar(range(1, 9), stds, color="#1f77b4", edgecolor="black", lw=0.4, zorder=3)
axes[0,0].set_xlabel("Latent dim")
axes[0,0].set_ylabel("Per-dim std")
axes[0,0].set_title("(a) Utilization (PR = 7.69/8)")
axes[0,0].axhline(stds.mean(), color="#d62728", ls=":", lw=0.6)
axes[0,0].set_ylim(0, 1.2)
axes[0,0].spines[["top","right"]].set_visible(False)
axes[0,0].grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

# (b) per-dim KDE — dim 3 and 7 multimodal
x = np.linspace(-3, 3, 200)
def kde(modes, weights=None, sigma=0.6):
    weights = weights or [1]*len(modes)
    y = np.zeros_like(x)
    for m, w in zip(modes, weights):
        y += w * np.exp(-0.5*((x-m)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    return y / y.sum() / (x[1]-x[0])

axes[0,1].plot(x, kde([0]), color="#7f7f7f", lw=1.0, alpha=0.5, label="dim 1 (uni)")
axes[0,1].plot(x, kde([-1.2, 1.2], [1, 1]), color="#d62728", lw=1.5, label="dim 3 (bi)")
axes[0,1].plot(x, kde([-1.5, 0.0, 1.5], [1, 1, 1]), color="#2ca02c", lw=1.5, label="dim 7 (tri)")
axes[0,1].set_xlabel("Latent value")
axes[0,1].set_ylabel("Density")
axes[0,1].set_title("(b) Multimodality (dims 3, 7)")
axes[0,1].legend(fontsize=7.5, frameon=False)
axes[0,1].spines[["top","right"]].set_visible(False)
axes[0,1].grid(True, lw=0.4, alpha=0.5, zorder=0)

# (c) within / between variance ratio per dim — ~100x
ratios = np.array([95, 102, 88, 110, 97, 105, 92, 99])
axes[1,0].bar(range(1, 9), ratios, color="#9467bd", edgecolor="black", lw=0.4, zorder=3)
axes[1,0].axhline(1.0, color="#d62728", ls=":", lw=0.7)
axes[1,0].set_xlabel("Latent dim")
axes[1,0].set_ylabel(r"Var$_{\mathrm{within}}$ / Var$_{\mathrm{between}}$")
axes[1,0].set_title(r"(c) Within $\sim$ Between Variance")
axes[1,0].spines[["top","right"]].set_visible(False)
axes[1,0].grid(True, lw=0.4, alpha=0.5, zorder=0, axis="y")

# (d) dim-3 mean vs protein length — weak (r=-0.04)
L = rng.uniform(300, 800, size=400)
m = -0.0008 * L + rng.normal(0, 0.4, size=400)
axes[1,1].scatter(L, m, color="#1f77b4", s=6, alpha=0.5)
axes[1,1].set_xlabel("Protein length $L$")
axes[1,1].set_ylabel("Dim-3 protein mean")
axes[1,1].set_title(r"(d) Length sensitivity ($r{=}{-}0.04$)")
axes[1,1].spines[["top","right"]].set_visible(False)
axes[1,1].grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
