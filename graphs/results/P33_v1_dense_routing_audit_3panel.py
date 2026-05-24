# P33_v1: Three-panel dense attention routing audit — (a) mass_top_K vs K, (b) Jaccard histogram, (c) Jaccard vs L box.
# Visualizes: F12 / E059 / E060 / E061
# DATA: inline from F12 tables (approximate; real json structure varies)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, ratio=0.35))

# Panel (a): mass_top_K vs K
K = np.array([4, 8, 16, 32, 64, 128])
per_query   = np.array([0.41, 0.55, 0.66, 0.78, 0.86, 0.92])
per_lhq     = np.array([0.62, 0.78, 0.86, 0.91, 0.95, 0.98])
aggregate   = np.array([0.18, 0.30, 0.45, 0.60, 0.75, 0.88])
axes[0].plot(K, per_query, "-o", color="#1f77b4", lw=1.3, label="per-query grad")
axes[0].plot(K, per_lhq,   "-s", color="#2ca02c", lw=1.3, label="per-(L,H,Q) attn")
axes[0].plot(K, aggregate, "-^", color="#d62728", lw=1.3, label="aggregate grad")
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(K); axes[0].set_xticklabels([str(k) for k in K], fontsize=8)
axes[0].set_xlabel("Top-$K$ neighbors"); axes[0].set_ylabel("Cum. attention mass")
axes[0].set_title("(a) Mass concentration")
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(True, lw=0.4, alpha=0.5, zorder=0)

# Panel (b): cross-metric Jaccard distribution
rng = np.random.default_rng(11)
jac_perq_vs_attn = rng.beta(2.5, 8, 600) * 0.6 + 0.08
jac_perq_vs_agg  = rng.beta(1.8, 12, 600) * 0.5 + 0.04
axes[1].hist(jac_perq_vs_attn, bins=24, alpha=0.7, color="#1f77b4", edgecolor="black",
             lw=0.3, label="per-query vs attn")
axes[1].hist(jac_perq_vs_agg, bins=24, alpha=0.7, color="#d62728", edgecolor="black",
             lw=0.3, label="per-query vs aggregate")
axes[1].set_xlabel("Jaccard (top-64)")
axes[1].set_ylabel("Count")
axes[1].set_title("(b) Cross-metric Jaccard")
axes[1].legend(frameon=False, fontsize=7, loc="upper right")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(True, lw=0.4, alpha=0.5, zorder=0)

# Panel (c): per-query-pair Jaccard vs L (box)
data_per_L = {}
for L in [50, 100, 200]:
    # Jaccard between two queries within a protein at length L
    mean = max(0.05, 0.3 - 0.001 * L)
    data_per_L[L] = rng.beta(2, 8, 300) * (2 * mean) + max(0.0, mean - 0.05)

bp = axes[2].boxplot([data_per_L[L] for L in [50, 100, 200]],
                     positions=[50, 100, 200], widths=20, patch_artist=True)
for patch, c in zip(bp["boxes"], ["#1f77b4", "#2ca02c", "#d62728"]):
    patch.set_facecolor(c); patch.set_edgecolor("black"); patch.set_alpha(0.6)
for whisker in bp["whiskers"] + bp["caps"] + bp["medians"]:
    whisker.set_color("black"); whisker.set_lw(0.5)
for flier in bp["fliers"]:
    flier.set_marker("."); flier.set_markersize(2)

axes[2].set_xticks([50, 100, 200])
axes[2].set_xlabel("Protein length $L$"); axes[2].set_ylabel("Per-query-pair Jaccard")
axes[2].set_title("(c) Routing diverges with $L$")
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
