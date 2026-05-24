# P16_v2: per-t val loss as variant-minus-canonical delta per t-bucket
# Differs from v1: emphasizes gap (no crossover) vs absolute curves
# Visualizes: F11 / E043
# Source data: results/per_t_val/*.json
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ks2218/la-proteina/results/per_t_val")
with open(ROOT / "canonical_2646.json") as f:
    canon = json.load(f)
canon_means = np.array([b["mean"] for b in canon["buckets"].values()])
t_mid = np.array([0.5*(b["t_lo"] + b["t_hi"]) for b in canon["buckets"].values()])

variants = {
    "conv_2331":      "conv_2331.json",
    "scnbr_t04_1133": "scnbr_t04_1133.json",
    "sparse_vanilla_1259": "sparse_vanilla_1259.json",
}
COLORS = ["#d62728", "#2ca02c", "#7f7f7f"]

fig, ax = plt.subplots(figsize=figsize(0.85))
ax.axhline(0, color="#1f77b4", lw=1.2, label="canonical = baseline")
for (label, fname), color in zip(variants.items(), COLORS):
    with open(ROOT / fname) as f:
        d = json.load(f)
    m = np.array([b["mean"] for b in d["buckets"].values()])
    ax.plot(t_mid, m - canon_means, marker="o", lw=1.2, ms=4, color=color,
            label=label.replace("_", "\\_"))

ax.set_xlabel(r"Diffusion-time bucket midpoint $t$")
ax.set_ylabel("Variant minus canonical val loss")
ax.set_title("No Variant Wins at Any $t$ (curves don't cross)")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
