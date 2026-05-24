# P16_v1: per-t val loss across CA-only variants — multi-line, real JSON data
# Visualizes: F11 / E043
# Source data: results/per_t_val/*.json
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ks2218/la-proteina/results/per_t_val")
files = {
    "canonical_2646": "canonical_2646.json",
    "conv_2331":      "conv_2331.json",
    "scnbr_t04_1133": "scnbr_t04_1133.json",
    "sparse_vanilla_1259": "sparse_vanilla_1259.json",
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"]

fig, ax = plt.subplots(figsize=figsize(0.85))
for (label, fname), color in zip(files.items(), COLORS):
    with open(ROOT / fname) as f:
        d = json.load(f)
    buckets = d["buckets"]
    t_mid = [0.5*(b["t_lo"] + b["t_hi"]) for b in buckets.values()]
    mean = [b["mean"] for b in buckets.values()]
    sem  = [b["sem"] for b in buckets.values()]
    ax.errorbar(t_mid, mean, yerr=sem, marker="o", lw=1.2, ms=4,
                color=color, label=label.replace("_", "\\_"), capsize=2)

ax.set_xlabel(r"Diffusion-time bucket midpoint $t$")
ax.set_ylabel(r"Paired val loss (seed=42, 600-subset)")
ax.set_title("Per-$t$ Validation Loss: Curves are Parallel, Not Crossing")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
