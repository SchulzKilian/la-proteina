# P10_v2: per-t paired-protein val loss across canonical vs variants — uses real JSON
# Differs from v1: replaces unreliable wandb cross-run curves with paired re-eval (E054)
# Visualizes: F5 / F11 / E054
# Source data: results/per_t_val/*.json
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ks2218/la-proteina/results/per_t_val")
files = {
    "canonical (step 2646)": "canonical_2646.json",
    "v2 last (step 1952)":   "canonical_lastv2.json",
    "conv (step 2331)":      "conv_2331.json",
    "scnbr t0.4 (1133)":     "scnbr_t04_1133.json",
    "sparse vanilla (1259)": "sparse_vanilla_1259.json",
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

fig, ax = plt.subplots(figsize=figsize(0.9))
for (label, fname), color in zip(files.items(), COLORS):
    with open(ROOT / fname) as f:
        d = json.load(f)
    buckets = d["buckets"]
    t_mid = [0.5*(b["t_lo"] + b["t_hi"]) for b in buckets.values()]
    mean = [b["mean"] for b in buckets.values()]
    sem  = [b["sem"] for b in buckets.values()]
    ax.errorbar(t_mid, mean, yerr=sem, marker="o", lw=1.0, ms=4,
                color=color, label=label, capsize=2)

ax.set_xlabel(r"Diffusion time bucket midpoint $t$")
ax.set_ylabel("Paired val loss")
ax.set_title("Per-$t$ Validation Loss (paired, seed=42, 600-protein subset)")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
