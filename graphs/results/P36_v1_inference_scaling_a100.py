# P36_v1: Dense vs sparse-K40 inference scaling on A100-80GB.
# Visualizes: E120 inference compute audit (peak GPU memory + wall-clock per protein vs L).
# DATA: real CSV results/inference_compute_audit/scaling_a100.csv (parsed via stdlib csv).
#       Dense OOMs at L=2400 (sidecar .oom.txt, ~79.4 GB peak before failure);
#       extrapolated dense 80 GB ceiling at L~2256 from fit mem ~ 614 MB + 0.0156*L^2.
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

CSV = Path(__file__).parent.parent.parent / "results" / "inference_compute_audit" / "scaling_a100.csv"

# Parse CSV (stdlib only — no pandas in this env).
data = {"canonical_dense": {"L": [], "mem_gb": [], "wall_pp": []},
        "sparse_K40":      {"L": [], "mem_gb": [], "wall_pp": []}}
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        arm = row["arm"]
        data[arm]["L"].append(int(row["L"]))
        data[arm]["mem_gb"].append(float(row["peak_gpu_mb"]) / 1024.0)
        data[arm]["wall_pp"].append(float(row["wall_s_per_protein"]))

series = [
    ("canonical_dense", "Dense", "#d62728", "o"),
    ("sparse_K40",      r"Sparse-$K40$", "#1f77b4", "s"),
]

fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize(1.0, 0.5))

# --- Panel (a): Peak GPU memory (GB) vs L, linear y ---
for arm, label, c, m in series:
    axa.plot(data[arm]["L"], data[arm]["mem_gb"], "-", color=c, marker=m,
             lw=1.4, markersize=4.5, label=label)
axa.axhline(80, color="#7f7f7f", lw=0.8, ls="--")
axa.text(700, 80, "A100 80 GB", fontsize=8, color="#7f7f7f", va="bottom", ha="left")

# Dense OOM at L=2400 (off the top of the panel near the wall).
axa.scatter([2400], [80], marker="x", color="#d62728", s=55, lw=1.8, zorder=5)
axa.annotate("OOM\n($L{=}2400$)", xy=(2400, 80), xytext=(2400, 64),
             fontsize=7.5, color="#d62728", ha="center", va="top")
# Extrapolated dense 80 GB ceiling.
axa.axvline(2256, color="#d62728", lw=0.6, ls=":", alpha=0.6)
axa.annotate(r"$L\approx2256$", xy=(2256, 40), fontsize=7.5, color="#d62728",
             ha="right", va="center", rotation=90)

axa.set_xlabel(r"Protein length $L$ (residues)")
axa.set_ylabel("Peak GPU memory (GB)")
axa.set_title("(a) Peak GPU memory")
axa.set_ylim(0, 88)
axa.spines[["top", "right"]].set_visible(False)
axa.grid(True, lw=0.4, alpha=0.5, zorder=0)

# --- Panel (b): Wall-clock per protein (s) vs L, linear y ---
for arm, label, c, m in series:
    axb.plot(data[arm]["L"], data[arm]["wall_pp"], "-", color=c, marker=m,
             lw=1.4, markersize=4.5, label=label)
axb.annotate(r"$12.6\times$ faster" "\n" r"at $L{=}2200$", xy=(2200, 120),
             fontsize=7.5, color="black", ha="right", va="center")

axb.set_xlabel(r"Protein length $L$ (residues)")
axb.set_ylabel("Wall-clock (s / protein)")
axb.set_title("(b) Wall-clock")
axb.spines[["top", "right"]].set_visible(False)
axb.grid(True, lw=0.4, alpha=0.5, zorder=0)

axa.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.05, 0.65))
fig.suptitle(r"Dense vs sparse-$K40$ inference scaling (CA-only, A100-80GB, nsteps$=400$)",
             fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))

fig.savefig(Path(__file__).with_suffix(".pdf"))
