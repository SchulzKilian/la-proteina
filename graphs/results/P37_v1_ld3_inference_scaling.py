# P37_v1: Official full-atom LD3 inference scaling on a single A100-80GB.
# Visualizes: E126 inference compute audit — how peak GPU memory and wall-clock
#   per protein grow with protein length L, and where the model runs out of memory.
# DATA: real CSV results/inference_compute_audit/scaling_a100_ld3.csv (parsed via
#   stdlib csv). Single arm (official_LD3, LD3_ucond_notri_800.ckpt), L runs 100->2200.
#   Model OOMs at the next step L=2400 (sidecar .oom.txt, ~80384 MB peak before failure).
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

CSV = Path(__file__).parent.parent.parent / "results" / "inference_compute_audit" / "scaling_a100_ld3.csv"

# Parse CSV (stdlib only — no pandas in this env).
L, mem_gb, wall_pp = [], [], []
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        L.append(int(row["L"]))
        mem_gb.append(float(row["peak_gpu_mb"]) / 1024.0)
        wall_pp.append(float(row["wall_s_per_protein"]))

fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize(1.0, 0.5))

# --- Panel (a): Peak GPU memory (GB) vs L ---
axa.plot(L, mem_gb, "-o", lw=1.4, markersize=4)
axa.set_xlabel(r"Protein length $L$ (residues)")
axa.set_ylabel("Peak GPU memory (GB)")
axa.set_title("(a) Peak GPU memory")
axa.set_ylim(bottom=0)
axa.spines[["top", "right"]].set_visible(False)
axa.grid(True, lw=0.4, alpha=0.5)

# --- Panel (b): Wall-clock per protein (s) vs L ---
axb.plot(L, wall_pp, "-o", lw=1.4, markersize=4)
axb.set_xlabel(r"Protein length $L$ (residues)")
axb.set_ylabel("Wall-clock (s / protein)")
axb.set_title("(b) Wall-clock")
axb.set_ylim(bottom=0)
axb.spines[["top", "right"]].set_visible(False)
axb.grid(True, lw=0.4, alpha=0.5)

fig.tight_layout()

fig.savefig(Path(__file__).with_suffix(".pdf"))
