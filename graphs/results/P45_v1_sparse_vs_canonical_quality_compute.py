# P45_v1: the honest sparse-vs-canonical story in three panels:
#   (A) quality  — canonical beats sparse on designability (N=30 matched-seed, E014/E019)
#   (B) wall     — sparse turns dense's super-linear inference cost into linear (E124, A100)
#   (C) memory   — dense OOMs at the 80 GB wall (L~2256); sparse runs where dense cannot (E124)
# Narrative: baseline is strictly better at the lengths it can do; sparse is the only thing that
# runs at the lengths it can't, at linear instead of quadratic cost.
# DATA: panel A from experiments.md E014 N=30 table (nsteps=200, matched seed=100 — relative only);
#       panels B/C from results/inference_compute_audit/scaling_a100.csv (E124, on-disk, real).
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

CANON = "#1f77b4"   # canonical dense
SPARSE = "#d62728"  # sparse K40
REPO = Path(__file__).parent.parent.parent

# ---- Panel A data: N=30 matched-seed scRMSD (Å), mean bars + min "ceiling" marker (E014/E019) ----
# Framing: mean compresses the gap to ~2-3x (vs 20x for the hard-threshold rate); the min marker
# shows sparse's *ceiling* sits within ~0.35 A of canonical at L=50/100 — near-canonical at lower yield.
Ls_q = ["50", "100", "200"]
canon_mean = [2.89, 2.21, 5.87]
sparse_mean = [5.67, 6.04, 11.08]
canon_min = [0.76, 0.86, 1.50]
sparse_min = [1.05, 1.21, 3.34]

# ---- Panels B/C data: on-disk A100 scaling ladder (E124) ----
csv_path = REPO / "results/inference_compute_audit/scaling_a100.csv"
dense_L, dense_wall, dense_mem = [], [], []
sparse_L, sparse_wall, sparse_mem = [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        L = float(row["L"])
        wall = float(row["wall_s_per_protein"])
        mem = float(row["peak_gpu_mb"]) / 1024.0  # -> GB
        if row["arm"] == "canonical_dense":
            dense_L.append(L); dense_wall.append(wall); dense_mem.append(mem)
        elif row["arm"] == "sparse_K40":
            sparse_L.append(L); sparse_wall.append(wall); sparse_mem.append(mem)

# dense OOM point (from scaling_a100.csv.oom.txt): L=2400 OOMs, peak ~79406 MB before death
OOM_L = 2400
GPU_CAP = 80.0  # GB wall

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(TEXTWIDTH := 6.30, 6.30 * 0.30))

# ============================ Panel A — quality (gentle: mean scRMSD + min ceiling) ============================
x = np.arange(len(Ls_q)); w = 0.38
axA.bar(x - w / 2, canon_mean, width=w, color=CANON, edgecolor="black",
        lw=0.4, label="canonical (mean)", zorder=3)
axA.bar(x + w / 2, sparse_mean, width=w, color=SPARSE, edgecolor="black",
        lw=0.4, label="sparse K=40 (mean)", zorder=3)
# min ("best-of-30") ceiling markers — sparse's best sits within ~0.35 A of canonical at L=50/100
axA.scatter(x - w / 2, canon_min, marker="D", s=14, color="white",
            edgecolor=CANON, lw=0.9, zorder=5, label="best-of-30")
axA.scatter(x + w / 2, sparse_min, marker="D", s=14, color="white",
            edgecolor=SPARSE, lw=0.9, zorder=5)
axA.axhline(2.0, color="black", lw=0.7, ls=":", zorder=2)
axA.text(2.42, 2.0, "2 \\AA", fontsize=6, va="bottom", ha="right")
axA.set_xticks(x); axA.set_xticklabels(Ls_q)
axA.set_xlabel("Length $L$")
axA.set_ylabel("scRMSD$_{ca}$ (\\AA)")
axA.set_ylim(0, 12.5)
axA.set_title("(A) Quality: baseline ahead, ceiling close", fontsize=8)
axA.legend(frameon=False, loc="upper left", fontsize=5.8)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(True, lw=0.4, alpha=0.5, axis="y", zorder=0)

# ============================ Panel B — wall-clock ============================
axB.plot(dense_L, dense_wall, "o-", color=CANON, ms=2.5, lw=1.0, label="dense $\\sim L^{1.72}$")
axB.plot(sparse_L, sparse_wall, "s-", color=SPARSE, ms=2.5, lw=1.0, label="sparse $\\sim L^{0.91}$")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("Length $L$")
axB.set_ylabel("Wall (s / protein)")
axB.set_title("(B) Compute: sparse scales linearly", fontsize=8)
axB.legend(frameon=False, loc="upper left", fontsize=6.5)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(True, lw=0.4, alpha=0.5, which="both", zorder=0)
# crossover band + 12.6x annotation at L=2200
axB.axvspan(200, 300, color="gray", alpha=0.12, zorder=0)
axB.annotate("12.6$\\times$", xy=(2200, 220.6), xytext=(900, 230),
             fontsize=6.5, color=CANON,
             arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))

# ============================ Panel C — memory / OOM wall ============================
axC.plot(dense_L, dense_mem, "o-", color=CANON, ms=2.5, lw=1.0, label="dense $\\propto L^2$")
axC.plot(sparse_L, sparse_mem, "s-", color=SPARSE, ms=2.5, lw=1.0, label="sparse $\\propto L$")
axC.axhline(GPU_CAP, color="black", lw=0.8, ls="--")
axC.text(120, GPU_CAP - 4.5, "80 GB wall", fontsize=6, va="top")
# dense OOM marker
axC.plot([OOM_L], [GPU_CAP], marker="x", color=CANON, ms=6, mew=1.4, zorder=5)
axC.annotate("dense OOM\n$L\\approx2256$", xy=(OOM_L, GPU_CAP), xytext=(1250, 60),
             fontsize=6, color=CANON, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.5, color=CANON))
axC.annotate("34$\\times$ less\nat $L{=}2200$", xy=(2200, sparse_mem[-2]), xytext=(1500, 14),
             fontsize=6, color=SPARSE, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.5, color=SPARSE))
axC.set_xlabel("Length $L$")
axC.set_ylabel("Peak GPU (GB)")
axC.set_ylim(0, 86)
axC.set_title("(C) Memory: sparse runs where dense can't", fontsize=8)
axC.legend(frameon=False, loc="center left", fontsize=6.5)
axC.spines[["top", "right"]].set_visible(False)
axC.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.tight_layout(pad=0.4)
fig.savefig(Path(__file__).with_suffix(".pdf"))
print("wrote", Path(__file__).with_suffix(".pdf").name)
