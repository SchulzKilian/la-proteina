# P47_v1: the "sparse is only marginally behind" story at the velocity-field level (E043 / F11).
#   (A) per-t FM validation loss, canonical vs sparse — curves are PARALLEL, not crossing;
#       sparse sits a small constant offset above canonical (no regime where it diverges).
#   (B) the offset itself (sparse - canonical) per t-bucket — bounded ~+0.13..+0.21 nat, +12% at
#       the loss-minimum bucket. This small offset is what the hard 2 A designability threshold
#       AMPLIFIES into the scary 67%->3.3% rate gap (P45 panel A).
# Same 600-protein paired subset, seed=42, identical per-protein per-t draws across both ckpts,
# so bucket-mean differences are PURE model differences. Real JSON, results/per_t_val/*.json.
# Caveat: sparse is step 1259 (under-trained) vs canonical 2646 — the offset conflates
# architecture with ~1400 fewer training steps; it is an UPPER bound on the architectural gap.
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

CANON = "#1f77b4"
SPARSE = "#d62728"
ROOT = Path(__file__).parent.parent.parent / "results/per_t_val"


def load(fname):
    d = json.load(open(ROOT / fname))
    b = d["buckets"]
    t_mid = np.array([0.5 * (v["t_lo"] + v["t_hi"]) for v in b.values()])
    mean = np.array([v["mean"] for v in b.values()])
    sem = np.array([v["sem"] for v in b.values()])
    return t_mid, mean, sem


t, c_mean, c_sem = load("canonical_2646.json")
_, s_mean, s_sem = load("sparse_vanilla_1259.json")
delta = s_mean - c_mean
# offset at the loss-minimum bucket (t in [0.6,0.8)) — the regime sampling spends most steps in
imin = int(np.argmin(c_mean))
pct = 100.0 * delta[imin] / c_mean[imin]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.30, 6.30 * 0.34))

# ---------------- Panel A: the two curves ----------------
axA.errorbar(t, c_mean, yerr=c_sem, marker="o", lw=1.2, ms=3.5, color=CANON,
             capsize=2, label="canonical (step 2646)")
axA.errorbar(t, s_mean, yerr=s_sem, marker="s", lw=1.2, ms=3.5, color=SPARSE,
             capsize=2, label="sparse K=40 (step 1259)")
# shade the loss-minimum bucket where sampling spends most of its budget
axA.axvspan(0.6, 0.8, color="gray", alpha=0.12, zorder=0)
axA.annotate(f"+{delta[imin]:.3f} nat\n(+{pct:.0f}\\%)", xy=(t[imin], s_mean[imin]),
             xytext=(0.45, 1.9), fontsize=6.5, color=SPARSE,
             arrowprops=dict(arrowstyle="->", lw=0.5, color=SPARSE))
axA.set_xlabel(r"Diffusion time $t$ (bucket midpoint)")
axA.set_ylabel("FM val loss (nat / protein)")
axA.set_title("(A) Parallel, not crossing", fontsize=8)
axA.legend(frameon=False, loc="upper right", fontsize=6.2)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(True, lw=0.4, alpha=0.5, zorder=0)

# ---------------- Panel B: the bounded offset ----------------
axB.bar(t, delta, width=0.12, color=SPARSE, edgecolor="black", lw=0.4, zorder=3)
for ti, dv in zip(t, delta):
    axB.text(ti, dv + 0.006, f"{dv:.2f}", ha="center", va="bottom", fontsize=6)
axB.axvspan(0.6, 0.8, color="gray", alpha=0.12, zorder=0)
axB.axhline(0, color="black", lw=0.6)
axB.set_xlabel(r"Diffusion time $t$ (bucket midpoint)")
axB.set_ylabel(r"sparse $-$ canonical (nat)")
axB.set_ylim(0, max(delta) * 1.35)
axB.set_title("(B) Offset is small + bounded", fontsize=8)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(True, lw=0.4, alpha=0.5, axis="y", zorder=0)

fig.tight_layout(pad=0.5)
fig.savefig(Path(__file__).with_suffix(".pdf"))
print("wrote", Path(__file__).with_suffix(".pdf").name, "| min-bucket offset",
      f"+{delta[imin]:.3f} nat (+{pct:.1f}%)")
