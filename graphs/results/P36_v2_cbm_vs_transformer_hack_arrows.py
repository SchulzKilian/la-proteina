# P36_v2: same F17 w=32 head-to-head as v1, but framed as "which way did REALITY move?"
# Both methods steer DOWN from the same unsteered prior; arrow = real (IUPred3) displacement.
# Transformer arrow points UP (away from target = hack); CBM arrow points DOWN (toward target).
# Hollow tick = predicted final (both ~0.11 — both predictors "claim" success).
# Visualizes: F17 / E109 (transformer NA-v1) vs E110 (CBM), iupred3 down-reg, n=48 each.
# DATA: inline from content_masterarbeit.md F17.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

# (label, predicted final, real mean, colour)
methods = [
    ("Transformer\n(NA-v1)", 0.110, 0.180, "#d62728"),
    ("CBM\n(bottleneck)",    0.114, 0.128, "#1f77b4"),
]
PRIOR  = 0.158
TARGET = 0.123

fig, ax = plt.subplots(figsize=figsize(0.55, ratio=1.0))
xs = [0, 1]

ax.axhline(PRIOR,  color="#888", ls=(0, (4, 3)), lw=0.8, zorder=1)
ax.axhline(TARGET, color="#2ca02c", ls=(0, (4, 3)), lw=0.9, zorder=1)
ax.text(1.62, PRIOR,  r"unsteered" + "\n" + r"prior", color="#888", fontsize=8,
        va="center", ha="left")
ax.text(1.62, TARGET, r"target", color="#2ca02c", fontsize=8, va="center", ha="left")

for x, (name, pred, real, c) in zip(xs, methods):
    # real displacement from the prior — the arrow tells the whole story
    ax.annotate("", xy=(x, real), xytext=(x, PRIOR),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=c, mutation_scale=18),
                zorder=3)
    ax.scatter([x], [real], color=c, s=36, zorder=4)
    ax.text(x + 0.10, real, f"{real:.3f}", color=c, fontsize=8, va="center", ha="left")
    # predicted final (what the steering optimised) — both land at ~0.11
    ax.scatter([x], [pred], facecolors="white", edgecolors=c, s=30, lw=1.1, zorder=4)
    ax.text(x + 0.10, pred, f"pred {pred:.3f}", color=c, fontsize=7, va="center",
            ha="left", alpha=0.85)

ax.text(0, 0.205, r"\textbf{moved away}" + "\n(hack)", color="#d62728", fontsize=8,
        ha="center", va="bottom")
ax.text(1, 0.090, r"\textbf{toward target}" + "\n(honest)", color="#1f77b4", fontsize=8,
        ha="center", va="top")

ax.set_xticks(xs)
ax.set_xticklabels([m[0] for m in methods])
ax.set_xlim(-0.5, 2.4)
ax.set_ylim(0.0, 0.23)
ax.set_ylabel(r"IUPred3 fraction disordered (real)")
ax.set_title(r"Steering down ($w{=}32$): real displacement from prior")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="y")
fig.savefig(Path(__file__).with_suffix(".pdf"))
