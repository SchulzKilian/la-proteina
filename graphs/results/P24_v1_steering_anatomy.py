# P24_v1: 3-panel anatomy — (a) w(t) schedule, (b) data-flow schematic, (c) example grad_norm trace.
# Visualizes: F10 / steering hook design
# DATA: inline (schematic + synthesized grad_norm)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, ratio=0.32))

# Panel (a): w(t) schedule
t = np.linspace(0, 1, 200)
w = np.where((t >= 0.3) & (t <= 0.8), (t - 0.3) / 0.5, 0.0)
w = np.where(t > 0.9, 0.0, w)
axes[0].plot(t, w, color="#1f77b4", lw=1.5)
axes[0].axvspan(0.3, 0.8, color="#1f77b4", alpha=0.10)
axes[0].axvspan(0.8, 0.9, color="#7f7f7f", alpha=0.15)
axes[0].axvline(0.9, color="#d62728", lw=0.6, ls=":")
axes[0].text(0.92, 0.05, "hard stop", fontsize=7, color="#d62728", rotation=90, va="bottom")
axes[0].set_xlabel("$t$ (flow time)")
axes[0].set_ylabel("$w(t)$")
axes[0].set_title("(a) Schedule")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(True, lw=0.4, alpha=0.5)

# Panel (b): data-flow schematic
axes[1].axis("off")
axes[1].set_xlim(0, 10); axes[1].set_ylim(0, 6)
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec="black", lw=0.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8)
def arr(ax, x0, y0, x1, y1):
    ax.add_patch(mpatches.FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=8, lw=0.7, color="black"))

box(axes[1], 0.2, 4.5, 2.0, 0.8, r"$z_t$", "#e7e7e7")
box(axes[1], 3.5, 4.5, 2.5, 0.8, r"$\hat z_1 {=} z_t {+} (1{-}t)v$", "#e7e7e7")
box(axes[1], 7.5, 4.5, 2.0, 0.8, r"predictor", "#cfe2f3")
box(axes[1], 7.5, 2.5, 2.0, 0.8, r"$L(z_t)$", "#fce5cd")
box(axes[1], 3.5, 2.5, 2.5, 0.8, r"$\nabla_{z_t}L$", "#fce5cd")
box(axes[1], 0.2, 2.5, 2.0, 0.8, r"unit-norm $\cdot w(t)$", "#d9ead3")
box(axes[1], 0.2, 0.5, 2.0, 0.8, r"$v {+} g$", "#d4edda")

arr(axes[1], 2.2, 4.9, 3.5, 4.9)
arr(axes[1], 6.0, 4.9, 7.5, 4.9)
arr(axes[1], 8.5, 4.5, 8.5, 3.3)
arr(axes[1], 7.5, 2.9, 6.0, 2.9)
arr(axes[1], 3.5, 2.9, 2.2, 2.9)
arr(axes[1], 1.2, 2.5, 1.2, 1.3)

axes[1].text(5, 5.7, "(b) Gradient flow (predictor only)", ha="center", fontsize=8)

# Panel (c): example grad_norm trace
rng = np.random.default_rng(7)
t_steps = np.linspace(0.0, 1.0, 200)
grad_norm = np.exp(-((t_steps - 0.6) ** 2) / 0.05) * (1 + 0.05 * rng.standard_normal(200))
grad_norm[t_steps < 0.3] = 0
grad_norm[t_steps > 0.9] = 0
w_sched = np.where((t_steps >= 0.3) & (t_steps <= 0.8), (t_steps - 0.3) / 0.5, 0.0)
w_sched = np.where(t_steps > 0.9, 0.0, w_sched)
axes[2].plot(t_steps, grad_norm, color="#1f77b4", lw=1.2, label="raw $\\|\\nabla L\\|$")
axes[2].plot(t_steps, w_sched * grad_norm / max(grad_norm.max(), 1e-6),
             color="#d62728", lw=1.2, ls="--", label="$w(t) \\cdot$ unit grad")
axes[2].set_xlabel("$t$")
axes[2].set_ylabel("Gradient magnitude")
axes[2].set_title("(c) Diagnostic trace")
axes[2].legend(frameon=False, fontsize=7, loc="upper right")
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].grid(True, lw=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
