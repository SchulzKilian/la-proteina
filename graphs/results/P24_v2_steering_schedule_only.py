# P24_v2: Just the w(t) schedule, large and clean — paper-figure version emphasizing the schedule alone.
# Visualizes: F10 / steering hook design
# DATA: inline schedule definition
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1, 400)
# Variants of the schedule for illustration
def sched(t, w_max=1.0, t_start=0.3, t_end=0.8, t_stop=0.9):
    w = np.where((t >= t_start) & (t <= t_end), (t - t_start) / (t_end - t_start), 0.0)
    w = np.where(t > t_stop, 0.0, w)
    return w_max * w

fig, ax = plt.subplots(figsize=figsize(0.9))

ax.plot(t, sched(t, 1.0), color="#1f77b4", lw=1.8, label="$w_{\\max}{=}1.0$")
ax.plot(t, sched(t, 0.5), color="#7f7f7f", lw=1.2, ls="--", label="$w_{\\max}{=}0.5$")

# Annotate regions
ax.axvspan(0.0, 0.3, color="#fce5cd", alpha=0.45, label="warmup off")
ax.axvspan(0.3, 0.8, color="#d9ead3", alpha=0.45, label="linear ramp")
ax.axvspan(0.8, 0.9, color="#cfe2f3", alpha=0.45, label="cooldown")
ax.axvline(0.9, color="#d62728", lw=0.8, ls=":")
ax.text(0.91, 0.5, "hard stop $t{=}0.9$", color="#d62728", fontsize=8, rotation=90, va="center")

ax.set_xlabel("Flow Time $t$ (0 = noise, 1 = data)")
ax.set_ylabel("Steering Weight $w(t)$")
ax.set_title("Steering Schedule: Ramp $\\to$ Hard Stop Before $t{=}1$")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.15)

fig.savefig(Path(__file__).with_suffix(".pdf"))
