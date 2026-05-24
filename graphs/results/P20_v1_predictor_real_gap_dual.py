# P20_v1: Predictor-vs-real TANGO gap dose-response — two-line per recipe (clean + NA-v1),
# predictor solid, real dashed, gap shaded.
# Visualizes: F10 / E028 / E032 / E050
# DATA: inline from F10 tables (content_masterarbeit.md lines 870-905)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

w = np.array([1, 2, 4, 8, 16])

# Clean ensemble (E028): predictor goes down hard, real moves opposite way (gradient hacking)
clean_pred = np.array([-10.5, -38.2, -74.6, -125.4, -195.6])
clean_real = np.array([1.2, 3.5, 4.8, 6.3, 7.4])

# NA-v1 ensemble (E032): predictor and real both go down, gap ~+3.8 at w=16
na_pred = np.array([-12.4, -28.6, -54.1, -88.5, -132.8])
na_real = np.array([-9.2, -22.3, -42.7, -75.4, -129.0])

fig, ax = plt.subplots(figsize=figsize(0.9))

# Clean ensemble: shaded gap shows the failure
ax.fill_between(w, clean_pred, clean_real, color="#d62728", alpha=0.15)
ax.plot(w, clean_pred, "-",  color="#d62728", lw=1.4, marker="o", label="clean: predictor")
ax.plot(w, clean_real, "--", color="#d62728", lw=1.0, marker="o", mfc="white", label="clean: real")

# NA-v1: tight band
ax.fill_between(w, na_pred, na_real, color="#1f77b4", alpha=0.15)
ax.plot(w, na_pred, "-",  color="#1f77b4", lw=1.4, marker="s", label="NA-v1: predictor")
ax.plot(w, na_real, "--", color="#1f77b4", lw=1.0, marker="s", mfc="white", label="NA-v1: real")

ax.axhline(0, color="#7f7f7f", lw=0.5, ls=":")

ax.annotate("gap = $-203$", xy=(16, (clean_pred[-1]+clean_real[-1])/2),
            xytext=(8.5, -80), fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", lw=0.4, color="#d62728"))
ax.annotate("gap = $+3.8$", xy=(16, (na_pred[-1]+na_real[-1])/2),
            xytext=(8, -180), fontsize=8, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", lw=0.4, color="#1f77b4"))

ax.set_xscale("log", base=2)
ax.set_xticks(w); ax.set_xticklabels([str(v) for v in w])
ax.set_xlabel("Steering Weight $w$")
ax.set_ylabel(r"TANGO ($\Delta$ vs unsteered)")
ax.set_title("Predictor:Real Gap Closes with Noise-Aware Ensemble")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

fig.savefig(Path(__file__).with_suffix(".pdf"))
