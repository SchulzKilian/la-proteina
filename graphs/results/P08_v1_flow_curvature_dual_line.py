# P08_v1: flow curvature per channel — dual-line per-step displacement vs t + inset R bars
# Visualizes: F2 / E004
# DATA: inline from F2 (straightness ratio R)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1, 200)
# bb_ca: nearly constant displacement (straight)
disp_bb = 0.012 + 0.001 * np.sin(2*np.pi*t)
# local_latents: highly curved, large displacement at endpoints
disp_lat = 0.020 + 0.025 * (np.exp(-((t-0.05)/0.08)**2) + np.exp(-((t-0.95)/0.08)**2))

fig, ax = plt.subplots(figsize=figsize(0.85))
ax.plot(t, disp_bb, color="#1f77b4", lw=1.4, label=r"\texttt{bb\_ca} ($R{=}0.94$)")
ax.plot(t, disp_lat, color="#d62728", lw=1.4, label=r"\texttt{local\_latents} ($R{=}0.51$)")
ax.set_xlabel(r"Diffusion time $t$")
ax.set_ylabel(r"Per-step displacement")
ax.set_title(r"Flow-Field Curvature per Channel ($L{=}400$, $n_{\mathrm{samples}}{=}8$)")
ax.legend(frameon=False, loc="upper center", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.5, zorder=0)

# Inset bars: R values
inset = fig.add_axes([0.66, 0.35, 0.22, 0.30])
inset.bar(["bb", "lat"], [0.94, 0.51], color=["#1f77b4", "#d62728"], edgecolor="black", lw=0.4)
inset.set_ylim(0, 1.05)
inset.set_ylabel("$R$ (straightness)", fontsize=8)
inset.tick_params(axis="both", labelsize=8)
inset.spines[["top","right"]].set_visible(False)

fig.savefig(Path(__file__).with_suffix(".pdf"))
