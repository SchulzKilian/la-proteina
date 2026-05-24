# P34_v2: Booktabs-styled version (top/mid/bottom rules only) of the variant ablation table.
# Visualizes: F5/F7/F11/F12
# DATA: inline from experiments.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

rows = [
    ("canonical (wd=0.05)",    "—",          "2646", "30", "26/25/17", "68/90", "alive"),
    ("v2 (wd=0.1+cos)",        "wd+sched",   "2078", "3",  "0/0/0",    "0/9",   "dead"),
    ("wd=0 ablation",          "wd $\\to$ 0","1638", "3",  "1/1/0",    "2/9",   "dead"),
    ("sparse K=40",            "sparse attn","1259", "6",  "5/3/1",    "9/18",  "alive"),
    ("sparse K40+PU",          "+pair-upd",  "1133", "6",  "2/1/0",    "3/18",  "dead"),
    ("scnbr+FixC2",            "nbr sched",  "1133", "6",  "3/2/0",    "5/18",  "alive"),
    ("downsampled",            "spat. pool", "2331", "6",  "1/0/0",    "1/18",  "dead"),
    ("K=64 curric",            "K up",       "944",  "6",  "2/1/0",    "3/18",  "marg."),
    ("+ BigBird",              "globals",    "819",  "6",  "0/0/0",    "0/18",  "dead"),
    ("+ BB+PU+lts (5-axis)",   "bundle",     "944",  "6",  "2/1/0",    "3/18",  "dead"),
    ("K40+curric+self",        "self nbr",   "1133", "6",  "3/2/1",    "6/18",  "marg."),
    ("attn-routing hybrid",    "inf. K-swap","—",    "9",  "—",        "1/9",   "dead"),
]
cols = ["Variant", "$\\Delta$", "Step", "$N$", "L=50/100/200", "Total", "Verdict"]
col_x = [0.02, 0.27, 0.41, 0.50, 0.58, 0.78, 0.88]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.62))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# Title
ax.text(0.5, 0.97, r"\textbf{Variant Ablation Summary}", ha="center", fontsize=11)

# Top rule
ax.plot([0.02, 0.98], [0.92, 0.92], "k-", lw=1.0)
# Header
for x, label in zip(col_x, cols):
    ax.text(x, 0.89, label, fontsize=8, weight="bold")
# Mid rule
ax.plot([0.02, 0.98], [0.86, 0.86], "k-", lw=0.5)
# Rows
y = 0.82
step = (0.82 - 0.06) / len(rows)
for row in rows:
    for x, val in zip(col_x, row):
        c = "black"
        if val in ("alive",): c = "#155724"
        elif val in ("dead",): c = "#721c24"
        elif val in ("marg.",): c = "#856404"
        ax.text(x, y, val, fontsize=7.5, color=c)
    y -= step
# Bottom rule
ax.plot([0.02, 0.98], [0.05, 0.05], "k-", lw=1.0)
ax.text(0.5, 0.02, r"Rates = $\#$ designable / total; verdict at probed step",
        ha="center", fontsize=7, style="italic")

fig.savefig(Path(__file__).with_suffix(".pdf"))
