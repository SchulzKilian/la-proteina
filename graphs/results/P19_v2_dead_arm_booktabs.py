# P19_v2: booktabs-styled dead-arm table — using matplotlib but mimicking LaTeX booktabs
# (no vertical lines, top/mid/bottom rules only).
# Visualizes: E034 / E053 / E055 / E056 / E058 / E063 / E077
# DATA: inline from experiments.md entries
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

rows = [
    ("downsampled",               "2331", "6", "17/0/0",   "spatial pool drops local detail"),
    ("downsampled (late)",        "3716", "6", "17/0/0",   "no recovery at later ckpt"),
    ("BigBird only",              "819",  "6", "0/0/0",    "globals position-unaware"),
    ("K64 curric+BB+PU+lts",      "944",  "6", "33/17/0",  "5-axis bundle, no gain"),
    ("K40+curric+self",           "1133", "6", "50/33/17", "below canon at $L{\\geq}100$"),
    ("attn-routing hybrid",       "—",    "9", "—",        "1/9 < canon 2/9"),
    ("sparse K40+PU",             "1133", "6", "33/17/0",  "pair-update needs dense grid"),
]

col_labels = ["Variant", "Step", "$N$", "Rate 50/100/200 (\\%)", "Kill Reason"]
col_widths = [0.20, 0.08, 0.06, 0.20, 0.42]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.42))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# Top rule
ax.axhline(0.92, xmin=0.02, xmax=0.98, color="black", lw=1.0)
# Mid rule
ax.axhline(0.84, xmin=0.02, xmax=0.98, color="black", lw=0.6)
# Bottom rule
ax.axhline(0.08, xmin=0.02, xmax=0.98, color="black", lw=1.0)

# Header text
x = 0.03
for label, w in zip(col_labels, col_widths):
    ax.text(x, 0.87, label, fontsize=9, weight="bold", va="center")
    x += w

# Row text
y = 0.78
row_step = 0.10
for row in rows:
    x = 0.03
    for cell, w in zip(row, col_widths):
        ax.text(x, y, cell, fontsize=8.5, va="center")
        x += w
    y -= row_step

ax.text(0.5, 0.98, r"\textbf{Dead-Arm Gallery: Booktabs Layout}",
        ha="center", va="center", fontsize=11)

fig.savefig(Path(__file__).with_suffix(".pdf"))
