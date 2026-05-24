# P19_v1: Dead-arm gallery table — matplotlib.table rendering, clean borders.
# Lists called-dead architectural arms with kill reason.
# Visualizes: E034 / E053 / E055 / E056 / E058 / E063 / E077
# DATA: inline from experiments.md entries
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

# Columns: Variant | Step | N | L=50 | L=100 | L=200 | Reason
rows = [
    ["downsampled",                "2331",  "6", "17\\%", "0\\%",  "0\\%",  "spatial pooling drops local detail"],
    ["downsampled (late)",         "3716",  "6", "17\\%", "0\\%",  "0\\%",  "no recovery at later ckpt"],
    ["BigBird only",               "819",   "6", "0\\%",  "0\\%",  "0\\%",  "position-unaware globals"],
    ["K64 curric+BB+PU+lowtsoft",  "944",   "6", "33\\%", "17\\%", "0\\%",  "5-axis bundle, no gain"],
    ["K40+curric+self",            "1133",  "6", "50\\%", "33\\%", "17\\%", "still below canon at $L{\\geq}100$"],
    ["attn-routing hybrid",        "—",     "9", "—",     "—",     "—",     "1/9 vs canon 2/9 at $L{\\geq}300$"],
    ["sparse K40+PU",              "1133",  "6", "33\\%", "17\\%", "0\\%",  "pair-update needs dense grid"],
]

col_labels = ["Variant", "Step", "$N$", "$L{=}50$", "$L{=}100$", "$L{=}200$", "Kill Reason"]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.42))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
               cellLoc="left", colLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.4)

# Make header bold-ish (matplotlib has no native bold; use color)
for i in range(len(col_labels)):
    cell = tbl[0, i]
    cell.set_facecolor("#ececec")
    cell.set_text_props(weight="bold")
    cell.set_edgecolor("black")

# Subtle borders
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.3)

ax.set_title(r"\textbf{Dead-Arm Gallery: Architectural Variants Killed}", pad=14)

fig.savefig(Path(__file__).with_suffix(".pdf"))
