# P07_v2: latent geometry as a one-table summary (rendered with matplotlib.table)
# Differs from v1: collapses 4 panels to a numeric readout
# Visualizes: F3 / E003
# DATA: inline from content_masterarbeit.md F3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=figsize(0.8, ratio=0.5))
ax.axis("off")

headers = ["Dim", "Per-dim std", "Kurtosis", "Multimodal?", "Length r"]
rows = [
    ["1",  "1.03", "0.18", "no",  "+0.02"],
    ["2",  "1.01", "0.21", "no",  "-0.01"],
    ["3",  "0.99", "-1.05", "yes (bi)", "-0.04"],
    ["4",  "0.96", "0.09", "no",  "+0.03"],
    ["5",  "0.94", "-0.42", "weak", "-0.02"],
    ["6",  "0.91", "0.31", "no",  "-0.01"],
    ["7",  "0.88", "-0.91", "yes (tri)", "+0.16"],
    ["8",  "0.85", "0.05", "no",  "+0.00"],
    ["all", "PR=7.69 / 8",   r"max $|\rho|{=}0.10$", "—", r"$|r|\leq 0.16$"],
]
tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.25)

# Style: bold header
for j in range(len(headers)):
    cell = tbl[0, j]
    cell.set_text_props(weight="bold")
    cell.set_facecolor("#e7e7e7")

# Highlight multimodal rows
for i in [3, 7]:  # 1-indexed dim 3 and 7
    for j in range(len(headers)):
        tbl[i, j].set_facecolor("#fde9e9")

# Summary row at bottom
for j in range(len(headers)):
    tbl[len(rows), j].set_facecolor("#e6f3ff")
    tbl[len(rows), j].set_text_props(weight="bold")

ax.set_title(r"Latent Geometry Summary (N=56K, $L\in[300, 800]$)", fontsize=10, pad=12)

fig.savefig(Path(__file__).with_suffix(".pdf"))
