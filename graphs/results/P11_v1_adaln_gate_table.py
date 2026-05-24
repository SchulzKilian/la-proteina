# P11_v1: AdaLN-Zero gate-norm collapse — matplotlib.table rendering
# Visualizes: F5 / E009 / E015 (Table format)
# DATA: inline from content_masterarbeit.md F5 table (lines 415-426)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=figsize(0.9, ratio=0.55))
ax.axis("off")

# 10 worst-affected layers
rows = [
    ["transformer\\_layers.7.transition.scale\\_output", "0.26"],
    ["transformer\\_layers.7.mhba.scale\\_output",        "0.31"],
    ["transformer\\_layers.8.transition.scale\\_output", "0.33"],
    ["transformer\\_layers.8.mhba.scale\\_output",        "0.36"],
    ["transformer\\_layers.9.transition.scale\\_output", "0.39"],
    ["transformer\\_layers.9.mhba.scale\\_output",        "0.42"],
    ["transformer\\_layers.10.transition.scale\\_output","0.45"],
    ["transformer\\_layers.11.transition.scale\\_output","0.48"],
    ["transformer\\_layers.12.transition.scale\\_output","0.52"],
    ["transformer\\_layers.13.transition.scale\\_output","0.60"],
]
headers = ["AdaLN-Zero gate parameter", "$\\|w_{\\mathrm{v2}}\\| / \\|w_{\\mathrm{old}}\\|$"]
tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left",
               colWidths=[0.72, 0.22])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.4)

for j in range(2):
    tbl[0, j].set_facecolor("#e7e7e7")
    tbl[0, j].set_text_props(weight="bold")

# Color-code ratio cells by severity
for i, row in enumerate(rows, start=1):
    r = float(row[1])
    color = "#fde9e9" if r < 0.40 else ("#fff4d6" if r < 0.55 else "#e6f3ff")
    tbl[i, 1].set_facecolor(color)
    tbl[i, 1].set_text_props(ha="center")

ax.set_title("AdaLN-Zero Gate-Norm Collapse: 10 Worst Layers (v2 / canonical)",
             fontsize=10, pad=12)

# Footer
ax.text(0.5, -0.05, "Layers 7-13: gates collapse to 26-60\\% of canonical magnitude.",
        ha="center", fontsize=8, color="#555", transform=ax.transAxes)

fig.savefig(Path(__file__).with_suffix(".pdf"))
