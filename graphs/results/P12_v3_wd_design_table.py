# P12_v3: headline 4-arm comparison as a clean numerical table
# Third version for this headline figure (sometimes a table is the cleanest presentation)
# Visualizes: F5 / F7 / E014 / E019
# DATA: inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=figsize(0.85, ratio=0.45))
ax.axis("off")

headers = ["Arm", "Best step", "L=50", "L=100", "L=200", "Pooled /90"]
rows_data = [
    ("canonical (wd=0.05)", "2646", 26, 25, 17, 68),
    ("v2 (wd=0.1)",         "2078",  0,  0,  0,  0),
    ("wd=0",                "1638",  9,  4,  2, 15),
    ("sparse K40",          "1259", 12, 10,  5, 27),
]

cells = []
for arm, step, a, b, c, total in rows_data:
    cells.append([arm, step,
                  f"{a}/30 ({a/30*100:.0f}\\%)",
                  f"{b}/30 ({b/30*100:.0f}\\%)",
                  f"{c}/30 ({c/30*100:.0f}\\%)",
                  f"{total}/90"])

tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.45)
for j in range(len(headers)):
    tbl[0, j].set_facecolor("#e7e7e7")
    tbl[0, j].set_text_props(weight="bold")

# Highlight winner (canonical) and loser (v2)
for j in range(len(headers)):
    tbl[1, j].set_facecolor("#d4edda")
    tbl[2, j].set_facecolor("#fde9e9")

ax.set_title("Designability by Recipe / Architecture (N=30 per cell)",
             fontsize=10, pad=12)
fig.savefig(Path(__file__).with_suffix(".pdf"))
