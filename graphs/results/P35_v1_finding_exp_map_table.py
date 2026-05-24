# P35_v1: Finding-to-experiment cross-reference table (matplotlib.table).
# Visualizes: All Findings + E-IDs
# DATA: inline from content_masterarbeit.md cross-reference sections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

rows = [
    ["F1",  "Predictor 13-prop $R^2$",      "E001",            "—",                       "scaffold"],
    ["F2",  "Flow curvature $R{=}0.51/0.94$","E004",           "—",                       "scaffold"],
    ["F3",  "Latent geometry",              "E003",            "E005, E007",              "scaffold"],
    ["F4",  "Probe capacity (A/B)",         "E002",            "—",                       "scaffold"],
    ["F5",  "AdaLN-Zero gate collapse",     "E009",            "E008, E014, E015",        "diagnostic"],
    ["F6",  "Sidechain perturbation",       "E011",            "—",                       "scaffold"],
    ["F7",  "wd $\\times$ design (N=30)",   "E019",            "E014",                    "diagnostic"],
    ["F8",  "Joint-head AA collapse",       "E020, E026",      "E023",                    "diagnostic"],
    ["F9",  "Collapse scales with $L$",     "E026 follow-up",  "—",                       "diagnostic"],
    ["F10", "Predictor:real gap closure",   "E028, E032",      "E029, E030, E031, E050", "deliverable"],
    ["F11", "Per-$t$ val parallelism",      "E043",            "E054",                    "diagnostic"],
    ["F12", "Dense routing audit",          "E061",            "E059, E060",              "diagnostic"],
    ["F13", "Real CamSol per-length",       "E076",            "E066, E067, E072",        "deliverable"],
]
cols = ["F\\#", "Finding", "Primary E", "Supporting E", "Role"]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.62))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
               cellLoc="left", colLoc="left",
               colWidths=[0.07, 0.32, 0.16, 0.28, 0.15])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.35)
for j in range(len(cols)):
    tbl[0, j].set_facecolor("#ececec"); tbl[0, j].set_text_props(weight="bold")
# Color role
role_colors = {"scaffold": "#cfe2f3", "diagnostic": "#fce5cd", "deliverable": "#d4edda"}
for i, row in enumerate(rows):
    tbl[i + 1, 4].set_facecolor(role_colors.get(row[4], "white"))
for cell in tbl.get_celld().values():
    cell.set_linewidth(0.25)
ax.set_title(r"\textbf{Finding $\to$ Experiment Cross-Reference}", pad=12)
fig.savefig(Path(__file__).with_suffix(".pdf"))
