# P34_v1: Variant ablation summary table (matplotlib.table rendering).
# Visualizes: F5/F7/F11/F12 + variant E-IDs (E008/E009/E014/E019/E021/E034/E039/E053/E055/E056/E077)
# DATA: inline from experiments.md entries
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt

# columns: Variant | Delta | Step | N | L50/100/200 | Total/(3*N) | Val | Verdict
rows = [
    ["canonical (wd=0.05)", "—",                              "2646", "30", "26/25/17", "68/90",  "4.77", "alive"],
    ["v2 (wd=0.1+cos)",     "wd, sched",                      "2078", "3",  "0/0/0",    "0/9",    "4.44", "dead"],
    ["wd=0 ablation",       "wd $\\to$ 0",                    "1638", "3",  "1/1/0",    "2/9",    "—",    "dead"],
    ["sparse K=40",         "sparse attn",                    "1259", "6",  "5/3/1",    "9/18",   "—",    "alive"],
    ["sparse K40+PU",       "+ pair-update",                  "1133", "6",  "2/1/0",    "3/18",   "—",    "dead"],
    ["scnbr$_{t{=}0.4}$+FixC2", "neighbor sched",             "1133", "6",  "3/2/0",    "5/18",   "—",    "alive"],
    ["downsampled",         "$2{\\times}$ pool",              "2331", "6",  "1/0/0",    "1/18",   "—",    "dead"],
    ["K=64 curric",         "K $50{\\to}64$",                 "944",  "6",  "2/1/0",    "3/18",   "—",    "marg."],
    ["+ BigBird only",      "+ globals",                      "819",  "6",  "0/0/0",    "0/18",   "—",    "dead"],
    ["+ BB+PU+lts",         "5-axis bundle",                  "944",  "6",  "2/1/0",    "3/18",   "—",    "dead"],
    ["K40+curric+self",     "self in nbrs",                   "1133", "6",  "3/2/1",    "6/18",   "—",    "marg."],
    ["attn-routing hybrid", "infer-time K swap",              "—",    "9",  "—",        "1/9",    "—",    "dead"],
]
cols = ["Variant", "$\\Delta$ from canon", "Step", "$N$", "Rate $L{=}50/100/200$",
        "Total", "Val", "Verdict"]

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.62))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
               cellLoc="left", colLoc="left",
               colWidths=[0.20, 0.16, 0.06, 0.05, 0.16, 0.07, 0.06, 0.10])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.0, 1.35)
for j in range(len(cols)):
    tbl[0, j].set_facecolor("#ececec"); tbl[0, j].set_text_props(weight="bold")
# Color verdict cells
for i, row in enumerate(rows):
    verdict = row[-1]
    bg = {"alive": "#d4edda", "dead": "#f8d7da", "marg.": "#fff3cd"}.get(verdict, "white")
    tbl[i + 1, len(cols) - 1].set_facecolor(bg)
for cell in tbl.get_celld().values():
    cell.set_linewidth(0.25)
ax.set_title(r"\textbf{Variant Ablation Summary: No Architectural Variant Beats Canonical}", pad=12)
fig.savefig(Path(__file__).with_suffix(".pdf"))
