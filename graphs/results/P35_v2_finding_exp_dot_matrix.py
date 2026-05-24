# P35_v2: Dot-matrix view of Finding -> E-ID mapping (alternative framing).
# Y axis: Findings F1-F13. X axis: E-IDs (selected). Dot if cited; dot color = primary vs supporting.
# Visualizes: All Findings + E-IDs
# DATA: inline from content_masterarbeit.md
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np

mapping = {
    "F1":  {"E001": "p"},
    "F2":  {"E004": "p"},
    "F3":  {"E003": "p", "E005": "s", "E007": "s"},
    "F4":  {"E002": "p"},
    "F5":  {"E008": "s", "E009": "p", "E014": "s", "E015": "s"},
    "F6":  {"E011": "p"},
    "F7":  {"E014": "s", "E019": "p"},
    "F8":  {"E020": "p", "E023": "s", "E026": "p"},
    "F9":  {"E026": "p"},
    "F10": {"E028": "p", "E029": "s", "E030": "s", "E031": "s", "E032": "p", "E050": "s"},
    "F11": {"E043": "p", "E054": "s"},
    "F12": {"E059": "s", "E060": "s", "E061": "p"},
    "F13": {"E066": "s", "E067": "s", "E072": "s", "E076": "p"},
}

findings = list(mapping.keys())
all_es = sorted({e for d in mapping.values() for e in d.keys()})

fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.45))
for fi, f in enumerate(findings):
    for ei, e in enumerate(all_es):
        kind = mapping[f].get(e)
        if kind is None: continue
        c = "#d62728" if kind == "p" else "#1f77b4"
        ax.scatter(ei, fi, color=c, s=70 if kind == "p" else 32,
                   edgecolor="black", lw=0.3, zorder=3)
ax.set_xticks(range(len(all_es))); ax.set_xticklabels(all_es, rotation=70, fontsize=7.5)
ax.set_yticks(range(len(findings))); ax.set_yticklabels(findings, fontsize=9)
ax.set_title("Finding $\\to$ Experiment Mapping")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.4, alpha=0.4, zorder=0)

# Legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", color="#d62728", lw=0, ms=8, label="Primary"),
           Line2D([], [], marker="o", color="#1f77b4", lw=0, ms=5, label="Supporting")]
ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right",
          bbox_to_anchor=(1.18, 1.0))

fig.savefig(Path(__file__).with_suffix(".pdf"))
