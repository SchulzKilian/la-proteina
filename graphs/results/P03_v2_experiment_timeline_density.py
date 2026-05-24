# P03_v2: timeline as stacked daily-count bar chart by route — alternative to dotplot
# Aggregates the same E001-E079 chronology to show experiment intensity over time
# DATA: inline from experiments.md index
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
from datetime import date
import collections

# Same entries as v1 (copied compactly)
entries = [
    ("E001","2026-04-21","scaffold"),("E002","2026-04-21","scaffold"),("E003","2026-04-21","scaffold"),
    ("E004","2026-04-22","scaffold"),("E005","2026-04-22","diagnostic"),("E006","2026-04-22","scaffold"),
    ("E007","2026-04-23","diagnostic"),("E008","2026-04-23","baseline"),
    ("E009","2026-04-24","baseline"),
    ("E010","2026-04-25","variant"),("E011","2026-04-25","scaffold"),
    ("E012","2026-04-26","diagnostic"),("E013","2026-04-26","variant"),
    ("E014","2026-04-27","baseline"),("E015","2026-04-27","diagnostic"),
    ("E016","2026-04-28","scaffold"),("E017","2026-04-28","scaffold"),
    ("E018","2026-04-29","diagnostic"),("E019","2026-04-29","baseline"),
    ("E020","2026-04-30","diagnostic"),("E021","2026-04-30","variant"),
    ("E022","2026-05-01","diagnostic"),("E023","2026-05-01","diagnostic"),
    ("E024","2026-05-02","scaffold"),("E025","2026-05-02","diagnostic"),
    ("E026","2026-05-03","diagnostic"),("E027","2026-05-03","steering"),
    ("E028","2026-05-04","steering"),("E029","2026-05-04","steering"),
    ("E030","2026-05-05","steering"),("E031","2026-05-05","steering"),
    ("E032","2026-05-06","steering"),("E033","2026-05-06","steering"),
    ("E034","2026-05-07","variant"),("E035","2026-05-07","variant"),
    ("E036","2026-05-08","steering"),("E037","2026-05-08","diagnostic"),
    ("E038","2026-05-09","variant"),("E039","2026-05-09","variant"),
    ("E040","2026-05-10","variant"),("E041","2026-05-10","variant"),
    ("E042","2026-05-11","diagnostic"),("E043","2026-05-11","diagnostic"),
    ("E044","2026-05-12","steering"),("E045","2026-05-12","steering"),
    ("E046","2026-05-13","variant"),("E047","2026-05-13","variant"),
    ("E048","2026-05-14","steering"),("E049","2026-05-14","steering"),
    ("E050","2026-05-15","steering"),("E051","2026-05-15","steering"),
    ("E052","2026-05-16","steering"),("E053","2026-05-16","variant"),
    ("E054","2026-05-17","diagnostic"),("E055","2026-05-17","variant"),
    ("E056","2026-05-18","variant"),("E057","2026-05-18","steering"),
    ("E058","2026-05-19","variant"),("E059","2026-05-19","diagnostic"),
    ("E060","2026-05-20","diagnostic"),("E061","2026-05-20","diagnostic"),
    ("E062","2026-05-21","variant"),("E063","2026-05-21","variant"),
    ("E064","2026-05-22","steering"),("E065","2026-05-22","steering"),("E066","2026-05-22","steering"),
    ("E067","2026-05-23","steering"),("E068","2026-05-23","steering"),("E069","2026-05-23","steering"),
    ("E070","2026-05-23","steering"),("E071","2026-05-23","steering"),("E072","2026-05-23","steering"),
    ("E073","2026-05-23","diagnostic"),("E074","2026-05-23","diagnostic"),
    ("E075","2026-05-23","steering"),("E076","2026-05-23","steering"),
    ("E077","2026-05-23","variant"),("E078","2026-05-23","variant"),("E079","2026-05-23","variant"),
]

counts = collections.defaultdict(lambda: collections.Counter())
for _, d, r in entries:
    yr, mo, da = map(int, d.split("-"))
    counts[date(yr, mo, da)][r] += 1

dates = sorted(counts.keys())
routes = ["scaffold", "baseline", "diagnostic", "variant", "steering"]
COLORS = {"scaffold":"#9467bd","baseline":"#7f7f7f","diagnostic":"#2ca02c",
          "variant":"#1f77b4","steering":"#d62728"}

import numpy as np
fig, ax = plt.subplots(figsize=figsize(1.0, ratio=0.45))
bot = np.zeros(len(dates))
for r in routes:
    h = np.array([counts[d][r] for d in dates])
    ax.bar(dates, h, bottom=bot, color=COLORS[r], label=r.title(), width=0.85,
           edgecolor="white", linewidth=0.3, zorder=3)
    bot += h

import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_ylabel("Experiments / Day")
ax.set_xlabel("Date (2026)")
ax.set_title(r"Experiment Intensity by Route")
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0, axis="y")

fig.savefig(Path(__file__).with_suffix(".pdf"))
