"""Aggregate scRMSD across the noise-aware-ensemble sweep.

Reads results/noise_aware_ensemble_sweep/<cell>/scRMSD_guided.csv.
Reports per-cell designability rate (<2 Å), mean scRMSD, and per-length
breakdown so we can answer "what w can I steer at without destroying
the protein?"
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/noise_aware_ensemble_sweep"
CELLS = []
for d in ["camsol_max", "tango_min"]:
    for w in [1, 2, 4, 8, 16]:
        CELLS.append(f"{d}_w{w}")

def parse(pid):
    s, n = pid.split("_")
    return int(s[1:]), int(n[1:])

rows = []
for cell in CELLS:
    csv = SWEEP / cell / "scRMSD_guided.csv"
    if not csv.exists():
        print(f"MISSING: {cell}")
        continue
    df = pd.read_csv(csv)
    for _, r in df.iterrows():
        try:
            v = float(r["scRMSD_ca_min"])
        except Exception:
            v = float("inf")
        seed, length = parse(r["protein_id"])
        rows.append({"cell": cell, "direction": cell.split("_w")[0],
                     "w": int(cell.split("_w")[1]), "seed": seed,
                     "length": length, "scRMSD": v})
full = pd.DataFrame(rows)
full.to_csv(SWEEP / "scRMSD_summary.csv", index=False)
print(f"Aggregate -> {SWEEP/'scRMSD_summary.csv'}  ({len(full)} rows)")

# Persistent outlier: s45_n500 was broken at every cell. Strip from "fair" stats.
S45_N500_MASK = ~((full.seed == 45) & (full.length == 500))
def fmt_pct(n_des, n_tot):
    return f"{n_des}/{n_tot} ({100*n_des/n_tot:.0f}%)"

print("\n## Per-cell designability (<2 Å)\n")
print("| direction | w | n | designable | mean | excl s45_n500: designable | mean |")
print("|---|---|---|---|---|---|---|")
for d in ["camsol_max", "tango_min"]:
    for w in [1, 2, 4, 8, 16]:
        cell = full[(full.direction == d) & (full.w == w)]
        n = len(cell)
        finite = cell[cell.scRMSD < 1e9]  # drop inf (failed runs)
        des = (finite.scRMSD < 2.0).sum()
        m = finite.scRMSD.mean()
        clean = full[(full.direction == d) & (full.w == w) & S45_N500_MASK]
        clean_finite = clean[clean.scRMSD < 1e9]
        des_c = (clean_finite.scRMSD < 2.0).sum()
        m_c = clean_finite.scRMSD.mean()
        print(f"| {d} | {w} | {n} | {fmt_pct(des, len(finite))} | {m:.2f} | "
              f"{fmt_pct(des_c, len(clean_finite))} | {m_c:.2f} |")

print("\n## Per-length × w (excl s45_n500), tango_min\n")
print("| L | w=1 | w=2 | w=4 | w=8 | w=16 |")
print("|---|---|---|---|---|---|")
for L in [300, 400, 500]:
    line = f"| {L} |"
    for w in [1, 2, 4, 8, 16]:
        cell = full[(full.direction == "tango_min") & (full.w == w) &
                    (full.length == L) & S45_N500_MASK]
        finite = cell[cell.scRMSD < 1e9]
        des = (finite.scRMSD < 2.0).sum()
        line += f" {des}/{len(finite)} (mean {finite.scRMSD.mean():.2f}) |"
    print(line)

print("\n## Per-length × w (excl s45_n500), camsol_max\n")
print("| L | w=1 | w=2 | w=4 | w=8 | w=16 |")
print("|---|---|---|---|---|---|")
for L in [300, 400, 500]:
    line = f"| {L} |"
    for w in [1, 2, 4, 8, 16]:
        cell = full[(full.direction == "camsol_max") & (full.w == w) &
                    (full.length == L) & S45_N500_MASK]
        finite = cell[cell.scRMSD < 1e9]
        des = (finite.scRMSD < 2.0).sum()
        line += f" {des}/{len(finite)} (mean {finite.scRMSD.mean():.2f}) |"
    print(line)
