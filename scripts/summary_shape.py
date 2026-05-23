"""Combined property + codesign summary for the shape ablation."""
import csv
import json
import statistics
from pathlib import Path

import pandas as pd

CELLS = ["cosine", "constant_in_window"]
ROOT = Path("results/ablation_2026_05_17")


def codesign_stats(csv_path: Path):
    if not csv_path.exists():
        return None
    rmsds = []
    for r in csv.DictReader(open(csv_path)):
        try:
            x = float(r["coScRMSD_ca"])
            if x != float("inf"):
                rmsds.append(x)
        except ValueError:
            continue
    n = sum(1 for _ in csv.DictReader(open(csv_path)))
    if not rmsds:
        return None
    return {
        "n": n,
        "n_finite": len(rmsds),
        "mean_rmsd": statistics.mean(rmsds),
        "med_rmsd": statistics.median(rmsds),
        "lt_2A": sum(1 for x in rmsds if x < 2.0),
        "lt_3A": sum(1 for x in rmsds if x < 3.0),
    }


def property_stats(cell_dir: Path):
    csv_path = cell_dir / "properties_guided.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = []
    for pid in df["protein_id"]:
        dp = cell_dir / "diagnostics" / f"{pid}_diagnostics.json"
        if not dp.exists():
            continue
        steps = json.loads(dp.read_text())
        pred = float(steps[-1]["predicted_properties"]["tango"])
        real = float(df.set_index("protein_id").loc[pid, "tango_total"])
        rows.append({"pid": pid, "pred": pred, "real": real, "gap": pred - real})
    if not rows:
        return None
    return {
        "n": len(rows),
        "pred": statistics.mean(r["pred"] for r in rows),
        "real": statistics.mean(r["real"] for r in rows),
        "gap": statistics.mean(r["gap"] for r in rows),
    }


# Baseline references — pulled from prior runs (same w=32, L=300, n=16, fixt1 ens denoised)
baseline_real = 449.4
baseline_gap = -82.4
baseline_codesign_mean = 2.71
baseline_codesign_lt2 = "13/16"

print()
print("=" * 116)
print(f"{'cell':40s}  {'n':>3} {'real':>7} {'gap':>7}    {'coScRMSD μ':>10}  {'median':>7}  {'<2Å':>5}  {'<3Å':>5}")
print("=" * 116)
print(f"{'fixt1 baseline [0.3,0.8]→0.9':40s}  {16:3d} {baseline_real:7.1f} {baseline_gap:+7.1f}    {baseline_codesign_mean:10.2f}  {1.12:7.2f}  {baseline_codesign_lt2:>5}  {'13/16':>5}")
print("-" * 116)

for cell in CELLS:
    p = property_stats(ROOT / cell)
    c = codesign_stats(ROOT / cell / "codesign_guided.csv")
    if not p:
        print(f"{cell:40s}  NO PROPERTY DATA")
        continue
    if not c:
        print(f"{cell:40s}  {p['n']:3d} {p['real']:7.1f} {p['gap']:+7.1f}    NO CODESIGN")
        continue
    print(
        f"{cell:40s}  {p['n']:3d} {p['real']:7.1f} {p['gap']:+7.1f}    "
        f"{c['mean_rmsd']:10.2f}  {c['med_rmsd']:7.2f}  "
        f"{c['lt_2A']:2d}/{c['n']:<2d}  {c['lt_3A']:2d}/{c['n']:<2d}"
    )

print()
print("Baseline ref (current deployment): real=449.4, gap=-82.4, coScRMSD mean=2.71, 13/16 <2Å.")
