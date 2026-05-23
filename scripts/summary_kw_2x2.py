"""Summary for the K × w 2×2 at L=300, n=32 (seeds 42-73)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import statistics

import pandas as pd

from steering.property_evaluate import evaluate_directory

CELL_DIRS = {
    "K1_w32": Path("results/fixt1_smoke/tango_min_w32_fixt1_ensemble_denoised_n48"),
    "K5_w32": Path("results/ablation_2026_05_17/ug_K5"),
    "K1_w64": Path("results/ablation_2026_05_17/K1_w64"),
    "K5_w64": Path("results/ablation_2026_05_17/K5_w64"),
}

SEEDS_32 = [f"s{i}" for i in range(42, 74)]


def collect(cell_dir: Path):
    guided = cell_dir / "guided"
    if not guided.exists():
        return []
    csv_path = cell_dir / "properties_guided.csv"
    if not csv_path.exists():
        df = evaluate_directory(guided, skip_tango=False)
        df.to_csv(csv_path, index=False)
    else:
        # Re-eval to pick up any newly generated PDBs
        df = evaluate_directory(guided, skip_tango=False)
        df.to_csv(csv_path, index=False)
    rows = []
    for pid in df["protein_id"]:
        seed = pid.split("_")[0]
        if seed not in SEEDS_32:
            continue
        if not pid.endswith("_n300"):
            continue  # filter to L=300 only
        dp = cell_dir / "diagnostics" / f"{pid}_diagnostics.json"
        if not dp.exists():
            continue
        steps = json.loads(dp.read_text())
        pred = float(steps[-1]["predicted_properties"]["tango"])
        real = float(df.set_index("protein_id").loc[pid, "tango_total"])
        rows.append({"pid": pid, "pred": pred, "real": real, "gap": pred - real})
    return rows


def codesign_at_L300(csv_path: Path):
    if not csv_path.exists():
        return None
    rmsds_by_pid = {}
    for r in csv.DictReader(open(csv_path)):
        pid = r["protein_id"]
        seed = pid.split("_")[0]
        if not pid.endswith("_n300"):
            continue
        if seed not in SEEDS_32:
            continue
        try:
            x = float(r["coScRMSD_ca"])
            if x != float("inf"):
                rmsds_by_pid[pid] = x
        except ValueError:
            continue
    rmsds = list(rmsds_by_pid.values())
    if not rmsds:
        return None
    return {
        "n": len(rmsds),
        "mean": statistics.mean(rmsds),
        "median": statistics.median(rmsds),
        "lt_2A": sum(1 for x in rmsds if x < 2.0),
        "lt_3A": sum(1 for x in rmsds if x < 3.0),
        "max": max(rmsds),
    }


print()
print("=" * 110)
print(f"{'cell':12s}  {'n_prop':>6}  {'pred':>7}  {'real':>7}  {'gap':>7}    "
      f"{'n_cs':>5}  {'cs_μ':>5}  {'cs_med':>6}  {'<2Å':>5}  {'<3Å':>5}  {'max':>5}")
print("=" * 110)

for name, d in CELL_DIRS.items():
    rows = collect(d)
    cs = codesign_at_L300(d / "codesign_guided.csv")
    if not rows:
        print(f"{name:12s}  NO PROPERTY DATA YET")
        continue
    pred = statistics.mean(r["pred"] for r in rows)
    real = statistics.mean(r["real"] for r in rows)
    gap = statistics.mean(r["gap"] for r in rows)
    if cs:
        print(f"{name:12s}  {len(rows):6d}  {pred:7.1f}  {real:7.1f}  {gap:+7.1f}    "
              f"{cs['n']:5d}  {cs['mean']:5.2f}  {cs['median']:6.2f}  "
              f"{cs['lt_2A']:2d}/{cs['n']:<2d}  {cs['lt_3A']:2d}/{cs['n']:<2d}  {cs['max']:5.2f}")
    else:
        print(f"{name:12s}  {len(rows):6d}  {pred:7.1f}  {real:7.1f}  {gap:+7.1f}    NO CODESIGN")

print()
print("Cell defs: K=denoising_steps; w=w_max; all at L=300, fixt1 5-fold ensemble, denoised, tango_min, linear_ramp [0.3,0.8]→0.9")
