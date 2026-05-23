"""Eval all cells from the 2026-05-17 ablation sweep + comparison vs baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import statistics

import pandas as pd

from steering.property_evaluate import evaluate_directory

ROOT = Path("/home/ks2218/la-proteina/results/ablation_2026_05_17")

# Cells generated at L=300, n=16 seeds, fixt1, denoised, tango_min
CELLS_L300 = [
    "constant_all", "late_only", "early_only", "wide_ramp",
    "cosine", "constant_in_window",
    "fold3",
    "ug_K2", "ug_K3", "ug_K5",
]

# Special cell: clean predictor at L=300/400/500 × 16 seeds (paired with E066)
CELL_CLEAN = "clean_ens_n48"


def collect(cell_dir: Path):
    guided = cell_dir / "guided"
    if not guided.exists() or not any(guided.glob("*.pdb")):
        return []
    csv_path = cell_dir / "properties_guided.csv"
    if not csv_path.exists():
        try:
            df = evaluate_directory(guided, skip_tango=False)
        except Exception:
            return []
        df.to_csv(csv_path, index=False)
    else:
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
    return rows


def summary(name, rows):
    if not rows:
        return f"{name}: NO DATA"
    p = statistics.mean(r["pred"] for r in rows)
    r = statistics.mean(r["real"] for r in rows)
    g = statistics.mean(r["gap"] for r in rows)
    return f"{name:36s}  n={len(rows):3d}  pred={p:7.1f}  real={r:7.1f}  gap={g:+7.1f}"


# Baseline references at L=300 (today's n=4 + n=48 L=300 subset)
print("=" * 90)
print("BASELINES at w=32, fixt1 unless noted")
print("=" * 90)

# fixt1 ensemble denoised L=300 (subset of today's n=48 grid)
na_root = Path("/home/ks2218/la-proteina/results/noise_aware_high_w_scout/tango_min_w32")
fixt1_root = Path("/home/ks2218/la-proteina/results/fixt1_smoke/tango_min_w32_fixt1_ensemble_denoised_n48")
na_rows = collect(na_root)
fixt1_rows = collect(fixt1_root)
na_L300 = [r for r in na_rows if r["pid"].endswith("_n300")]
fixt1_L300 = [r for r in fixt1_rows if r["pid"].endswith("_n300")]
print(summary("fixt1 ens denoised L=300 (E32 baseline)", fixt1_L300))
print(summary("NA-v1 ens denoised L=300", na_L300))
print(summary("fixt1 ens denoised L=300+400+500 n=48", fixt1_rows))
print(summary("NA-v1 ens denoised L=300+400+500 n=48", na_rows))

# Today's single-fold fixt1 at L=300 n=4 — for the folds=1 reading
single_root = Path("/home/ks2218/la-proteina/results/fixt1_smoke/tango_min_w32_fixt1_single_legacy")
single_rows = collect(single_root) if single_root.exists() else []
if single_rows:
    print(summary("fixt1 SINGLE FOLD denoised L=300 n=4 (folds=1)", single_rows))

print()
print("=" * 90)
print("ABLATION CELLS at L=300, n=16, fixt1, denoised, tango_min")
print("=" * 90)

baseline_l300 = fixt1_L300
b_pred = statistics.mean(r["pred"] for r in baseline_l300)
b_real = statistics.mean(r["real"] for r in baseline_l300)
b_gap = statistics.mean(r["gap"] for r in baseline_l300)
print(f"  Baseline (linear_ramp t∈[0.3,0.8], stop=0.9, K=1, 5-fold): pred={b_pred:.1f} real={b_real:.1f} gap={b_gap:+.1f}")
print()

cell_results = {}
for cell in CELLS_L300:
    rows = collect(ROOT / cell)
    cell_results[cell] = rows
    if not rows:
        print(f"{cell:36s}  NO DATA"); continue
    p = statistics.mean(r["pred"] for r in rows)
    r = statistics.mean(r["real"] for r in rows)
    g = statistics.mean(r["gap"] for r in rows)
    dp = p - b_pred
    dr = r - b_real
    dg = g - b_gap
    print(f"{cell:36s}  n={len(rows):3d}  pred={p:7.1f}  real={r:7.1f}  gap={g:+7.1f}    Δreal={dr:+6.1f}  Δgap={dg:+6.1f}")

print()
print("=" * 90)
print("CLEAN PREDICTOR n=48 (paired vs NA-v1 + fixt1 at n=48)")
print("=" * 90)

clean_rows = collect(ROOT / CELL_CLEAN)
print(summary("clean ens denoised n=48", clean_rows))
print(summary("fixt1 ens denoised n=48 (today)", fixt1_rows))
print(summary("NA-v1 ens denoised n=48 (E066)", na_rows))

# Paired Δ vs NA-v1
clean_map = {r["pid"]: r for r in clean_rows}
na_map = {r["pid"]: r for r in na_rows}
common = sorted(set(clean_map) & set(na_map))
if common:
    dr = [clean_map[p]["real"] - na_map[p]["real"] for p in common]
    dg = [clean_map[p]["gap"]  - na_map[p]["gap"]  for p in common]
    mr = statistics.mean(dr); sr = statistics.stdev(dr)
    mg = statistics.mean(dg); sg = statistics.stdev(dg)
    print(f"  Paired (n={len(common)}): Δreal (clean - NA) = {mr:+.1f} ± {sr/len(common)**0.5:.1f} SEM")
    print(f"  Paired (n={len(common)}): Δgap  (clean - NA) = {mg:+.1f} ± {sg/len(common)**0.5:.1f} SEM")

# Per-length breakdown for clean predictor
for L in [300, 400, 500]:
    Lc = [r for r in clean_rows if r["pid"].endswith(f"_n{L}")]
    Lna = [r for r in na_rows if r["pid"].endswith(f"_n{L}")]
    Lfx = [r for r in fixt1_rows if r["pid"].endswith(f"_n{L}")]
    if Lc:
        cp = statistics.mean(r["pred"] for r in Lc); cr = statistics.mean(r["real"] for r in Lc); cg = statistics.mean(r["gap"] for r in Lc)
        npp = statistics.mean(r["pred"] for r in Lna); nr = statistics.mean(r["real"] for r in Lna); ng = statistics.mean(r["gap"] for r in Lna)
        fp = statistics.mean(r["pred"] for r in Lfx); fr = statistics.mean(r["real"] for r in Lfx); fg = statistics.mean(r["gap"] for r in Lfx)
        print(f"  L={L}: clean real={cr:.1f} gap={cg:+.1f} | NA real={nr:.1f} gap={ng:+.1f} | fixt1 real={fr:.1f} gap={fg:+.1f}")
