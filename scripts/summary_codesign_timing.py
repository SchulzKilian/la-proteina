"""Print coScRMSD summary table for the timing ablation."""
import csv
import statistics
from pathlib import Path


def stats(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path)))
    rmsds = []
    for r in rows:
        try:
            x = float(r["coScRMSD_ca"])
            if x == float("inf"):
                continue
            rmsds.append(x)
        except ValueError:
            continue
    if not rmsds:
        return None
    return {
        "n": len(rows),
        "n_finite": len(rmsds),
        "mean": statistics.mean(rmsds),
        "median": statistics.median(rmsds),
        "lt_2A": sum(1 for x in rmsds if x < 2.0),
        "lt_3A": sum(1 for x in rmsds if x < 3.0),
        "lt_4A": sum(1 for x in rmsds if x < 4.0),
    }


def row(name, s):
    if s is None:
        return f"{name:48s}  NO DATA"
    return (
        f"{name:48s}  "
        f"n={s['n']:3d}  finite={s['n_finite']:3d}  "
        f"mean={s['mean']:5.2f}  med={s['median']:5.2f}  "
        f"<2A={s['lt_2A']:3d}  <3A={s['lt_3A']:3d}  <4A={s['lt_4A']:3d}"
    )


# NA-v1 baseline at w=32 L=300 (E066), filtered from full grid
na_csv = Path("results/noise_aware_high_w_scout/tango_min_w32/codesign_guided.csv")
na_L300_rmsds = []
if na_csv.exists():
    for r in csv.DictReader(open(na_csv)):
        if r["protein_id"].endswith("_n300"):
            try:
                x = float(r["coScRMSD_ca"])
                if x != float("inf"):
                    na_L300_rmsds.append(x)
            except ValueError:
                pass

na_summary = None
if na_L300_rmsds:
    na_summary = {
        "n": 16,
        "n_finite": len(na_L300_rmsds),
        "mean": statistics.mean(na_L300_rmsds),
        "median": statistics.median(na_L300_rmsds),
        "lt_2A": sum(1 for x in na_L300_rmsds if x < 2.0),
        "lt_3A": sum(1 for x in na_L300_rmsds if x < 3.0),
        "lt_4A": sum(1 for x in na_L300_rmsds if x < 4.0),
    }

fixt1_baseline = stats(
    Path("results/fixt1_smoke/tango_min_w32_fixt1_ensemble_denoised_n48/codesign_guided.csv")
)

print("=" * 110)
print(f"coScRMSD on the timing ablation (w=32, L=300, n=16, fixt1, denoised, tango_min)")
print("=" * 110)
print(row("NA-v1 ens denoised L=300 (E066)", na_summary))
print(row("fixt1 ens denoised L=300 (today, baseline)", fixt1_baseline))
print("-" * 110)
for cell in ["constant_all", "late_only", "early_only", "wide_ramp"]:
    s = stats(Path(f"results/ablation_2026_05_17/{cell}/codesign_guided.csv"))
    print(row(cell, s))
