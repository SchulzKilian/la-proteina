"""Eval pass on the full noise-aware-ensemble sweep + predictor-vs-real gap.

For each cell in results/noise_aware_ensemble_sweep/<dir>_w<W>/:
  1. Compute real-property panel on guided/ via steering.property_evaluate's
     evaluate_directory (writes properties_guided.csv next to the run).
  2. Pull predictor's last-step claim per protein from diagnostics JSON.
  3. Match on protein_id, compute gap = predictor_claim - real_value.

Outputs:
  - properties_guided.csv per cell (skipped if already present).
  - results/noise_aware_ensemble_sweep/gap_summary.csv with one row per
    (direction, w, length) and aggregate stats.
  - Markdown summary printed to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steering.property_evaluate import evaluate_directory

SWEEP = ROOT / "results/noise_aware_ensemble_sweep"
WLEVELS = [1, 2, 4, 8, 16]
DIRECTIONS = [("camsol_max", "camsol_intrinsic", None),  # no real metric locally
              ("tango_min",  "tango",            "tango_total")]
LENGTHS = [300, 400, 500]


def predictor_final(diag_path: Path, key: str) -> float:
    steps = json.loads(diag_path.read_text())
    return float(steps[-1]["predicted_properties"][key])


def parse_pid(pid: str) -> tuple[int, int]:
    """sX_nL -> (X, L)"""
    s_part, n_part = pid.split("_")
    return int(s_part[1:]), int(n_part[1:])


def main():
    rows = []
    for direction, pred_key, real_col in DIRECTIONS:
        for w in WLEVELS:
            cell = SWEEP / f"{direction}_w{w}"
            if not cell.exists():
                print(f"MISSING: {cell.name}", flush=True)
                continue

            csv_path = cell / "properties_guided.csv"
            if not csv_path.exists():
                print(f"[eval] computing real properties for {cell.name} ...", flush=True)
                df = evaluate_directory(cell / "guided", skip_tango=False)
                df.to_csv(csv_path, index=False)
            else:
                df = pd.read_csv(csv_path)

            df = df.set_index("protein_id")
            diag_dir = cell / "diagnostics"

            for diag in sorted(diag_dir.glob("*_diagnostics.json")):
                pid = diag.stem.replace("_diagnostics", "")
                seed, length = parse_pid(pid)
                pred = predictor_final(diag, pred_key)
                real = float("nan")
                if real_col is not None and pid in df.index and real_col in df.columns:
                    v = df.loc[pid, real_col]
                    if pd.notna(v):
                        real = float(v)
                rows.append({
                    "direction": direction, "w": w, "length": length, "seed": seed,
                    "protein_id": pid,
                    "predictor": pred,
                    "real": real,
                    "gap": pred - real if not np.isnan(real) else float("nan"),
                })

    full = pd.DataFrame(rows)
    out_path = SWEEP / "gap_summary.csv"
    full.to_csv(out_path, index=False)
    print(f"\nFull table -> {out_path.relative_to(ROOT)}  ({len(full)} rows)")

    # ---- Aggregate tables ----
    print("\n## Aggregate per (direction, w) — averaged across all 48 proteins per cell")
    print("\n| direction | w | n | predictor mean | real mean | gap mean | gap std | gap p5 | gap p95 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for direction, _, real_col in DIRECTIONS:
        sub = full[full.direction == direction]
        for w in WLEVELS:
            cell = sub[sub.w == w]
            n = len(cell)
            pred_m = cell.predictor.mean()
            if real_col is None:
                print(f"| {direction} | {w} | {n} | {pred_m:.1f} | n/a | n/a | n/a | n/a | n/a |")
            else:
                real_m = cell.real.mean()
                gap_m = cell.gap.mean()
                gap_s = cell.gap.std()
                gap_p5 = cell.gap.quantile(0.05)
                gap_p95 = cell.gap.quantile(0.95)
                print(f"| {direction} | {w} | {n} | {pred_m:.1f} | {real_m:.1f} | "
                      f"{gap_m:+.1f} | {gap_s:.1f} | {gap_p5:+.1f} | {gap_p95:+.1f} |")

    print("\n## Per-length breakdown for tango_min (real `tango_total` available)")
    print("\n| w | L | n | predictor mean | real mean | gap mean | gap std |")
    print("|---|---|---|---|---|---|---|")
    sub = full[(full.direction == "tango_min")]
    for w in WLEVELS:
        for L in LENGTHS:
            cell = sub[(sub.w == w) & (sub.length == L)]
            n = len(cell)
            if n == 0:
                continue
            print(f"| {w} | {L} | {n} | "
                  f"{cell.predictor.mean():.1f} | {cell.real.mean():.1f} | "
                  f"{cell.gap.mean():+.1f} | {cell.gap.std():.1f} |")

    # ---- Δ vs w=1 reference ----
    print("\n## Δ predictor vs Δ real, ratio (tango_min only)")
    print("\n| w | Δ predictor (vs w=1) | Δ real (vs w=1) | ratio Δpred/Δreal |")
    print("|---|---|---|---|")
    sub = full[full.direction == "tango_min"]
    base_pred = sub[sub.w == 1].predictor.mean()
    base_real = sub[sub.w == 1].real.mean()
    for w in WLEVELS:
        cell = sub[sub.w == w]
        dp = cell.predictor.mean() - base_pred
        dr = cell.real.mean() - base_real
        ratio = dp / dr if abs(dr) > 1e-9 else float("inf")
        print(f"| {w} | {dp:+.1f} | {dr:+.1f} | {ratio:.2f}× |")


if __name__ == "__main__":
    main()
