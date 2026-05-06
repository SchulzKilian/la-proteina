"""Pairwise TM-score across the noise-aware-ensemble sweep + an unsteered baseline.

For each (direction, w, L) cell in results/noise_aware_ensemble_sweep/<cell>/guided/:
  - Take up to 16 PDBs at length L (the seeds 42-57 grid)
  - Compute pairwise TM-score (mean of tm_norm_chain1 + tm_norm_chain2)
  - Report median / mean / p10 / p90 of the 16C2 = 120 pairs

Baseline: results/generated_stratified_300_800_nsteps400/samples/. Pairs proteins
in a length window [L-15, L+15] of each target L, takes up to 16, runs the same
pairwise TM-score.

Higher mean pairwise TM-score = lower structural diversity (proteins look more
similar to each other). The "steering reduces diversity" hypothesis predicts
mean pairwise TM-score should rise with w.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/noise_aware_ensemble_sweep"
BASELINE_DIR = ROOT / "results/generated_stratified_300_800_nsteps400/samples"

DIRECTIONS = ["camsol_max", "tango_min"]
WLEVELS = [1, 2, 4, 8, 16]
LENGTHS = [300, 400, 500]
SEEDS = list(range(42, 58))


def coords_seq(pdb_path: Path):
    s = get_structure(str(pdb_path))
    chain = next(s.get_chains())
    return get_residue_data(chain)


def pairwise_tm(pdb_paths: list[Path]) -> list[float]:
    """All-pairs mean(TM-norm-chain1, TM-norm-chain2). Returns list len = N*(N-1)/2."""
    cs_data = []
    for p in pdb_paths:
        try:
            cs_data.append(coords_seq(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}", flush=True)
            cs_data.append(None)

    out = []
    for i, j in combinations(range(len(pdb_paths)), 2):
        if cs_data[i] is None or cs_data[j] is None:
            continue
        c1, s1 = cs_data[i]
        c2, s2 = cs_data[j]
        try:
            r = tm_align(c1, c2, s1, s2)
            out.append(0.5 * (r.tm_norm_chain1 + r.tm_norm_chain2))
        except Exception as e:
            print(f"  align fail {pdb_paths[i].name} <-> {pdb_paths[j].name}: {e}", flush=True)
    return out


def stats(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"label": name, "n_pairs": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    arr = np.array(vals)
    return {"label": name, "n_pairs": len(vals),
            "mean": arr.mean(), "median": np.median(arr),
            "p10": np.percentile(arr, 10), "p90": np.percentile(arr, 90)}


def main():
    rows = []

    # Steered cells
    for direction in DIRECTIONS:
        for w in WLEVELS:
            for L in LENGTHS:
                cell_dir = SWEEP / f"{direction}_w{w}" / "guided"
                pdbs = sorted([cell_dir / f"s{s}_n{L}.pdb" for s in SEEDS
                               if (cell_dir / f"s{s}_n{L}.pdb").exists()])
                label = f"{direction}_w{w}_L{L}"
                print(f"[{label}] {len(pdbs)} PDBs", flush=True)
                tms = pairwise_tm(pdbs)
                row = stats(label, tms)
                row.update({"set": "steered", "direction": direction, "w": w, "L": L})
                rows.append(row)

    # Baseline — unsteered. Group by length window [L-15, L+15], pick up to 16.
    for L in LENGTHS:
        all_pdbs = sorted(BASELINE_DIR.glob("*.pdb"))
        candidates = []
        for p in all_pdbs:
            try:
                stem = p.stem  # e.g. s1000_n310
                _, npart = stem.split("_")
                length = int(npart[1:])
                if abs(length - L) <= 15:
                    candidates.append(p)
            except ValueError:
                continue
        candidates = candidates[:16]
        label = f"baseline_L{L}"
        print(f"[{label}] {len(candidates)} PDBs (window L±15)", flush=True)
        tms = pairwise_tm(candidates)
        row = stats(label, tms)
        row.update({"set": "baseline", "direction": "unsteered", "w": 0, "L": L})
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = SWEEP / "diversity_pairwise_tm.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.relative_to(ROOT)}\n", flush=True)

    # Aggregate per (direction, w) — mean across lengths
    print("## Mean pairwise TM-score per (direction, w), aggregated across L∈{300, 400, 500}\n")
    print("| direction | w | mean TM | median TM | p10 | p90 | n_pairs |")
    print("|---|---|---|---|---|---|---|")
    for direction in DIRECTIONS + ["unsteered"]:
        for w in ([0] if direction == "unsteered" else WLEVELS):
            sub = df[(df.direction == direction) & (df.w == w)]
            if len(sub) == 0:
                continue
            # Combine all pairs across the 3 lengths by weighting by n_pairs
            n_total = sub.n_pairs.sum()
            mean_w = float((sub["mean"] * sub.n_pairs).sum() / max(n_total, 1))
            med_w = float((sub["median"] * sub.n_pairs).sum() / max(n_total, 1))
            p10_w = float((sub["p10"] * sub.n_pairs).sum() / max(n_total, 1))
            p90_w = float((sub["p90"] * sub.n_pairs).sum() / max(n_total, 1))
            label = "unsteered (baseline)" if direction == "unsteered" else f"{direction} w={w}"
            print(f"| {label} | {w} | {mean_w:.3f} | {med_w:.3f} | {p10_w:.3f} | {p90_w:.3f} | {n_total} |")

    print("\n## Per-length × w (mean pairwise TM-score)\n")
    for direction in DIRECTIONS:
        print(f"### {direction}\n")
        print("| L | w=1 | w=2 | w=4 | w=8 | w=16 | unsteered baseline |")
        print("|---|---|---|---|---|---|---|")
        for L in LENGTHS:
            line = f"| {L} |"
            for w in WLEVELS:
                cell = df[(df.direction == direction) & (df.w == w) & (df.L == L)]
                if len(cell):
                    line += f" {cell['mean'].iloc[0]:.3f} |"
                else:
                    line += " — |"
            base = df[(df.direction == "unsteered") & (df.L == L)]
            if len(base):
                line += f" {base['mean'].iloc[0]:.3f} |"
            else:
                line += " — |"
            print(line)
        print()


if __name__ == "__main__":
    main()
