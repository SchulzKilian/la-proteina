"""Run scRMSD evaluation on hybrid_grad_routing PDBs and report
paired baseline-vs-hybrid stats.

Walks results/hybrid_grad_routing/<label>/{baseline,hybrid}/L{50,100,200}/
sample_{0,1,2}.pdb, runs ProteinMPNN+ESMFold (CA-only) per PDB, records
best-scRMSD per arm/L/sample, and prints a per-(L) paired summary plus a
pooled designability count.

Usage:
    /home/ks2218/.conda/envs/laproteina_env/bin/python \
        script_utils/eval_hybrid_grad_routing.py \
        --label grad_routing_smoke_n3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

import pandas as pd
import torch

from proteinfoundation.metrics.designability import scRMSD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--root_dir", default="results/hybrid_grad_routing",
        help="Parent dir of <label>/",
    )
    parser.add_argument(
        "--num_seq_per_target", type=int, default=8,
        help="ProteinMPNN sequences per PDB. evaluate.py default is 8.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir) / args.label
    assert root.exists(), f"No such dir: {root}"
    arms = ["baseline", "hybrid"]
    lengths = [50, 100, 200]
    samples = [0, 1, 2]

    rows = []
    for arm in arms:
        for L in lengths:
            for s in samples:
                pdb_path = root / arm / f"L{L}" / f"sample_{s}.pdb"
                if not pdb_path.exists():
                    logger.warning(f"  MISSING {pdb_path}")
                    continue
                tmp_dir = root / arm / f"L{L}" / f"sample_{s}_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    res = scRMSD(
                        pdb_file_path=str(pdb_path.resolve()),
                        tmp_path=str(tmp_dir.resolve()),
                        num_seq_per_target=args.num_seq_per_target,
                        use_pdb_seq=False,
                        ret_min=True,
                        rmsd_modes=["ca", "bb3o"],
                        folding_models=["esmfold"],
                        keep_outputs=True,  # else run_esmfold deletes tmp_path before scRMSD reads it (folding_models.py:159)
                    )
                except Exception as e:
                    logger.exception(f"  ERROR on {pdb_path}: {e}")
                    rows.append({
                        "arm": arm, "L": L, "sample_idx": s,
                        "ca_esmfold": float("nan"),
                        "bb3o_esmfold": float("nan"),
                        "wall_s": time.time() - t0,
                        "error": str(e),
                    })
                    continue
                # res structure: {"ca": {"esmfold": min_scrmsd_float}, "bb3o": {...}}
                ca = res.get("ca", {}).get("esmfold", float("nan"))
                bb3o = res.get("bb3o", {}).get("esmfold", float("nan"))
                rows.append({
                    "arm": arm, "L": L, "sample_idx": s,
                    "ca_esmfold": float(ca),
                    "bb3o_esmfold": float(bb3o),
                    "wall_s": time.time() - t0,
                })
                logger.info(
                    f"  {arm:>8} L={L:>3} s={s}: ca={ca:.3f} bb3o={bb3o:.3f}  ({time.time()-t0:.1f}s)"
                )

    df = pd.DataFrame(rows)
    out_csv = root / "results_scrmsd.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nWrote {out_csv}")

    # ---------- Per-(arm, L) summary ----------
    print("\n" + "=" * 70)
    print(f"Hybrid gradient-routing — paired scRMSD eval ({args.label})")
    print("=" * 70)
    print("\nPer-(arm, L) scRMSD_ca_esmfold (best over MPNN sequences, ESMFold refold):")
    print("\n  arm       L   N  designable(<2Å)  best Å   median Å   mean Å")
    for arm in arms:
        for L in lengths:
            sub = df[(df.arm == arm) & (df.L == L) & df.ca_esmfold.notna()]
            n = len(sub)
            if n == 0:
                continue
            d = int((sub.ca_esmfold < 2.0).sum())
            print(f"  {arm:>8} {L:>3}  {n:>1}    {d}/{n}             "
                  f"{sub.ca_esmfold.min():.2f}    {sub.ca_esmfold.median():.2f}      "
                  f"{sub.ca_esmfold.mean():.2f}")

    # ---------- Pooled ----------
    print("\nPooled across all lengths:")
    for arm in arms:
        sub = df[(df.arm == arm) & df.ca_esmfold.notna()]
        n = len(sub)
        d = int((sub.ca_esmfold < 2.0).sum())
        print(f"  {arm:>8}: {d}/{n} = {100*d/n:.1f}% designable, "
              f"best {sub.ca_esmfold.min():.2f} Å, median {sub.ca_esmfold.median():.2f} Å")

    # ---------- Paired diff (hybrid - baseline) per seed ----------
    print("\nPaired Δ scRMSD_ca_esmfold (hybrid − baseline) per (L, sample_idx):")
    pivot = df.pivot_table(index=["L", "sample_idx"], columns="arm", values="ca_esmfold")
    if "baseline" in pivot.columns and "hybrid" in pivot.columns:
        pivot["delta"] = pivot["hybrid"] - pivot["baseline"]
        pivot["winner"] = pivot.apply(
            lambda r: "hybrid" if r["delta"] < -0.1
            else ("baseline" if r["delta"] > 0.1 else "tie"), axis=1
        )
        print(pivot[["baseline", "hybrid", "delta", "winner"]].round(3).to_string())
        print(f"\nNet over 9 paired samples:")
        print(f"  hybrid wins   (Δ<-0.1):  {(pivot.delta < -0.1).sum()}")
        print(f"  ties (|Δ|≤0.1):          {((pivot.delta >= -0.1) & (pivot.delta <= 0.1)).sum()}")
        print(f"  baseline wins (Δ>+0.1):  {(pivot.delta > 0.1).sum()}")
        print(f"  mean Δ:   {pivot.delta.mean():+.3f} Å")
        print(f"  median Δ: {pivot.delta.median():+.3f} Å")


if __name__ == "__main__":
    main()
