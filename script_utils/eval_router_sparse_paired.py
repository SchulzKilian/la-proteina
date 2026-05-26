"""Move-2 paired eval: scRMSD + paired ΔscRMSD vs canonical dense.

Reads PDBs from results/sparse_trunk_move2/<label>/{sparse,baseline}/L{...}/sample_{i}.pdb,
runs ProteinMPNN + ESMFold (CA-only) per PDB, prints per-(L) paired summary +
pooled designability + paired ΔscRMSD distribution.

Mirrors script_utils/eval_hybrid_grad_routing.py (same scRMSD plumbing, just
new arms/dir layout).
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import pandas as pd
import torch
from loguru import logger

from proteinfoundation.metrics.designability import scRMSD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--root_dir", default="results/sparse_trunk_move2")
    ap.add_argument("--lengths", default="100,200",
                    help="Comma-sep lengths to scan.")
    ap.add_argument("--n_samples", type=int, default=12,
                    help="Per-length sample count (must match sampling --n_samples).")
    ap.add_argument("--num_seq_per_target", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root_dir) / args.label
    assert root.exists(), f"No such dir: {root}"
    arms = ["sparse", "baseline"]
    lengths = [int(x) for x in args.lengths.split(",")]
    samples = list(range(args.n_samples))

    rows = []
    for arm in arms:
        for L in lengths:
            for s in samples:
                pdb_path = root / arm / f"L{L}" / f"sample_{s}.pdb"
                if not pdb_path.exists():
                    logger.warning(f"  MISSING {pdb_path}")
                    continue
                tmp_dir = root / arm / f"L{L}" / f"sample_{s}_tmp"
                # Eval-tmp-dir sweep on retry (MEMORY.md: feedback_eval_tmp_dir_sweep).
                if tmp_dir.exists():
                    import shutil
                    shutil.rmtree(tmp_dir)
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
                        keep_outputs=True,
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
                ca = res.get("ca", {}).get("esmfold", float("nan"))
                bb3o = res.get("bb3o", {}).get("esmfold", float("nan"))
                rows.append({
                    "arm": arm, "L": L, "sample_idx": s,
                    "ca_esmfold": float(ca),
                    "bb3o_esmfold": float(bb3o),
                    "wall_s": time.time() - t0,
                })
                logger.info(
                    f"  {arm:>8} L={L:>3} s={s:>2}: ca={ca:.3f} bb3o={bb3o:.3f}  ({time.time()-t0:.1f}s)"
                )

    df = pd.DataFrame(rows)
    out_csv = root / "results_scrmsd.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nWrote {out_csv}")

    # ---------- Per-(arm, L) summary ----------
    print("\n" + "=" * 70)
    print(f"Move-2 sparse-trunk paired scRMSD eval ({args.label})")
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
            print(f"  {arm:>8} {L:>3}  {n:>2}    {d}/{n}             "
                  f"{sub.ca_esmfold.min():.2f}    {sub.ca_esmfold.median():.2f}      "
                  f"{sub.ca_esmfold.mean():.2f}")

    # ---------- Pooled ----------
    print("\nPooled across all lengths:")
    for arm in arms:
        sub = df[(df.arm == arm) & df.ca_esmfold.notna()]
        n = len(sub)
        if n == 0:
            continue
        d = int((sub.ca_esmfold < 2.0).sum())
        print(f"  {arm:>8}: {d}/{n} = {100*d/n:.1f}% designable, "
              f"best {sub.ca_esmfold.min():.2f} Å, median {sub.ca_esmfold.median():.2f} Å")

    # ---------- Paired Δ (sparse − baseline) per seed ----------
    print("\nPaired Δ scRMSD_ca_esmfold (sparse − baseline) per (L, sample_idx):")
    pivot = df.pivot_table(index=["L", "sample_idx"], columns="arm", values="ca_esmfold")
    if "baseline" in pivot.columns and "sparse" in pivot.columns:
        pivot["delta"] = pivot["sparse"] - pivot["baseline"]
        pivot["winner"] = pivot.apply(
            lambda r: "sparse" if r["delta"] < -0.1
            else ("baseline" if r["delta"] > 0.1 else "tie"), axis=1
        )
        print(pivot[["baseline", "sparse", "delta", "winner"]].round(3).to_string())
        valid = pivot.dropna(subset=["delta"])
        print(f"\nNet over {len(valid)} paired samples:")
        print(f"  sparse wins   (Δ<-0.1):  {(valid.delta < -0.1).sum()}")
        print(f"  ties (|Δ|≤0.1):          {((valid.delta >= -0.1) & (valid.delta <= 0.1)).sum()}")
        print(f"  baseline wins (Δ>+0.1):  {(valid.delta > 0.1).sum()}")
        print(f"  mean Δ:   {valid.delta.mean():+.3f} Å")
        print(f"  median Δ: {valid.delta.median():+.3f} Å")


if __name__ == "__main__":
    main()
