"""w=0 sanity check: run the same codesignability pipeline on length-matched
UNSTEERED PDBs from `generated_stratified_300_800_nsteps400/`.

Same canonical inference config (`inference_ucond_notri_long`, nsteps=400, SDE),
same codesign call (use_pdb_seq=True, num_seq=1, ESMFold + CA-RMSD), so the
codesign rate on these is the *base rate* the steered cells in E042 should be
compared against.
"""
from __future__ import annotations
import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_unsteered")

BASELINE_DIR = _ROOT / "results/generated_stratified_300_800_nsteps400"
OUT_CSV = _ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_baseline.csv"


def evaluate_one(pdb_path: Path, tmp_root: Path) -> float:
    name = pdb_path.stem
    tmp_dir = tmp_root / name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    res = scRMSD(
        pdb_file_path=str(pdb_path),
        tmp_path=str(tmp_dir),
        num_seq_per_target=1,
        use_pdb_seq=True,
        rmsd_modes=["ca"],
        folding_models=["esmfold"],
        keep_outputs=True,
        ret_min=False,
    )
    rmsds = res["ca"]["esmfold"]
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return rmsds[0] if rmsds else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[300, 400, 500])
    ap.add_argument("--per-target", type=int, default=4)
    ap.add_argument("--tol", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(BASELINE_DIR / "manifest.csv")
    chosen = []
    for t in args.targets:
        sub = df[(df["length"] >= t - args.tol) & (df["length"] <= t + args.tol)].head(args.per_target)
        chosen.append(sub.assign(target=t))
    pick = pd.concat(chosen).reset_index(drop=True)
    print(pick[["protein_id", "seed", "length", "target"]].to_string(index=False))

    tmp_root = _ROOT / "tmp" / "codesign_unsteered"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in pick.iterrows():
        pdb = BASELINE_DIR / r["pdb"]
        t0 = time.time()
        rmsd = evaluate_one(pdb, tmp_root)
        dt = time.time() - t0
        logger.info("%s (L=%d, target~%d) -> coScRMSD_ca=%.3f Å (%.1fs)",
                    r["protein_id"], r["length"], r["target"], rmsd, dt)
        rows.append({"protein_id": r["protein_id"],
                     "seed": int(r["seed"]),
                     "length": int(r["length"]),
                     "target": int(r["target"]),
                     "coScRMSD_ca": rmsd})

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")
    print(f"\nOverall codesign rate (coScRMSD_ca < 2 Å): "
          f"{(out.coScRMSD_ca < 2.0).sum()}/{len(out)} = "
          f"{(out.coScRMSD_ca < 2.0).mean():.0%}")
    print(f"Per-target rate:")
    print(out.groupby("target").agg(
        n=("coScRMSD_ca", "count"),
        n_codes=("coScRMSD_ca", lambda s: int((s < 2.0).sum())),
        rate=("coScRMSD_ca", lambda s: round((s < 2.0).mean(), 2)),
        mean=("coScRMSD_ca", "mean"),
        med=("coScRMSD_ca", "median"),
    ).round(2).to_string())


if __name__ == "__main__":
    main()
