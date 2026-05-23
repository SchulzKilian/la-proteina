"""Codesign-extend seeds 46-51 (3 lengths each) for the noise-aware-ensemble
sweep. Appends to existing codesign_guided.csv per cell, idempotent.

Usage:
  python scripts/codesign_extend_seeds_46_51.py --cells camsol_max_w1 ...
  python scripts/codesign_extend_seeds_46_51.py --cells tango_min_w1 ...

Used to scale n=12 (seeds 42-45) → n=30 (seeds 42-51) per cell.
"""
from __future__ import annotations
import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_extend")

SEEDS = [46, 47, 48, 49, 50, 51]
LENGTHS = [300, 400, 500]
SWEEP_ROOT = _ROOT / "results/noise_aware_ensemble_sweep"


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


def already_done(csv_path: Path) -> set[str]:
    """Read existing codesign_guided.csv and return set of done protein_ids."""
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                done.add(row[0])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="Cell names like camsol_max_w1 tango_min_w16")
    args = ap.parse_args()

    tmp_root = _ROOT / f"tmp/codesign_extend_{Path(sys.argv[0]).stem}_{time.time():.0f}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    overall = []
    for cell in args.cells:
        guided_dir = SWEEP_ROOT / cell / "guided"
        csv_path = SWEEP_ROOT / cell / "codesign_guided.csv"
        if not guided_dir.exists():
            logger.error("Missing guided/ for %s, skipping", cell)
            continue

        done = already_done(csv_path)
        targets = [(seed, L) for seed in SEEDS for L in LENGTHS]
        targets = [(s, L) for s, L in targets
                   if f"s{s}_n{L}" not in done]
        if not targets:
            logger.info("[%s] all 18 already done, skipping", cell)
            continue

        logger.info("[%s] %d new codesigns to run", cell, len(targets))
        # Append mode (header only if file is new)
        write_header = not csv_path.exists()
        f_out = csv_path.open("a", newline="")
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["protein_id", "coScRMSD_ca"])
        f_out.flush()

        for seed, L in targets:
            pid = f"s{seed}_n{L}"
            pdb = guided_dir / f"{pid}.pdb"
            if not pdb.exists():
                logger.warning("[%s] missing PDB %s, skipping", cell, pid)
                continue
            t0 = time.time()
            rmsd = evaluate_one(pdb, tmp_root)
            dt = time.time() - t0
            logger.info("[%s] %s -> coScRMSD=%.3f Å (%.1fs)", cell, pid, rmsd, dt)
            writer.writerow([pid, f"{rmsd:.4f}"])
            f_out.flush()
            overall.append({"cell": cell, "pid": pid, "rmsd": rmsd})
        f_out.close()

    print(f"\nTotal codesigns: {len(overall)}")
    if overall:
        n_under_2 = sum(1 for r in overall if r["rmsd"] < 2.0)
        print(f"Under 2 Å: {n_under_2}/{len(overall)} = {n_under_2/len(overall):.0%}")


if __name__ == "__main__":
    main()
