"""Extend the matched-seed unsteered codesign to seeds 46-51.

Reads from results/sanity_unsteered_seed42_45/unguided/ and appends to
results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv.
Idempotent (skips already-done protein_ids).
"""
from __future__ import annotations
import csv, logging, shutil, sys, time
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_unsteered_ext")

UNGUIDED_DIR = _ROOT / "results/sanity_unsteered_seed42_45/unguided"
OUT_CSV = _ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv"
SEEDS = [46, 47, 48, 49, 50, 51]
LENGTHS = [300, 400, 500]


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
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        done = set(existing["protein_id"].tolist())
    else:
        existing = pd.DataFrame()
        done = set()

    targets = [(s, L) for s in SEEDS for L in LENGTHS
               if f"s{s}_n{L}" not in done]
    logger.info("%d new unsteered codesigns to run", len(targets))

    tmp_root = _ROOT / f"tmp/codesign_unsteered_ext_{int(time.time())}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    new_rows = []
    for seed, L in targets:
        pid = f"s{seed}_n{L}"
        pdb = UNGUIDED_DIR / f"{pid}.pdb"
        if not pdb.exists():
            logger.warning("Missing PDB %s, skipping", pid)
            continue
        t0 = time.time()
        rmsd = evaluate_one(pdb, tmp_root)
        dt = time.time() - t0
        logger.info("%s -> coScRMSD=%.3f Å (%.1fs)", pid, rmsd, dt)
        new_rows.append({"protein_id": pid, "seed": seed, "length": L,
                         "coScRMSD_ca": rmsd})

    if new_rows:
        # Merge with existing CSV preserving the same columns
        new_df = pd.DataFrame(new_rows)
        if not existing.empty:
            full = pd.concat([existing, new_df], ignore_index=True)
        else:
            full = new_df
        full.to_csv(OUT_CSV, index=False)
        print(f"Appended {len(new_rows)} rows; total now {len(full)}")
        print(f"<2 Å: {(full['coScRMSD_ca'] < 2).sum()}/{len(full)} = {(full['coScRMSD_ca'] < 2).mean():.0%}")


if __name__ == "__main__":
    main()
