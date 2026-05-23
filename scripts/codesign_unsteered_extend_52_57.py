"""Extend the matched-seed unsteered codesign to seeds 52-57.

Reads from results/sanity_unsteered_seed42_45/unguided/ (the dir is misnamed
historically — it now holds seeds 42-57 unsteered PDBs once the 52-57
extension generation has dropped its outputs there). Appends rows to
results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv.
Idempotent (skips already-done protein_ids).
"""
from __future__ import annotations
import logging, shutil, sys, time
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_unsteered_ext")

UNGUIDED_DIR = _ROOT / "results/sanity_unsteered_seed42_45/unguided"
OUT_CSV = _ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv"
SEEDS = [52, 53, 54, 55, 56, 57]
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
        new_df = pd.DataFrame(new_rows)
        full = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        full.to_csv(OUT_CSV, index=False)
        logger.info("Appended %d rows; CSV now %d total.", len(new_rows), len(full))
    else:
        logger.info("Nothing new to append.")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
