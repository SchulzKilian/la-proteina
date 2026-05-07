"""Codesign on the matched-seed apples-to-apples unsteered run.

Same seeds (42-45), same lengths (300/400/500), same nsteps=400, same
inference_ucond_notri_long config as the steered cells in
results/noise_aware_ensemble_sweep/, but with model.steering_guide=None.

PDBs are at results/sanity_unsteered_seed42_45/unguided/*.pdb (12 PDBs).
"""
from __future__ import annotations
import logging, shutil, sys, time
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_matched_seed")

UNGUIDED_DIR = _ROOT / "results/sanity_unsteered_seed42_45/unguided"
OUT_CSV = _ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv"


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
    pdbs = sorted(UNGUIDED_DIR.glob("*.pdb"))
    print(f"Codesign on {len(pdbs)} unguided matched-seed PDBs")
    tmp_root = _ROOT / "tmp" / "codesign_matched_seed"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for pdb in pdbs:
        seed = int(pdb.stem.split("_")[0][1:])
        L = int(pdb.stem.split("_n")[-1])
        t0 = time.time()
        rmsd = evaluate_one(pdb, tmp_root)
        dt = time.time() - t0
        logger.info("%s -> coScRMSD_ca=%.3f Å (%.1fs)", pdb.stem, rmsd, dt)
        rows.append({"protein_id": pdb.stem, "seed": seed, "length": L,
                     "coScRMSD_ca": rmsd})

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")
    print(out.to_string(index=False))
    print()
    print(f"Overall codesign rate (<2 Å): "
          f"{(out.coScRMSD_ca < 2).sum()}/{len(out)} = "
          f"{(out.coScRMSD_ca < 2).mean():.0%}")
    print(f"By length:")
    print(out.groupby("length").agg(
        n=("coScRMSD_ca", "count"),
        n_codes=("coScRMSD_ca", lambda s: int((s < 2).sum())),
        rate_2=("coScRMSD_ca", lambda s: round((s < 2).mean(), 2)),
        rate_3=("coScRMSD_ca", lambda s: round((s < 3).mean(), 2)),
        rate_4=("coScRMSD_ca", lambda s: round((s < 4).mean(), 2)),
        mean=("coScRMSD_ca", "mean"),
        med=("coScRMSD_ca", "median"),
    ).round(2))


if __name__ == "__main__":
    main()
