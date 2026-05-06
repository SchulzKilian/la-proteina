"""Codesignability across the noise-aware-ensemble sweep.

For each (direction, w, L) cell, take the model's joint-head sequence + backbone
from each PDB and ask: does ESMFold fold the sequence onto the backbone?

This is the test that matters for steering, since the steering perturbs the
sequence-determining latent while leaving the backbone near-frozen — the
standard MPNN-redesigns-the-sequence designability check is largely insensitive
to what steering did.

Output: codesign_guided.csv per cell (next to the existing scRMSD_guided.csv).
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from proteinfoundation.metrics.designability import scRMSD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codesign_sweep")

OUT_BASE = _ROOT / os.environ.get("OUT_BASE", "results/noise_aware_ensemble_sweep")
CFGS = [
    "camsol_max_w1", "camsol_max_w2", "camsol_max_w4", "camsol_max_w8", "camsol_max_w16",
    "tango_min_w1", "tango_min_w2", "tango_min_w4", "tango_min_w8", "tango_min_w16",
]


def evaluate_one(pdb_path: Path, tmp_root: Path) -> dict:
    """use_pdb_seq=True: fold the protein's own joint-head sequence with ESMFold."""
    name = pdb_path.stem
    tmp_dir = tmp_root / name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    res = scRMSD(
        pdb_file_path=str(pdb_path),
        tmp_path=str(tmp_dir),
        num_seq_per_target=1,            # only the joint-head's sequence
        use_pdb_seq=True,                 # codesignability mode
        rmsd_modes=["ca"],
        folding_models=["esmfold"],
        keep_outputs=True,
        ret_min=False,
    )
    rmsds = res["ca"]["esmfold"]
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"protein_id": name,
            "coScRMSD_ca": rmsds[0] if rmsds else float("inf")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[300, 400, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = ap.parse_args()

    tmp_root = _ROOT / "tmp" / "codesign_sweep"
    tmp_root.mkdir(parents=True, exist_ok=True)

    pdb_set: list[tuple[str, Path]] = []
    for cfg in CFGS:
        guided_dir = OUT_BASE / cfg / "guided"
        if not guided_dir.exists():
            logger.warning("Missing %s, skipping", guided_dir)
            continue
        for seed in args.seeds:
            for L in args.lengths:
                pdb = guided_dir / f"s{seed}_n{L}.pdb"
                if pdb.exists():
                    pdb_set.append((cfg, pdb))
                else:
                    logger.warning("Missing %s", pdb)

    grand_total = len(pdb_set)
    logger.info("Codesignability over %d PDBs (lengths=%s seeds=%s)",
                grand_total, args.lengths, args.seeds)

    grand_done = 0
    t_start = time.time()

    by_cfg: dict[str, list[Path]] = {}
    for cfg, pdb in pdb_set:
        by_cfg.setdefault(cfg, []).append(pdb)

    for cfg, pdbs in by_cfg.items():
        out_csv = OUT_BASE / cfg / "codesign_guided.csv"
        done_ids: set[str] = set()
        if out_csv.exists():
            with open(out_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(row["protein_id"])

        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(["protein_id", "coScRMSD_ca"])

            for pdb in pdbs:
                name = pdb.stem
                if name in done_ids:
                    grand_done += 1
                    continue
                try:
                    res = evaluate_one(pdb, tmp_root)
                    grand_done += 1
                    elapsed = time.time() - t_start
                    rate = grand_done / max(elapsed, 1.0)
                    remaining = (grand_total - grand_done) / max(rate, 1e-6)
                    logger.info(
                        "[%s] %s coScRMSD=%.2fA  (%d/%d, ETA %.0fmin)",
                        cfg, name, res["coScRMSD_ca"],
                        grand_done, grand_total, remaining / 60,
                    )
                    w.writerow([name, f"{res['coScRMSD_ca']:.4f}"])
                    fh.flush()
                except Exception as e:
                    logger.exception("FAILED on %s: %s", name, e)
                    w.writerow([name, "inf"])
                    fh.flush()

    logger.info("Codesign pass complete. Total wall: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
