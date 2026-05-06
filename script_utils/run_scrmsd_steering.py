"""scRMSD on the nsteps=400 steering sweep using the OFFICIAL scRMSD() pipeline.

Subsample: lengths from --lengths CLI flag (default 300), 4 seeds per length.
Iterates the 10 configs sequentially. Per-config CSV at
results/steering_camsol_tango_L500_nsteps400/{config}/scRMSD_guided.csv.

Resumes safely: if the CSV already has a row for a protein_id, that protein
is skipped.

Why this script and not the custom one from yesterday: the custom path gave
30 A scRMSD on what we now know are designable structures (after the
nsteps=100 -> nsteps=400 fix). The official scRMSD() returns sub-1 A on the
same PDBs. Differences must be in the ProteinMPNN / ESMFold piping; not
worth debugging when the official function works as long as we pass
keep_outputs=True (otherwise it deletes the ESM PDBs before reading them).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("scrmsd_steering")

_OUT_BASE_ENV = os.environ.get("OUT_BASE")
OUT_BASE = _ROOT / _OUT_BASE_ENV if _OUT_BASE_ENV else _ROOT / "results" / "steering_camsol_tango_L500_nsteps400"
CFGS = [
    "camsol_max_w1", "camsol_max_w2", "camsol_max_w4", "camsol_max_w8", "camsol_max_w16",
    "tango_min_w1", "tango_min_w2", "tango_min_w4", "tango_min_w8", "tango_min_w16",
]


def evaluate_one(pdb_path: Path, tmp_root: Path) -> dict:
    """Run official scRMSD with N=8, keep_outputs to avoid the cleanup-bug, mode=ca."""
    name = pdb_path.stem
    tmp_dir = tmp_root / name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    res = scRMSD(
        pdb_file_path=str(pdb_path),
        tmp_path=str(tmp_dir),
        num_seq_per_target=8,
        use_pdb_seq=False,
        rmsd_modes=["ca"],
        folding_models=["esmfold"],
        keep_outputs=True,
        ret_min=False,
    )
    rmsds = res["ca"]["esmfold"]
    # Cleanup tmp once we have the numbers
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "protein_id": name,
        "scRMSD_ca_min": min(rmsds) if rmsds else float("inf"),
        "scRMSD_ca_all": rmsds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--csv-suffix", type=str, default="",
                    help="Optional suffix on the per-config CSV filename — useful if "
                         "you want to keep multiple sweeps separate (e.g. _L300, _L400).")
    args = ap.parse_args()

    tmp_root = _ROOT / "tmp" / "scrmsd_steering"
    tmp_root.mkdir(parents=True, exist_ok=True)

    pdb_set: list[tuple[str, Path]] = []  # (cfg, pdb_path)
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
    logger.info("scRMSD over %d PDBs (lengths=%s seeds=%s)", grand_total, args.lengths, args.seeds)

    grand_done = 0
    t_start = time.time()

    csv_name = f"scRMSD_guided{args.csv_suffix}.csv"

    # Group by config so we open one CSV per config
    by_cfg: dict[str, list[Path]] = {}
    for cfg, pdb in pdb_set:
        by_cfg.setdefault(cfg, []).append(pdb)

    for cfg, pdbs in by_cfg.items():
        out_csv = OUT_BASE / cfg / csv_name

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
                w.writerow(["protein_id", "scRMSD_ca_min", "scRMSD_ca_all"])

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
                        "[%s] %s scRMSD_min=%.2fA  (%d/%d total, ETA %.0fmin)",
                        cfg, name, res["scRMSD_ca_min"],
                        grand_done, grand_total, remaining / 60,
                    )
                    w.writerow([
                        name,
                        f"{res['scRMSD_ca_min']:.4f}",
                        ";".join(f"{x:.4f}" for x in res["scRMSD_ca_all"]),
                    ])
                    fh.flush()
                except Exception as e:
                    logger.exception("FAILED on %s: %s", name, e)
                    w.writerow([name, "inf", "inf"])
                    fh.flush()

    logger.info("scRMSD pass complete. Total wall: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
