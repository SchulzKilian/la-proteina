"""Designability (scRMSD via ProteinMPNN -> ESMFold -> CA-RMSD) on the steering sweep.

Walks results/steering_camsol_tango_L500/{config}/guided/*.pdb across all 10
configs. ESMFold weights are loaded ONCE at startup and reused for all 480
proteins; ProteinMPNN is invoked per PDB via the existing wrapper.

Per-config CSV (`scRMSD_guided.csv`) with columns:
    protein_id, length, scRMSD_ca_min, scRMSD_ca_all, n_seqs

Single GPU (cuda:0). ProteinMPNN uses ca_only=True, matching the canonical
La-Proteina post-fix designability convention (E017 / E018).
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from proteinfoundation.metrics.designability import (
    run_proteinmpnn,
    pdb_name_from_path,
    rmsd_metric,
)
from proteinfoundation.metrics.folding_models import _convert_esm_outputs_to_pdb
from proteinfoundation.utils.pdb_utils import load_pdb

from transformers import AutoTokenizer, EsmForProteinFolding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("designability_steering")

OUT_BASE = _ROOT / "results" / "steering_camsol_tango_L500"
CFGS = [
    "camsol_max_w1", "camsol_max_w2", "camsol_max_w4", "camsol_max_w8", "camsol_max_w16",
    "tango_min_w1", "tango_min_w2", "tango_min_w4", "tango_min_w8", "tango_min_w16",
]
NUM_SEQ = 4         # 4 ProteinMPNN seqs per structure (halves wall vs paper-default 8)
SAMPLING_TEMP = 0.1
DEVICE = "cuda:0"


def fold_with_persistent_esmfold(
    sequences: List[str],
    tokenizer,
    esm_model,
) -> List[str]:
    """Run ESMFold on a list of sequences using an already-loaded model.

    Returns a list of PDB strings (same order as input). Batches into chunks of
    1 when max-length is large to avoid OOM at L=500.
    """
    pdb_strings: List[str] = []
    if not sequences:
        return pdb_strings

    max_nres = max(len(s) for s in sequences)
    if max_nres > 450:
        batch_size = 1
    elif max_nres > 350:
        batch_size = 2
    else:
        batch_size = 4

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=False, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = esm_model(**inputs)
        pdb_strings.extend(_convert_esm_outputs_to_pdb(outputs))
    return pdb_strings


def ca_rmsd(gen_coors: torch.Tensor, gen_mask: torch.Tensor,
            rec_coors: torch.Tensor, rec_mask: torch.Tensor) -> float:
    """Kabsch-aligned CA-RMSD in Angstroms. Inputs are atom37 [L,37,3]."""
    if gen_coors.shape != rec_coors.shape:
        # ESMFold may return a different residue count; rmsd_metric requires
        # matched shapes — clip the longer one to the min length.
        L = min(gen_coors.shape[0], rec_coors.shape[0])
        gen_coors = gen_coors[:L]
        gen_mask = gen_mask[:L]
        rec_coors = rec_coors[:L]
        rec_mask = rec_mask[:L]
    return rmsd_metric(
        coors_1_atom37=gen_coors,
        coors_2_atom37=rec_coors,
        mask_atom_37=gen_mask & rec_mask,
        mode="ca",
        align=True,
    )


def evaluate_pdb(pdb_path: str, tmp_dir: str, tokenizer, esm_model) -> dict:
    """Run ProteinMPNN -> ESMFold -> CA-RMSD on a single PDB."""
    name = pdb_name_from_path(pdb_path)
    pdb_tmp = Path(tmp_dir) / name
    pdb_tmp.mkdir(parents=True, exist_ok=True)

    # 1) ProteinMPNN (ca_only=True, 8 sequences)
    seqs = run_proteinmpnn(
        pdb_file_path=pdb_path,
        out_dir_root=str(pdb_tmp),
        num_seq_per_target=NUM_SEQ,
        sampling_temp=SAMPLING_TEMP,
        ca_only=True,
        verbose=False,
    )
    seq_strs = [s["seq"] for s in seqs]

    # 2) ESMFold all 8 sequences, persistent model
    pdb_strings = fold_with_persistent_esmfold(seq_strs, tokenizer, esm_model)

    # 3) Load gen structure once
    gen_prot = load_pdb(pdb_path)
    gen_coors = torch.tensor(gen_prot.atom_positions, dtype=torch.float32)
    gen_mask = torch.tensor(gen_prot.atom_mask).bool()
    L = gen_coors.shape[0]

    # 4) RMSD vs each folded sequence
    rmsds = []
    for k, pdb_str in enumerate(pdb_strings):
        out_pdb_tmp = pdb_tmp / f"esm_{k}.pdb"
        out_pdb_tmp.write_text(pdb_str)
        try:
            rec_prot = load_pdb(str(out_pdb_tmp))
            rec_coors = torch.tensor(rec_prot.atom_positions, dtype=torch.float32)
            rec_mask = torch.tensor(rec_prot.atom_mask).bool()
            rmsds.append(ca_rmsd(gen_coors, gen_mask, rec_coors, rec_mask))
        except Exception as e:
            logger.warning("RMSD failed for %s seq %d: %s", name, k, e)
            rmsds.append(float("inf"))

    # Cleanup tmp
    try:
        shutil.rmtree(pdb_tmp)
    except Exception:
        pass

    return {
        "protein_id": name,
        "length": L,
        "scRMSD_ca_min": min(rmsds) if rmsds else float("inf"),
        "scRMSD_ca_all": rmsds,
        "n_seqs": len(rmsds),
    }


def main():
    tmp_root = _ROOT / "tmp" / "designability_steering"
    tmp_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ESMFold (one-time)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(DEVICE)
    esm_model.eval()
    logger.info("ESMFold ready in %.1fs", time.time() - t0)

    grand_total = 0
    grand_done = 0
    for cfg in CFGS:
        guided_dir = OUT_BASE / cfg / "guided"
        if not guided_dir.exists():
            logger.warning("Missing %s, skipping", guided_dir)
            continue
        grand_total += len(list(guided_dir.glob("*.pdb")))

    t_start = time.time()
    for ci, cfg in enumerate(CFGS, start=1):
        guided_dir = OUT_BASE / cfg / "guided"
        out_csv = OUT_BASE / cfg / "scRMSD_guided.csv"
        if not guided_dir.exists():
            continue

        pdbs = sorted(guided_dir.glob("*.pdb"))
        logger.info("[%d/%d] %s — %d PDBs", ci, len(CFGS), cfg, len(pdbs))

        # Resume-safe: if CSV already has all rows, skip
        if out_csv.exists():
            done_ids = set()
            with open(out_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(row["protein_id"])
            if len(done_ids) >= len(pdbs):
                logger.info("  already complete (%d rows), skipping", len(done_ids))
                grand_done += len(pdbs)
                continue
        else:
            done_ids = set()

        # Append mode if some rows exist already
        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["protein_id", "length", "scRMSD_ca_min", "scRMSD_ca_all_min2"])

            for i, pdb_path in enumerate(pdbs, start=1):
                name = pdb_name_from_path(str(pdb_path))
                if name in done_ids:
                    grand_done += 1
                    continue
                tmp_dir = str(tmp_root / cfg)
                Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                try:
                    res = evaluate_pdb(str(pdb_path), tmp_dir, tokenizer, esm_model)
                    grand_done += 1
                    elapsed = time.time() - t_start
                    rate = grand_done / max(elapsed, 1.0)
                    remaining = (grand_total - grand_done) / max(rate, 1e-6)
                    if i % 4 == 0 or i == len(pdbs):
                        logger.info(
                            "    %s: scRMSD_ca_min=%.3fA  (%d/%d in cfg, %d/%d total, ETA %.0fmin)",
                            res["protein_id"], res["scRMSD_ca_min"],
                            i, len(pdbs), grand_done, grand_total, remaining / 60,
                        )
                    rmsds_sorted = sorted([float(x) for x in res["scRMSD_ca_all"]])
                    second_best = rmsds_sorted[1] if len(rmsds_sorted) > 1 else rmsds_sorted[0] if rmsds_sorted else float("inf")
                    writer.writerow([
                        res["protein_id"], res["length"],
                        f"{res['scRMSD_ca_min']:.4f}",
                        f"{second_best:.4f}",
                    ])
                    fh.flush()
                except Exception as e:
                    logger.exception("FAILED on %s: %s", name, e)
                    writer.writerow([name, -1, "inf", "inf"])
                    fh.flush()

        logger.info("  %s done -> %s", cfg, out_csv)

    logger.info("Sweep designability complete. Total wall: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
