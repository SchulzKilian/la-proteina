"""GPU-side probe of the CA-only designability eval pipeline.

For each of two real native PDBs (5L33 109aa, 6MRR 71aa), build three test
inputs and run the full ProteinMPNN -> ESMFold -> RMSD pipeline on each:

  A. RECON-vanillaMPNN: native PDB -> strip to CA -> ca_to_backbone_atom37
     reconstruction -> save -> run scRMSD as the codebase currently does
     (run_proteinmpnn ca_only=False, vanilla weights). This mirrors what
     evaluate.py does on a CA-only generation output.
  B. BARECA-caMPNN: native PDB -> strip to CA only (no fake N/C/O) -> save
     -> run a parallel scRMSD path with run_proteinmpnn ca_only=True
     (CA-only ProteinMPNN weights). Reference: how a clean CA-only design
     pipeline is supposed to work.
  C. NATIVE-vanillaMPNN: native PDB unchanged -> run scRMSD with
     run_proteinmpnn ca_only=False. Sanity check: the rest of the
     pipeline (ProteinMPNN, ESMFold, RMSD) is healthy on real backbones.

Per-condition output: best CA-RMSD across N MPNN sequences (lower = more
designable). Native should be ~< 2 A. The gap A vs B vs C tells us how
much of any "0/3 designable" result is the eval recipe vs the model.

Run from repo root with the laproteina env on PATH:
    python script_utils/probe_ca_eval/run_ca_eval_probe.py
"""
import os
import shutil
import sys
import time
import traceback

ROOT = "/home/ks2218/la-proteina"
sys.path.insert(0, ROOT)

import numpy as np
import torch
from loguru import logger

from proteinfoundation.metrics.designability import (
    extract_seq_from_pdb,
    rmsd_metric,
    run_proteinmpnn,
)
from proteinfoundation.metrics.folding_models import run_esmfold
from proteinfoundation.utils.coors_utils import ca_to_backbone_atom37
from proteinfoundation.utils.pdb_utils import (
    create_full_prot,
    load_pdb,
    to_pdb,
    write_prot_to_pdb,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_pdb_recon(native_path: str, out_path: str):
    """A: reconstruct N/C/O from CA via ca_to_backbone_atom37."""
    prot = load_pdb(native_path)
    atom37 = prot.atom_positions
    mask = prot.atom_mask
    keep = (mask[:, 0] > 0.5) & (mask[:, 1] > 0.5) & (mask[:, 2] > 0.5)
    atom37 = atom37[keep]
    aatype = prot.aatype[keep]
    ca = torch.from_numpy(atom37[:, 1, :]).float()
    recon = ca_to_backbone_atom37(ca).numpy()  # [n, 37, 3] in A
    write_prot_to_pdb(
        prot_pos=recon.astype(np.float32),
        aatype=aatype,
        file_path=out_path,
        overwrite=True,
        no_indexing=True,
    )
    return aatype


def make_pdb_bareca(native_path: str, out_path: str):
    """B: CA-only PDB. Only CA position is non-zero so write_prot_to_pdb
    masks the rest out automatically."""
    prot = load_pdb(native_path)
    atom37 = prot.atom_positions
    mask = prot.atom_mask
    keep = mask[:, 1] > 0.5  # any residue with a CA
    atom37 = atom37[keep]
    aatype = prot.aatype[keep]
    bare = np.zeros_like(atom37)
    bare[:, 1, :] = atom37[:, 1, :]  # only CA
    write_prot_to_pdb(
        prot_pos=bare.astype(np.float32),
        aatype=aatype,
        file_path=out_path,
        overwrite=True,
        no_indexing=True,
    )
    return aatype


def copy_native(native_path: str, out_path: str):
    """C: pass-through copy of the native PDB."""
    shutil.copy(native_path, out_path)
    return None


def run_pipeline(pdb_path: str, tmp_dir: str, ca_only_flag: bool, num_seq: int = 8):
    """Run ProteinMPNN -> ESMFold -> CA-RMSD for one PDB. Return list of CA-RMSDs."""
    os.makedirs(tmp_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(pdb_path))[0]

    t0 = time.time()
    seqs_meta = run_proteinmpnn(
        pdb_path,
        tmp_dir,
        num_seq_per_target=num_seq,
        sampling_temp=0.1,
        ca_only=ca_only_flag,
        verbose=False,
    )
    seqs = [s["seq"] for s in seqs_meta]
    t_mpnn = time.time() - t0
    logger.info(f"  MPNN done: {len(seqs)} seqs in {t_mpnn:.1f}s")

    fold_dir = os.path.join(tmp_dir, "esmfold_output")
    os.makedirs(fold_dir, exist_ok=True)
    t0 = time.time()
    fold_paths = run_esmfold(seqs, fold_dir, name, suffix="mpnn", keep_outputs=False)
    t_fold = time.time() - t0
    logger.info(f"  ESMFold done in {t_fold:.1f}s ({len(fold_paths)} structures)")

    # Compare folded vs original CA trace
    gen = load_pdb(pdb_path)
    gen_coors = torch.tensor(gen.atom_positions)
    gen_mask = torch.tensor(gen.atom_mask).bool()

    rmsds = []
    for fp in fold_paths:
        if fp is None:
            rmsds.append(float("inf"))
            continue
        rec = load_pdb(fp)
        rec_coors = torch.tensor(rec.atom_positions)
        rec_mask = torch.tensor(rec.atom_mask).bool()
        m = gen_mask * rec_mask
        rmsds.append(rmsd_metric(gen_coors, rec_coors, mask_atom_37=m, mode="ca"))

    return rmsds, t_mpnn, t_fold


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    out_root = os.path.join(ROOT, "script_utils", "probe_ca_eval", "outputs")
    os.makedirs(out_root, exist_ok=True)

    # Set ProteinMPNN's PYTHON_EXEC env var so the subprocess uses the same env.
    os.environ.setdefault("PYTHON_EXEC", sys.executable)

    natives = [
        ("5L33", f"{ROOT}/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"),
        ("6MRR", f"{ROOT}/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb"),
    ]

    summary = []

    for pdb_id, native_path in natives:
        logger.info(f"\n========== {pdb_id}  ({native_path}) ==========")

        # Build the three test PDBs
        case_dir = os.path.join(out_root, pdb_id)
        os.makedirs(case_dir, exist_ok=True)

        recon_pdb = os.path.join(case_dir, f"{pdb_id}_recon.pdb")
        bareca_pdb = os.path.join(case_dir, f"{pdb_id}_bareca.pdb")
        native_pdb = os.path.join(case_dir, f"{pdb_id}_native.pdb")

        try:
            make_pdb_recon(native_path, recon_pdb)
            make_pdb_bareca(native_path, bareca_pdb)
            copy_native(native_path, native_pdb)
        except Exception as e:
            logger.error(f"  PDB build failed for {pdb_id}: {e}")
            traceback.print_exc()
            continue

        for label, pdb, ca_only_flag in [
            ("A_RECON_vanillaMPNN", recon_pdb, False),
            ("B_BARECA_caMPNN", bareca_pdb, True),
            ("C_NATIVE_vanillaMPNN", native_pdb, False),
        ]:
            logger.info(f"--- {label} on {pdb_id} ---")
            tmp = os.path.join(case_dir, label)
            try:
                rmsds, t_mpnn, t_fold = run_pipeline(pdb, tmp, ca_only_flag)
                rmsds_arr = np.array([r for r in rmsds if np.isfinite(r)])
                if len(rmsds_arr) == 0:
                    best, mean = float("inf"), float("inf")
                else:
                    best, mean = float(rmsds_arr.min()), float(rmsds_arr.mean())
                logger.info(f"  ==> {label}: min CA-RMSD={best:.2f} A, mean={mean:.2f} A")
                summary.append({
                    "pdb_id": pdb_id, "label": label,
                    "min_rmsd": best, "mean_rmsd": mean,
                    "n_ok": int(len(rmsds_arr)),
                    "all_rmsds": [round(float(r), 3) for r in rmsds],
                    "t_mpnn": round(t_mpnn, 1), "t_fold": round(t_fold, 1),
                })
            except Exception as e:
                logger.error(f"  {label} failed: {e}")
                traceback.print_exc()
                summary.append({
                    "pdb_id": pdb_id, "label": label,
                    "min_rmsd": float("inf"), "error": str(e),
                })

    # Print final table
    logger.info("\n\n========== SUMMARY ==========")
    logger.info(f"{'pdb_id':<8}{'label':<26}{'min_rmsd':>10}{'mean_rmsd':>11}{'n_ok':>5}")
    for r in summary:
        logger.info(
            f"{r['pdb_id']:<8}"
            f"{r['label']:<26}"
            f"{r.get('min_rmsd', float('inf')):>10.2f}"
            f"{r.get('mean_rmsd', float('inf')):>11.2f}"
            f"{r.get('n_ok', 0):>5}"
        )

    # Save full summary as JSON for later
    import json
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved {os.path.join(out_root, 'summary.json')}")


if __name__ == "__main__":
    main()
