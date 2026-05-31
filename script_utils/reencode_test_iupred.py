#!/usr/bin/env python
"""
RE-ENCODE TEST (draft, for review — run on Cambridge HPC with laproteina_env).

Chain:
  generated full-atom PDB
    -> graphein protein_to_pyg  (atom37-like coords, residues)
    -> PDB_TO_OPENFOLD_INDEX_TENSOR reindex + coord_mask + residue_type
    -> CoordsToNanometers + CenterStructureTransform (NO rotation)
    -> AE encoder  ->  mean[L,8]   (on-manifold latent of the ACTUAL structure)
    -> SteeringPredictor (5-fold NA-v1 ensemble, t=1) -> iupred3_fraction_disordered

Then compare predicted_disorder (on re-encoded latent) vs real disorder from
properties_guided.csv, per protein + Pearson + means.

IMPORTANT — AE choice:
  The steering PREDICTOR (multitask_t1_noise_aware/20260505_110348) was trained on
  latents in data/pdb_train/processed_latents_300_800, which precompute_latents.py
  produced with **AE1_ucond_512.ckpt** (verified in slurm_precompute_27754627.out:
  "Loading AutoEncoder from .../AE1_ucond_512.ckpt").
  So to feed the predictor latents from the SAME encoder distribution it learned,
  re-encode with AE1_ucond_512.ckpt (default below).

  NOTE the mismatch: the steering SAMPLER (inference_ucond_notri_long) runs the
  flow on AE2_ucond_800 latents. The predictor scoring AE2 latents is the live
  pipeline's own (pre-existing) train/serve AE mismatch — not introduced here.
  Set AE_PATH = AE2 if you want to reproduce what the predictor sees at sample time
  instead of what it was trained on. Both are reported as options.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
from torch_geometric.transforms import Compose

# --- repo on path ---
REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.transforms import (
    CoordsToNanometers,
    CenterStructureTransform,
)
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from steering.predictor import SteeringPredictor
from steering.registry import PROPERTY_TO_INDEX

from openfold.np.residue_constants import resname_to_idx
from graphein.protein.tensor.io import protein_to_pyg

# ==============================================================================
# CONFIG
# ==============================================================================
PDB_DIR = f"{REPO}/results/iupred_target_sweep/iupred_target_w32/guided"
PROPS_CSV = f"{REPO}/results/iupred_target_sweep/iupred_target_w32/properties_guided.csv"

# AE that produced the predictor's TRAINING latents (see header note).
# Override with argv[1] = path to an AE ckpt (e.g. AE2_ucond_800) to compare.
AE_PATH = f"{REPO}/checkpoints_laproteina/AE1_ucond_512.ckpt"
if len(sys.argv) > 1:
    AE_PATH = sys.argv[1]
AE_TAG = os.path.splitext(os.path.basename(AE_PATH))[0]
DIAG_DIR = f"{REPO}/results/iupred_target_sweep/iupred_target_w32/diagnostics"

PREDICTOR_CKPTS = [
    f"{REPO}/laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_{k}_best.pt"
    for k in range(5)
]

PROPERTY = "iupred3_fraction_disordered"
PROP_IDX = PROPERTY_TO_INDEX[PROPERTY]          # == 5
REAL_COL = "iupred3_fraction_disordered"        # column in properties_guided.csv
ID_COL = "protein_id"                            # == pdb basename, e.g. s42_n300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILL = 1e-5  # graphein fill value for missing atoms (matches pdb_data.py)

# ==============================================================================
def pdb_to_encoder_input(pdb_path, baking_pipeline, device):
    """PDB file -> dict ready for ae.encoder(.). Mirrors precompute_latents.py."""
    graph = protein_to_pyg(
        path=pdb_path,
        chain_selection="all",
        keep_insertions=True,
        store_het=False,
        store_bfactor=False,
        fill_value_coords=FILL,
    )
    # coord_mask from fill value, residue types (pdb_data.py:71-75)
    coord_mask = (graph.coords != FILL)[..., 0]            # [L, 37] (PDB order)
    graph.coord_mask = coord_mask
    graph.residue_type = torch.tensor(
        [resname_to_idx[r] for r in graph.residues]
    ).long()

    # PDB atom order -> OpenFold atom order (constants.py:9-11; precompute:74-75)
    graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

    # Å -> nm (coords_nm) + center on CA. NO GlobalRotationTransform: encode the
    # structure as-is. (Latents are not rotation-invariant; centering matches the
    # precompute recipe and removes only the global translation.)
    graph = baking_pipeline(graph)   # CoordsToNanometers, CenterStructureTransform

    L = graph.coord_mask.shape[0]

    # Build encoder input dict (precompute_latents.py:150-167)
    enc = {}
    for k in graph.keys():
        v = graph[k]
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device).unsqueeze(0)   # add batch dim
        else:
            enc[k] = v

    na_idx = 0  # mask read at atom slot 0 (N) — verbatim from precompute:160-161
    seq_mask = enc["coord_mask"][:, :, na_idx].bool()   # [1, L]
    enc["mask_dict"] = {
        "coords": seq_mask.unsqueeze(-1).unsqueeze(-1),  # [1, L, 1, 1]
        "residue_type": seq_mask,
    }
    enc["mask"] = seq_mask
    return enc, seq_mask, L


def main():
    assert os.path.exists(AE_PATH), AE_PATH

    print(f"[load] AE  {AE_PATH}")
    ae = AutoEncoder.load_from_checkpoint(AE_PATH).to(DEVICE).eval()
    print(f"[load] AE latent_dim = {ae.latent_dim}")

    print(f"[load] predictor ensemble ({len(PREDICTOR_CKPTS)} folds)")
    predictor = SteeringPredictor(PREDICTOR_CKPTS, device=DEVICE)
    # de-norm uses predictor._stats_mean/_std internally; predict() returns real units

    bake = Compose([CoordsToNanometers(), CenterStructureTransform()])

    df = pd.read_csv(PROPS_CSV)
    real_map = dict(zip(df[ID_COL].astype(str), df[REAL_COL].astype(float)))

    pdbs = sorted(glob.glob(os.path.join(PDB_DIR, "*.pdb")))
    print(f"[data] {len(pdbs)} PDBs in {PDB_DIR}")

    rows = []
    with torch.no_grad():
        for p in pdbs:
            pid = os.path.splitext(os.path.basename(p))[0]
            if pid not in real_map:
                print(f"  [skip] {pid}: not in CSV")
                continue
            try:
                enc, seq_mask, L = pdb_to_encoder_input(p, bake, DEVICE)
                out = ae.encoder(enc)
                mean = out["mean"][0]                 # [L, 8]  (encoder.py:159)
                assert mean.shape[0] == L and mean.shape[1] == ae.latent_dim, mean.shape

                z = mean.unsqueeze(0).float()         # [1, L, 8]
                mask = seq_mask                        # [1, L] bool
                t = torch.ones(1, device=DEVICE)       # clean (predictor.predict default)
                pred_all = predictor.predict(z, mask, t)   # [1, P] de-normalised, real units
                pred = float(pred_all[0, PROP_IDX].item())
            except Exception as e:
                print(f"  [fail] {pid}: {e}")
                continue

            real = real_map[pid]
            rows.append((pid, pred, real))
            print(f"  {pid:12s}  pred={pred:.4f}  real={real:.4f}")

    if not rows:
        print("No rows.")
        return

    res = pd.DataFrame(rows, columns=["protein_id", "pred_disorder_reencoded", "real_disorder"])

    # Join the STEERED predicted disorder (diagnostics final step) for the 3-way compare.
    import json
    steered = {}
    for pid in res["protein_id"]:
        dp = os.path.join(DIAG_DIR, f"{pid}_diagnostics.json")
        if os.path.exists(dp):
            d = json.load(open(dp))
            steered[pid] = d[-1]["predicted_properties"]["iupred3_fraction_disordered"]
    res["pred_disorder_steered"] = res["protein_id"].map(steered)

    pr = np.corrcoef(res["pred_disorder_reencoded"], res["real_disorder"])[0, 1]
    sub = res.dropna(subset=["pred_disorder_steered"])
    pr_steer = np.corrcoef(sub["pred_disorder_steered"], sub["real_disorder"])[0, 1] if len(sub) > 2 else float("nan")
    print(f"\n==== SUMMARY (AE = {AE_TAG}) ====")
    print(f"n                       = {len(res)}")
    print(f"mean real               = {res['real_disorder'].mean():.4f}   (target 0.123)")
    print(f"mean pred (STEERED z)    = {sub['pred_disorder_steered'].mean():.4f}   r_vs_real = {pr_steer:.3f}")
    print(f"mean pred (RE-ENCODED z) = {res['pred_disorder_reencoded'].mean():.4f}   r_vs_real = {pr:.3f}")
    print("Interpretation: if RE-ENCODED pred jumps UP toward real (and r is high),")
    print("the predictor is correct on the on-manifold latent => steered z was off-manifold.")
    out_csv = os.path.join(os.path.dirname(PROPS_CSV), f"reencode_test_iupred_w32_{AE_TAG}.csv")
    res.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    main()
