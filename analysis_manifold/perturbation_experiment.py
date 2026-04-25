"""Sidechain manifold comparison: coord-space vs. latent-space perturbations.

For each protein in a small length-stratified eval set:
  - Encode through AE1 to get the latent mean.
  - At each k in NOISE_SCALES and for each space in {coord, latent}, build a
    perturbed all-atom structure that differs from the original ONLY in the
    sidechain atoms (atom37 indices not in {0:N, 1:CA, 2:C, 4:O}).
  - Write the perturbed structure as a PDB under
    inference/<EVAL_CONFIG_NAME>/<protein_id>/job_<cell_idx>_<space>_<k>_<protein_id>.pdb
    where cell_idx = k_idx*2 + space_idx, so each (k, space) cell maps to a
    distinct evaluate.py --job_id and can be evaluated as one batch.

Coord-arm noise: per-(restype, atom_idx) std of the atom's offset from CA in
the residue-local (N, CA, C) frame, computed across the eval set itself.
Noise is sampled in the local frame and rotated back to global before adding.

Latent-arm noise: per-dim std of encoder `mean` over the eval set (≈ 1 since
the AE is KL-regularised toward N(0,1)).

After decoding the latent arm, we splice the original N/CA/C/O back so the
ONLY difference vs. the original is sidechain placement.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

# La-Proteina sources its constants from the project root via sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


# OpenFold atom37 ordering: ['N', 'CA', 'C', 'CB', 'O', 'CG', ...]
N_IDX, CA_IDX, C_IDX, O_IDX = 0, 1, 2, 4
BACKBONE_IDX = (N_IDX, CA_IDX, C_IDX, O_IDX)
N_ATOM37 = 37
SIDECHAIN_MASK_37 = torch.tensor(
    [i not in BACKBONE_IDX for i in range(N_ATOM37)], dtype=torch.bool
)
N_RESTYPES = 21  # 20 AAs + unknown bucket; we store stats with a 21-row table.

NOISE_SCALES = (0.1, 0.3, 0.5, 1.0, 2.0)
SPACES = ("coord", "latent")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def list_eval_proteins(processed_dir: Path, length_range: Tuple[int, int],
                       n_per_bin: int, n_bins: int, seed: int) -> List[Path]:
    """Sample a length-stratified subset of .pt files from processed_dir.

    We avoid loading every file just to read its length. Instead, we walk the
    shard directories and probe randomly until each length bin is filled.
    """
    rng = np.random.default_rng(seed)
    lo, hi = length_range
    bin_edges = np.linspace(lo, hi, n_bins + 1).astype(int)
    bins: List[List[Path]] = [[] for _ in range(n_bins)]

    all_files = []
    for shard in sorted(processed_dir.iterdir()):
        if shard.is_dir():
            all_files.extend(shard.glob("*.pt"))
    rng.shuffle(all_files)

    for f in all_files:
        if all(len(b) >= n_per_bin for b in bins):
            break
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not hasattr(data, "coord_mask"):
            continue
        L = int(data.coord_mask.shape[0])
        if L < lo or L > hi:
            continue
        # Find bin
        b_idx = int(np.searchsorted(bin_edges[1:], L, side="right"))
        b_idx = min(b_idx, n_bins - 1)
        if len(bins[b_idx]) < n_per_bin:
            bins[b_idx].append(f)

    selected = [f for b in bins for f in b]
    print(f"Selected {len(selected)} proteins, lengths/bin: "
          f"{[len(b) for b in bins]} (bin edges: {bin_edges.tolist()})")
    return selected


def load_protein(pt_path: Path) -> Dict:
    """Load a raw processed .pt file and convert to the dict format the AE wants.

    Replicates the relevant pieces of precompute_latents.py:
      - Reindex atom37 axis from PDB to OpenFold ordering.
      - Convert coords from Å to nm.
      - Center coords by mean CA position.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    L = int(data.coord_mask.shape[0])

    # Truncate any over-long auxiliary tensors (mirrors precompute_latents.py).
    for key in list(data.keys()):
        v = data[key]
        if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] > L:
            data[key] = v[:L]

    # Reindex atom dimension to OpenFold order.
    coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]            # [L, 37, 3] in Å
    coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]       # [L, 37]
    residue_type = data.residue_type.long()                             # [L]

    # Å -> nm
    coords_nm = coords * 0.1                                            # [L, 37, 3]

    # Center on mean CA position (mean over residues that have a CA).
    ca_mask = coord_mask[:, CA_IDX].bool()
    com = coords_nm[ca_mask, CA_IDX, :].mean(dim=0)
    coords_nm = coords_nm - com[None, None, :]

    return {
        "id": data.id,
        "L": L,
        "coords_nm": coords_nm.float(),     # [L, 37, 3], OpenFold order, nm, centered
        "coord_mask": coord_mask.bool(),    # [L, 37]
        "residue_type": residue_type,       # [L]
    }


# ---------------------------------------------------------------------------
# Local-frame construction (Gram–Schmidt from N, CA, C)
# ---------------------------------------------------------------------------


def build_residue_frames(coords_nm: torch.Tensor, coord_mask: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-residue local rotation matrices R[L, 3, 3] from N, CA, C.

    Returns (R, frame_mask). R is in nm-units (it is rotation-only). For
    residues missing any of N/CA/C, frame_mask[i] is False and R[i] = identity
    so downstream code is safe to apply unconditionally; callers should
    consult frame_mask before trusting per-residue stats.

    Convention: e1 along (N - CA), e3 along normal to (N-CA, C-CA), e2 = e3 x e1.
    """
    L = coords_nm.shape[0]
    n = coords_nm[:, N_IDX, :]
    ca = coords_nm[:, CA_IDX, :]
    c = coords_nm[:, C_IDX, :]

    bb_present = coord_mask[:, N_IDX] & coord_mask[:, CA_IDX] & coord_mask[:, C_IDX]

    e1 = n - ca
    e1 = _safe_normalize(e1)
    u = c - ca
    # Remove component along e1
    u = u - (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = _safe_normalize(u)
    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # [L, 3, 3] columns are basis vecs in global
    R = torch.where(bb_present[:, None, None], R, torch.eye(3).expand(L, 3, 3))
    return R, bb_present


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v / n


# ---------------------------------------------------------------------------
# Sidechain stats (per restype, per atom_idx, in residue-local frame)
# ---------------------------------------------------------------------------


def accumulate_sidechain_stats(proteins: List[Dict]) -> torch.Tensor:
    """Returns per-(restype, atom_idx) stddev in nm of sidechain offsets from CA
    measured in the residue-local (N, CA, C) frame.

    Output shape: [N_RESTYPES, 37, 3]. For (restype, atom_idx) pairs with no
    observations the value is set to a fallback (mean over present atoms of
    same restype, or 0.1 nm = 1 Å as a global fallback).
    """
    # accumulate sums and squared sums per (restype, atom_idx, dim)
    sums = torch.zeros(N_RESTYPES, N_ATOM37, 3, dtype=torch.float64)
    sumsq = torch.zeros_like(sums)
    counts = torch.zeros(N_RESTYPES, N_ATOM37, dtype=torch.int64)

    for prot in proteins:
        coords = prot["coords_nm"]              # [L, 37, 3] (centered, nm)
        mask = prot["coord_mask"]               # [L, 37]
        rtype = prot["residue_type"].clamp(0, N_RESTYPES - 1)  # [L]
        R, frame_ok = build_residue_frames(coords, mask)        # [L, 3, 3], [L]
        if not frame_ok.any():
            continue

        ca = coords[:, CA_IDX, :]                                # [L, 3]
        offset_global = coords - ca[:, None, :]                  # [L, 37, 3]
        # Express in local frame: local = R^T @ global  (R columns are e1,e2,e3)
        offset_local = torch.einsum("lij,laj->lai", R, offset_global)  # [L, 37, 3]

        atom_present = mask & frame_ok[:, None]                  # [L, 37]
        # zero out absent rows so they don't contribute
        offset_local = offset_local * atom_present[..., None]

        for L_idx in range(coords.shape[0]):
            r = int(rtype[L_idx])
            for a in range(N_ATOM37):
                if not atom_present[L_idx, a]:
                    continue
                v = offset_local[L_idx, a].double()
                sums[r, a] += v
                sumsq[r, a] += v * v
                counts[r, a] += 1

    # Compute std safely
    means = torch.where(counts[..., None] > 0, sums / counts.clamp_min(1)[..., None], torch.zeros_like(sums))
    var = torch.where(counts[..., None] > 1,
                      sumsq / counts.clamp_min(1)[..., None] - means * means,
                      torch.zeros_like(sums))
    var = var.clamp_min(0.0)
    std = var.sqrt().float()

    # Fallbacks for unseen (restype, atom_idx). Prefer mean over the same
    # restype's seen atoms; final fallback is 0.1 nm (= 1 Å).
    GLOBAL_FALLBACK = 0.1
    for r in range(N_RESTYPES):
        seen_atoms = (counts[r] > 0)
        if seen_atoms.any():
            row_fallback = std[r][seen_atoms].mean(dim=0)        # [3]
        else:
            row_fallback = torch.full((3,), GLOBAL_FALLBACK)
        for a in range(N_ATOM37):
            if a in BACKBONE_IDX:
                continue  # never perturbed; std left as 0
            if counts[r, a] == 0:
                std[r, a] = row_fallback

    # Backbone rows forced to 0 — we never sample noise into them.
    for a in BACKBONE_IDX:
        std[:, a, :] = 0.0

    print(f"Sidechain std stats: nonzero entries = "
          f"{int((std.norm(dim=-1) > 0).sum())}/{N_RESTYPES * N_ATOM37}, "
          f"mean nonzero std = {std[std.norm(dim=-1) > 0].mean().item():.4f} nm")
    return std  # [N_RESTYPES, 37, 3], in nm, in residue-local frame


# ---------------------------------------------------------------------------
# AE encode / decode helpers
# ---------------------------------------------------------------------------


def make_encoder_batch(prot: Dict, device) -> Dict:
    """Wrap a single protein dict into the batched form the AE expects."""
    coords_nm = prot["coords_nm"].to(device).unsqueeze(0)         # [1, L, 37, 3]
    coord_mask = prot["coord_mask"].to(device).unsqueeze(0)       # [1, L, 37]
    residue_type = prot["residue_type"].to(device).unsqueeze(0)   # [1, L]
    seq_mask = coord_mask[:, :, N_IDX].bool()                     # [1, L]
    ca_coors_nm = coords_nm[:, :, CA_IDX, :]                      # [1, L, 3]

    return {
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "ca_coors_nm": ca_coors_nm,
        "mask": seq_mask,
        "mask_dict": {
            "coords": seq_mask.unsqueeze(-1).unsqueeze(-1),       # [1, L, 1, 1]
            "residue_type": seq_mask,                              # [1, L]
        },
    }


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def perturb_coord_arm(prot: Dict, sidechain_std_local: torch.Tensor,
                      k: float, rng: torch.Generator) -> torch.Tensor:
    """Add k * sigma noise to sidechain atoms (in residue-local frame),
    rotate back to global, return the perturbed [L, 37, 3] (nm) coords.

    Backbone atoms are returned unchanged.
    """
    coords_nm = prot["coords_nm"].clone()
    mask = prot["coord_mask"]
    rtype = prot["residue_type"].clamp(0, N_RESTYPES - 1)
    R, frame_ok = build_residue_frames(coords_nm, mask)            # [L, 3, 3]

    sigma = sidechain_std_local[rtype]                              # [L, 37, 3]
    noise_local = torch.randn(coords_nm.shape, generator=rng, dtype=coords_nm.dtype) * sigma * k
    # Only sidechain atoms that are present and have a valid frame:
    apply_mask = mask & SIDECHAIN_MASK_37[None, :] & frame_ok[:, None]
    noise_local = noise_local * apply_mask[..., None]

    # Rotate noise to global: global_delta = R @ local_delta  (R columns are basis)
    noise_global = torch.einsum("lij,laj->lai", R, noise_local)
    return coords_nm + noise_global


def perturb_latent_arm(prot: Dict, ae: AutoEncoder, z_mean: torch.Tensor,
                       latent_std: torch.Tensor, k: float,
                       rng: torch.Generator, device) -> torch.Tensor:
    """Add k * sigma noise to latent mean, decode, splice original backbone,
    return [L, 37, 3] (nm) all-atom coords.

    To keep the comparison apples-to-apples with the coord arm, we enforce
    that the *atom set* present in the latent-arm output is identical to the
    original `coord_mask`:

      - Backbone (N/CA/C/O): always replaced by the original (CA was already
        passed to the decoder; we splice N, C, O back too so backbone is
        bit-identical to the original).
      - Sidechain atoms present in original:
          * if decoder also has it  -> use decoder's perturbed prediction
          * if decoder DROPPED it   -> fall back to the original position
            (so the atom is still present in the PDB; there's just no
            perturbation on that one atom).
      - Sidechain atoms absent from original: zeroed (omitted from PDB by
        write_prot_to_pdb's nonzero-mask logic).

    Without this, scRMSD's `mask = gen_mask * rec_mask` would compare the
    two arms over different atom counts and the result would be confounded
    by the AE's own atom-presence predictions.
    """
    noise = torch.randn(z_mean.shape, generator=rng, dtype=z_mean.dtype)  # [1, L, D]
    z_perturbed = z_mean + (noise * latent_std[None, None, :] * k).to(z_mean.device)

    seq_mask = prot["coord_mask"][:, N_IDX].bool().to(device).unsqueeze(0)   # [1, L]
    ca_coors_nm = prot["coords_nm"][:, CA_IDX, :].to(device).unsqueeze(0)    # [1, L, 3]
    out = ae.decode(z_perturbed, ca_coors_nm, seq_mask)
    decoded = out["coors_nm"][0].cpu()                                       # [L, 37, 3]
    decoded_atom_mask = out["atom_mask"][0].cpu().bool()                     # [L, 37]

    orig = prot["coords_nm"]
    orig_mask = prot["coord_mask"]

    perturbed = torch.zeros_like(orig)

    # Backbone: bit-identical to original where original has the atom.
    for a in BACKBONE_IDX:
        keep = orig_mask[:, a]
        perturbed[keep, a, :] = orig[keep, a, :]

    # Sidechains: enforce original atom set; fall back to original position
    # if the decoder dropped the atom.
    for a in range(N_ATOM37):
        if a in BACKBONE_IDX:
            continue
        present_in_orig = orig_mask[:, a]
        decoder_has = decoded_atom_mask[:, a]
        # Where original AND decoder both have it: use decoder's prediction.
        both = present_in_orig & decoder_has
        perturbed[both, a, :] = decoded[both, a, :]
        # Where original has it but decoder dropped: fall back to original.
        only_orig = present_in_orig & ~decoder_has
        perturbed[only_orig, a, :] = orig[only_orig, a, :]
        # Else: stays zero (omitted from PDB by write_prot_to_pdb).

    return perturbed


# ---------------------------------------------------------------------------
# PDB writing
# ---------------------------------------------------------------------------


def write_pdb_for_cell(coords_nm: torch.Tensor, prot: Dict, k_idx: int,
                       k: float, space: str, out_root: Path) -> Path:
    """Write a single PDB. coords_nm is [L, 37, 3] in nm (OpenFold order)."""
    space_idx = SPACES.index(space)
    cell_idx = k_idx * len(SPACES) + space_idx
    sub = out_root / prot["id"]
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"job_{cell_idx}_{space}_k{k:g}_{prot['id']}.pdb"
    pdb_path = sub / fname
    if pdb_path.exists():
        pdb_path.unlink()
    # Also remove the sibling tmp_dir evaluate.py would create.
    tmp_dir = pdb_path.with_suffix("")
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)

    coords_ang = coords_nm.numpy() * 10.0                                    # nm -> Å
    aatype = prot["residue_type"].numpy()

    write_prot_to_pdb(
        prot_pos=coords_ang,
        file_path=str(pdb_path),
        aatype=aatype,
        no_indexing=True,
        overwrite=True,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=str,
                   default="/rds/user/ks2218/hpc-work/processed")
    p.add_argument("--ae-ckpt", type=str,
                   default="/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt")
    p.add_argument("--length-min", type=int, default=50)
    p.add_argument("--length-max", type=int, default=300)
    p.add_argument("--n-per-bin", type=int, default=7)
    p.add_argument("--n-bins", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", type=str,
                   default="./inference/eval_manifold_perturbation")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Sample eval proteins.
    pt_paths = list_eval_proteins(
        processed_dir=Path(args.processed_dir),
        length_range=(args.length_min, args.length_max),
        n_per_bin=args.n_per_bin,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    proteins = []
    for f in pt_paths:
        try:
            proteins.append(load_protein(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    print(f"Loaded {len(proteins)} proteins, lengths "
          f"{[p['L'] for p in proteins]}")

    # 2. Load AE.
    print(f"Loading AE from {args.ae_ckpt}")
    ae = AutoEncoder.load_from_checkpoint(args.ae_ckpt, strict=False).to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    latent_dim = ae.latent_dim
    print(f"AE latent_dim = {latent_dim}")

    # 3. Compute coord-arm sidechain stds in residue-local frame.
    print("Computing sidechain stds in residue-local frame...")
    sidechain_std_local = accumulate_sidechain_stats(proteins)            # [21, 37, 3] (nm)

    # 4. Encode all proteins, collect z_means.
    z_means = {}
    with torch.no_grad():
        for prot in proteins:
            batch = make_encoder_batch(prot, device)
            enc_out = ae.encoder(batch)
            z_means[prot["id"]] = enc_out["mean"].detach()                # [1, L, D] on device

    # 5. Latent-arm std: per-dim std of mean across all positions of all proteins.
    flat = []
    for pid, m in z_means.items():
        seq_mask = next(p for p in proteins if p["id"] == pid)["coord_mask"][:, N_IDX].bool()
        flat.append(m[0, seq_mask, :].cpu())                              # [n_res, D]
    flat = torch.cat(flat, dim=0)                                         # [Nres_total, D]
    latent_std = flat.std(dim=0)
    print(f"Per-dim latent std: {latent_std.tolist()}")

    # 6. Save stats sidecar so the experiment is reproducible.
    stats_path = out_root / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "noise_scales": list(NOISE_SCALES),
            "spaces": list(SPACES),
            "n_proteins": len(proteins),
            "lengths": [p["L"] for p in proteins],
            "ids": [p["id"] for p in proteins],
            "ae_ckpt": args.ae_ckpt,
            "latent_dim": int(latent_dim),
            "latent_std_per_dim": latent_std.tolist(),
            "sidechain_std_summary_nm": {
                "mean_nonzero": float(
                    sidechain_std_local[sidechain_std_local.norm(dim=-1) > 0].mean()),
                "n_nonzero": int((sidechain_std_local.norm(dim=-1) > 0).sum()),
            },
            "backbone_atom_indices": list(BACKBONE_IDX),
            "cell_index_formula": "k_idx * 2 + space_idx (space_idx: coord=0, latent=1)",
        }, f, indent=2)
    print(f"Wrote {stats_path}")

    # 7. Generate perturbed PDBs. One RNG per (protein, k, space) cell so
    #    re-running with the same seed reproduces every file deterministically.
    written = 0
    for k_idx, k in enumerate(NOISE_SCALES):
        for prot in proteins:
            # coord arm
            rng_c = torch.Generator().manual_seed(
                args.seed * 10_000 + k_idx * 1_000 + abs(hash(prot["id"])) % 997)
            coords_pert = perturb_coord_arm(prot, sidechain_std_local, k, rng_c)
            write_pdb_for_cell(coords_pert, prot, k_idx, k, "coord", out_root)
            written += 1

            # latent arm
            rng_l = torch.Generator().manual_seed(
                args.seed * 10_000 + k_idx * 1_000 + abs(hash(prot["id"])) % 997 + 1)
            with torch.no_grad():
                coords_pert = perturb_latent_arm(
                    prot, ae, z_means[prot["id"]], latent_std.to(device),
                    k, rng_l, device,
                )
            write_pdb_for_cell(coords_pert, prot, k_idx, k, "latent", out_root)
            written += 1
        print(f"  k={k}: wrote {len(proteins) * 2} PDBs "
              f"(coord+latent for {len(proteins)} proteins)")

    print(f"Total PDBs written: {written}")
    print(f"Output root: {out_root}")
    print("Cells: k_idx in 0..{} -> job_id 0..{}".format(
        len(NOISE_SCALES) - 1, len(NOISE_SCALES) * len(SPACES) - 1))


if __name__ == "__main__":
    main()
