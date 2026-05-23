"""
Slot-permutation invariance diagnostic for sparse-attention checkpoints.

Hypothesis under test: in mixed-sparse training (slot k → residue identity varies
per query / step), the model could have learned to use slot index k itself as a
position signal via any K-indexed learnable, K-indexed positional bias, or any
operation that mixes along the K dim with learned weights.

Test design:
  - Load a trained mixed-sparse checkpoint.
  - Build a synthetic batch at L=50.
  - Run a single forward pass with default neighbour slot order (Run A).
  - Same batch and same neighbour set (residue identities unchanged), but slots
    re-ordered (sorted by residue index along the K dim, slot_valid moved
    consistently). Run again (Run B).
  - Compare outputs. If they differ by more than fp32 round-off, slot index
    carries learned signal. If they're (numerically) identical, slot index is
    not the issue.

This is a single-forward-pass test; nsteps=400 is irrelevant.
"""
import argparse
import os
import sys

import torch
from dotenv import load_dotenv
from loguru import logger

# Make repo imports work whether invoked from repo root or elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proteinfoundation.proteina import Proteina  # noqa: E402
from proteinfoundation.nn.modules import sparse_neighbors as sn_mod  # noqa: E402


def build_synthetic_batch(L: int, B: int, device, dtype=torch.float32):
    """A minimal batch the NN can consume.

    Fields are chosen to match what the trained mixed-K=40 ckpt's feature set
    expects (xt_bb_ca, x_sc_bb_ca, optional_ca_coors_nm seq, optional res_type;
    pair: rel_seq_sep, xt_bb_ca_pair_dists, x_sc_bb_ca_pair_dists,
    optional_ca_pair_dist).
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    # Synthesise Cα coords roughly the right scale (nm). A chain-like prior is
    # nice for spatial neighbours to have some structure, but for the slot-perm
    # test only the *set* of neighbours matters.
    coords_ca = torch.randn(B, L, 3, generator=g) * 1.5  # nm, ~3-5 Å typical
    coords_ca = coords_ca.to(device=device, dtype=dtype)

    # Build atom37 coords with Cα slot filled.
    coords37 = torch.zeros(B, L, 37, 3, device=device, dtype=dtype)
    coords37[:, :, 1, :] = coords_ca  # OpenFold atom_order: 1 = CA
    coord_mask = torch.zeros(B, L, 37, dtype=torch.bool, device=device)
    coord_mask[:, :, 1] = True  # only CA present
    residue_type = torch.zeros(B, L, dtype=torch.long, device=device)  # ALA
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    residue_pdb_idx = (
        torch.arange(1, L + 1, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, L)
        .contiguous()
    )

    # Fixed mid-range t — irrelevant to the slot-perm test as long as it's the
    # same for both runs.
    t = torch.full((B,), 0.5, device=device, dtype=dtype)

    # x_sc default = zeros (same shape as x_t). Matches inference step==0.
    batch = {
        "x_t": {"bb_ca": coords_ca.clone()},
        "t": {"bb_ca": t},
        "mask": mask,
        "coords_nm": coords37,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "residue_pdb_idx": residue_pdb_idx,
        # Self-cond placeholder (zeros) — the model's seq features include
        # x_sc_bb_ca, and the corresponding pair feature falls back to zeros
        # when x_sc isn't present.
        "x_sc": {"bb_ca": torch.zeros_like(coords_ca)},
        "use_ca_coors_nm_feature": False,
    }
    return batch


def sort_slots_by_residue_idx(neighbor_idx, slot_valid):
    """Sort each query's K slots by neighbour-residue index along the K dim.

    Returns reordered (neighbor_idx, slot_valid). The set of neighbours per
    query is identical to the input; only slot positions change.
    """
    # stable sort along K
    sorted_idx, perm = torch.sort(neighbor_idx, dim=-1, stable=True)
    slot_valid_sorted = torch.gather(slot_valid, dim=-1, index=perm)
    return sorted_idx, slot_valid_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/ks2218/la-proteina/sparse_K40_step1259.ckpt",
        help="Path to a mixed-sparse trained checkpoint.",
    )
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda:0", help="Set to cuda:N to pin a single GPU."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"Slot-permutation invariance test")
    logger.info(f"  ckpt:   {args.ckpt}")
    logger.info(f"  L={args.L}  B={args.B}  device={device}")
    logger.info("=" * 70)

    assert os.path.exists(args.ckpt), f"Missing ckpt: {args.ckpt}"

    logger.info("Loading model...")
    model = Proteina.load_from_checkpoint(args.ckpt, strict=False, autoencoder_ckpt_path=None)
    model = model.to(device).eval()

    # Make sure the model is the sparse one
    assert getattr(model.nn, "sparse_attention", False), (
        f"This audit targets sparse-attention models; ckpt nn has "
        f"sparse_attention={getattr(model.nn, 'sparse_attention', None)}"
    )
    n_seq = model.nn.n_seq_neighbors
    n_sp = model.nn.n_spatial_neighbors
    n_rd = model.nn.n_random_neighbors
    K = 2 * n_seq + n_sp + n_rd
    logger.info(
        f"  sparse: n_seq={n_seq}/side, n_spatial={n_sp}, n_random={n_rd}  → K={K}"
    )
    logger.info(
        f"  curriculum_neighbors={getattr(model.nn, 'curriculum_neighbors', False)}, "
        f"n_global_tokens={getattr(model.nn, 'n_global_tokens', 0)}"
    )

    # Disable curriculum/globals/sc_neighbors so the slot-perm test compares
    # plain sparse-only behaviour. For the K=40 mixed ckpt these are already
    # off; this is defensive.
    model.nn.curriculum_neighbors = False
    if hasattr(model.nn, "sc_neighbors"):
        model.nn.sc_neighbors = False

    batch = build_synthetic_batch(args.L, args.B, device)

    # --------- 1. Compute neighbour_idx normally (Run A) and a sorted copy (Run B)
    # We use the canonical builder (no curriculum) directly.
    with torch.no_grad():
        ca_coors = batch["x_t"]["bb_ca"]
        mask = batch["mask"]
        nbr_idx_A, slot_valid_A = sn_mod.build_neighbor_idx(
            ca_coors, mask, n_seq=n_seq, n_spatial=n_sp, n_random=n_rd
        )
    nbr_idx_B, slot_valid_B = sort_slots_by_residue_idx(nbr_idx_A, slot_valid_A)

    # Sanity: identical neighbour SETS per query (only order differs).
    set_A = torch.sort(nbr_idx_A, dim=-1)[0]
    set_B = torch.sort(nbr_idx_B, dim=-1)[0]
    assert torch.equal(set_A, set_B), "Sorted slot sets should match; bug in test."
    n_perm_changed_rows = (nbr_idx_A != nbr_idx_B).any(dim=-1).sum().item()
    logger.info(
        f"  neighbour sets identical between runs; "
        f"{n_perm_changed_rows} / {args.B * args.L} (b,i) rows had their slot order changed."
    )

    # --------- 2. Monkey-patch the model's neighbour-builder so the forward
    # uses the precomputed indices. We replace the bound method
    # `LocalLatentsTransformer._build_neighbor_idx` to return either A or B.
    container = {"idx": nbr_idx_A, "valid": slot_valid_A}

    def patched_build_neighbor_idx(self, ca_coors, mask, t=None):
        # Ignore inputs; return the precomputed (idx, valid) so both runs see
        # exactly the same neighbour set with only slot-order changes.
        return container["idx"], container["valid"]

    import types

    model.nn._build_neighbor_idx = types.MethodType(patched_build_neighbor_idx, model.nn)

    # --------- 3. Two forwards in fp32 with deterministic settings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def fwd():
        with torch.no_grad():
            return model.nn(batch)

    # Run A: default slot order
    container["idx"] = nbr_idx_A.clone()
    container["valid"] = slot_valid_A.clone()
    out_A = fwd()
    # Run B: slots sorted by residue index
    container["idx"] = nbr_idx_B.clone()
    container["valid"] = slot_valid_B.clone()
    out_B = fwd()

    # --------- 4. Compare outputs (bb_ca is the primary trunk output; latents
    # only present if the model has a latent head).
    def diffs(name, a, b):
        d = (a - b).abs()
        return {
            "shape": tuple(a.shape),
            "max_diff": d.max().item(),
            "mean_diff": d.mean().item(),
            "max_a": a.abs().max().item(),
            "max_b": b.abs().max().item(),
        }

    logger.info("=" * 70)
    logger.info("Outputs:")
    report = {}
    for head, val_A in out_A.items():
        for sub, t_A in val_A.items():
            t_B = out_B[head][sub]
            stats = diffs(f"{head}/{sub}", t_A, t_B)
            report[f"{head}/{sub}"] = stats
            logger.info(
                f"  {head}/{sub:>4}  shape={stats['shape']}  "
                f"max|A-B|={stats['max_diff']:.3e}  "
                f"mean|A-B|={stats['mean_diff']:.3e}  "
                f"max|A|={stats['max_a']:.3e}"
            )

    # Verdict heuristic:
    primary = report.get("bb_ca/v", None) or next(iter(report.values()))
    max_diff = primary["max_diff"]
    rel = max_diff / max(primary["max_a"], 1e-12)
    logger.info("-" * 70)
    logger.info(f"  Primary output max|A-B| = {max_diff:.3e}")
    logger.info(f"  relative to max|A|       = {rel:.3e}")
    if max_diff < 1e-4 and rel < 1e-4:
        logger.info(
            "  VERDICT: outputs match to numerical precision → slot index "
            "carries NO learned signal under this checkpoint."
        )
    elif max_diff < 1e-3:
        logger.info(
            "  VERDICT: outputs differ at >fp32-roundoff but <1e-3. Could be "
            "non-determinism in op order; investigate."
        )
    else:
        logger.info(
            "  VERDICT: outputs differ by >1e-3 between runs → slot index "
            "DOES carry learned signal; sparse path is NOT slot-permutation-equivariant."
        )


if __name__ == "__main__":
    main()
