"""Decision-gate audit for content-adaptive top-K distillation idea.

Two checks against the canonical dense baseline (CA-only, no sparse):

  CHECK 1 — concentration. For each (layer, head, t-step), compute the fraction
            of attention mass captured by the top-K residues, K ∈ {8, 16, 32,
            48, 64}. If mass_top_16 >= 0.7 and mass_top_32 >= 0.85 broadly,
            top-K distillation has surface to land on. If attention is diffuse
            (mass_top_64 ~ uniform = 64/N), distillation cannot help — stop.

  CHECK 2 — stability. For each (layer, head, query), record the top-16
            attended residues. Then measure Jaccard overlap of those sets:
              - across adjacent layers (does layer 5 agree with layer 6?)
              - across adjacent t-steps (does t=0.3 agree with t=0.5?)
              - across heads within a (layer, t) (do heads agree?)
            Tells us whether a single shared K-set across the trunk is viable
            (high Jaccard) or whether per-layer / per-t routing is required
            (low Jaccard).

Loads canonical dense ckpt at
    /rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/
        checkpoints/best_val_00000026_000000002646.ckpt
and a small deterministic protein subset from
    data/pdb_train/processed_latents/<shard>/*.pt
binned into L ~ {50, 100, 200}.

Output: results/dense_attn_audit/<label>.json with all per-bucket stats.

Compute: 1x A100, bf16 forward. ~5-15 min wall for the default settings.
Memory peak: [B=2, H=12, N=200, N=200] fp32 ~ 11 MB per layer, transient
during the hook; total model+batch < 8 GB.

Usage (from repo root):
    /home/ks2218/conda_envs/laproteina_env/bin/python \\
        script_utils/audit_dense_attention_concentration.py \\
        --label canonical_2646_dense_attn
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from loguru import logger
from torch_geometric.data import Data

from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    GlobalRotationTransform,
)
from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
from proteinfoundation.proteina import Proteina


CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)

# Top-K grid for Check 1.
K_GRID = (8, 16, 32, 48, 64)

# Top-K for Jaccard (Check 2). 16 is small enough to be a meaningful "important set".
JACCARD_K = 16


# ---------------------------------------------------------------------------- #
# Hook: replaces PairBiasAttention._attn with an instrumented version.
# Records per-call (per-layer) stats keyed by id(self) -> layer_idx.
# ---------------------------------------------------------------------------- #
def make_hooked_attn(records: List[dict]):
    """Replacement for PairBiasAttention._attn that returns the same value
    but additionally appends a dict of stats to `records`.

    Computes (in-hook, discards the [B, H, N, N] tensor right after):
      - mass_top_K[H, |K_GRID|]: fraction of total per-query attention mass
                                 captured by top-K (mean over queries) per head
      - top_idx[H, N, JACCARD_K]: indices of top-JACCARD_K keys per (head, query)
      - layer_idx: int
      - t_label: float (set externally on the module before forward)
      - protein_label: str (set externally on the module before forward)
      - N_real: int (real, non-padding length set externally on the module)
    """
    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def _attn(self, q, k, v, b, mask):
        # Exact reproduction of pair_bias_attn.PairBiasAttention._attn
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)  # [B, H, N, N]
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # Record stats. Only batch=0 for compactness; user can re-run if they
        # want per-batch records.
        B, H, N, _ = attn.shape
        layer_idx = getattr(self, "_layer_idx", -1)
        t_label = getattr(self, "_t_label", float("nan"))
        protein_label = getattr(self, "_protein_label", "?")
        N_real = getattr(self, "_N_real", N)

        # Restrict to real queries × real keys to avoid padding-row noise.
        # attn[..., :N_real, :N_real] preserves the valid subblock.
        attn_real = attn[0, :, :N_real, :N_real].float()  # [H, N_real, N_real]

        # mass_top_K per (head, query): for each query row, sort desc, take top K.
        # Then average over queries → [H, |K_GRID|].
        sorted_attn, _ = attn_real.sort(dim=-1, descending=True)  # [H, N_real, N_real]
        cum = sorted_attn.cumsum(dim=-1)                          # [H, N_real, N_real]
        denom = sorted_attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        mass_curve = cum / denom                                  # [H, N_real, N_real]
        # Index at K-1 for each K in grid (top-K mass).
        mass_top_K_per_query = torch.stack(
            [mass_curve[..., min(k - 1, N_real - 1)] for k in K_GRID],
            dim=-1,
        )  # [H, N_real, |K_GRID|]
        mass_top_K_mean = mass_top_K_per_query.mean(dim=1)  # [H, |K_GRID|]
        mass_top_K_median = mass_top_K_per_query.median(dim=1).values  # [H, |K_GRID|]

        # Top-JACCARD_K indices per (head, query) — only over real keys.
        Kjac = min(JACCARD_K, N_real)
        top_idx = attn_real.topk(k=Kjac, dim=-1).indices  # [H, N_real, Kjac]

        records.append(
            {
                "layer_idx": int(layer_idx),
                "t_label": float(t_label),
                "protein_label": str(protein_label),
                "N_real": int(N_real),
                "mass_top_K_mean": mass_top_K_mean.detach().cpu().tolist(),
                "mass_top_K_median": mass_top_K_median.detach().cpu().tolist(),
                "top_idx": top_idx.detach().cpu().tolist(),  # nested list, small
                "K_grid": list(K_GRID),
                "Kjac": Kjac,
            }
        )

        return out

    return _attn


# ---------------------------------------------------------------------------- #
# Data loading — mirrors run_per_t_val.py but length-bins the subset.
# ---------------------------------------------------------------------------- #
def list_processed_files(data_dir: str) -> List[str]:
    pattern = os.path.join(
        data_dir, "pdb_train", "processed_latents", "*", "*.pt"
    )
    return sorted(glob.glob(pattern))


def load_one(path: str) -> Data:
    return torch.load(path, map_location="cpu", weights_only=False)


def pick_per_length_bin(
    files: List[str],
    length_targets: Tuple[int, ...],
    proteins_per_bin: int,
    tol: int,
    seed: int,
    max_scan: int = 5000,
) -> Dict[int, List[Data]]:
    """For each target length L ∈ length_targets, find `proteins_per_bin`
    proteins with |actual_length - L| <= tol. Scans up to `max_scan` files."""
    rng = random.Random(seed)
    pool = list(files)
    rng.shuffle(pool)
    pool = pool[:max_scan]

    bins: Dict[int, List[Data]] = {L: [] for L in length_targets}
    needed = sum(proteins_per_bin for _ in length_targets)
    found = 0
    for fpath in pool:
        if found >= needed:
            break
        try:
            d = load_one(fpath)
        except Exception:
            continue
        n = int(d.coords_nm.shape[0])
        for L in length_targets:
            if abs(n - L) <= tol and len(bins[L]) < proteins_per_bin:
                bins[L].append(d)
                found += 1
                break
    return bins


def apply_transforms(items: List[Data], seed: int) -> List[Data]:
    out = []
    for i, it in enumerate(items):
        it = CenterStructureTransform()(it)
        it = ChainBreakPerResidueTransform()(it)
        torch.manual_seed(seed + 1_000 + i)
        it = GlobalRotationTransform()(it)
        out.append(it)
    return out


def pad_and_collate(items: List[Data], max_pad: int) -> Dict[str, torch.Tensor]:
    bs = len(items)
    coords = torch.zeros(bs, max_pad, 37, 3)
    cmask = torch.zeros(bs, max_pad, 37, dtype=torch.bool)
    rtype = torch.zeros(bs, max_pad, dtype=torch.long)
    mask = torch.zeros(bs, max_pad, dtype=torch.bool)
    real_lens = []
    for i, it in enumerate(items):
        n = it.coords_nm.shape[0]
        if n > max_pad:
            n = max_pad
        coords[i, :n] = it.coords_nm[:n]
        cmask[i, :n] = it.coord_mask[:n]
        rtype[i, :n] = it.residue_type[:n]
        mask[i, :n] = True
        real_lens.append(n)
    return {
        "coords_nm": coords,
        "coord_mask": cmask,
        "residue_type": rtype,
        "mask": mask,
        "_real_lens": real_lens,
    }


# ---------------------------------------------------------------------------- #
# Aggregation: turn raw per-call records into Check-1 / Check-2 summaries.
# ---------------------------------------------------------------------------- #
def jaccard_top16(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Pairwise Jaccard of top-K index sets, mean over (head, query).

    idx_a, idx_b: [H, N, K] int tensors of indices (already restricted to
    real queries/keys).
    """
    H, N, K = idx_a.shape
    assert idx_b.shape == idx_a.shape, (idx_a.shape, idx_b.shape)
    # Build [H, N, max_idx+1] one-hot membership; tractable since N <= 200.
    max_idx = max(int(idx_a.max()), int(idx_b.max())) + 1
    a = torch.zeros(H, N, max_idx, dtype=torch.bool)
    b = torch.zeros(H, N, max_idx, dtype=torch.bool)
    a.scatter_(2, idx_a, True)
    b.scatter_(2, idx_b, True)
    inter = (a & b).sum(dim=-1).float()  # [H, N]
    union = (a | b).sum(dim=-1).float().clamp(min=1.0)
    return (inter / union).mean().item()


def aggregate_check1(records: List[dict]) -> dict:
    """Aggregate top-K mass capture across (layer, t, length-bin, protein)."""
    # Group by (length_bin, t, layer)
    grouped: Dict[Tuple[int, float, int], List[List[List[float]]]] = defaultdict(
        list
    )
    for r in records:
        key = (r["N_real"], r["t_label"], r["layer_idx"])
        grouped[key].append(r["mass_top_K_mean"])  # [H, |K_GRID|]
    summary = {}
    for (N_real, t, layer), per_protein in grouped.items():
        # Mean over proteins, then mean over heads → single number per K
        arr = torch.tensor(per_protein)  # [P, H, |K_GRID|]
        mean_over_proteins_and_heads = arr.mean(dim=(0, 1)).tolist()  # [|K_GRID|]
        median_over_proteins_and_heads = arr.median(dim=0).values.median(dim=0).values.tolist()
        summary[f"L{N_real}_t{t:.2f}_layer{layer}"] = {
            "N_real": int(N_real),
            "t": float(t),
            "layer": int(layer),
            "n_proteins": int(arr.shape[0]),
            "K_grid": list(K_GRID),
            "mass_top_K_mean": mean_over_proteins_and_heads,
            "mass_top_K_median": median_over_proteins_and_heads,
            "uniform_baseline_K_over_N": [k / N_real for k in K_GRID],
        }
    return summary


def aggregate_check2(records: List[dict]) -> dict:
    """Compute layer-, t-, and head-Jaccard of top-K attended sets."""
    # Group by (protein, t, layer): each record has top_idx [H, N, K]
    by_pt_l: Dict[Tuple[str, float, int], dict] = {}
    for r in records:
        key = (r["protein_label"], r["t_label"], r["layer_idx"])
        # Note: top_idx is a list of lists of lists (head, query, K)
        by_pt_l[key] = r

    # Collect groups
    proteins = sorted(set(k[0] for k in by_pt_l))
    t_values = sorted(set(k[1] for k in by_pt_l))
    layers = sorted(set(k[2] for k in by_pt_l))

    layer_jaccard = {}  # (protein, t) -> mean Jaccard between adjacent layers
    t_jaccard = {}  # (protein, layer) -> mean Jaccard between adjacent t-steps
    head_jaccard = {}  # (protein, t, layer) -> mean Jaccard between heads

    # Layer-adjacent Jaccard
    for p in proteins:
        for t in t_values:
            scores = []
            for l1, l2 in zip(layers[:-1], layers[1:]):
                r1 = by_pt_l.get((p, t, l1))
                r2 = by_pt_l.get((p, t, l2))
                if r1 is None or r2 is None:
                    continue
                a = torch.tensor(r1["top_idx"], dtype=torch.long)
                b = torch.tensor(r2["top_idx"], dtype=torch.long)
                if a.shape != b.shape:
                    continue
                scores.append(jaccard_top16(a, b))
            if scores:
                layer_jaccard[f"{p}_t{t:.2f}"] = sum(scores) / len(scores)

    # t-adjacent Jaccard
    for p in proteins:
        for l in layers:
            scores = []
            for t1, t2 in zip(t_values[:-1], t_values[1:]):
                r1 = by_pt_l.get((p, t1, l))
                r2 = by_pt_l.get((p, t2, l))
                if r1 is None or r2 is None:
                    continue
                a = torch.tensor(r1["top_idx"], dtype=torch.long)
                b = torch.tensor(r2["top_idx"], dtype=torch.long)
                if a.shape != b.shape:
                    continue
                scores.append(jaccard_top16(a, b))
            if scores:
                t_jaccard[f"{p}_layer{l}"] = sum(scores) / len(scores)

    # Head-Jaccard within (p, t, l): average pairwise across heads
    for p in proteins:
        for t in t_values:
            for l in layers:
                r = by_pt_l.get((p, t, l))
                if r is None:
                    continue
                idx = torch.tensor(r["top_idx"], dtype=torch.long)  # [H, N, K]
                H = idx.shape[0]
                scores = []
                # All pairs (h1, h2) with h1 < h2. For H=12 → 66 pairs.
                for h1 in range(H):
                    for h2 in range(h1 + 1, H):
                        a = idx[h1:h1 + 1]  # [1, N, K]
                        b = idx[h2:h2 + 1]
                        scores.append(jaccard_top16(a, b))
                if scores:
                    head_jaccard[f"{p}_t{t:.2f}_layer{l}"] = sum(scores) / len(scores)

    # Aggregates over proteins.
    def aggregate(d: dict) -> dict:
        if not d:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
        vals = list(d.values())
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    return {
        "layer_jaccard_per_group": layer_jaccard,
        "t_jaccard_per_group": t_jaccard,
        "head_jaccard_per_group": head_jaccard,
        "layer_jaccard_summary": aggregate(layer_jaccard),
        "t_jaccard_summary": aggregate(t_jaccard),
        "head_jaccard_summary": aggregate(head_jaccard),
    }


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", default=CKPT_DEFAULT)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length_targets", type=str, default="50,100,200")
    parser.add_argument("--proteins_per_bin", type=int, default=3)
    parser.add_argument("--length_tol", type=int, default=5)
    parser.add_argument(
        "--t_values", type=str, default="0.10,0.30,0.50,0.70,0.90",
        help="Comma-sep list of t values to probe.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pad", type=int, default=256)
    parser.add_argument("--out_dir", default="results/dense_attn_audit")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("DATA_PATH", "/home/ks2218/la-proteina/data"),
    )
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    length_targets = tuple(int(x) for x in args.length_targets.split(","))
    t_values = tuple(float(x) for x in args.t_values.split(","))

    logger.info("=" * 80)
    logger.info(f"Dense-attention concentration audit | ckpt={args.ckpt_file}")
    logger.info(f"  length_targets={length_targets} (+/- {args.length_tol}), "
                f"proteins_per_bin={args.proteins_per_bin}")
    logger.info(f"  t_values={t_values}")
    logger.info("=" * 80)

    # ------------------- Load model -------------------
    assert os.path.exists(args.ckpt_file), f"Missing ckpt: {args.ckpt_file}"
    logger.info("Loading model...")
    model = Proteina.load_from_checkpoint(
        args.ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    cfg_exp = model.cfg_exp
    run_name = cfg_exp.get("run_name_")
    logger.info(f"  run_name_   = {run_name}")
    # Confirm this ckpt uses dense (not sparse) attention.
    nn_cfg = cfg_exp.get("nn", {})
    sparse_flag = nn_cfg.get("sparse_attention", False) or nn_cfg.get(
        "sparse_attention", None
    )
    logger.info(f"  sparse_attention flag in ckpt: {sparse_flag}")
    assert not sparse_flag, (
        f"This audit targets the DENSE path; ckpt has sparse_attention={sparse_flag}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ------------------- Install dense hook -------------------
    records: List[dict] = []
    PairBiasAttention._attn = make_hooked_attn(records)
    # Tag every attention layer with its index so the hook can attribute records.
    for i, layer in enumerate(model.nn.transformer_layers):
        layer.mhba.mha._layer_idx = i

    # ------------------- Build protein subset -------------------
    logger.info("Scanning protein files for length bins...")
    all_files = list_processed_files(args.data_dir)
    logger.info(f"  found {len(all_files)} .pt files; scanning for length-binned subset")
    bins = pick_per_length_bin(
        all_files,
        length_targets,
        proteins_per_bin=args.proteins_per_bin,
        tol=args.length_tol,
        seed=args.seed,
    )
    for L_target, items in bins.items():
        actual = [int(it.coords_nm.shape[0]) for it in items]
        logger.info(f"  L~{L_target}: got {len(items)} proteins, actual lengths={actual}")
        if len(items) < args.proteins_per_bin:
            logger.warning(
                f"  L~{L_target}: only {len(items)} found, may want a higher --max_scan"
            )

    # ------------------- Forward each (protein, t) and record attention -------------------
    n_layers = len(model.nn.transformer_layers)
    expected_records = sum(
        len(items) * len(t_values) * n_layers for items in bins.values()
    )
    logger.info(
        f"Expected total attention-layer calls: {expected_records} "
        f"({sum(len(v) for v in bins.values())} proteins × {len(t_values)} t × {n_layers} layers)"
    )

    forward_count = 0
    for L_target, items in bins.items():
        items_t = apply_transforms(items, args.seed)
        for prot_idx, it in enumerate(items_t):
            protein_label = f"L{L_target}_p{prot_idx}"
            for t_val in t_values:
                # Build a batch of size 1 padded to max_pad
                bs_items = [it]
                batch_cpu = pad_and_collate(bs_items, args.max_pad)
                real_len = batch_cpu.pop("_real_lens")[0]
                batch = {k: v.to(device) for k, v in batch_cpu.items()}

                # Set per-layer hook context
                for layer in model.nn.transformer_layers:
                    layer.mhba.mha._t_label = float(t_val)
                    layer.mhba.mha._protein_label = protein_label
                    layer.mhba.mha._N_real = int(real_len)

                # Construct x_t at the requested t value
                batch = model.add_clean_samples(batch)
                x_1_dict, mask_proc, batch_shape, n_pad, dtype, dev = (
                    model.fm.process_batch(batch)
                )
                x_0 = model.fm.sample_noise(
                    n=n_pad, shape=batch_shape, mask=mask_proc, device=dev
                )
                B = batch_shape[0]
                t_bb = torch.full((B,), float(t_val), device=dev)
                t = {"bb_ca": t_bb}
                x_t = model.fm.interpolate(
                    x_0=x_0, x_1=x_1_dict, t=t, mask=mask_proc
                )
                batch["x_0"] = x_0
                batch["x_1"] = x_1_dict
                batch["x_t"] = x_t
                batch["t"] = t
                batch["mask"] = mask_proc

                with torch.no_grad():
                    _ = model.call_nn(batch, n_recycle=0)
                forward_count += 1

                if forward_count % 5 == 0:
                    logger.info(
                        f"  forwards={forward_count}, records={len(records)} "
                        f"(latest: {protein_label}, t={t_val})"
                    )

    logger.info(f"All forwards complete: {forward_count} forwards, "
                f"{len(records)} attention-layer records.")

    # ------------------- Aggregate Check 1 -------------------
    logger.info("Aggregating Check 1 (top-K mass concentration)...")
    check1_summary = aggregate_check1(records)
    # Headline: averaged across all (L, t, layer)
    all_means = torch.tensor(
        [v["mass_top_K_mean"] for v in check1_summary.values()]
    )  # [n_groups, |K_GRID|]
    grand_mean = all_means.mean(dim=0).tolist()
    grand_med = all_means.median(dim=0).values.tolist()
    headline_check1 = {
        "K_grid": list(K_GRID),
        "grand_mean_mass_top_K": grand_mean,
        "grand_median_mass_top_K": grand_med,
        "n_groups": int(all_means.shape[0]),
    }
    logger.info("  HEADLINE Check 1:")
    logger.info(f"    K_grid       = {list(K_GRID)}")
    logger.info(f"    mass_top_K   = {[f'{x:.3f}' for x in grand_mean]}  (mean over groups)")
    logger.info(f"    median       = {[f'{x:.3f}' for x in grand_med]}")
    logger.info("  Decision rule:")
    logger.info("    mass_top_16 >= 0.70 AND mass_top_32 >= 0.85 → GO (distillation viable)")
    logger.info("    mass_top_64 close to 64/N (uniform)         → STOP (diffuse attention)")

    # ------------------- Aggregate Check 2 -------------------
    logger.info("Aggregating Check 2 (Jaccard stability)...")
    check2_summary = aggregate_check2(records)
    logger.info("  HEADLINE Check 2 (mean pairwise Jaccard of top-16 attended sets):")
    logger.info(f"    layer-adjacent : {check2_summary['layer_jaccard_summary']}")
    logger.info(f"    t-adjacent     : {check2_summary['t_jaccard_summary']}")
    logger.info(f"    head-within    : {check2_summary['head_jaccard_summary']}")
    logger.info("  Decision rule:")
    logger.info("    layer Jaccard >= 0.7  → shared K-set across the trunk works (cheap student)")
    logger.info("    layer Jaccard <= 0.3  → per-layer routing required (expensive student)")
    logger.info("    t Jaccard >= 0.7      → routing computed once per trajectory")
    logger.info("    t Jaccard <= 0.3      → routing must update per ODE step")

    # ------------------- Write JSON -------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "ckpt_file": args.ckpt_file,
        "label": args.label,
        "length_targets": list(length_targets),
        "t_values": list(t_values),
        "n_proteins_per_bin": args.proteins_per_bin,
        "K_grid": list(K_GRID),
        "JACCARD_K": JACCARD_K,
        "n_records": len(records),
        "check1_headline": headline_check1,
        "check1_per_group": check1_summary,
        "check2": check2_summary,
    }
    out_path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
