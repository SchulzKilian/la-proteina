"""Gradient-saliency audit on the canonical dense baseline — PER-QUERY mode.

This is the per-query version (replaces the earlier aggregate one): for each
sampled query residue i, compute ‖∂‖v_pred[i]‖ / ∂x_t[j]‖₂ for every residue j.
That gives one saliency vector PER query per protein per t, structurally
identical to sparse's per-residue K-set (each query has its own neighbor list).

Rationale for going per-query (user's observation 2026-05-13):
  - The aggregate version (sum of per-residue losses, single backward) summed
    across all queries' loss terms, so per-query specialization was averaged
    out. Empirically that path showed *more diffuse* gradient than the
    softmax attention top-K of E059 — the wrong question being asked.
  - In sparse attention each residue i HAS its own attention list of K residues
    it routes to. The gradient analog of "important set for query i" is the
    top-K residues j by ‖∂scalar_i / ∂x_t[j]‖, where scalar_i is some scalar
    summary of i's prediction. We use scalar_i = ‖v_pred[i]‖₂ (no ground
    truth needed; pure functional dependency through the model).
  - With per-query saliency, the comparison to dense's per-query attention
    pattern is one-to-one: same query, same protein, same t, top-K from
    gradient vs top-K from attention.

What this records, per (protein, t, sampled_query_i):
  - saliency_grad[N_real]: ‖∂‖v_pred[i]‖ / ∂x_t[j]‖₂ for each residue j
  - top_grad_idx[Kj]: indices of top-Kj most-influential residues for query i
  - mass_top_K_grad[|K_grid|]: concentration at K ∈ {8, 16, 32, 48, 64}

What the hook records, per (protein, t, layer, sampled_query_i):
  - top_attn_idx_per_query[H, Kj]: dense's top-Kj attended residues for that
    same query at that layer, per head. One-to-one comparable with top_grad_idx.

Cross-metric outputs (the headline numbers for the routing-prior question):
  - Per (protein, t, sampled_query_i, layer, head): Jaccard(top_K_grad, top_K_attn)
  - "Max over (layer, head)" per (protein, t, query): does dense's attention
    agree with gradient AT SOME (layer, head)? If high, attention and gradient
    converge somewhere in the trunk on the same routing prior for this query.
  - "Mean over (layer, head)" per (protein, t, query): typical agreement.
  - Query-pair Jaccard within protein-t (across pairs of sampled queries): do
    different queries care about different residues? If LOW, user's intuition
    holds; if HIGH, queries share a global important set.

Run from repo root (needs GPU; backward pass is non-negotiable for gradient):
    /home/ks2218/conda_envs/laproteina_env/bin/python \\
        script_utils/audit_dense_gradient_saliency.py \\
        --label canonical_2646_grad_per_query \\
        --queries_per_protein 8

Compute: 1× A100. With queries_per_protein=8: one forward per (protein, t)
plus 8 backwards (using retain_graph) + 1 aggregate backward. ~15-20 min wall.
Memory: forward activations + retained graph during 8 sequential backwards.
~8-12 GB peak at L=200. Set --queries_per_protein lower if memory tight.
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

# Same K-grid as E059 for direct comparison.
K_GRID = (8, 16, 32, 48, 64)
JACCARD_K = 16


# ---------------------------------------------------------------------------- #
# Hook: records per-query top-K attended for the sampled queries listed in
# self._sampled_queries (set externally before forward). One-to-one comparable
# with the per-query gradient top-K computed via vector-Jacobian product.
# ---------------------------------------------------------------------------- #
def make_attn_hook(records: List[dict]):
    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def _attn(self, q, k, v, b, mask):
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask_rs = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        with torch.no_grad():
            B, H, N, _ = attn.shape
            layer_idx = getattr(self, "_layer_idx", -1)
            t_label = getattr(self, "_t_label", float("nan"))
            protein_label = getattr(self, "_protein_label", "?")
            N_real = getattr(self, "_N_real", N)
            sampled_queries = getattr(self, "_sampled_queries", None)

            attn_real = attn[0, :, :N_real, :N_real].float()  # [H, N_real, N_real]
            Kj = min(JACCARD_K, N_real)

            # Head-averaged attention-received per key (the global metric).
            attn_received_per_head = attn_real.sum(dim=1)  # [H, N_real]
            attn_received_mean_heads = attn_received_per_head.mean(dim=0)  # [N_real]
            top_attn_global_idx = attn_received_mean_heads.topk(k=Kj).indices.cpu().tolist()

            # Per-query top-K for the sampled queries only.
            top_attn_per_query = None  # [H, |sampled|, Kj] indices, or None
            if sampled_queries is not None and len(sampled_queries) > 0:
                sq = torch.tensor(sampled_queries, device=attn_real.device, dtype=torch.long)
                sampled_attn = attn_real.index_select(dim=1, index=sq)  # [H, |sq|, N_real]
                top_attn_per_query = sampled_attn.topk(k=Kj, dim=-1).indices.cpu().tolist()

            records.append(
                {
                    "layer_idx": int(layer_idx),
                    "t_label": float(t_label),
                    "protein_label": str(protein_label),
                    "N_real": int(N_real),
                    "Kj": Kj,
                    "top_attn_global_idx": top_attn_global_idx,
                    "sampled_queries": list(sampled_queries) if sampled_queries is not None else None,
                    "top_attn_per_query": top_attn_per_query,
                }
            )
        return out

    return _attn


# ---------------------------------------------------------------------------- #
# Protein loading — identical scaffolding to E059.
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
        for L_t in length_targets:
            if abs(n - L_t) <= tol and len(bins[L_t]) < proteins_per_bin:
                bins[L_t].append(d)
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
# Helpers.
# ---------------------------------------------------------------------------- #
def mass_top_K(saliency: torch.Tensor, K_grid: Tuple[int, ...]) -> List[float]:
    s = saliency.float().abs()
    sorted_s, _ = s.sort(descending=True)
    total = sorted_s.sum().clamp(min=1e-12)
    cum = sorted_s.cumsum(dim=0)
    return [(cum[min(K - 1, s.numel() - 1)] / total).item() for K in K_grid]


def jaccard_set(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return float("nan")
    return len(sa & sb) / len(union)


def sample_queries(N_real: int, k: int, rng: random.Random) -> List[int]:
    """Deterministic sample of `k` distinct query indices from [0, N_real)."""
    if k >= N_real:
        return list(range(N_real))
    return sorted(rng.sample(range(N_real), k))


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
        "--queries_per_protein", type=int, default=8,
        help="Number of query residues per protein to compute per-query gradient for. "
             "Set to 0 to skip per-query (aggregate-only) — but the aggregate path "
             "previously showed diffuse gradients, so 8 is the recommended default.",
    )
    parser.add_argument(
        "--also_aggregate", action="store_true",
        help="Additionally backprop the FM loss (sum over all queries) for an aggregate "
             "gradient — costs one extra backward per (protein, t). Default off since "
             "the aggregate is known to be diffuse.",
    )
    parser.add_argument(
        "--t_values", type=str, default="0.10,0.30,0.50,0.70,0.90",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pad", type=int, default=256)
    parser.add_argument("--out_dir", default="results/dense_attn_audit")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("DATA_PATH", "/home/ks2218/la-proteina/data"),
    )
    parser.add_argument(
        "--save_raw_saliency", action="store_true",
        help="Dump every per-query saliency vector to the JSON. Without this flag, "
             "only the top-K indices and mass_top_K curves are saved (much smaller).",
    )
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    length_targets = tuple(int(x) for x in args.length_targets.split(","))
    t_values = tuple(float(x) for x in args.t_values.split(","))

    logger.info("=" * 80)
    logger.info(f"Dense per-query gradient-saliency audit | ckpt={args.ckpt_file}")
    logger.info(f"  length_targets={length_targets} (±{args.length_tol}), "
                f"proteins_per_bin={args.proteins_per_bin}")
    logger.info(f"  t_values={t_values}")
    logger.info(f"  queries_per_protein={args.queries_per_protein}, "
                f"also_aggregate={args.also_aggregate}")
    logger.info("=" * 80)

    # ------------------- Load model -------------------
    assert os.path.exists(args.ckpt_file), f"Missing ckpt: {args.ckpt_file}"
    logger.info("Loading model...")
    model = Proteina.load_from_checkpoint(
        args.ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    cfg_exp = model.cfg_exp
    nn_cfg = cfg_exp.get("nn", {})
    sparse_flag = nn_cfg.get("sparse_attention", False)
    assert not sparse_flag, (
        f"This audit targets the DENSE path; ckpt has sparse_attention={sparse_flag}"
    )
    logger.info(f"  run_name_   = {cfg_exp.get('run_name_')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Gradient saliency requires CUDA (backward pass)."
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------- Install attention hook -------------------
    attn_records: List[dict] = []
    PairBiasAttention._attn = make_attn_hook(attn_records)
    for i, layer in enumerate(model.nn.transformer_layers):
        layer.mhba.mha._layer_idx = i

    # ------------------- Build protein subset -------------------
    logger.info("Scanning protein files for length bins...")
    all_files = list_processed_files(args.data_dir)
    logger.info(f"  found {len(all_files)} .pt files")
    bins = pick_per_length_bin(
        all_files,
        length_targets,
        proteins_per_bin=args.proteins_per_bin,
        tol=args.length_tol,
        seed=args.seed,
    )
    for L_target, items in bins.items():
        actual = [int(it.coords_nm.shape[0]) for it in items]
        logger.info(f"  L~{L_target}: got {len(items)} proteins, lengths={actual}")

    n_layers = len(model.nn.transformer_layers)

    # ------------------- Per-(protein, t) loop -------------------
    grad_records: List[dict] = []
    forward_count = 0
    backward_count = 0
    for L_target, items in bins.items():
        items_t = apply_transforms(items, args.seed)
        for prot_idx, it in enumerate(items_t):
            protein_label = f"L{L_target}_p{prot_idx}"
            for t_val in t_values:
                bs_items = [it]
                batch_cpu = pad_and_collate(bs_items, args.max_pad)
                real_len = batch_cpu.pop("_real_lens")[0]
                batch = {k: v.to(device) for k, v in batch_cpu.items()}

                # Sample queries deterministically per (protein, t).
                rng_q = random.Random(args.seed * 1000 + int(t_val * 100) + prot_idx)
                sampled_queries = sample_queries(
                    real_len, args.queries_per_protein, rng_q
                )

                # Push hook context onto every attention layer.
                for layer in model.nn.transformer_layers:
                    layer.mhba.mha._t_label = float(t_val)
                    layer.mhba.mha._protein_label = protein_label
                    layer.mhba.mha._N_real = int(real_len)
                    layer.mhba.mha._sampled_queries = list(sampled_queries)

                # Build x_t (same RNG order as E059 → matched noise sample).
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
                # Make x_t a leaf with grad.
                x_t_bb = x_t["bb_ca"].detach().clone().requires_grad_(True)
                x_t_grad = {"bb_ca": x_t_bb}

                batch["x_0"] = x_0
                batch["x_1"] = x_1_dict
                batch["x_t"] = x_t_grad
                batch["t"] = t
                batch["mask"] = mask_proc

                # Forward — grad enabled because x_t_bb.requires_grad=True.
                nn_out = model.call_nn(batch, n_recycle=0)
                forward_count += 1
                v_pred = nn_out["bb_ca"]["v"]  # [B, n_pad, 3]
                v_pred_real = v_pred[0, :real_len]  # [N_real, 3]

                # ----- Per-query gradient (vector-Jacobian product) -----
                per_query_entries = []
                for qi_pos, query_i in enumerate(sampled_queries):
                    x_t_bb.grad = None
                    scalar_i = v_pred_real[query_i].norm()
                    # Retain graph for all but the last per-query backward
                    # (or until aggregate backward if --also_aggregate).
                    is_last_per_query = (qi_pos == len(sampled_queries) - 1)
                    retain = (not is_last_per_query) or args.also_aggregate
                    scalar_i.backward(retain_graph=retain)
                    backward_count += 1
                    sal = (
                        x_t_bb.grad[0, :real_len]
                        .norm(dim=-1)
                        .detach()
                        .float()
                        .cpu()
                    )  # [N_real]
                    mtk = mass_top_K(sal, K_GRID)
                    Kj = min(JACCARD_K, int(sal.numel()))
                    top_idx = sal.topk(k=Kj).indices.tolist()
                    entry = {
                        "query_i": int(query_i),
                        "K_grid": list(K_GRID),
                        "mass_top_K_grad": mtk,
                        "top_grad_idx": top_idx,
                        "Kj": Kj,
                        "scalar_i_val": float(scalar_i.item()),
                    }
                    if args.save_raw_saliency:
                        entry["saliency"] = sal.tolist()
                    per_query_entries.append(entry)

                # ----- Optional aggregate backward (final, no retain) -----
                aggregate_record = None
                if args.also_aggregate:
                    x_t_bb.grad = None
                    losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
                    per_proto = sum(losses[k] for k in losses if "_justlog" not in k)
                    loss = per_proto.sum()
                    loss.backward()  # graph released
                    backward_count += 1
                    sal_agg = (
                        x_t_bb.grad[0, :real_len].norm(dim=-1).detach().float().cpu()
                    )
                    aggregate_record = {
                        "mass_top_K_grad_agg": mass_top_K(sal_agg, K_GRID),
                        "top_grad_idx_agg": sal_agg.topk(
                            k=min(JACCARD_K, int(sal_agg.numel()))
                        ).indices.tolist(),
                        "loss": float(loss.item()),
                    }

                grad_records.append(
                    {
                        "protein_label": protein_label,
                        "t_label": float(t_val),
                        "N_real": int(real_len),
                        "sampled_queries": list(sampled_queries),
                        "per_query": per_query_entries,
                        "aggregate": aggregate_record,
                    }
                )

                if forward_count % 5 == 0:
                    mass16_first3 = [f"{e['mass_top_K_grad'][1]:.3f}" for e in per_query_entries[:3]]
                    logger.info(
                        f"  fwd={forward_count}, bwd={backward_count} "
                        f"(latest: {protein_label}, t={t_val}, "
                        f"per-query mass_top_16 [first 3]: {mass16_first3})"
                    )

                del nn_out, v_pred, v_pred_real, x_t_bb
                torch.cuda.empty_cache()

    logger.info(
        f"Done. forwards={forward_count}, backwards={backward_count}, "
        f"grad_records={len(grad_records)}, attn_records={len(attn_records)}"
    )

    # ------------------- Check 1' (per-query concentration) -------------------
    logger.info("=" * 80)
    logger.info("Check 1' — per-query gradient concentration")
    logger.info("=" * 80)
    all_per_query_mtk = []
    for r in grad_records:
        for e in r["per_query"]:
            all_per_query_mtk.append(e["mass_top_K_grad"])
    mtk_arr = torch.tensor(all_per_query_mtk)  # [n_queries_total, |K|]
    grand_mean = mtk_arr.mean(dim=0).tolist()
    grand_med = mtk_arr.median(dim=0).values.tolist()
    headline_check1 = {
        "K_grid": list(K_GRID),
        "grand_mean_mass_top_K_per_query": grand_mean,
        "grand_median_mass_top_K_per_query": grand_med,
        "n_queries_total": int(mtk_arr.shape[0]),
    }
    logger.info(f"  K_grid       = {list(K_GRID)}")
    logger.info(f"  mass_top_K   = {[f'{x:.3f}' for x in grand_mean]}  "
                f"(mean over {mtk_arr.shape[0]} per-query records)")
    logger.info(f"  median       = {[f'{x:.3f}' for x in grand_med]}")
    logger.info(f"  E059 attention reference (per-query top-K of softmax weights):")
    logger.info(f"    mass_top_K_attn ≈ [0.510, 0.656, 0.794, 0.866, 0.907]")

    # ------------------- Check 2' (per-query t-Jaccard) -------------------
    logger.info("=" * 80)
    logger.info("Check 2' — per-query gradient t-Jaccard (does the important set "
                "for query i stay stable across t?)")
    logger.info("=" * 80)
    by_pq: Dict[Tuple[str, int], Dict[float, List[int]]] = defaultdict(dict)
    for r in grad_records:
        for e in r["per_query"]:
            by_pq[(r["protein_label"], e["query_i"])][r["t_label"]] = e["top_grad_idx"]
    t_jaccards: List[float] = []
    for (_, _), t_to_idx in by_pq.items():
        ts = sorted(t_to_idx.keys())
        for t1, t2 in zip(ts[:-1], ts[1:]):
            t_jaccards.append(jaccard_set(t_to_idx[t1], t_to_idx[t2]))
    t_jacc_summary = (
        {
            "mean": sum(t_jaccards) / len(t_jaccards),
            "min": min(t_jaccards),
            "max": max(t_jaccards),
            "n": len(t_jaccards),
        }
        if t_jaccards
        else {"mean": float("nan"), "n": 0}
    )
    logger.info(f"  per-query t-Jaccard: {t_jacc_summary}")
    logger.info(f"  E059 attention reference: t-Jaccard_attn = 0.475 mean (n=126)")

    # ------------------- New check: query-pair Jaccard (within protein, t) -------------------
    logger.info("=" * 80)
    logger.info("New check — query-pair Jaccard: do different queries care about "
                "different residues?")
    logger.info("  (LOW ≈ user's intuition holds: routing must be per-query)")
    logger.info("  (HIGH ≈ shared 'important set' exists per protein-t)")
    logger.info("=" * 80)
    pair_jaccards: List[float] = []
    for r in grad_records:
        per_q = r["per_query"]
        for i1 in range(len(per_q)):
            for i2 in range(i1 + 1, len(per_q)):
                pair_jaccards.append(
                    jaccard_set(per_q[i1]["top_grad_idx"], per_q[i2]["top_grad_idx"])
                )
    pair_jacc_summary = (
        {
            "mean": sum(pair_jaccards) / len(pair_jaccards),
            "min": min(pair_jaccards),
            "max": max(pair_jaccards),
            "n": len(pair_jaccards),
        }
        if pair_jaccards
        else {"mean": float("nan"), "n": 0}
    )
    logger.info(f"  query-pair Jaccard: {pair_jacc_summary}")

    # ------------------- Cross-metric (per-query): grad vs attn -------------------
    logger.info("=" * 80)
    logger.info("Cross-metric (per-query) — Jaccard(top_K_grad[i], top_K_attn[l, h, i])")
    logger.info("=" * 80)
    # Build attn lookup by (protein, t, layer)
    attn_by_key: Dict[Tuple[str, float, int], dict] = {}
    for ar in attn_records:
        attn_by_key[(ar["protein_label"], ar["t_label"], ar["layer_idx"])] = ar

    # For each (protein, t, query, layer, head): one Jaccard number.
    cross_records = []
    for r in grad_records:
        for qi_pos, e in enumerate(r["per_query"]):
            qi = e["query_i"]
            for layer_idx in range(n_layers):
                ar = attn_by_key.get((r["protein_label"], r["t_label"], layer_idx))
                if ar is None or ar["top_attn_per_query"] is None:
                    continue
                # ar["top_attn_per_query"] is [H, |sampled|, Kj] — find qi's position
                # in ar["sampled_queries"].
                try:
                    qi_in_attn = ar["sampled_queries"].index(qi)
                except ValueError:
                    continue
                # For each head, Jaccard of grad-top-K vs head's top-K-attended for qi
                per_head_jacc = []
                for h in range(len(ar["top_attn_per_query"])):
                    top_a_h = ar["top_attn_per_query"][h][qi_in_attn]
                    per_head_jacc.append(
                        jaccard_set(e["top_grad_idx"], top_a_h)
                    )
                cross_records.append(
                    {
                        "protein_label": r["protein_label"],
                        "t": r["t_label"],
                        "query_i": qi,
                        "layer_idx": layer_idx,
                        "per_head_jaccard": per_head_jacc,
                        "max_head_jaccard": max(per_head_jacc),
                        "mean_head_jaccard": sum(per_head_jacc) / len(per_head_jacc),
                    }
                )

    # Summaries
    all_max_lh = []  # max over (layer, head) per (protein, t, query)
    by_ptq: Dict[Tuple[str, float, int], List[float]] = defaultdict(list)
    for c in cross_records:
        by_ptq[(c["protein_label"], c["t"], c["query_i"])].extend(c["per_head_jaccard"])
    for key, vals in by_ptq.items():
        all_max_lh.append(max(vals))
    max_summary = (
        {
            "mean_of_max_per_ptq": sum(all_max_lh) / len(all_max_lh),
            "min_of_max_per_ptq": min(all_max_lh),
            "max_of_max_per_ptq": max(all_max_lh),
            "n": len(all_max_lh),
        }
        if all_max_lh
        else {"mean_of_max_per_ptq": float("nan")}
    )

    all_jaccards = []
    for c in cross_records:
        all_jaccards.extend(c["per_head_jaccard"])
    overall_summary = (
        {
            "mean": sum(all_jaccards) / len(all_jaccards),
            "min": min(all_jaccards),
            "max": max(all_jaccards),
            "n": len(all_jaccards),
        }
        if all_jaccards
        else {"mean": float("nan"), "n": 0}
    )
    logger.info(f"  overall mean Jaccard(grad,attn) per (q, l, h): {overall_summary}")
    logger.info(f"  per (protein, t, query) MAX over (l, h)      : {max_summary}")

    logger.info("=" * 80)
    logger.info("Decision rules:")
    logger.info("  - per-query mass_top_K ≫ aggregate's previous numbers → splitting "
                "by query recovered concentration that was washed out by aggregation.")
    logger.info("  - per-query mass_top_K ≈ or ≪ aggregate → diffuseness is intrinsic, "
                "not an artifact of aggregation.")
    logger.info("  - query-pair Jaccard low → different queries genuinely care about "
                "different residues; routing must be per-query (user's intuition).")
    logger.info("  - query-pair Jaccard high → shared important set per protein-t.")
    logger.info("  - cross-metric MAX-over-(l,h) ≥ 0.5 → dense attention DOES agree "
                "with gradient at some (l, h) for typical queries.")
    logger.info("=" * 80)

    # ------------------- Write JSON -------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "ckpt_file": args.ckpt_file,
        "label": args.label,
        "length_targets": list(length_targets),
        "t_values": list(t_values),
        "n_proteins_per_bin": args.proteins_per_bin,
        "queries_per_protein": args.queries_per_protein,
        "also_aggregate": args.also_aggregate,
        "K_grid": list(K_GRID),
        "JACCARD_K": JACCARD_K,
        "n_grad_records": len(grad_records),
        "n_attn_records": len(attn_records),
        "n_cross_records": len(cross_records),
        "check1_per_query_headline": headline_check1,
        "check2_per_query_t_jaccard": t_jacc_summary,
        "query_pair_jaccard_summary": pair_jacc_summary,
        "cross_metric_overall": overall_summary,
        "cross_metric_max_per_ptq": max_summary,
        "grad_records": grad_records,
        "attn_records": attn_records,
        "cross_records": cross_records,
    }
    out_path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
