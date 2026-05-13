"""Gradient-saliency sibling of E059's audit.

Replaces softmax attention weight with ‖∂L/∂x_t[j]‖ as the per-residue
importance metric, on the canonical dense baseline. Same scaffolding as
`audit_dense_attention_concentration.py` (same model, same seed, same
length bins, same t-grid), so outputs are paired with E059's JSON by
(protein_label, t_label).

Per-(protein, t) we record both metrics from the SAME forward pass so the
cross-metric overlap is intrinsic, not RNG-fragile.

What this measures:
  - GRADIENT (the new metric):
      ‖∂L_fm/∂x_t[j]‖_2 per residue j, where L_fm is the standard
      flow-matching loss (sum of non-_justlog components) computed against
      the protein's clean structure. One scalar per residue per (protein, t).
      → mass_top_K_grad, top-K_grad indices, t-Jaccard of top-K_grad sets.
  - ATTENTION (re-recorded for cross-metric comparison, single per-layer
    aggregate rather than E059's per-(layer, head) top_idx tensor):
      Attention-received per key j = Σ_i α[h, i, j], aggregated over heads
      via mean. One scalar per (residue, layer) per (protein, t).
      → top-K_attn indices per layer.
  - CROSS-METRIC: Jaccard(top_K_grad, top_K_attn_per_layer). For each
    (protein, t), 14 numbers (one per layer); report mean/max/min/per-layer.

Decision rules (in addition to E059's):
  - If mass_top_K_grad ≫ E059's mass_top_K_attn (e.g., top-16 captures ≥0.9
    vs E059's 0.66): gradient saliency is a sharper importance metric than
    attention; the E059 STOP conclusion needs revisiting.
  - If t-Jaccard_grad ≥ 0.7: gradient-derived important set is stable along
    the trajectory; per-protein-per-t routing computed once would suffice.
  - If max-over-layers Jaccard(grad, attn) ≥ 0.5: dense attention DOES reflect
    loss-importance at some layer(s); they're not orthogonal metrics.
  - If overlap is uniformly low (≤ 0.2 across layers): gradient saliency
    identifies a different set than dense's routing — neither metric alone
    is the right routing prior, and the question of "what should a sparse
    K-set contain" needs an inference-time test (Step 2 in the sequence).

Run from repo root, on a GPU node (needs CUDA for the backward pass):
    /home/ks2218/conda_envs/laproteina_env/bin/python \\
        script_utils/audit_dense_gradient_saliency.py \\
        --label canonical_2646_gradient

Compute: ~12-15 min on 1× A100. Forward + backward through dense; activation
memory bounded by L_max × n_layers × d × B which at L=200, n=14, d=768, B=1
fits comfortably (peak ~6 GB).
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
# Light attention hook — records ONLY attention-received-per-key, aggregated
# over heads. No per-query top_idx tensors here; this is for cross-metric
# overlap, not E059's stability check.
# ---------------------------------------------------------------------------- #
def make_light_attn_hook(records: List[dict]):
    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def _attn(self, q, k, v, b, mask):
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask_rs = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)  # [B, H, N, N]
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # Stat recording — DETACHED (does not affect backward graph).
        B, H, N, _ = attn.shape
        layer_idx = getattr(self, "_layer_idx", -1)
        t_label = getattr(self, "_t_label", float("nan"))
        protein_label = getattr(self, "_protein_label", "?")
        N_real = getattr(self, "_N_real", N)

        with torch.no_grad():
            # Attention received per key, summed over queries, mean over heads.
            attn_real = attn[0, :, :N_real, :N_real].float()  # [H, N_real, N_real]
            attn_received_per_head = attn_real.sum(dim=1)  # [H, N_real]
            attn_received_mean_heads = attn_received_per_head.mean(dim=0)  # [N_real]
            # top-K_attn (head-averaged) per layer
            Kj = min(JACCARD_K, N_real)
            top_attn_idx = attn_received_mean_heads.topk(k=Kj).indices  # [Kj]

            records.append(
                {
                    "layer_idx": int(layer_idx),
                    "t_label": float(t_label),
                    "protein_label": str(protein_label),
                    "N_real": int(N_real),
                    "attn_received_mean_heads": attn_received_mean_heads.cpu().tolist(),
                    "top_attn_idx_head_avg": top_attn_idx.cpu().tolist(),
                    "Kj": Kj,
                }
            )
        return out

    return _attn


# ---------------------------------------------------------------------------- #
# Protein loading — IDENTICAL to E059's audit_dense_attention_concentration.py
# so the protein subset and per-t draws line up across both audits.
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
# Helpers — concentration mass_top_K and Jaccard over index sets.
# ---------------------------------------------------------------------------- #
def mass_top_K(saliency: torch.Tensor, K_grid: Tuple[int, ...]) -> List[float]:
    """saliency: [N_real]. Returns [|K_grid|] = cumulative-mass fraction at top K."""
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
        help="Match E059 default; controls cross-metric overlap pairing.",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Match E059 default seed=42 to share protein subset.")
    parser.add_argument("--max_pad", type=int, default=256)
    parser.add_argument("--out_dir", default="results/dense_attn_audit")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("DATA_PATH", "/home/ks2218/la-proteina/data"),
    )
    parser.add_argument(
        "--force_precision_f32", action="store_true",
        help="Run forward+backward in fp32. Gradient saliency is less noisy in "
             "fp32 but ~2x memory/time. Default: bf16 (matches training and E059).",
    )
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    length_targets = tuple(int(x) for x in args.length_targets.split(","))
    t_values = tuple(float(x) for x in args.t_values.split(","))

    logger.info("=" * 80)
    logger.info(f"Dense-gradient-saliency audit | ckpt={args.ckpt_file}")
    logger.info(f"  length_targets={length_targets} (±{args.length_tol}), "
                f"proteins_per_bin={args.proteins_per_bin}")
    logger.info(f"  t_values={t_values}")
    logger.info(f"  precision={'fp32' if args.force_precision_f32 else 'bf16'}")
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
    nn_cfg = cfg_exp.get("nn", {})
    sparse_flag = nn_cfg.get("sparse_attention", False)
    logger.info(f"  sparse_attention flag in ckpt: {sparse_flag}")
    assert not sparse_flag, (
        f"This audit targets the DENSE path; ckpt has sparse_attention={sparse_flag}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Gradient saliency requires CUDA (backward pass)."
    model.to(device).eval()
    # Freeze all model parameters — we only want gradient w.r.t. the input x_t.
    # Saves backward memory by not accumulating gradients on the ~160M params.
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------- Install attention hook (light variant) -------------------
    attn_records: List[dict] = []
    PairBiasAttention._attn = make_light_attn_hook(attn_records)
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
        if len(items) < args.proteins_per_bin:
            logger.warning(
                f"  L~{L_target}: only {len(items)}; consider raising --max_scan"
            )

    n_layers = len(model.nn.transformer_layers)
    expected_attn_records = sum(
        len(items) * len(t_values) * n_layers for items in bins.values()
    )
    logger.info(
        f"Expected attn-layer call count: {expected_attn_records} "
        f"({sum(len(v) for v in bins.values())} proteins × "
        f"{len(t_values)} t × {n_layers} layers)"
    )

    # ------------------- Forward+backward per (protein, t) -------------------
    grad_records: List[dict] = []
    forward_count = 0
    for L_target, items in bins.items():
        items_t = apply_transforms(items, args.seed)
        for prot_idx, it in enumerate(items_t):
            protein_label = f"L{L_target}_p{prot_idx}"
            for t_val in t_values:
                bs_items = [it]
                batch_cpu = pad_and_collate(bs_items, args.max_pad)
                real_len = batch_cpu.pop("_real_lens")[0]
                batch = {k: v.to(device) for k, v in batch_cpu.items()}

                for layer in model.nn.transformer_layers:
                    layer.mhba.mha._t_label = float(t_val)
                    layer.mhba.mha._protein_label = protein_label
                    layer.mhba.mha._N_real = int(real_len)

                # Build x_t at the requested t value (matches E059's draw).
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
                # Enable grad on x_t[bb_ca]. Must clone-and-set requires_grad
                # because the original interpolate output may not be a leaf.
                x_t_bb = x_t["bb_ca"].detach().clone().requires_grad_(True)
                x_t_grad = {"bb_ca": x_t_bb}

                batch["x_0"] = x_0
                batch["x_1"] = x_1_dict
                batch["x_t"] = x_t_grad
                batch["t"] = t
                batch["mask"] = mask_proc

                # Forward through the network — graph is built since x_t_bb.requires_grad.
                # We deliberately do NOT wrap in autocast — gradient saliency is sensitive
                # to bf16 underflow, so use fp32 here unless --force_precision_f32 is off.
                if args.force_precision_f32:
                    nn_out = model.call_nn(batch, n_recycle=0)
                else:
                    nn_out = model.call_nn(batch, n_recycle=0)
                losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
                per_proto = sum(
                    losses[k] for k in losses if "_justlog" not in k
                )  # [B]
                loss = per_proto.sum()  # scalar

                # Backward: gradient w.r.t. x_t_bb only (model params are frozen).
                loss.backward()
                grad = x_t_bb.grad  # [B, n_pad, 3]
                assert grad is not None, "x_t.grad is None — graph broken?"
                # Saliency per residue = L2 norm of the 3-D gradient at each residue.
                saliency = grad[0, :real_len].norm(dim=-1).detach().float().cpu()  # [N_real]

                mtk = mass_top_K(saliency, K_GRID)
                Kj = min(JACCARD_K, int(saliency.numel()))
                top_grad_idx = saliency.topk(k=Kj).indices.tolist()

                grad_records.append(
                    {
                        "protein_label": protein_label,
                        "t_label": float(t_val),
                        "N_real": int(real_len),
                        "saliency": saliency.tolist(),
                        "K_grid": list(K_GRID),
                        "mass_top_K": mtk,
                        "top_grad_idx": top_grad_idx,
                        "Kj": Kj,
                        "loss": float(loss.item()),
                    }
                )

                forward_count += 1
                if forward_count % 5 == 0:
                    logger.info(
                        f"  fwd+bwd={forward_count} (latest: {protein_label}, "
                        f"t={t_val}, loss={loss.item():.4f}, "
                        f"mass_top_16={mtk[1]:.3f})"
                    )

                # Free graph memory.
                del nn_out, losses, per_proto, loss, x_t_bb, grad
                torch.cuda.empty_cache()

    logger.info(
        f"Forwards complete: {forward_count}. "
        f"grad_records={len(grad_records)}, attn_records={len(attn_records)}"
    )

    # ------------------- Aggregate Check 1' (gradient concentration) -------------------
    logger.info("Aggregating Check 1' (gradient saliency concentration)...")
    mtk_all = torch.tensor([r["mass_top_K"] for r in grad_records])  # [n, |K|]
    grand_mean = mtk_all.mean(dim=0).tolist()
    grand_med = mtk_all.median(dim=0).values.tolist()
    headline_check1 = {
        "K_grid": list(K_GRID),
        "grand_mean_mass_top_K_grad": grand_mean,
        "grand_median_mass_top_K_grad": grand_med,
        "n_records": int(mtk_all.shape[0]),
    }
    logger.info("  HEADLINE Check 1' (gradient saliency):")
    logger.info(f"    K_grid       = {list(K_GRID)}")
    logger.info(f"    mass_top_K   = {[f'{x:.3f}' for x in grand_mean]}  (mean over {len(grad_records)} records)")
    logger.info(f"    median       = {[f'{x:.3f}' for x in grand_med]}")
    logger.info("  E059 attention reference (recall):")
    logger.info("    mass_top_K_attn ≈ [0.510, 0.656, 0.794, 0.866, 0.907] at K=[8,16,32,48,64]")

    # ------------------- Aggregate Check 2' (t-Jaccard of top-K_grad) -------------------
    logger.info("Aggregating Check 2' (t-Jaccard of gradient top-K)...")
    by_pt: Dict[str, Dict[float, List[int]]] = defaultdict(dict)
    for r in grad_records:
        by_pt[r["protein_label"]][r["t_label"]] = r["top_grad_idx"]
    t_jaccards_grad: List[float] = []
    for prot, t_to_idx in by_pt.items():
        ts = sorted(t_to_idx.keys())
        for t1, t2 in zip(ts[:-1], ts[1:]):
            t_jaccards_grad.append(jaccard_set(t_to_idx[t1], t_to_idx[t2]))
    t_jacc_grad_summary = (
        {
            "mean": sum(t_jaccards_grad) / len(t_jaccards_grad),
            "min": min(t_jaccards_grad),
            "max": max(t_jaccards_grad),
            "n": len(t_jaccards_grad),
        }
        if t_jaccards_grad
        else {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    )
    logger.info(f"  t-Jaccard_grad: {t_jacc_grad_summary}")
    logger.info("  E059 attention reference: t-Jaccard_attn = 0.475 mean (n=126)")

    # ------------------- Cross-metric: Jaccard(top_K_grad, top_K_attn_per_layer) -------------------
    logger.info("Aggregating cross-metric overlap (grad ∩ attn-head-avg)...")
    # Index attn records by (protein, t, layer)
    attn_by_key: Dict[Tuple[str, float, int], List[int]] = {}
    for r in attn_records:
        attn_by_key[(r["protein_label"], r["t_label"], r["layer_idx"])] = r[
            "top_attn_idx_head_avg"
        ]

    cross_per_record: Dict[str, dict] = {}
    layer_means: Dict[int, List[float]] = defaultdict(list)
    layer_max_record: Dict[str, dict] = {}
    for r in grad_records:
        key_prefix = (r["protein_label"], r["t_label"])
        per_layer = {}
        for layer_idx in range(n_layers):
            top_a = attn_by_key.get((*key_prefix, layer_idx))
            if top_a is None:
                continue
            j = jaccard_set(r["top_grad_idx"], top_a)
            per_layer[layer_idx] = j
            layer_means[layer_idx].append(j)
        if per_layer:
            vals = list(per_layer.values())
            cross_per_record[f"{r['protein_label']}_t{r['t_label']:.2f}"] = {
                "per_layer": per_layer,
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "max_at_layer": max(per_layer, key=per_layer.get),
            }
            layer_max_record[
                f"{r['protein_label']}_t{r['t_label']:.2f}"
            ] = {
                "max_jaccard": max(vals),
                "max_at_layer": max(per_layer, key=per_layer.get),
            }

    # Per-layer aggregates
    per_layer_summary = {
        layer_idx: {
            "mean": sum(v) / len(v) if v else float("nan"),
            "min": min(v) if v else float("nan"),
            "max": max(v) if v else float("nan"),
            "n": len(v),
        }
        for layer_idx, v in layer_means.items()
    }
    # Headline: overall mean and max across all (protein, t, layer) triples
    all_cross = []
    for r in grad_records:
        for layer_idx in range(n_layers):
            top_a = attn_by_key.get((r["protein_label"], r["t_label"], layer_idx))
            if top_a is not None:
                all_cross.append(jaccard_set(r["top_grad_idx"], top_a))
    cross_overall = (
        {
            "mean": sum(all_cross) / len(all_cross),
            "min": min(all_cross),
            "max": max(all_cross),
            "n": len(all_cross),
        }
        if all_cross
        else {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    )
    # Per-(protein, t) MAX (best agreement at any layer): the headline for
    # "do attention and gradient EVER agree somewhere?"
    per_pt_max = [v["max_jaccard"] for v in layer_max_record.values()]
    cross_max_summary = (
        {
            "mean_of_per_pt_max": sum(per_pt_max) / len(per_pt_max),
            "min_of_per_pt_max": min(per_pt_max),
            "max_of_per_pt_max": max(per_pt_max),
            "n": len(per_pt_max),
        }
        if per_pt_max
        else {"mean_of_per_pt_max": float("nan")}
    )
    logger.info(f"  cross-metric Jaccard(grad, attn-head-avg):")
    logger.info(f"    overall (all layer × protein × t): {cross_overall}")
    logger.info(f"    per-(protein,t) MAX over layers : {cross_max_summary}")
    logger.info(f"    per-layer means: {{layer_idx: mean}}")
    for layer_idx in sorted(per_layer_summary):
        s = per_layer_summary[layer_idx]
        logger.info(
            f"      layer {layer_idx:>2d}: mean={s['mean']:.3f}, "
            f"min={s['min']:.3f}, max={s['max']:.3f}, n={s['n']}"
        )

    logger.info("Decision rules:")
    logger.info("  - mass_top_K_grad ≫ mass_top_K_attn (E059 ref) → gradient is sharper.")
    logger.info("  - t-Jaccard_grad ≥ 0.7 → gradient important set stable over trajectory.")
    logger.info("  - max-over-layers Jaccard ≥ 0.5 → attention DOES reflect loss-importance somewhere.")
    logger.info("  - mean Jaccard ≤ 0.2 across layers → metrics orthogonal; routing-prior question open.")

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
        "force_precision_f32": bool(args.force_precision_f32),
        "n_grad_records": len(grad_records),
        "n_attn_records": len(attn_records),
        "check1_grad_headline": headline_check1,
        "check2_grad_t_jaccard_summary": t_jacc_grad_summary,
        "cross_metric_overall": cross_overall,
        "cross_metric_per_pt_max_summary": cross_max_summary,
        "cross_metric_per_layer": per_layer_summary,
        "cross_metric_per_record": cross_per_record,
        # Per-record dumps for any post-hoc analysis you want to redo.
        "grad_records": grad_records,
        "attn_records": attn_records,
    }
    out_path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
