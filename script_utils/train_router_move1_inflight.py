"""In-flight router training using the canonical PDB dataloader.

Differences vs script_utils/train_router_move1.py:
  - No offline teacher .pt files; teacher signal is extracted on every step
    by running the frozen dense trunk forward and capturing top-K attention
    via the same monkey-patch + forward-pre-hook pattern as the offline
    extraction.
  - Uses PDBLightningDataModule (canonical training distribution: L=50-512,
    cluster-random sampling, sequence-similarity 50% train/val split).
  - t and x_t are produced by dense's own `fm.corrupt_batch` (canonical
    flow-matching corruption recipe), so the router sees exactly the
    (layer_inputs, t, x_t) the dense forward saw — bit-perfect alignment.
  - Validation set is cached once at start (raw batches) and re-evaluated
    each round; teacher is re-captured per eval, so the val recall numbers
    are deterministic.

Architectural default: bigrouter (hidden_dim=256, score_dim=64, mlp_block) +
pair_features=True + use_t_emb=True. All three Move-1 axes stacked.

Memory: dense fp16 + router fp32 at B=6 N=512 should land ~30-40 GB on A100.
If OOM, reduce --batch_size to 4.
"""
import argparse
import contextlib
import csv
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
from proteinfoundation.nn.modules.router import TopKRouter, TopKRouterT
from proteinfoundation.proteina import Proteina


CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)
N_LAYERS = 14
N_HEADS = 12
K_TOP = 64
TRUNK_DIM = 768


# ---------------------------------------------------------------------------- #
# Capture state. Tensors live on GPU during a single forward; cleared after
# each router step.
# ---------------------------------------------------------------------------- #
class CaptureBuf:
    def __init__(self):
        self.layer_inputs: Dict[int, torch.Tensor] = {}
        self.top_k_indices: Dict[int, torch.Tensor] = {}
        self.top_k_weights: Dict[int, torch.Tensor] = {}

    def reset(self):
        self.layer_inputs.clear()
        self.top_k_indices.clear()
        self.top_k_weights.clear()

    def stack(self):
        """Stack the per-layer captures into batched tensors.
        Returns:
            layer_inputs: [B, L, N, D] fp32
            top_k_indices: [B, L, H, N, K] long
            top_k_weights: [B, L, H, N, K] fp32
        """
        L_ = len(self.layer_inputs)
        li = torch.stack([self.layer_inputs[i] for i in range(L_)], dim=1).float()
        idx = torch.stack([self.top_k_indices[i] for i in range(L_)], dim=1).long()
        wts = torch.stack([self.top_k_weights[i] for i in range(L_)], dim=1).float()
        return li, idx, wts


BUF = CaptureBuf()


def make_capture_attn():
    """Class-level replacement for PairBiasAttention._attn. Bit-identical
    compute path, plus a top-K snapshot per layer recorded into BUF."""
    from einops import rearrange
    from torch import einsum

    max_neg = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def _attn(self_attn, q, k, v, b, mask):
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self_attn.scale
        if mask is not None:
            mask_rs = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # Snapshot top-K. Padding keys have attn=0 because of -inf mask, so
        # top-K picks real keys whenever N_real >= K. For N_real < K, the
        # weight==0 slot_valid mask handles the loss correctly.
        layer_idx = getattr(self_attn, "_layer_idx", -1)
        assert 0 <= layer_idx < N_LAYERS, f"_layer_idx missing on layer: {layer_idx}"
        top = attn.float().topk(K_TOP, dim=-1)
        BUF.top_k_indices[layer_idx] = top.indices.to(torch.int16)
        BUF.top_k_weights[layer_idx] = top.values.to(torch.float16)
        return out

    return _attn


def make_layer_input_hook(layer_idx: int):
    """Pre-hook on each transformer_layers[i]. Captures the incoming seqs
    tensor (shape [B, N, D]) and stores it in BUF."""
    def _pre(module, args, kwargs):
        seqs = args[0] if len(args) > 0 else kwargs.get("seqs", None)
        assert seqs is not None and seqs.ndim == 3, f"unexpected seqs at layer {layer_idx}"
        BUF.layer_inputs[layer_idx] = seqs.detach().to(torch.float16)
        return None
    return _pre


# ---------------------------------------------------------------------------- #
# Loss + recall — same KL definition as the offline trainer.
# ---------------------------------------------------------------------------- #
def kl_loss(scores, top_k_idx, top_k_wts, slot_valid, query_mask):
    B, L_, H, N, _ = scores.shape
    key_mask = query_mask.view(B, 1, 1, 1, N).expand(B, L_, H, N, N)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    log_p = F.log_softmax(scores, dim=-1)
    log_p_at_top = log_p.gather(-1, top_k_idx)
    eps = 1e-12
    log_w = torch.log(top_k_wts.clamp(min=eps))
    contrib = top_k_wts * (log_w - log_p_at_top) * slot_valid.float()
    per_query = contrib.sum(dim=-1)
    q_mask_lh = query_mask.view(B, 1, 1, N).expand_as(per_query).float()
    n_valid = q_mask_lh.sum().clamp(min=1.0)
    loss = (per_query * q_mask_lh).sum() / n_valid
    return loss, n_valid


@torch.no_grad()
def compute_recall_stats(scores, top_k_idx, slot_valid, query_mask, t_vals):
    """Mirror of eval_recall_at_64 from the offline trainer, simplified for
    a single batch. Returns dicts of running totals to accumulate over batches."""
    B, L_, H, N, _ = scores.shape
    key_mask = query_mask.view(B, 1, 1, 1, N).expand_as(scores)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    router_top = scores.topk(K_TOP, dim=-1).indices

    r_oh = torch.zeros(B, L_, H, N, N, dtype=torch.bool, device=scores.device)
    r_oh.scatter_(-1, router_top, True)
    t_oh = torch.zeros_like(r_oh)
    valid_idx = top_k_idx.masked_fill(~slot_valid, 0)
    t_oh.scatter_(-1, valid_idx, True)
    # Clear spurious "0" entries that came purely from invalid slots.
    any_zero_valid = (slot_valid & (top_k_idx == 0)).any(dim=-1)
    any_zero_invalid = (~slot_valid & (top_k_idx == 0)).any(dim=-1)
    clear_zero = (~any_zero_valid) & any_zero_invalid
    t_oh[..., 0] = t_oh[..., 0] & ~clear_zero

    inter = (r_oh & t_oh).sum(dim=-1).float()
    denom = slot_valid.float().sum(dim=-1).clamp(min=1.0)
    recall = inter / denom

    q_mask = query_mask.view(B, 1, 1, N).expand_as(recall).float()
    return {
        "overall_sum": (recall * q_mask).sum().item(),
        "overall_cnt": q_mask.sum().item(),
        "per_layer": [(recall[:, li] * q_mask[:, li]).sum().item()
                      / max(1.0, q_mask[:, li].sum().item()) for li in range(L_)],
        "per_layer_cnt": [q_mask[:, li].sum().item() for li in range(L_)],
        "per_lh": [[(recall[:, li, hi] * q_mask[:, li, hi]).sum().item(),
                    q_mask[:, li, hi].sum().item()] for hi in range(H) for li in range(L_)],
        "per_t": {
            float(t_vals[bi].item()): ((recall[bi] * q_mask[bi]).sum().item(),
                                        q_mask[bi].sum().item())
            for bi in range(B)
        },
    }


# ---------------------------------------------------------------------------- #
# Single-batch teacher-extraction pass through dense.
# ---------------------------------------------------------------------------- #
def _to_device_recursive(obj, device):
    """Recursively move tensors in nested dicts/lists. The PDB batch is a
    PyG DataBatch object — handled via its built-in .to(device)."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    try:
        from torch_geometric.data.data import BaseData
        if isinstance(obj, BaseData):
            return obj.to(device)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_device_recursive(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(_to_device_recursive(x, device) for x in obj)
    return obj


_DEBUG_BATCH_ONCE = {"done": False}


def _debug_batch(batch, label: str):
    """One-time print of batch structure + tensor devices."""
    if _DEBUG_BATCH_ONCE["done"]:
        return
    _DEBUG_BATCH_ONCE["done"] = True
    logger.info(f"=== DEBUG {label} ===")
    def walk(obj, prefix):
        if torch.is_tensor(obj):
            logger.info(f"  {prefix}: Tensor dev={obj.device} dtype={obj.dtype} shape={tuple(obj.shape)}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, (list, tuple)):
            logger.info(f"  {prefix}: list/tuple len={len(obj)}")
        else:
            logger.info(f"  {prefix}: type={type(obj).__name__}")
    walk(batch, "")


def dense_forward_capture(dense, batch, device, use_amp=True):
    """Move batch to device, do canonical-style corruption, run dense forward
    in eval mode. After this call, BUF is populated with the layer inputs +
    top-K captures. The batch dict is mutated to add x_t, x_0, x_1, t, etc."""
    BUF.reset()
    _debug_batch(batch, "raw batch from dataloader")
    batch = _to_device_recursive(batch, device)
    _debug_batch(batch, "after _to_device_recursive")
    batch = dense.add_clean_samples(batch)
    batch = dense.fm.corrupt_batch(batch)
    _debug_batch(batch, "after corrupt_batch (pre-dense)")
    ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp \
        else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        _ = dense.call_nn(batch, n_recycle=0)
    return batch


def router_forward_from_capture(router, batch, use_t_emb, use_pair):
    """Stack the captured teacher tensors, then run the router forward."""
    li, idx, wts = BUF.stack()  # all on GPU
    slot_valid = wts > 0
    query_mask = batch["mask"].bool()
    t = batch["t"]["bb_ca"].float()
    coords = batch["x_t"]["bb_ca"] if use_pair else None

    if use_t_emb:
        scores = router(li, t, coords_nm=coords)
    else:
        scores = router(li, coords_nm=coords)
    return scores, idx, wts, slot_valid, query_mask, t


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_file", default=CKPT_DEFAULT)
    ap.add_argument("--out_dir", default="results/router_move1_inflight")
    ap.add_argument("--data_cfg",
                    default="configs/dataset/pdb/pdb_train_ucond.yaml")
    ap.add_argument("--n_steps", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=6,
                    help="Per-step training batch (matches canonical dense training).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_padding_size", type=int, default=512)
    ap.add_argument("--worst_resolution", type=float, default=2.0,
                    help="Match dense ckpt's training_ca_only.yaml override (2.0 Å).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_val_batches", type=int, default=20)
    # Router architecture flags (defaults: bigrouter + pair + t-emb)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--score_dim", type=int, default=64)
    ap.add_argument("--mlp_block", action="store_true", default=True)
    ap.add_argument("--no_mlp_block", dest="mlp_block", action="store_false")
    ap.add_argument("--pair_features", action="store_true", default=True)
    ap.add_argument("--no_pair_features", dest="pair_features", action="store_false")
    ap.add_argument("--use_t_emb", action="store_true", default=True)
    ap.add_argument("--no_use_t_emb", dest="use_t_emb", action="store_false")
    ap.add_argument("--mlp_dim", type=int, default=256)
    # Early-stop
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_improve", type=float, default=1e-3)
    # Resume from a checkpoint of the router (not the dense)
    ap.add_argument("--resume_from", type=str, default=None)
    args = ap.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"out_dir = {out_dir}")
    logger.info(f"router arch: hidden={args.hidden_dim} score={args.score_dim} "
                f"mlp_block={args.mlp_block} pair_features={args.pair_features} "
                f"use_t_emb={args.use_t_emb}")

    # ---- Load dense ----
    logger.info(f"loading dense ckpt {args.ckpt_file}")
    dense = Proteina.load_from_checkpoint(
        args.ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    nn_cfg = dense.cfg_exp.get("nn", {})
    assert not nn_cfg.get("sparse_attention", False), "dense ckpt has sparse_attention"
    assert nn_cfg.get("token_dim", -1) == TRUNK_DIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "in-flight training requires CUDA"
    # Unwrap torch.compile wrapper BEFORE moving to device — otherwise
    # _orig_mod's submodules may not get walked by .to(device).
    if hasattr(dense.nn, "_orig_mod"):
        dense.nn = dense.nn._orig_mod
    dense.to(device).eval()
    dense.nn.to(device)  # belt and suspenders
    for p in dense.parameters():
        p.requires_grad_(False)
    # Sanity log: confirm every dense submodule is on cuda.
    cpu_params = [n for n, p in dense.named_parameters() if p.device.type != "cuda"]
    cpu_bufs   = [n for n, b in dense.named_buffers()    if b.device.type != "cuda"]
    if cpu_params or cpu_bufs:
        logger.warning(f"params on CPU after .to(device): {cpu_params}")
        logger.warning(f"buffers on CPU after .to(device): {cpu_bufs}")
    else:
        logger.info("all dense params + buffers on cuda")

    # ---- Install capture hooks ----
    PairBiasAttention._attn = make_capture_attn()
    hook_handles = []
    for i, layer in enumerate(dense.nn.transformer_layers):
        layer.mhba.mha._layer_idx = i
        hook_handles.append(
            layer.register_forward_pre_hook(make_layer_input_hook(i), with_kwargs=True)
        )

    # ---- Build canonical dataloader ----
    logger.info(f"loading data config {args.data_cfg}")
    cfg = OmegaConf.load(args.data_cfg)
    OmegaConf.resolve(cfg)
    # Override a few knobs for our use case (fewer workers; same batch size).
    cfg.datamodule.batch_size = args.batch_size
    cfg.datamodule.num_workers = args.num_workers
    cfg.datamodule.max_padding_size = args.max_padding_size
    # The base pdb_train_ucond.yaml has worst_resolution=5.0; training_ca_only.yaml
    # (which produced our dense ckpt) overrides this to 2.0. Apply the same
    # override here so the CSV the splitter expects actually exists on disk.
    cfg.datamodule.dataselector.worst_resolution = args.worst_resolution
    dm = instantiate(cfg.datamodule)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    logger.info(f"train loader: ~{len(train_loader)} batches / epoch")
    logger.info(f"val loader:   ~{len(val_loader)} batches available")

    # Cache the first n_val_batches val batches (raw, on CPU).
    val_batches_cpu = []
    val_iter = iter(val_loader)
    for _ in range(args.n_val_batches):
        try:
            b = next(val_iter)
        except StopIteration:
            break
        val_batches_cpu.append(
            {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()}
        )
    logger.info(f"cached {len(val_batches_cpu)} val batches for fixed eval")

    # ---- Build router ----
    router_kwargs = dict(
        trunk_dim=TRUNK_DIM, hidden_dim=args.hidden_dim, score_dim=args.score_dim,
        n_layers=N_LAYERS, n_heads=N_HEADS,
        pair_features=args.pair_features, mlp_block=args.mlp_block, mlp_dim=args.mlp_dim,
    )
    router = TopKRouterT(**router_kwargs).to(device) if args.use_t_emb \
        else TopKRouter(**router_kwargs).to(device)
    logger.info(f"router params: {router.num_params():,}")

    start_step = 0
    resume_recall = 0.0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        router.load_state_dict(ckpt["state_dict"])
        start_step = int(ckpt.get("step", 0))
        resume_recall = float(ckpt.get("recall@64", ckpt.get("best_recall@64", 0.0)))
        logger.info(f"resumed from {args.resume_from} step={start_step} "
                    f"prev_recall@64={resume_recall:.4f}")

    opt = torch.optim.AdamW(router.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    csv_mode = "a" if args.resume_from is not None else "w"
    log_csv = out_dir / "training_log.csv"
    csv_f = open(log_csv, csv_mode, newline="")
    csv_w = csv.writer(csv_f)
    if csv_mode == "w":
        csv_w.writerow(["step", "loss", "n_valid_queries", "wall_s_per_step"])

    # ---- Training loop ----
    router.train()
    step = start_step
    t0 = time.time()
    last_log = time.time()
    best_recall = resume_recall
    best_step = start_step
    no_improve = 0
    stopped_early = False
    train_iter = iter(train_loader)
    target_step = start_step + args.n_steps

    while step < target_step:
        try:
            batch = next(train_iter)
        except StopIteration:
            logger.info("re-iter train loader (epoch boundary)")
            train_iter = iter(train_loader)
            batch = next(train_iter)

        try:
            batch = dense_forward_capture(dense, batch, device, use_amp=True)
            scores, idx, wts, slot_valid, query_mask, t = router_forward_from_capture(
                router, batch, args.use_t_emb, args.pair_features
            )
            loss, n_valid = kl_loss(scores, idx, wts, slot_valid, query_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at step {step}; skipping batch.")
            torch.cuda.empty_cache()
            continue

        step += 1
        if step % args.log_every == 0 or step == start_step + 1:
            dt = (time.time() - last_log) / max(1, args.log_every)
            logger.info(f"step {step:>5d} | loss {loss.item():.4f} | "
                        f"nq {int(n_valid.item())} | {dt:.2f}s/step")
            csv_w.writerow([step, float(loss.item()), int(n_valid.item()), dt])
            csv_f.flush()
            last_log = time.time()

        if step % args.eval_every == 0 or step == target_step:
            logger.info(f"[eval @ step {step}] re-extracting val teacher ...")
            t_eval = time.time()
            router.eval()
            overall_sum, overall_cnt = 0.0, 0.0
            per_layer_sum = [0.0] * N_LAYERS
            per_layer_cnt = [0.0] * N_LAYERS
            per_lh_sum = [[0.0, 0.0] for _ in range(N_LAYERS * N_HEADS)]
            per_t = defaultdict(lambda: [0.0, 0.0])
            for vb in val_batches_cpu:
                vb_in = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in vb.items()}
                vb_in = dense_forward_capture(dense, vb_in, device, use_amp=True)
                scores, idx, wts, slot_valid, query_mask, t_val = router_forward_from_capture(
                    router, vb_in, args.use_t_emb, args.pair_features
                )
                with torch.no_grad():
                    stats = compute_recall_stats(scores, idx, slot_valid, query_mask, t_val)
                overall_sum += stats["overall_sum"]
                overall_cnt += stats["overall_cnt"]
                for i, (s, c) in enumerate(zip(stats["per_layer"], stats["per_layer_cnt"])):
                    per_layer_sum[i] += s * c
                    per_layer_cnt[i] += c
                for idx_lh, (s, c) in enumerate(stats["per_lh"]):
                    per_lh_sum[idx_lh][0] += s
                    per_lh_sum[idx_lh][1] += c
                for t_v, (s, c) in stats["per_t"].items():
                    per_t[round(t_v, 2)][0] += s
                    per_t[round(t_v, 2)][1] += c

            overall_recall = overall_sum / max(1.0, overall_cnt)
            layer_table = {f"layer_{li}": per_layer_sum[li] / max(1.0, per_layer_cnt[li])
                           for li in range(N_LAYERS)}
            lh_table = {}
            for hi in range(N_HEADS):
                for li in range(N_LAYERS):
                    s, c = per_lh_sum[hi * N_LAYERS + li]
                    lh_table[f"L{li}_H{hi}"] = s / max(1.0, c)
            t_table = {f"t_{tv:.2f}": per_t[tv][0] / max(1.0, per_t[tv][1])
                       for tv in sorted(per_t)}
            stats_out = {
                "overall_recall@64": overall_recall,
                "per_layer": layer_table,
                "per_t": t_table,
                "n_queries_evaluated": int(overall_cnt),
            }
            router.train()

            logger.info(f"  overall recall@64 = {overall_recall:.4f} "
                        f"({time.time() - t_eval:.1f}s)")
            with open(out_dir / f"eval_step{step:05d}.json", "w") as f:
                json.dump(stats_out, f, indent=2)
            with open(out_dir / f"eval_step{step:05d}_perLH.json", "w") as f:
                json.dump(lh_table, f, indent=2)

            if overall_recall > best_recall + args.min_improve:
                best_recall = overall_recall; best_step = step; no_improve = 0
                torch.save({
                    "state_dict": router.state_dict(),
                    "config": {
                        "trunk_dim": TRUNK_DIM, "hidden_dim": args.hidden_dim,
                        "score_dim": args.score_dim, "n_layers": N_LAYERS,
                        "n_heads": N_HEADS, "use_t_emb": args.use_t_emb,
                        "pair_features": args.pair_features,
                        "mlp_block": args.mlp_block, "mlp_dim": args.mlp_dim,
                    },
                    "step": step, "recall@64": overall_recall,
                    "args": vars(args),
                }, out_dir / "router_best.pt")
                logger.info(f"  new best recall@64 = {overall_recall:.4f} @ step {step}")
            else:
                no_improve += 1
                logger.info(f"  no-improve {no_improve}/{args.patience} "
                            f"(best={best_recall:.4f} @ step {best_step})")
                if args.patience > 0 and no_improve >= args.patience:
                    logger.info(f"Early stop: best={best_recall:.4f} @ step {best_step}")
                    stopped_early = True
                    break

    csv_f.close()
    final_path = out_dir / "router_final.pt"
    torch.save({
        "state_dict": router.state_dict(),
        "config": {
            "trunk_dim": TRUNK_DIM, "hidden_dim": args.hidden_dim,
            "score_dim": args.score_dim, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "use_t_emb": args.use_t_emb, "pair_features": args.pair_features,
            "mlp_block": args.mlp_block, "mlp_dim": args.mlp_dim,
        },
        "args": vars(args),
        "wall_total_s": time.time() - t0,
        "stopped_early": stopped_early,
        "best_recall@64": best_recall, "best_step": best_step, "final_step": step,
    }, final_path)
    logger.info(f"saved {final_path}; wall {time.time()-t0:.1f}s; "
                f"final_step={step}; best={best_recall:.4f} @ step {best_step}; "
                f"stopped_early={stopped_early}")

    for h in hook_handles:
        h.remove()


if __name__ == "__main__":
    main()
