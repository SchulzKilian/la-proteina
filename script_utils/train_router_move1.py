"""Move-1 router training (E065).

Trains a tiny TopKRouter to predict the canonical dense model's per-(layer,
head, query) top-64 attention pattern from each layer's input hidden state.
Teacher data: precomputed .pt files from extract_router_teacher_data.py.

Loss: per-query KL(teacher || predicted), where teacher is a sparse softmax
over the top-64 keys (the saved weights, already normalized) and predicted
is full-softmax over all N real keys. Since teacher has zero mass on
non-top-64 keys, KL reduces to:
    Σ_{j ∈ top_64} w_j * (log w_j - log_softmax(scores)[j])

Train split: proteins 0..449   (first 90% of unique IDs in the manifest).
Eval split:  proteins 450..499 (last 10%).
Decision metric: held-out mean recall@64 of router top-64 vs teacher top-64.

Outputs:
  results/router_move1/training_log.csv
  results/router_move1/eval_step<N>.json
  results/router_move1/router_final.pt
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset

from proteinfoundation.nn.modules.router import TopKRouter, TopKRouterT


N_LAYERS = 14
N_HEADS = 12
K_TOP = 64
TRUNK_DIM = 768


class TeacherDataset(Dataset):
    """Each item = one (protein, t) .pt file. Returns tensors directly."""
    def __init__(self, files: List[str]):
        self.files = sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        rec = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return rec


def load_record(path: str, with_coords: bool = False) -> Dict:
    """Load the main teacher .pt; optionally also load the coords sidecar
    (<pdb_id>_t<t>_coords.pt) and merge bb_ca into the record."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    if with_coords:
        sidecar = path.replace(".pt", "_coords.pt")
        if path.endswith("_coords.pt"):
            raise ValueError(f"unexpected coords path passed as main: {path}")
        try:
            cr = torch.load(sidecar, map_location="cpu", weights_only=False)
            assert int(cr["N"]) == int(rec["N"]), \
                f"N mismatch for {os.path.basename(path)}: main={rec['N']} coords={cr['N']}"
            rec["bb_ca"] = cr["bb_ca"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"--pair_features set but coords sidecar missing: {sidecar}. "
                f"Run extract_router_teacher_coords.py first."
            )
    return rec


def split_by_protein(all_files: List[str], train_frac: float = 0.9, seed: int = 42
                     ) -> Tuple[List[str], List[str]]:
    """Group files by protein_id (filename prefix), shuffle proteins, split."""
    by_pid: Dict[str, List[str]] = defaultdict(list)
    for f in all_files:
        # Filename pattern: <pdb_id>_t<t:.2f>.pt
        base = os.path.basename(f)
        pid = base.rsplit("_t", 1)[0]
        by_pid[pid].append(f)
    pids = sorted(by_pid.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_train = int(round(train_frac * len(pids)))
    train_pids = set(pids[:n_train])
    train, val = [], []
    for pid, fs in by_pid.items():
        (train if pid in train_pids else val).extend(fs)
    return sorted(train), sorted(val)


def make_length_buckets(files: List[str], bucket_step: int = 16
                        ) -> Dict[int, List[str]]:
    """Bucket by N (length) for length-balanced batching. Loading every file
    just for shape is wasteful — peek the saved tensor metadata only.
    """
    buckets: Dict[int, List[str]] = defaultdict(list)
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        n = int(rec["N"])
        b = (n // bucket_step) * bucket_step
        buckets[b].append(f)
    return buckets


def collate_batch(records: List[Dict]) -> Dict:
    """Pad-to-longest-in-batch. Records can have different N.
    Returns:
        layer_inputs: [B, L, N_pad, D] fp32
        top_k_indices: [B, L, H, N_pad, K] long  (padding rows: zeros, masked)
        top_k_weights: [B, L, H, N_pad, K] fp32  (padding rows: zeros)
        slot_valid:   [B, L, H, N_pad, K] bool   (True for real teacher entries)
        query_mask:   [B, N_pad] bool   (True for real queries)
        key_mask:     [B, N_pad] bool   (True for real keys; ==query_mask here)
        N_real:       [B] long
    """
    B = len(records)
    Ns = [int(r["N"]) for r in records]
    N_max = max(Ns)
    layer_inputs = torch.zeros(B, N_LAYERS, N_max, TRUNK_DIM, dtype=torch.float32)
    top_k_indices = torch.zeros(B, N_LAYERS, N_HEADS, N_max, K_TOP, dtype=torch.long)
    top_k_weights = torch.zeros(B, N_LAYERS, N_HEADS, N_max, K_TOP, dtype=torch.float32)
    slot_valid = torch.zeros(B, N_LAYERS, N_HEADS, N_max, K_TOP, dtype=torch.bool)
    query_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, r in enumerate(records):
        n = int(r["N"])
        layer_inputs[i, :, :n, :] = r["layer_inputs"].float()
        idx_i = r["top_k_indices"].long()            # [L, H, n, K]
        wts_i = r["top_k_weights"].float()           # [L, H, n, K]
        top_k_indices[i, :, :, :n, :] = idx_i
        top_k_weights[i, :, :, :n, :] = wts_i
        # A teacher slot is valid iff (real query) AND (real key index < n)
        # AND the weight is > 0. Real-key-index < n is implied by extraction
        # (we only took top-K of the [n, n] subblock). The wts>0 check covers
        # the K_eff < K_TOP pad-with-zero case for short proteins.
        slot_valid[i, :, :, :n, :] = wts_i > 0
        query_mask[i, :n] = True
    t_values = torch.tensor([float(r["t_value"]) for r in records], dtype=torch.float32)
    out = {
        "layer_inputs": layer_inputs,
        "top_k_indices": top_k_indices,
        "top_k_weights": top_k_weights,
        "slot_valid": slot_valid,
        "query_mask": query_mask,
        "N_real": torch.tensor(Ns, dtype=torch.long),
        "t_values": t_values,
    }
    if "bb_ca" in records[0]:
        coords = torch.zeros(B, N_max, 3, dtype=torch.float32)
        for i, r in enumerate(records):
            n = int(r["N"])
            coords[i, :n, :] = r["bb_ca"].float()
        out["coords_nm"] = coords
    return out


def kl_loss(
    scores: torch.Tensor,       # [B, L, H, N, N]
    top_k_idx: torch.Tensor,    # [B, L, H, N, K]
    top_k_wts: torch.Tensor,    # [B, L, H, N, K]  (sparse teacher weights)
    slot_valid: torch.Tensor,   # [B, L, H, N, K]  bool
    query_mask: torch.Tensor,   # [B, N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KL(teacher || predicted). Predicted softmax is computed over real keys
    only (mask padding to -inf). Returns (mean_loss, n_valid_queries).
    """
    B, L, H, N, _ = scores.shape
    # Mask padding keys before log-softmax.
    key_mask = query_mask.view(B, 1, 1, 1, N).expand(B, L, H, N, N)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    log_p = F.log_softmax(scores, dim=-1)                  # [B, L, H, N, N]
    # Gather log_p at top-K teacher indices.
    log_p_at_top = log_p.gather(-1, top_k_idx)             # [B, L, H, N, K]
    # KL contribution: w_j * (log w_j - log p_j). Use eps to safely log w_j;
    # slot_valid masks the eps-only entries out.
    eps = 1e-12
    log_w = torch.log(top_k_wts.clamp(min=eps))
    contrib = top_k_wts * (log_w - log_p_at_top)           # [B, L, H, N, K]
    contrib = contrib * slot_valid.float()
    # Sum over K slots → per-(b, l, h, n) KL. Then average over (l, h, valid n).
    per_query_kl = contrib.sum(dim=-1)                     # [B, L, H, N]
    q_mask_lh = query_mask.view(B, 1, 1, N).expand_as(per_query_kl)
    n_valid = q_mask_lh.float().sum().clamp(min=1.0)
    loss = (per_query_kl * q_mask_lh.float()).sum() / n_valid
    return loss, n_valid


@torch.no_grad()
def eval_recall_at_64(
    model,
    val_files: List[str],
    device: torch.device,
    batch_size: int = 4,
    use_t_emb: bool = False,
    use_pair_features: bool = False,
) -> Dict:
    model.eval()
    sums = defaultdict(lambda: [0.0, 0])  # key -> (sum_recall, count_queries)
    # Per-(layer, head, t) breakdowns + per-(L, H) heatmap.
    per_layer = defaultdict(lambda: [0.0, 0])
    per_head = defaultdict(lambda: [0.0, 0])
    per_t = defaultdict(lambda: [0.0, 0])
    per_lh = defaultdict(lambda: [0.0, 0])  # (l, h) -> (sum, count)
    overall = [0.0, 0]

    # Group val files by length for tight padding.
    val_buckets = defaultdict(list)
    for f in val_files:
        rec_meta = torch.load(f, map_location="cpu", weights_only=False)
        val_buckets[int(rec_meta["N"])].append((f, float(rec_meta["t_value"])))

    for n, items in sorted(val_buckets.items()):
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            recs = [load_record(p, with_coords=use_pair_features) for p, _ in chunk]
            batch = collate_batch(recs)
            layer_inputs = batch["layer_inputs"].to(device)
            top_k_idx = batch["top_k_indices"].to(device)
            slot_valid = batch["slot_valid"].to(device)
            query_mask = batch["query_mask"].to(device)
            coords = batch["coords_nm"].to(device) if use_pair_features else None
            N_pad = layer_inputs.shape[2]

            if use_t_emb:
                t_v = batch["t_values"].to(device)
                scores = model(layer_inputs, t_v, coords_nm=coords)
            else:
                scores = model(layer_inputs, coords_nm=coords)  # [B, L, H, N, N]
            # Mask padding keys before topk so they don't sneak in.
            key_mask = query_mask.view(-1, 1, 1, 1, N_pad).expand_as(scores)
            scores = scores.masked_fill(~key_mask, float("-inf"))
            router_top = scores.topk(K_TOP, dim=-1).indices   # [B, L, H, N, K]

            # Recall: |router_top ∩ teacher_top| / K, per (b, l, h, real-q).
            # Use one-hot membership trick (N is small enough).
            B, L, H, N_b, K = router_top.shape
            assert K == K_TOP
            r_oh = torch.zeros(B, L, H, N_b, N_pad, dtype=torch.bool, device=device)
            t_oh = torch.zeros_like(r_oh)
            r_oh.scatter_(-1, router_top, True)
            t_oh.scatter_(-1, top_k_idx, True)
            # For teacher-side, only count slots that were actually valid.
            # We can shadow t_oh by ANDing teacher one-hot with slot_valid via
            # a second scatter that overwrites invalid slots; simpler: re-build
            # t_oh by scattering ONLY valid slots.
            t_oh = torch.zeros(B, L, H, N_b, N_pad, dtype=torch.bool, device=device)
            valid_idx = top_k_idx.masked_fill(~slot_valid, 0)  # invalid → 0
            t_oh.scatter_(-1, valid_idx, True)
            # Subtract the spurious "0" entries that came purely from invalid slots:
            any_zero_valid = (slot_valid & (top_k_idx == 0)).any(dim=-1)  # [B,L,H,N]
            any_zero_invalid = (~slot_valid & (top_k_idx == 0)).any(dim=-1)
            # If a row had a valid slot at index 0, leave the True alone; if NOT and we
            # still have True at index 0 because an invalid slot got remapped, clear it.
            clear_zero = (~any_zero_valid) & any_zero_invalid  # [B,L,H,N]
            zero_col = torch.zeros_like(clear_zero)             # [B,L,H,N]
            t_oh[..., 0] = t_oh[..., 0] & ~(clear_zero & ~zero_col)

            inter = (r_oh & t_oh).sum(dim=-1).float()       # [B, L, H, N]
            # Denominator: teacher's K_eff per (b, l, h, q) (sum of valid slots).
            denom = slot_valid.float().sum(dim=-1).clamp(min=1.0)  # [B, L, H, N]
            recall = inter / denom                          # [B, L, H, N]

            q_mask = query_mask.view(B, 1, 1, N_b).expand_as(recall).float()
            # Aggregate.
            t_values_batch = torch.tensor([t for _, t in chunk], device=device)
            tot = (recall * q_mask).sum().item()
            cnt = q_mask.sum().item()
            overall[0] += tot
            overall[1] += int(cnt)
            for li in range(L):
                v = (recall[:, li] * q_mask[:, li]).sum().item()
                c = q_mask[:, li].sum().item()
                per_layer[li][0] += v
                per_layer[li][1] += int(c)
                for hi in range(H):
                    v_h = (recall[:, li, hi] * q_mask[:, li, hi]).sum().item()
                    c_h = q_mask[:, li, hi].sum().item()
                    per_head[hi][0] += v_h / max(1, L)  # spread across layers for per-head average
                    per_head[hi][1] += int(c_h) / max(1, L)
                    per_lh[(li, hi)][0] += v_h
                    per_lh[(li, hi)][1] += int(c_h)
            for bi, (_, t_v) in enumerate(chunk):
                v_t = (recall[bi] * q_mask[bi]).sum().item()
                c_t = q_mask[bi].sum().item()
                per_t[round(t_v, 2)][0] += v_t
                per_t[round(t_v, 2)][1] += int(c_t)

    overall_recall = overall[0] / max(1.0, overall[1])
    layer_table = {f"layer_{li}": per_layer[li][0] / max(1.0, per_layer[li][1])
                   for li in sorted(per_layer)}
    head_table = {f"head_{hi}": per_head[hi][0] / max(1.0, per_head[hi][1])
                  for hi in sorted(per_head)}
    t_table = {f"t_{tv:.2f}": per_t[tv][0] / max(1.0, per_t[tv][1])
               for tv in sorted(per_t)}
    per_lh_table = {f"L{li}_H{hi}": per_lh[(li, hi)][0] / max(1.0, per_lh[(li, hi)][1])
                    for (li, hi) in sorted(per_lh)}
    model.train()
    return {
        "overall_recall@64": overall_recall,
        "per_layer": layer_table,
        "per_head": head_table,
        "per_t": t_table,
        "per_lh": per_lh_table,
        "n_queries_evaluated": int(overall[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir",
                    default="/rds/user/ks2218/hpc-work/store/router_teacher_data")
    ap.add_argument("--out_dir", default="results/router_move1")
    ap.add_argument("--n_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--score_dim", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_t_emb", action="store_true",
                    help="Enable explicit t-embedding conditioning on the router input.")
    ap.add_argument("--pair_features", action="store_true",
                    help="Enable pair-distance RBF features. Requires coords sidecar.")
    ap.add_argument("--mlp_block", action="store_true",
                    help="Insert a 2-layer GELU MLP between W_in and per-(L,H) projections.")
    ap.add_argument("--mlp_dim", type=int, default=256,
                    help="Hidden dim of the optional MLP block.")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Path to router_best.pt or router_final.pt to resume from. "
                         "Loads weights only; optimizer state is re-initialised. "
                         "Step counter continues from the checkpoint's step.")
    ap.add_argument("--patience", type=int, default=6,
                    help="Early-stop patience in #evals (0=disabled). "
                         "Default 6 evals = patience * eval_every steps without "
                         "meaningful improvement.")
    ap.add_argument("--min_improve", type=float, default=1e-3,
                    help="A new eval recall must exceed best by this margin to "
                         "count as improvement.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"out_dir = {out_dir}")

    all_files = sorted(
        f for f in glob.glob(os.path.join(args.teacher_dir, "*.pt"))
        if not f.endswith("_coords.pt")  # sidecars are loaded via load_record(with_coords=True)
    )
    logger.info(f"teacher main files found: {len(all_files)}")
    assert len(all_files) > 0, f"no main .pt files in {args.teacher_dir}"

    train_files, val_files = split_by_protein(all_files, train_frac=0.9, seed=args.seed)
    logger.info(f"train files: {len(train_files)}  val files: {len(val_files)}")

    # Length-bucket train for stable batches (sort by N to minimise padding).
    train_buckets = make_length_buckets(train_files, bucket_step=16)
    bucket_keys = sorted(train_buckets.keys())
    logger.info(f"train length buckets (step=16): {bucket_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_kwargs = dict(
        trunk_dim=TRUNK_DIM,
        hidden_dim=args.hidden_dim,
        score_dim=args.score_dim,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        pair_features=args.pair_features,
        mlp_block=args.mlp_block,
        mlp_dim=args.mlp_dim,
    )
    if args.use_t_emb:
        model = TopKRouterT(**router_kwargs).to(device)
    else:
        model = TopKRouter(**router_kwargs).to(device)
    logger.info(
        f"router params: {model.num_params():,}  use_t_emb={args.use_t_emb}  "
        f"pair_features={args.pair_features}  mlp_block={args.mlp_block}  "
        f"hidden_dim={args.hidden_dim}  score_dim={args.score_dim}"
    )

    # Resume from a saved router ckpt (weights only; optimizer state is fresh).
    start_step = 0
    resume_recall = 0.0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_step = int(ckpt.get("step", 0))
        resume_recall = float(ckpt.get("recall@64", ckpt.get("best_recall@64", 0.0)))
        logger.info(
            f"resumed from {args.resume_from}  start_step={start_step}  "
            f"prev_recall@64={resume_recall:.4f}"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Open the CSV in append mode when resuming so we keep the prior run's log.
    csv_mode = "a" if args.resume_from is not None else "w"
    log_csv = out_dir / "training_log.csv"
    csv_f = open(log_csv, csv_mode, newline="")
    csv_w = csv.writer(csv_f)
    if csv_mode == "w":
        csv_w.writerow(["step", "loss", "n_valid_queries", "lr", "wall_s_per_step"])

    model.train()
    step = start_step
    t0 = time.time()
    last_log = time.time()
    best_recall = resume_recall  # only save new "best" if we beat the resume point
    best_step = start_step
    no_improve = 0
    stopped_early = False
    # When resuming, args.n_steps is interpreted as ADDITIONAL steps on top of start_step.
    target_step = start_step + args.n_steps
    while step < target_step:
        # Sample one bucket, then sample batch_size files from it.
        bk = random.choice(bucket_keys)
        files_in_bucket = train_buckets[bk]
        if len(files_in_bucket) < args.batch_size:
            # mix across nearby buckets to fill
            pool = list(files_in_bucket)
            for shift in (16, -16, 32, -32):
                pool.extend(train_buckets.get(bk + shift, []))
                if len(pool) >= args.batch_size:
                    break
            chosen = random.sample(pool, min(args.batch_size, len(pool)))
        else:
            chosen = random.sample(files_in_bucket, args.batch_size)

        try:
            recs = [load_record(p, with_coords=args.pair_features) for p in chosen]
            batch = collate_batch(recs)
            layer_inputs = batch["layer_inputs"].to(device, non_blocking=True)
            top_k_idx = batch["top_k_indices"].to(device, non_blocking=True)
            top_k_wts = batch["top_k_weights"].to(device, non_blocking=True)
            slot_valid = batch["slot_valid"].to(device, non_blocking=True)
            query_mask = batch["query_mask"].to(device, non_blocking=True)
            coords = batch["coords_nm"].to(device, non_blocking=True) if args.pair_features else None

            if args.use_t_emb:
                t_v = batch["t_values"].to(device, non_blocking=True)
                scores = model(layer_inputs, t_v, coords_nm=coords)
            else:
                scores = model(layer_inputs, coords_nm=coords)
            loss, n_valid = kl_loss(scores, top_k_idx, top_k_wts, slot_valid, query_mask)

            opt.zero_grad()
            loss.backward()
            opt.step()
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at step {step} (bucket {bk}, B={args.batch_size}); skipping batch")
            torch.cuda.empty_cache()
            continue

        step += 1
        if step % args.log_every == 0 or step == 1:
            dt = (time.time() - last_log) / max(1, args.log_every)
            lr_now = opt.param_groups[0]["lr"]
            logger.info(f"step {step:>5d} | loss {loss.item():.4f} | "
                        f"nq {int(n_valid.item())} | {dt:.2f}s/step")
            csv_w.writerow([step, float(loss.item()), int(n_valid.item()), lr_now, dt])
            csv_f.flush()
            last_log = time.time()

        if step % args.eval_every == 0 or step == args.n_steps:
            logger.info(f"[eval @ step {step}] running held-out recall@64 ...")
            t_eval = time.time()
            stats = eval_recall_at_64(
                model, val_files, device, batch_size=4,
                use_t_emb=args.use_t_emb, use_pair_features=args.pair_features,
            )
            logger.info(f"  overall recall@64 = {stats['overall_recall@64']:.4f} "
                        f"({(time.time()-t_eval):.1f}s)")
            # Split per_lh out into a separate file to keep the main eval JSON tight.
            per_lh = stats.pop("per_lh")
            eval_path = out_dir / f"eval_step{step:05d}.json"
            with open(eval_path, "w") as f:
                json.dump(stats, f, indent=2)
            with open(out_dir / f"eval_step{step:05d}_perLH.json", "w") as f:
                json.dump(per_lh, f, indent=2)
            logger.info(f"  wrote {eval_path}")

            cur = float(stats["overall_recall@64"])
            if cur > best_recall + args.min_improve:
                best_recall = cur; best_step = step; no_improve = 0
                # Save best checkpoint — always have peak weights even if a later
                # regression triggers early-stop.
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {
                        "trunk_dim": TRUNK_DIM, "hidden_dim": args.hidden_dim,
                        "score_dim": args.score_dim, "n_layers": N_LAYERS,
                        "n_heads": N_HEADS, "use_t_emb": args.use_t_emb,
                        "pair_features": args.pair_features,
                        "mlp_block": args.mlp_block, "mlp_dim": args.mlp_dim,
                    },
                    "step": step, "recall@64": cur,
                    "args": vars(args),
                }, out_dir / "router_best.pt")
                logger.info(f"  new best recall@64 = {cur:.4f} @ step {step}")
            else:
                no_improve += 1
                logger.info(
                    f"  no-improve {no_improve}/{args.patience}  "
                    f"(best={best_recall:.4f} @ step {best_step})"
                )
                if args.patience > 0 and no_improve >= args.patience:
                    logger.info(
                        f"Early stop: {args.patience} evals without "
                        f"≥{args.min_improve:.4f} improvement. "
                        f"Best recall@64 = {best_recall:.4f} at step {best_step}."
                    )
                    stopped_early = True
                    break

    csv_f.close()
    final_path = out_dir / "router_final.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "trunk_dim": TRUNK_DIM,
            "hidden_dim": args.hidden_dim,
            "score_dim": args.score_dim,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "use_t_emb": args.use_t_emb,
            "pair_features": args.pair_features,
            "mlp_block": args.mlp_block,
            "mlp_dim": args.mlp_dim,
        },
        "args": vars(args),
        "wall_total_s": time.time() - t0,
        "stopped_early": stopped_early,
        "best_recall@64": best_recall,
        "best_step": best_step,
        "final_step": step,
    }, final_path)
    logger.info(
        f"saved {final_path}; total wall {time.time()-t0:.1f}s; "
        f"final_step={step}; best={best_recall:.4f} @ step {best_step}; "
        f"stopped_early={stopped_early}"
    )


if __name__ == "__main__":
    main()
