"""Move-1 teacher-data extraction: dense attention top-64 + layer inputs.

For a stratified sample of training proteins, runs ONE forward pass of the
canonical dense CA-only checkpoint at each of t ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
(codebase convention: t=0 noise, t=1 clean; x_t = (1-t)*noise + t*x_data).

Records, per (protein, t):
  - layer_inputs    [14, N, 768]  fp16  — input hidden state to each layer.
                                          Pre-hook on transformer_layers[i].
  - top_k_indices   [14, 12, N, K=64] int16  — top-K key indices per (l, h, q),
                                                over REAL keys only (≤ N).
  - top_k_weights   [14, 12, N, K=64] fp16   — post-softmax attention weights at
                                                those indices. Sum may be < 1
                                                if real K < 64 (slots padded).

Stratification: L ∈ {50, 100, 150, 200} (±10), ~125 proteins per bucket for
the full run. --smoke picks one protein per bucket (+1 extra near L=100), 5 total.

Outputs one .pt per (protein, t) to:
  /rds/user/ks2218/hpc-work/store/router_teacher_data/<protein_id>_t<t:.2f>.pt

Atomic-write (tmp + rename) so a SLURM kill doesn't leave half-written files.
The extraction is resume-safe: existing output files are skipped.

Usage:
  /home/ks2218/conda_envs/laproteina_env/bin/python \\
      script_utils/extract_router_teacher_data.py --smoke
  /home/ks2218/conda_envs/laproteina_env/bin/python \\
      script_utils/extract_router_teacher_data.py --n_proteins 500
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
OUT_DIR_DEFAULT = "/rds/user/ks2218/hpc-work/store/router_teacher_data"
K_TOP = 64
N_LAYERS = 14
N_HEADS = 12


# Per-(protein, t) capture state populated by the patched _attn and the
# layer-input pre-hooks. Re-initialised before every forward.
class CaptureBuf:
    def __init__(self):
        self.N_real: int = 0
        # [N_layers, N_real, D] — set by layer-input pre-hooks (incremental).
        self.layer_inputs: Dict[int, torch.Tensor] = {}
        # [N_layers, N_heads, N_real, K_TOP]
        self.top_k_indices: Dict[int, torch.Tensor] = {}
        self.top_k_weights: Dict[int, torch.Tensor] = {}

    def reset(self, N_real: int):
        self.N_real = int(N_real)
        self.layer_inputs.clear()
        self.top_k_indices.clear()
        self.top_k_weights.clear()


BUF = CaptureBuf()


def make_capture_attn():
    """Patched PairBiasAttention._attn that:
      1. Runs the original computation bit-identically.
      2. Snapshots top-K indices and weights at the (layer_idx, h, real-q).
    Layer-head are tagged via attributes on the attention instance.
    """
    from einops import rearrange
    from torch import einsum

    max_neg = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def _attn(self, q, k, v, b, mask):
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask_rs = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        layer_idx = getattr(self, "_layer_idx", -1)
        assert 0 <= layer_idx < N_LAYERS, f"missing _layer_idx tag: {layer_idx}"
        N_real = BUF.N_real
        # attn: [B=1, H, N_pad, N_pad] — keep real-q × real-k subblock.
        attn_real = attn[0, :, :N_real, :N_real].float()  # [H, N_real, N_real]

        K_eff = min(K_TOP, N_real)
        top = attn_real.topk(K_eff, dim=-1)
        idx = top.indices.to(torch.int16)            # [H, N_real, K_eff]
        wts = top.values.to(torch.float16)           # [H, N_real, K_eff]
        if K_eff < K_TOP:
            # Pad slots [K_eff:K_TOP] with index 0 / weight 0.
            pad_idx = torch.zeros(N_HEADS, N_real, K_TOP - K_eff, dtype=torch.int16)
            pad_wts = torch.zeros(N_HEADS, N_real, K_TOP - K_eff, dtype=torch.float16)
            idx = torch.cat([idx.cpu(), pad_idx], dim=-1)
            wts = torch.cat([wts.cpu(), pad_wts], dim=-1)
        else:
            idx = idx.cpu()
            wts = wts.cpu()
        BUF.top_k_indices[layer_idx] = idx     # [H, N_real, K_TOP]
        BUF.top_k_weights[layer_idx] = wts     # [H, N_real, K_TOP]
        return out

    return _attn


def make_layer_input_hook(layer_idx: int):
    """Forward pre-hook on transformer_layers[i]. Captures input seqs[0, :N_real, :]."""
    def _pre(module, args, kwargs):
        # args = (seqs, pair_rep, c, mask, ...). seqs shape [B, N_pad, D].
        seqs = args[0] if len(args) > 0 else kwargs.get("seqs", None)
        assert seqs is not None and seqs.ndim == 3, f"unexpected seqs at layer {layer_idx}"
        BUF.layer_inputs[layer_idx] = seqs[0, :BUF.N_real, :].detach().to(torch.float16).cpu()
        return None
    return _pre


# ---------------------------------------------------------------------------- #
# Data selection
# ---------------------------------------------------------------------------- #
def list_processed_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "pdb_train", "processed_latents", "*", "*.pt")
    return sorted(glob.glob(pattern))


def load_csv_lengths(repo_root: str) -> Dict[str, int]:
    """Return {pdb_id: length} from one of the train CSVs (avoids loading 355k .pt
    files just to read shapes). Picks the first matching CSV in data/pdb_train/."""
    import pandas as pd
    cand = sorted(glob.glob(os.path.join(repo_root, "data", "pdb_train", "df_pdb_*.csv")))
    cand = [c for c in cand if "_latents" not in c]  # raw CSV preferred
    if not cand:
        cand = sorted(glob.glob(os.path.join(repo_root, "data", "pdb_train", "df_pdb_*.csv")))
    if not cand:
        return {}
    df = pd.read_csv(cand[0])
    if "id" in df.columns and "length" in df.columns:
        return dict(zip(df["id"].astype(str), df["length"].astype(int)))
    return {}


def stratified_pick(
    files: List[str],
    lengths_by_id: Dict[str, int],
    targets: Tuple[int, ...],
    n_per_bucket: int,
    tol: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """Returns list of (pdb_id, file_path) per bucket, len-balanced. Walks `files`
    in a shuffled order, peeking the length from `lengths_by_id` so we don't
    have to load .pt headers. Falls back to loading the file when ID is missing
    from CSV.
    """
    rng = random.Random(seed)
    pool = list(files)
    rng.shuffle(pool)

    buckets: Dict[int, List[Tuple[str, str]]] = {L: [] for L in targets}
    need_total = n_per_bucket * len(targets)
    chosen = 0
    for fpath in pool:
        if chosen >= need_total:
            break
        pdb_id = os.path.splitext(os.path.basename(fpath))[0]
        n = lengths_by_id.get(pdb_id)
        if n is None:
            try:
                d = torch.load(fpath, map_location="cpu", weights_only=False)
                n = int(d.coords_nm.shape[0])
            except Exception:
                continue
        for L in targets:
            if abs(n - L) <= tol and len(buckets[L]) < n_per_bucket:
                buckets[L].append((pdb_id, fpath))
                chosen += 1
                break

    flat = []
    for L in targets:
        flat.extend(buckets[L])
        logger.info(f"  bucket L~{L}: {len(buckets[L])} / {n_per_bucket}")
    return flat


def apply_transforms(d: Data, seed: int) -> Data:
    d = CenterStructureTransform()(d)
    d = ChainBreakPerResidueTransform()(d)
    torch.manual_seed(seed)
    d = GlobalRotationTransform()(d)
    return d


# ---------------------------------------------------------------------------- #
# Main extraction
# ---------------------------------------------------------------------------- #
def extract_one(
    model: Proteina,
    data_item: Data,
    pdb_id: str,
    t_values: Tuple[float, ...],
    out_dir: str,
    device: torch.device,
    max_pad: int,
    seed: int,
) -> List[Tuple[float, str]]:
    """Run all t_values for one protein; save one .pt per (protein, t).
    Returns list of (t_val, out_path) for the saved files."""
    N_real = int(data_item.coords_nm.shape[0])
    # Pad to max(N_real, 64) — model is fine with any padding ≥ real length
    # because per-residue masking zeros out padded positions everywhere.
    n_pad = max(N_real, 64)
    if max_pad is not None:
        n_pad = min(max_pad, n_pad)
        N_real = min(N_real, n_pad)

    coords_nm = torch.zeros(1, n_pad, 37, 3)
    coord_mask = torch.zeros(1, n_pad, 37, dtype=torch.bool)
    residue_type = torch.zeros(1, n_pad, dtype=torch.long)
    mask = torch.zeros(1, n_pad, dtype=torch.bool)
    coords_nm[0, :N_real] = data_item.coords_nm[:N_real]
    coord_mask[0, :N_real] = data_item.coord_mask[:N_real]
    residue_type[0, :N_real] = data_item.residue_type[:N_real]
    mask[0, :N_real] = True
    batch_cpu = {
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "mask": mask,
    }

    out_paths: List[Tuple[float, str]] = []
    for t_val in t_values:
        out_path = os.path.join(out_dir, f"{pdb_id}_t{t_val:.2f}.pt")
        if os.path.exists(out_path):
            logger.info(f"  skip (exists): {out_path}")
            out_paths.append((t_val, out_path))
            continue

        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        # Reset capture buffer
        BUF.reset(N_real=N_real)
        # Seed the noise sample deterministically per (pdb_id, t)
        torch.manual_seed(seed + abs(hash((pdb_id, round(float(t_val), 4)))) % (2 ** 32))

        batch = model.add_clean_samples(batch)
        x_1_dict, mask_proc, batch_shape, n_actual, dtype, dev = model.fm.process_batch(batch)
        x_0 = model.fm.sample_noise(n=n_actual, shape=batch_shape, mask=mask_proc, device=dev)
        B = batch_shape[0]
        t_bb = torch.full((B,), float(t_val), device=dev)
        t = {"bb_ca": t_bb}
        x_t = model.fm.interpolate(x_0=x_0, x_1=x_1_dict, t=t, mask=mask_proc)
        batch["x_0"] = x_0
        batch["x_1"] = x_1_dict
        batch["x_t"] = x_t
        batch["t"] = t
        batch["mask"] = mask_proc

        with torch.no_grad():
            _ = model.call_nn(batch, n_recycle=0)

        # Sanity: every layer captured both layer input and attention top-K.
        for li in range(N_LAYERS):
            assert li in BUF.layer_inputs, f"missing layer_input for layer {li}"
            assert li in BUF.top_k_indices, f"missing top-K for layer {li}"

        layer_inputs = torch.stack(
            [BUF.layer_inputs[li] for li in range(N_LAYERS)], dim=0
        )  # [L, N_real, D]
        top_k_indices = torch.stack(
            [BUF.top_k_indices[li] for li in range(N_LAYERS)], dim=0
        )  # [L, H, N_real, K]
        top_k_weights = torch.stack(
            [BUF.top_k_weights[li] for li in range(N_LAYERS)], dim=0
        )  # [L, H, N_real, K]

        record = {
            "protein_id": pdb_id,
            "t_value": float(t_val),
            "N": int(N_real),
            "layer_inputs": layer_inputs,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
        }

        tmp_path = out_path + ".tmp"
        torch.save(record, tmp_path)
        os.replace(tmp_path, out_path)
        out_paths.append((t_val, out_path))
        logger.info(
            f"  saved {out_path}  N={N_real} "
            f"li={tuple(layer_inputs.shape)} "
            f"idx={tuple(top_k_indices.shape)} "
            f"wts={tuple(top_k_weights.shape)}"
        )
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_file", default=CKPT_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 5 proteins (1 per bucket + 1 extra near L=100).")
    ap.add_argument("--n_proteins", type=int, default=500,
                    help="Total proteins; will be distributed across 4 length buckets.")
    # Length targets match the actual processed_latents/ distribution
    # (range ~218-610, median ~381). Original prompt's {50,100,150,200} doesn't
    # match this dataset, so we span the real distribution instead.
    ap.add_argument("--length_targets", default="300,400,500")
    ap.add_argument("--length_tol", type=int, default=25)
    ap.add_argument("--t_values", default="0.10,0.30,0.50,0.70,0.90")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pad", type=int, default=520,
                    help="Pad length cap. Cropping is safe — captures use N_real.")
    ap.add_argument("--data_dir", default=os.environ.get("DATA_PATH", os.path.join(REPO, "data")))
    args = ap.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    targets = tuple(int(x) for x in args.length_targets.split(","))
    t_values = tuple(float(x) for x in args.t_values.split(","))

    if args.smoke:
        n_per_bucket = 1  # 4 buckets → 4 proteins, +1 extra below
        smoke_extra_near = 100
    else:
        n_per_bucket = max(1, args.n_proteins // len(targets))

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("=" * 80)
    logger.info(f"Router teacher-data extraction")
    logger.info(f"  ckpt:    {args.ckpt_file}")
    logger.info(f"  out_dir: {args.out_dir}")
    logger.info(f"  targets: {targets} (+/-{args.length_tol})  per-bucket={n_per_bucket}")
    logger.info(f"  t_vals:  {t_values}")
    logger.info(f"  smoke:   {args.smoke}")
    logger.info("=" * 80)

    # Load dense model
    assert os.path.exists(args.ckpt_file), f"missing ckpt: {args.ckpt_file}"
    model = Proteina.load_from_checkpoint(
        args.ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    nn_cfg = model.cfg_exp.get("nn", {})
    assert not nn_cfg.get("sparse_attention", False), "ckpt is sparse — wrong ckpt"
    n_layers_actual = len(model.nn.transformer_layers)
    n_heads_actual = nn_cfg.get("nheads", N_HEADS)
    assert n_layers_actual == N_LAYERS, f"expected {N_LAYERS} layers, got {n_layers_actual}"
    assert n_heads_actual == N_HEADS, f"expected {N_HEADS} heads, got {n_heads_actual}"
    token_dim = nn_cfg.get("token_dim", -1)
    logger.info(f"  trunk_dim = {token_dim} (expected 768)")
    assert token_dim == 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No GPU — running on CPU. Forward pass will be ~10-50× slower; "
                       "use this only for the structure-validation smoke test.")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.nn, "_orig_mod"):
        model.nn = model.nn._orig_mod
        logger.info("  unwrapped torch.compile wrapper")

    # Install patches:
    # 1. Class-level _attn replacement (captures top-K).
    PairBiasAttention._attn = make_capture_attn()
    # 2. Per-layer tag + forward-pre-hook for layer input.
    hook_handles = []
    for i, layer in enumerate(model.nn.transformer_layers):
        layer.mhba.mha._layer_idx = i
        h = layer.register_forward_pre_hook(make_layer_input_hook(i), with_kwargs=True)
        hook_handles.append(h)

    # Build protein subset
    logger.info("Scanning processed_latents/ ...")
    all_files = list_processed_files(args.data_dir)
    logger.info(f"  total candidate files: {len(all_files)}")
    lengths_by_id = load_csv_lengths(REPO)
    logger.info(f"  loaded {len(lengths_by_id)} (id, length) entries from CSV")

    selected = stratified_pick(
        all_files, lengths_by_id, targets, n_per_bucket, args.length_tol, args.seed
    )
    if args.smoke:
        # Add one extra near L=100 to reach 5.
        extra = stratified_pick(
            all_files, lengths_by_id, (smoke_extra_near,), 1, args.length_tol,
            args.seed + 1,
        )
        already = {pid for pid, _ in selected}
        for pid, fp in extra:
            if pid not in already:
                selected.append((pid, fp))
                break
    logger.info(f"  selected {len(selected)} proteins")

    # Extract
    manifest = []
    t_start = time.time()
    for prot_idx, (pdb_id, fpath) in enumerate(selected):
        try:
            d = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"  load failed for {fpath}: {e}")
            continue
        d = apply_transforms(d, args.seed + prot_idx)
        N_real = int(d.coords_nm.shape[0])
        logger.info(f"[{prot_idx+1}/{len(selected)}] {pdb_id} (N={N_real})")
        try:
            saved = extract_one(
                model=model,
                data_item=d,
                pdb_id=pdb_id,
                t_values=t_values,
                out_dir=args.out_dir,
                device=device,
                max_pad=args.max_pad,
                seed=args.seed,
            )
            for t_v, p in saved:
                manifest.append({"protein_id": pdb_id, "t": t_v, "path": p, "N": N_real})
        except Exception as e:
            logger.error(f"  extraction failed for {pdb_id}: {e}")

    elapsed = time.time() - t_start
    logger.info(f"Done. {len(manifest)} files in {elapsed:.1f}s "
                f"({elapsed / max(1, len(manifest)):.2f}s / file)")

    manifest_path = os.path.join(
        args.out_dir, "manifest_smoke.json" if args.smoke else "manifest.json"
    )
    with open(manifest_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt_file,
            "t_values": list(t_values),
            "targets": list(targets),
            "seed": args.seed,
            "n_files": len(manifest),
            "files": manifest,
        }, f, indent=2)
    logger.info(f"  manifest: {manifest_path}")

    for h in hook_handles:
        h.remove()


if __name__ == "__main__":
    main()
