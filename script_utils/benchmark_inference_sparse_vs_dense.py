"""Inference-time compute benchmark: sparse-K40 vs canonical dense.

Measures peak GPU memory and per-protein wall-clock at L ∈ {50, 100, 200}
(the length regime where the canonical model actually delivers designable
samples) for both ckpts on the same GPU, same nsteps=400 schedule, same seed.

Output: CSV with columns ckpt,L,wall_s,wall_s_per_protein,peak_gpu_mb.

Usage:
    CUDA_VISIBLE_DEVICES=4 python script_utils/benchmark_inference_sparse_vs_dense.py \
        --output_csv results/inference_compute_audit/sparse_vs_dense.csv

Per CLAUDE.md: nsteps=400 (hard rule), one GPU, runs sequentially.
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proteinfoundation.datasets.gen_dataset import GenDataset  # noqa: E402
from proteinfoundation.generate import (  # noqa: E402
    load_ckpt_n_configure_inference,
    setup,
)


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def benchmark_one(
    config_name: str,
    ckpt_name: str,
    lengths: list,
    nsamples: int,
    seed: int,
    device: torch.device,
):
    """Load one ckpt and measure (wall, peak_gpu) at each length.

    Returns list of dicts: [{"L": ..., "wall_s": ..., "wall_s_per_protein": ...,
    "peak_gpu_mb": ..., "n": nsamples}].
    """
    config_dir = str(ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"ckpt_name={ckpt_name}",
                f"generation.dataset.nsamples={nsamples}",
                f"generation.dataset.max_nsamples_per_batch={nsamples}",
                f"seed={seed}",
                "++gen_njobs=1",
                "++job_id=0",
            ],
        )
    # setup() expects a root_dir; we don't need its output dirs but we do need it
    # to initialise logging/paths.
    setup(cfg, create_root=False, config_name=config_name, job_id=0)

    print(f"[load] ckpt={ckpt_name}", flush=True)
    model = load_ckpt_n_configure_inference(cfg)
    model._generation_base_seed = seed
    model.to(device).eval()

    # Warmup: a single short forward at L=lengths[0] to amortise CUDA init /
    # JIT compile / cuBLAS handle creation. We do NOT record this.
    print(f"[warmup] L={lengths[0]} n=1", flush=True)
    warmup_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    warmup_cfg.generation.dataset.nsamples = 1
    warmup_cfg.generation.dataset.max_nsamples_per_batch = 1
    warmup_cfg.generation.dataset.nlens_cfg.nres_lens = [lengths[0]]
    warmup_ds = GenDataset(**warmup_cfg.generation.dataset)
    warmup_dl = DataLoader(warmup_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(warmup_dl):
            batch = _move_batch_to_device(batch, device)
            model.predict_step(batch, batch_idx)
            break
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    results = []
    for L in lengths:
        per_L_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        per_L_cfg.generation.dataset.nsamples = nsamples
        per_L_cfg.generation.dataset.max_nsamples_per_batch = nsamples
        per_L_cfg.generation.dataset.nlens_cfg.nres_lens = [L]
        ds = GenDataset(**per_L_cfg.generation.dataset)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch = _move_batch_to_device(batch, device)
                model.predict_step(batch, batch_idx)
                break  # one batch = nsamples proteins
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        wall = t1 - t0
        results.append(
            {
                "L": L,
                "n": nsamples,
                "wall_s": wall,
                "wall_s_per_protein": wall / nsamples,
                "peak_gpu_mb": peak_mb,
            }
        )
        print(
            f"[measure] L={L:>3d} n={nsamples} wall={wall:7.2f}s "
            f"({wall/nsamples:6.2f}s/protein) peak={peak_mb:7.1f} MB",
            flush=True,
        )
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--nsamples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA"
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES already selects
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[setup] device={gpu_name} lengths={args.lengths} "
          f"nsamples={args.nsamples} seed={args.seed}", flush=True)

    arms = [
        # (label, config_name, ckpt_name)
        ("canonical_dense",
         "inference_canonical_step2646_n6_nfe400",
         "baseline_wd0.05_step2646.ckpt"),
        ("sparse_K40",
         "inference_sparse_K40_step1259_baseline_n6_nfe400",
         "sparse_K40_step1259.ckpt"),
    ]

    all_rows = []
    for label, cfg_name, ckpt_name in arms:
        print(f"\n=== {label} ===", flush=True)
        rows = benchmark_one(
            config_name=cfg_name,
            ckpt_name=ckpt_name,
            lengths=args.lengths,
            nsamples=args.nsamples,
            seed=args.seed,
            device=device,
        )
        for r in rows:
            r["arm"] = label
            r["ckpt"] = ckpt_name
            r["gpu"] = gpu_name
            all_rows.append(r)

    # Write CSV
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arm", "ckpt", "L", "n", "wall_s",
                        "wall_s_per_protein", "peak_gpu_mb", "gpu"],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[done] wrote {out}", flush=True)

    # Print side-by-side
    print("\n========== SIDE-BY-SIDE ==========")
    by_L = {}
    for r in all_rows:
        by_L.setdefault(r["L"], {})[r["arm"]] = r
    print(f"{'L':>4s} | {'canonical wall':>14s} | {'sparse wall':>14s} | "
          f"{'wall ratio':>10s} | {'canonical mem':>14s} | "
          f"{'sparse mem':>14s} | {'mem ratio':>10s}")
    print("-" * 110)
    for L in sorted(by_L):
        c = by_L[L].get("canonical_dense", {})
        s = by_L[L].get("sparse_K40", {})
        if not c or not s:
            continue
        wr = s["wall_s_per_protein"] / c["wall_s_per_protein"]
        mr = s["peak_gpu_mb"] / c["peak_gpu_mb"]
        print(
            f"{L:>4d} | {c['wall_s_per_protein']:>11.2f}s/p | "
            f"{s['wall_s_per_protein']:>11.2f}s/p | {wr:>10.3f} | "
            f"{c['peak_gpu_mb']:>11.1f}MB | {s['peak_gpu_mb']:>11.1f}MB | "
            f"{mr:>10.3f}"
        )


if __name__ == "__main__":
    main()
