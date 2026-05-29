"""Micro-benchmark: time a single predict_for_sampling forward pass on
conv (ca_only_downsampled step 2961) vs canonical (test_ca_only_diffusion
step 2646) at various L. Tests the hypothesis that conv per-step is genuinely
faster than canonical at large L (production-pipeline regime), even though the
hybrid sampling probe at L=200 showed no wall-clock savings.

Output: per-L per-model mean/median/std forward-pass ms over N=20 timed iters,
plus peak GPU memory after the timed sweep.

Usage: CUDA_VISIBLE_DEVICES=<n> python script_utils/microbench_conv_vs_canonical.py
"""
import os
import sys
import time
import json
import gc

import torch

sys.path.insert(0, os.path.abspath("."))
from proteinfoundation.proteina import Proteina


CKPTS = {
    "conv_2961":      "/home/ks2218/la-proteina/best_val_00000029_000000002961.ckpt",
    "canonical_2646": "/home/ks2218/la-proteina/best_val_00000026_000000002646.ckpt",
}
L_VALUES = [50, 100, 200, 300, 500, 800]
BATCH = 1
WARMUP_ITERS = 3
TIMED_ITERS = 20
T_VALUE = 0.5  # mid-trajectory; any single t is fine for per-step timing


def build_batch(L: int, batch_size: int = 1, device: str = "cuda"):
    """Construct a minimal batch dict for CA-only predict_for_sampling.

    Replicates the keys the integrator sets in product_space_flow_matcher.py:
      batch["x_t"][data_mode] : [B, N, 3]
      batch["t"][data_mode]   : [B]
      batch["mask"]           : [B, N] bool
    For CA-only, data_mode = "bb_ca".
    """
    x_t = torch.randn(batch_size, L, 3, device=device, dtype=torch.float32)
    t_val = torch.full((batch_size,), T_VALUE, device=device, dtype=torch.float32)
    mask = torch.ones(batch_size, L, dtype=torch.bool, device=device)
    batch = {
        "x_t": {"bb_ca": x_t},
        "t": {"bb_ca": t_val},
        "mask": mask,
    }
    return batch


def time_forwards(model, L: int, n_warmup: int, n_timed: int):
    """Run warm-up + timed forwards. Returns list of per-iter wall ms."""
    batch = build_batch(L, BATCH, device="cuda")
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.predict_for_sampling(batch, mode="full", n_recycle=0)
    torch.cuda.synchronize()
    # timed
    times_ms = []
    with torch.no_grad():
        for _ in range(n_timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.predict_for_sampling(batch, mode="full", n_recycle=0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# batch_size: {BATCH}, t={T_VALUE}, warmup={WARMUP_ITERS}, timed={TIMED_ITERS}")
    print()
    results = {}
    for tag, ckpt_path in CKPTS.items():
        print(f"=== loading {tag} from {ckpt_path}", flush=True)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        model = Proteina.load_from_checkpoint(
            ckpt_path, strict=False, autoencoder_ckpt_path=None
        )
        model = model.to("cuda").eval()
        weights_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024) - baseline_alloc_mb
        print(f"  weight footprint on GPU: {weights_alloc_mb:.1f} MB")
        results[tag] = {"weights_mb": weights_alloc_mb, "per_L": {}}
        for L in L_VALUES:
            try:
                torch.cuda.reset_peak_memory_stats()
                ts = time_forwards(model, L, WARMUP_ITERS, TIMED_ITERS)
                peak_after_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                mean_ms = sum(ts) / len(ts)
                ts_sorted = sorted(ts)
                median_ms = ts_sorted[len(ts) // 2]
                std_ms = (sum((x - mean_ms) ** 2 for x in ts) / len(ts)) ** 0.5
                results[tag]["per_L"][L] = {
                    "mean_ms": mean_ms,
                    "median_ms": median_ms,
                    "std_ms": std_ms,
                    "min_ms": min(ts),
                    "max_ms": max(ts),
                    "peak_gpu_mb": peak_after_mb,
                    "per_iter_ms": ts,
                }
                print(
                    f"  L={L:4d}: mean {mean_ms:6.2f} ms (median {median_ms:.2f}, "
                    f"std {std_ms:.2f}, min {min(ts):.2f}, max {max(ts):.2f})  "
                    f"peak GPU {peak_after_mb:.0f} MB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"  L={L:4d}: OOM — {e}", flush=True)
                results[tag]["per_L"][L] = {"oom": True}
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  L={L:4d}: ERROR — {type(e).__name__}: {e}", flush=True)
                results[tag]["per_L"][L] = {"error": str(e)}
                torch.cuda.empty_cache()
        # free this model before loading the next
        del model
        torch.cuda.empty_cache()
        gc.collect()
    print()
    print("=== SUMMARY: per-L per-step wall ms (mean ± std over 20 iters) ===")
    print(f"{'L':>5} | {'conv mean ms':>14} | {'canonical mean ms':>18} | {'speedup (canon/conv)':>22}")
    for L in L_VALUES:
        conv = results.get("conv_2961", {}).get("per_L", {}).get(L, {})
        can = results.get("canonical_2646", {}).get("per_L", {}).get(L, {})
        if conv.get("mean_ms") and can.get("mean_ms"):
            speedup = can["mean_ms"] / conv["mean_ms"]
            print(
                f"{L:>5} | {conv['mean_ms']:>10.2f} ± {conv['std_ms']:<2.1f} | "
                f"{can['mean_ms']:>14.2f} ± {can['std_ms']:<2.1f} | {speedup:>20.2f}×"
            )
        elif conv.get("oom") or can.get("oom"):
            cstr = "OOM" if conv.get("oom") else f"{conv.get('mean_ms', 0):.1f}"
            kstr = "OOM" if can.get("oom") else f"{can.get('mean_ms', 0):.1f}"
            print(f"{L:>5} | {cstr:>14} | {kstr:>18} | (one OOM'd)")
    out_path = "/home/ks2218/la-proteina/inference/microbench_conv_vs_canonical.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
