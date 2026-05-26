"""Time a single forward+backward pass of the optimized router-sparse trunk.

Mirrors the training inner loop at canonical settings (B=6, max_padding_size=512,
bf16-mixed autocast) and measures the per-microbatch wall time. Multiply by
accumulate_grad_batches (=32) to estimate per-opt-step time, and compare to
canonical dense's ~27.7 s/opt-step from CLAUDE.md.

This validates the RBF-cache + bf16-router optimizations BEFORE committing to a
chained training slot.

Usage (1 GPU SLURM job; see submit wrapper):
    /home/ks2218/conda_envs/laproteina_env/bin/python \\
        script_utils/profile_router_sparse_microbatch.py --n_iters 20 --warmup 3
"""
import argparse
import sys
import time

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import torch
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from script_utils.load_frozen_router import load_frozen_router


def build_input(B: int, N: int, device: torch.device):
    """Build a Move-2 trunk input matching call_nn's batch shape."""
    return {
        "x_t": {"bb_ca": torch.randn(B, N, 3, device=device)},
        "x_sc": {"bb_ca": torch.randn(B, N, 3, device=device)},
        "t": {"bb_ca": torch.rand(B, device=device).clamp_(0.02, 0.98)},
        "mask": torch.ones(B, N, dtype=torch.bool, device=device),
        "ca_coors_nm": torch.randn(B, N, 3, device=device),
        "residue_type": torch.zeros(B, N, dtype=torch.long, device=device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--n_iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--router_ckpt",
                    default=f"{REPO}/results/router_move1_inflight/router_final.pt")
    ap.add_argument("--accum", type=int, default=32,
                    help="accumulate_grad_batches used in training. Per-opt-step time = "
                         "per-microbatch × accum.")
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.to_container(
        OmegaConf.load(f"{REPO}/configs/nn/ca_only_router_sparse_K64_160M.yaml"),
        resolve=True,
    )
    cfg.pop("name", None)

    print(f"Building trunk (router_sparse_K=64) ...")
    trunk = LocalLatentsTransformer(**cfg).to(device)
    router = load_frozen_router(args.router_ckpt, map_location="cpu")
    trunk.attach_router(router)
    trunk.to(device)
    trunk.train()
    n_trainable = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")

    inp = build_input(args.B, args.N, device)
    print(f"  input: B={args.B}, N={args.N}, on {device}")
    print()

    # Warm-up (cudnn auto-tune, allocator)
    print(f"Warmup {args.warmup} iters ...")
    for i in range(args.warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = trunk(inp)
        loss = out["bb_ca"]["v"].pow(2).mean()
        loss.backward()
        for p in trunk.parameters():
            if p.grad is not None:
                p.grad = None
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Timed iters
    print(f"Timing {args.n_iters} forward+backward iters ...")
    times = []
    for i in range(args.n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = trunk(inp)
        loss = out["bb_ca"]["v"].pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        for p in trunk.parameters():
            if p.grad is not None:
                p.grad = None
        if i < 3 or i == args.n_iters - 1:
            print(f"  iter {i:>2}: {dt*1000:>7.1f} ms")

    print()
    import statistics
    mean = statistics.mean(times)
    med = statistics.median(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"==== Per-microbatch (forward + backward) ====")
    print(f"  median: {med*1000:.1f} ms")
    print(f"  mean:   {mean*1000:.1f} ms  ± {sd*1000:.1f} ms")
    print()
    print(f"==== Per-opt-step estimate (accum={args.accum}) ====")
    print(f"  median: {med * args.accum:.1f} s")
    print(f"  mean:   {mean * args.accum:.1f} s")
    print()
    canonical_per_step = 27.7
    print(f"==== Comparison vs canonical dense ({canonical_per_step} s/opt-step) ====")
    print(f"  median slowdown: {(med * args.accum) / canonical_per_step:.2f}×")
    print(f"  opt steps/hour:  {3600 / (med * args.accum):.0f}")
    print(f"  time to step 1500: {1500 * (med * args.accum) / 3600:.1f} h")

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    mem_resv = torch.cuda.max_memory_reserved() / 1e9
    print()
    print(f"==== GPU memory peak ====")
    print(f"  allocated: {mem_alloc:.1f} GB")
    print(f"  reserved:  {mem_resv:.1f} GB")


if __name__ == "__main__":
    main()
