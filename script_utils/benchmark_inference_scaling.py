"""Inference-time compute scaling benchmark, OOM-resilient.

Extends E074 (script_utils/benchmark_inference_sparse_vs_dense.py) to long
lengths and a new GPU. Measures peak GPU memory + per-protein wall-clock at a
ladder of L for canonical-dense and sparse-K40, on the SAME GPU process, same
nsteps=400 schedule, same seed.

Differences from the E074 script (why this is a separate file):
  - **Incremental CSV writes.** Each (arm, L) row is appended and flushed the
    instant it is measured, so an OOM / SLURM-kill / Ctrl-C mid-sweep keeps
    every length measured so far. The E074 script wrote the CSV only at the
    very end (one OOM lost the whole run).
  - **Per-arm OOM handling.** A CUDA-OOM at length L is caught, logged to the
    sidecar `<csv>.oom.txt`, and BREAKS that arm's length loop (every longer L
    would OOM too). The other arm still runs its full ladder — this is exactly
    how the "sparse fits at L where dense OOMs" data point is produced.
  - **Default ladder from L=100 upward** (overlaps E074's 100..500 so the new
    curve is directly comparable, then 600..2400 to find the dense ceiling).

Output CSV schema is byte-compatible with E074's
`results/inference_compute_audit/sparse_vs_dense_scaling.csv`
(columns arm,ckpt,L,n,wall_s,wall_s_per_protein,peak_gpu_mb,gpu) so the two can
be concatenated. OOM events are NOT written as data rows (they would pollute
the numeric columns); they go to the `.oom.txt` sidecar instead.

Per CLAUDE.md: nsteps=400 (hard rule), one GPU, arms run sequentially.

Usage (single GPU already selected by CUDA_VISIBLE_DEVICES):
    python script_utils/benchmark_inference_scaling.py \
        --output_csv results/inference_compute_audit/scaling_a100.csv \
        --lengths 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400
"""

import argparse
import csv
import gc
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

# torch.cuda.OutOfMemoryError exists in torch>=1.13; fall back to RuntimeError
# string-match for safety across versions.
_OOM_TYPES = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else ()


def _is_oom(exc: Exception) -> bool:
    if _OOM_TYPES and isinstance(exc, _OOM_TYPES):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def benchmark_one(config_name, ckpt_name, label, lengths, nsamples, seed, device,
                  row_sink, oom_path, gpu_name):
    """Load one ckpt and measure (wall, peak_gpu) at each length.

    Calls row_sink(dict) for every SUCCESSFUL length (row written + flushed
    immediately). On the first CUDA-OOM, records it to oom_path and stops this
    arm (no longer length can fit).
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
    # create_root=True just makes ./inference/<config_name> if absent (idempotent);
    # the benchmark writes no outputs there, but setup() refuses if it's missing.
    setup(cfg, create_root=True, config_name=config_name, job_id=0)

    print(f"[load] arm={label} ckpt={ckpt_name}", flush=True)
    model = load_ckpt_n_configure_inference(cfg)
    model._generation_base_seed = seed
    model.to(device).eval()

    # Warmup at the shortest length to amortise CUDA init / JIT / cuBLAS handles.
    print(f"[warmup] L={lengths[0]} n=1", flush=True)
    warmup_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    warmup_cfg.generation.dataset.nsamples = 1
    warmup_cfg.generation.dataset.max_nsamples_per_batch = 1
    warmup_cfg.generation.dataset.nlens_cfg.nres_lens = [lengths[0]]
    warmup_dl = DataLoader(GenDataset(**warmup_cfg.generation.dataset), batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(warmup_dl):
            model.predict_step(_move_batch_to_device(batch, device), batch_idx)
            break
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for L in lengths:
        per_L_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        per_L_cfg.generation.dataset.nsamples = nsamples
        per_L_cfg.generation.dataset.max_nsamples_per_batch = nsamples
        per_L_cfg.generation.dataset.nlens_cfg.nres_lens = [L]
        dl = DataLoader(GenDataset(**per_L_cfg.generation.dataset), batch_size=1, shuffle=False)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dl):
                    model.predict_step(_move_batch_to_device(batch, device), batch_idx)
                    break  # one batch = nsamples proteins
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            msg = (f"[OOM] arm={label} L={L} n={nsamples} "
                   f"(peak before OOM ~{peak_mb:.0f} MB on {gpu_name}); "
                   f"stopping arm — all longer L would OOM too.")
            print(msg, flush=True)
            with open(oom_path, "a") as f:
                f.write(msg + "\n")
            gc.collect()
            torch.cuda.empty_cache()
            break  # ceiling for this arm reached

        t1 = time.perf_counter()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        wall = t1 - t0
        row = {
            "arm": label,
            "ckpt": ckpt_name,
            "L": L,
            "n": nsamples,
            "wall_s": wall,
            "wall_s_per_protein": wall / nsamples,
            "peak_gpu_mb": peak_mb,
            "gpu": gpu_name,
        }
        row_sink(row)
        print(f"[measure] arm={label} L={L:>4d} n={nsamples} wall={wall:8.2f}s "
              f"({wall/nsamples:7.2f}s/protein) peak={peak_mb:8.1f} MB", flush=True)
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--nsamples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                             1200, 1400, 1600, 1800, 2000, 2200, 2400])
    ap.add_argument("--arms", nargs="+",
                    choices=["canonical_dense", "sparse_K40", "official_LD3"],
                    default=["canonical_dense", "sparse_K40"])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA"
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES already selects
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[setup] device={gpu_name} lengths={args.lengths} "
          f"nsamples={args.nsamples} seed={args.seed} arms={args.arms}", flush=True)

    all_arms = {
        "canonical_dense": ("inference_canonical_step2646_n6_nfe400",
                            "baseline_wd0.05_step2646.ckpt"),
        "sparse_K40": ("inference_sparse_K40_step1259_baseline_n6_nfe400",
                       "sparse_K40_step1259.ckpt"),
        # Official full-atom La Proteina (AE + latent + full-atom decode).
        # Its config sets ckpt_path=./checkpoints_laproteina and autoencoder_ckpt_path
        # itself, so no repo-root symlink is needed. _ca_only_mode=False here.
        "official_LD3": ("inference_ucond_notri_long",
                         "LD3_ucond_notri_800.ckpt"),
    }

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    oom_path = out.with_suffix(out.suffix + ".oom.txt")
    fieldnames = ["arm", "ckpt", "L", "n", "wall_s", "wall_s_per_protein",
                  "peak_gpu_mb", "gpu"]

    # --- Resume: skip already-measured (arm, L) and arms that hit their OOM
    # ceiling on a previous run. Re-running the driver after a crash / Ctrl-C /
    # time-limit picks up exactly where it left off instead of redoing the
    # expensive dense tail.
    done = set()        # (arm, L) already in CSV
    oom_arms = set()    # arms that already OOM'd -> ceiling known, don't retry
    resuming = out.exists()
    if resuming:
        with out.open(newline="") as rf:
            for r in csv.DictReader(rf):
                try:
                    done.add((r["arm"], int(r["L"])))
                except (KeyError, ValueError):
                    continue
    if oom_path.exists():
        oom_txt = oom_path.read_text()
        for label in all_arms:
            if f"arm={label} " in oom_txt:
                oom_arms.add(label)
    if done or oom_arms:
        print(f"[resume] {len(done)} (arm,L) already measured; "
              f"OOM-ceiling arms to skip: {sorted(oom_arms) or 'none'}", flush=True)

    # Append if resuming (preserve prior rows); write header only for a fresh file.
    f = out.open("a" if resuming else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not resuming:
        writer.writeheader()
        f.flush()

    def row_sink(row):
        writer.writerow(row)
        f.flush()  # survive Ctrl-C / OOM / SLURM kill

    try:
        for label in args.arms:
            cfg_name, ckpt_name = all_arms[label]
            if label in oom_arms:
                print(f"\n=== {label} === [resume] already hit OOM ceiling; "
                      f"skipping arm.", flush=True)
                continue
            remaining = [L for L in args.lengths if (label, L) not in done]
            if not remaining:
                print(f"\n=== {label} === [resume] all requested lengths done; "
                      f"skipping arm.", flush=True)
                continue
            print(f"\n=== {label} === lengths to run: {remaining}", flush=True)
            benchmark_one(cfg_name, ckpt_name, label, remaining, args.nsamples,
                          args.seed, device, row_sink, oom_path, gpu_name)
    finally:
        f.close()

    print(f"\n[done] wrote {out}", flush=True)
    if oom_path.exists():
        print(f"[done] OOM events recorded in {oom_path}", flush=True)


if __name__ == "__main__":
    main()
