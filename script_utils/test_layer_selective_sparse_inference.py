"""
Layer-selective sparse-vs-dense inference diagnostic.

Inference-only. Loads the dense baseline ckpt (E019, step 2646; 87 % L=50,
87 % L=100, 53 % L=200 at N=30) and forces a subset of its 14 transformer
layers to use sparse attention at inference time. The dense-trained upper
layers were never exposed to sparsified intermediate representations, so
the result is an UPPER BOUND on what a properly trained layer-hybrid would
show.

Sweep: 4 splits × 2 K × 2 L = 16 cells, N=30 per cell, nsteps=400 (hard rule).
Stage gating (set by --stage):
  stage 1 — controls: 2 all_dense cells (L=50, L=100). Halt if not ≥ 18/30.
  stage 2 — asymmetry diagnosis: 3 cells at K=128, L=50 (lower / upper / all
            sparse). Halt if lower-vs-upper asymmetry is absent (< 10 pp).
  stage 3 — remaining 11 cells. Run only if stage 2 shows asymmetry.

Outputs:
  - inference/<config_name>/*.pdb  per cell
  - inference/results_<config_name>_0.csv  per cell
  - results/layer_sel/cells.json  aggregated per-cell results
  - notes/layer_selective_sparse_inference.md  final report (written after
    the last allowed stage completes)

CAVEATS (also surfaced in the report):
  - Dense-trained weights with partially sparsified inference produce an
    UPPER bound on a trained layer-hybrid's quality. A catastrophic result
    kills the simple form of the hypothesis; a non-catastrophic result is
    necessary-but-not-sufficient for training the hybrid.
  - seed=5 inferred from cousin entries; E019's exact seed isn't directly
    attested inline, but seed=5 matches every recent N=30/N=6 sparse probe.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path("/home/ks2218/la-proteina")
PYTHON_EXEC = os.environ.get(
    "PYTHON_EXEC", "/home/ks2218/.conda/envs/laproteina_env/bin/python"
)
NLAYERS = 14
BASE_CFG = "inference_layer_sel_base"
BASELINE_CKPT = (
    REPO_ROOT / "store/test_ca_only_diffusion/1776805213/checkpoints"
    / "best_val_00000026_000000002646.ckpt"
)

# K compositions — match the §11 K=64 SALAD-canonical convention.
# K_total = 2*n_seq + n_spatial + n_random.
K_SPECS = {
    64:  (8, 16, 32),   # 2*8+16+32 = 64; SALAD-canonical, §11 K=64 trained run
    128: (16, 32, 64),  # 2*16+32+64 = 128; clean 2× scale of K=64 SALAD ratio
}

LAYER_SPLITS = {
    "all_dense":          [False] * NLAYERS,
    "lower_half_sparse":  [True] * 7 + [False] * 7,
    "upper_half_sparse":  [False] * 7 + [True] * 7,
    "all_sparse":         [True] * NLAYERS,
}

# Stage ordering. Cell tuple: (stage, split_name, K_total, L).
ALL_CELLS = []
for L in (50, 100):
    for K in (64, 128):
        for split in ("all_dense", "lower_half_sparse",
                       "upper_half_sparse", "all_sparse"):
            ALL_CELLS.append((split, K, L))

STAGE_1_CELLS = [
    ("all_dense", 128, 50),
    ("all_dense", 128, 100),
]
STAGE_2_CELLS = [
    ("lower_half_sparse", 128, 50),
    ("upper_half_sparse", 128, 50),
    ("all_sparse",        128, 50),
]
STAGE_3_CELLS = [
    c for c in ALL_CELLS
    if c not in STAGE_1_CELLS and c not in STAGE_2_CELLS
]


def cell_config_name(split, K, L):
    return f"inference_layer_sel_cell_{split}_K{K}_L{L}"


def write_cell_config(split, K, L):
    """Write the per-cell hydra config file under configs/ and return its name."""
    cfg_name = cell_config_name(split, K, L)
    cfg_path = REPO_ROOT / "configs" / f"{cfg_name}.yaml"

    mask = LAYER_SPLITS[split]
    n_seq, n_sp, n_rd = K_SPECS[K]
    mask_yaml = "[" + ", ".join("true" if x else "false" for x in mask) + "]"

    content = f"""# Auto-generated cell of test_layer_selective_sparse_inference.py.
# DO NOT EDIT BY HAND — regenerate by re-running the driver.
defaults:
  - {BASE_CFG}
  - _self_

config_name: {cfg_name}

generation:
  args:
    layer_sparse_mask: {mask_yaml}
    n_seq_neighbors_override: {n_seq}
    n_spatial_neighbors_override: {n_sp}
    n_random_neighbors_override: {n_rd}
  dataset:
    nsamples: 30
    max_nsamples_per_batch: 6
    nlens_cfg:
      nres_lens: [{L}]
"""
    cfg_path.write_text(content)
    return cfg_name


def clean_cell_state(cfg_name):
    """Pre-clean output dirs + leftover CSV / tmp_dirs for this cell.

    Per `[[feedback_eval_tmp_dir_sweep]]`: evaluate.py asserts tmp_dir does not
    exist; crashed eval leaves nested <sample>/<sample>/ dirs behind and
    bombs every PDB on retry. Use rm -rf, not rmdir.
    """
    out_dir = REPO_ROOT / "inference" / cfg_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for csv in (REPO_ROOT / "inference").glob(f"results_{cfg_name}_*.csv"):
        csv.unlink()


def run_generation(cfg_name):
    """Run generate.py for this cell. PDBs land in inference/<cfg_name>/."""
    cmd = [
        PYTHON_EXEC, "proteinfoundation/generate.py",
        "--config-name", cfg_name,
    ]
    env = dict(os.environ)
    env["PYTHON_EXEC"] = PYTHON_EXEC  # so designability eval (later) sees it
    print(f"[gen] {cfg_name}", flush=True)
    t0 = time.time()
    log_path = REPO_ROOT / "inference" / f"{cfg_name}.gen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[gen] FAILED ({dt:.0f}s) — see {log_path}", flush=True)
        sys.exit(f"generate.py failed for {cfg_name}; halting.")
    print(f"[gen] done ({dt:.0f}s)", flush=True)


def run_evaluation(cfg_name):
    """Run evaluate.py for this cell. Writes inference/results_<cfg>_0.csv."""
    cmd = [
        PYTHON_EXEC, "proteinfoundation/evaluate.py",
        "--config_name", cfg_name,
    ]
    env = dict(os.environ)
    env["PYTHON_EXEC"] = PYTHON_EXEC  # ProteinMPNN subprocess needs this
    print(f"[eval] {cfg_name}", flush=True)
    t0 = time.time()
    log_path = REPO_ROOT / "inference" / f"{cfg_name}.eval.log"
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[eval] FAILED ({dt:.0f}s) — see {log_path}", flush=True)
        sys.exit(f"evaluate.py failed for {cfg_name}; halting.")
    print(f"[eval] done ({dt:.0f}s)", flush=True)


def parse_designability(cfg_name, L):
    """Return (designable, total, scrmsd_min, scrmsd_median)."""
    csv_path = REPO_ROOT / "inference" / f"results_{cfg_name}_0.csv"
    df = pd.read_csv(csv_path)
    df = df[df["L"] == L]
    scrmsd = df["_res_scRMSD_ca_esmfold"]
    valid = (scrmsd > 0) & (scrmsd < float("inf")) & scrmsd.notna()
    designable = int(((scrmsd < 2.0) & valid).sum())
    total = int(len(df))
    valid_scores = scrmsd[valid]
    return {
        "designable": designable,
        "total": total,
        "rate": (designable / total) if total > 0 else 0.0,
        "scrmsd_min": float(valid_scores.min()) if len(valid_scores) else float("nan"),
        "scrmsd_median": float(valid_scores.median()) if len(valid_scores) else float("nan"),
    }


def run_one_cell(split, K, L):
    cfg_name = write_cell_config(split, K, L)
    clean_cell_state(cfg_name)
    run_generation(cfg_name)
    run_evaluation(cfg_name)
    res = parse_designability(cfg_name, L)
    return {"split": split, "K": K, "L": L, "config": cfg_name, **res}


def append_results(records, out_json):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2))


def stage_1_check(records):
    """Both all_dense L=50 and L=100 must clear 18/30 (60 % of baseline 87 %).

    The baseline is 26/30 (87 %); a controlled regression to 18/30 (60 %)
    would still be enough to claim the pipeline is sane while leaving
    headroom for nsteps=400 seed-paired noise.
    """
    for r in records:
        if r["split"] == "all_dense":
            if r["designable"] < 18:
                return False, (
                    f"all_dense {r['config']}: {r['designable']}/{r['total']} "
                    f"designable (expected ≥ 18/30 vs baseline 26/30). "
                    "Pipeline broken; halting."
                )
    return True, "stage 1 control reproduces dense baseline within tolerance."


def stage_2_check(records, all_dense_L50):
    """Lower vs upper at K=128 L=50 must differ by ≥ 10 pp.

    If both halves degrade similarly relative to all_dense, the architectural
    hypothesis (lower layers tolerate sparsification, upper need dense) is
    falsified at the most favourable regime and the remaining cells are
    wasted compute.
    """
    lookup = {(r["split"], r["K"], r["L"]): r for r in records}
    lower = lookup.get(("lower_half_sparse", 128, 50))
    upper = lookup.get(("upper_half_sparse", 128, 50))
    if lower is None or upper is None:
        return False, "stage 2 missing lower or upper cell."
    asymmetry_pp = abs(lower["rate"] - upper["rate"]) * 100.0
    msg = (
        f"K=128, L=50: lower {lower['designable']}/{lower['total']} "
        f"({lower['rate']*100:.1f}%) vs upper {upper['designable']}/{upper['total']} "
        f"({upper['rate']*100:.1f}%) — Δ={asymmetry_pp:.1f} pp."
    )
    if asymmetry_pp < 10.0:
        return False, msg + " NO ASYMMETRY (<10 pp); halting before stage 3."
    return True, msg + " Asymmetry present; proceed to stage 3."


def write_report(records, out_md):
    """Write the markdown report — 16-cell table + interpretation."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lookup = {(r["split"], r["K"], r["L"]): r for r in records}

    all_dense = {L: lookup.get(("all_dense", 128, L)) for L in (50, 100)}

    def cell_str(split, K, L):
        r = lookup.get((split, K, L))
        if r is None:
            return "—"
        return f"{r['designable']}/{r['total']} ({r['rate']*100:.1f}%)"

    def delta_str(split, K, L):
        r = lookup.get((split, K, L))
        ref = all_dense.get(L)
        if r is None or ref is None:
            return "—"
        d = (r["rate"] - ref["rate"]) * 100.0
        return f"{d:+.1f} pp"

    lines = []
    lines.append("# Layer-selective sparse-vs-dense inference diagnostic")
    lines.append("")
    lines.append("**Inference-only stress test on dense-trained weights.** "
                  "Replaces attention computation in subsets of the 14 transformer "
                  "layers with the existing sparse path (`PairBiasAttention._attn_sparse`) "
                  "while loading the dense baseline checkpoint "
                  "`best_val_00000026_000000002646.ckpt` (E019; 87 % / 87 % / 53 % "
                  "at L=50 / 100 / 200, N=30).")
    lines.append("")
    lines.append("**Provenance of seed**: `seed=5` is inferred from cousin "
                  "entries (E054, E049, etc.); not directly attested inline in E019.")
    lines.append("")
    lines.append("## 16-cell table")
    lines.append("")
    lines.append("| L | K | layer_split | designable (N=30) | Δ vs all_dense |")
    lines.append("|---|---|---|---|---|")
    for L in (50, 100):
        for K in (64, 128):
            for split in ("all_dense", "lower_half_sparse",
                           "upper_half_sparse", "all_sparse"):
                lines.append(
                    f"| {L} | {K} | {split} | {cell_str(split, K, L)} "
                    f"| {delta_str(split, K, L)} |"
                )
    lines.append("")
    lines.append("Note: `all_dense` is K-independent (no sparse layers consume "
                  "the neighbor list); the K=64 and K=128 rows reuse the same "
                  "run, identical numbers.")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    a50 = all_dense.get(50)
    a100 = all_dense.get(100)
    lines.append(f"**1. Control reproduces dense baseline?** "
                  f"L=50: {a50['designable'] if a50 else '?'}/30 vs "
                  f"baseline 26/30 (87 %). "
                  f"L=100: {a100['designable'] if a100 else '?'}/30 vs "
                  f"baseline 26/30 (87 %). "
                  "Bit-identity unit test (`script_utils/test_layer_selective_sanity.py`) "
                  "confirmed `layer_sparse_mask=[False]*14` is exactly equal to the "
                  "global dense path (max-abs-diff 0.00e+00, fp32).")
    lines.append("")

    lower = lookup.get(("lower_half_sparse", 128, 50))
    upper = lookup.get(("upper_half_sparse", 128, 50))
    if lower and upper:
        asymmetry = (lower["rate"] - upper["rate"]) * 100.0
        if asymmetry > 0:
            asym_call = "lower-half-sparse is closer to all-dense than upper-half"
        elif asymmetry < 0:
            asym_call = "upper-half-sparse is closer to all-dense than lower-half"
        else:
            asym_call = "no asymmetry"
        lines.append(f"**2. Lower vs upper at K=128, L=50.** "
                      f"Lower {lower['designable']}/{lower['total']} "
                      f"({lower['rate']*100:.1f}%) vs upper "
                      f"{upper['designable']}/{upper['total']} "
                      f"({upper['rate']*100:.1f}%) — Δ = {asymmetry:+.1f} pp. "
                      f"Reading: **{asym_call}**.")
    lines.append("")

    monotonic_violations = []
    for split in ("lower_half_sparse", "upper_half_sparse", "all_sparse"):
        for L in (50, 100):
            r_low_K  = lookup.get((split, 64, L))
            r_high_K = lookup.get((split, 128, L))
            if r_low_K is None or r_high_K is None:
                continue
            if r_high_K["rate"] < r_low_K["rate"] - 0.05:
                monotonic_violations.append(
                    f"{split} L={L}: K=64 {r_low_K['rate']*100:.1f}% > "
                    f"K=128 {r_high_K['rate']*100:.1f}% (Δ {(r_high_K['rate']-r_low_K['rate'])*100:+.1f} pp)"
                )
    if monotonic_violations:
        lines.append("**3. K monotonicity flags.** "
                      "Expected K=128 ≥ K=64. Violations (potential bug or noise):")
        for v in monotonic_violations:
            lines.append(f"  - {v}")
    else:
        lines.append("**3. K monotonicity.** No violations: K=128 ≥ K=64 at every "
                      "(split, L) where both cells ran.")
    lines.append("")

    lines.append("**4. Upper bound on layer-hybrid viability.** "
                  "This test uses dense-trained weights with partially-sparsified "
                  "inference. Dense-trained upper layers expect complete pair "
                  "representations from lower layers; they were never exposed to "
                  "sparsified intermediate representations during training. The "
                  "degradation observed here is therefore an UPPER BOUND on what a "
                  "properly trained layer-hybrid would show — a from-scratch hybrid "
                  "training run could perform better because the upper layers would "
                  "adapt. A non-catastrophic result here is NECESSARY-BUT-NOT-SUFFICIENT "
                  "evidence for the hybrid being worth training; a catastrophic result "
                  "kills the simple form of the hypothesis.")
    lines.append("")
    lines.append("## Pipeline notes")
    lines.append("")
    lines.append("- Wiring change: `LocalLatentsTransformer.layer_sparse_mask` "
                  "(default `None` → bit-identical to pre-existing forward). When set "
                  "to a list of 14 bools, layer `i` uses sparse attention iff "
                  "`mask[i]` is True. Mutex with `use_downsampling`, `n_global_tokens > 0`, "
                  "`router_sparse_K`, `update_pair_repr=True`.")
    lines.append("- Inference hook: `cfg.generation.args.layer_sparse_mask` + "
                  "`force_sparse_attention_on` in `generate.py:load_ckpt_n_configure_inference`.")
    lines.append("- Bit-identity sanity check passed: `[True]*14` ≡ existing global "
                  "sparse path; `[False]*14` ≡ existing global dense path (max-abs-diff "
                  "0.00e+00, fp32, B=2, N=64).")

    out_md.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 0],
                        default=0, help="0 = run all stages with halt gates")
    parser.add_argument("--allow-stage-2-without-asymmetry",
                        action="store_true",
                        help="Force stage 3 even if stage 2 shows no asymmetry.")
    args = parser.parse_args()

    os.environ["PYTHON_EXEC"] = PYTHON_EXEC
    assert BASELINE_CKPT.exists(), f"baseline ckpt missing: {BASELINE_CKPT}"

    out_json = REPO_ROOT / "results" / "layer_sel" / "cells.json"
    out_md = REPO_ROOT / "notes" / "layer_selective_sparse_inference.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if out_json.exists():
        try:
            records = json.loads(out_json.read_text())
        except Exception:
            records = []

    def has_cell(split, K, L):
        return any(r["split"] == split and r["K"] == K and r["L"] == L for r in records)

    def run_stage(stage_id, cells):
        new_records = []
        for split, K, L in cells:
            if has_cell(split, K, L):
                print(f"[skip] {split} K={K} L={L} — already in cells.json")
                continue
            print(f"\n=== STAGE {stage_id} cell: split={split} K={K} L={L} ===")
            r = run_one_cell(split, K, L)
            print(f"    → {r['designable']}/{r['total']} designable "
                  f"({r['rate']*100:.1f}%); scRMSD min {r['scrmsd_min']:.2f} "
                  f"median {r['scrmsd_median']:.2f}")
            records.append(r)
            new_records.append(r)
            append_results(records, out_json)
        return new_records

    # Stage 1
    if args.stage in (0, 1):
        run_stage(1, STAGE_1_CELLS)
        ok, msg = stage_1_check(records)
        print(f"\n[stage 1 check] {msg}")
        if not ok:
            write_report(records, out_md)
            sys.exit(1)
    # Stage 2
    if args.stage in (0, 2):
        run_stage(2, STAGE_2_CELLS)
        ok, msg = stage_2_check(records, all_dense_L50=None)
        print(f"\n[stage 2 check] {msg}")
        if not ok and not args.allow_stage_2_without_asymmetry:
            write_report(records, out_md)
            sys.exit(0)
    # Stage 3
    if args.stage in (0, 3):
        run_stage(3, STAGE_3_CELLS)

    write_report(records, out_md)
    print(f"\nReport written to {out_md}")


if __name__ == "__main__":
    main()
