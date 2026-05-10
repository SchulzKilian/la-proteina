"""Fill the gaps in the steering predictor × ensemble × fold matrix.

The audit identified three categories of missing data and three reporting
inconsistencies. This script addresses them in one resumable pipeline.

Tier 1 — new experiments
  1.1  Clean predictor, single fold (fold 0 + fold 2), no smoothing, no ensemble
       — n=4 smoke at L=300, w=16. Plus a clean+ensemble-NO-smoothing cell to
       isolate the smoothing contribution from E028.
  1.2  Noise-aware-v1 single fold for folds 0, 1, 3, 4 — same n=4 smoke.
       Closes "is fold 2 specially smart?" question.
  1.3  Noise-aware-v1 single fold (fold 2) full sweep — w ∈ {1,2,4,8,16} at
       L ∈ {300,400,500} × 16 seeds. Brings single-fold to the same n=48
       statistical power as E032 (NA + ensemble) and E028 (clean + ensemble).

Tier 2 — unified reporting
  Single per-protein-gap metric across every cell. Two markdown tables:
  the n=4 smoke matrix and the n=48 sweep matrix, with smoothing status
  columns and a Δratio sidebar for n=48 cells.

Stages (each idempotent, runnable independently):
    write-configs   emit YAML configs under steering/config/audit_matrix/
    generate        run steering.generate for any cell missing PDBs
    eval            run real-TANGO eval on any cell missing properties_guided.csv
    report          aggregate everything and print/write the two unified tables
    all             run the four stages in order

Prerequisites on the target machine:
  - Clean predictor ckpts at  PRED_CLEAN/fold_{0..4}_best.pt
  - Noise-aware ckpts at      PRED_NA_V1/fold_{0..4}_best.pt
  - TANGO binary on PATH or via $TANGO_EXE
  - PYTHON_EXEC pointing at the laproteina env (only needed for the eval stage)
  - The reference cells used as comparison anchors:
      results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w16/
        (E028 clean+ensemble+smoothing, n=4 + n=48 anchor)
      results/noise_aware_smoke/tango_min_w16_fold2/
        (E029 NA-v1 fold 2 single, n=4 anchor)
      results/noise_aware_ensemble_sweep/tango_min_w*/
        (E032 NA-v1 ensemble full sweep, n=48 anchor)
    Override the paths via CLI flags if your layout differs.

Typical use:
    cd ~/la-proteina
    python script_utils/audit_steering_matrix.py write-configs
    python script_utils/audit_steering_matrix.py generate --device cuda:0
    PYTHON_EXEC=$(which python) python script_utils/audit_steering_matrix.py eval
    python script_utils/audit_steering_matrix.py report --markdown audit_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "steering/config/audit_matrix"
RESULTS_DIR = ROOT / "results/audit_matrix"

PRED_CLEAN_DEFAULT = ROOT / "laproteina_steerability/logs/multitask_t1/20260427_161809/checkpoints"
PRED_NA_V1_DEFAULT = ROOT / "laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints"

REF_E028 = ROOT / "results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w16"
REF_E029 = ROOT / "results/noise_aware_smoke/tango_min_w16_fold2"
REF_E032_SWEEP = ROOT / "results/noise_aware_ensemble_sweep"

SMOKE_SEEDS = [42, 43, 44, 45]
SWEEP_SEEDS = list(range(42, 58))
SWEEP_LENGTHS = [300, 400, 500]
SWEEP_W = [1, 2, 4, 8, 16]

PROTEINA_CFG = "inference_ucond_notri_long"
NSTEPS = 400


@dataclass
class Cell:
    name: str
    predictor_kind: str
    fold_label: str
    smoothing: bool
    seeds: list[int]
    lengths: list[int]
    w: int
    checkpoints: list[str] = field(default_factory=list)


def make_smoke_cells(pred_clean: Path, pred_na: Path) -> list[Cell]:
    """Tier 1.1 + 1.2: w=16, L=300, seeds 42-45, single-cell smokes."""
    cells: list[Cell] = []

    cells.append(Cell(
        name="clean_fold0",
        predictor_kind="clean", fold_label="fold0", smoothing=False,
        seeds=SMOKE_SEEDS, lengths=[300], w=16,
        checkpoints=[str(pred_clean / "fold_0_best.pt")],
    ))
    cells.append(Cell(
        name="clean_fold2",
        predictor_kind="clean", fold_label="fold2", smoothing=False,
        seeds=SMOKE_SEEDS, lengths=[300], w=16,
        checkpoints=[str(pred_clean / "fold_2_best.pt")],
    ))
    cells.append(Cell(
        name="clean_ensemble_nosmoothing",
        predictor_kind="clean", fold_label="ens5", smoothing=False,
        seeds=SMOKE_SEEDS, lengths=[300], w=16,
        checkpoints=[str(pred_clean / f"fold_{i}_best.pt") for i in range(5)],
    ))

    for f in [0, 1, 3, 4]:
        cells.append(Cell(
            name=f"na_v1_fold{f}",
            predictor_kind="na_v1", fold_label=f"fold{f}", smoothing=False,
            seeds=SMOKE_SEEDS, lengths=[300], w=16,
            checkpoints=[str(pred_na / f"fold_{f}_best.pt")],
        ))
    return cells


def make_sweep_cells(pred_na: Path) -> list[Cell]:
    """Tier 1.3: NA-v1 fold 2 full sweep — w ∈ {1,2,4,8,16}, L ∈ {300,400,500}, 16 seeds."""
    cells: list[Cell] = []
    for w in SWEEP_W:
        cells.append(Cell(
            name=f"na_v1_fold2_sweep_w{w}",
            predictor_kind="na_v1", fold_label="fold2", smoothing=False,
            seeds=SWEEP_SEEDS, lengths=SWEEP_LENGTHS, w=w,
            checkpoints=[str(pred_na / "fold_2_best.pt")],
        ))
    return cells


def emit_yaml(cell: Cell, out_dir: Path) -> Path:
    cfg = {
        "steering": {
            "enabled": True,
            "checkpoint": cell.checkpoints if len(cell.checkpoints) > 1 else cell.checkpoints[0],
            "objectives": [{"property": "tango", "direction": "minimize", "weight": 1.0}],
            "schedule": {
                "type": "linear_ramp",
                "w_max": float(cell.w),
                "t_start": 0.3,
                "t_end": 0.8,
                "t_stop": 0.9,
            },
            "gradient_norm": "unit",
            "gradient_clip": 10.0,
            "channel": "local_latents",
            "log_diagnostics": True,
        }
    }
    if cell.smoothing:
        cfg["steering"]["smoothing"] = {"sigma": 0.1, "n_samples": 4}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cell.name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def cell_result_dir(cell: Cell) -> Path:
    return RESULTS_DIR / cell.name


def cell_pdbs_present(cell: Cell) -> bool:
    guided = cell_result_dir(cell) / "guided"
    if not guided.is_dir():
        return False
    expected = {f"s{s}_n{L}.pdb" for s in cell.seeds for L in cell.lengths}
    actual = {p.name for p in guided.glob("*.pdb")}
    return expected.issubset(actual)


def stage_write_configs(args) -> None:
    pred_clean = Path(args.pred_clean)
    pred_na = Path(args.pred_na)
    for d, p in [("clean", pred_clean), ("noise-aware", pred_na)]:
        if not p.is_dir():
            print(f"WARNING: {d} predictor dir not found: {p}", file=sys.stderr)
    cells = make_smoke_cells(pred_clean, pred_na) + make_sweep_cells(pred_na)
    for cell in cells:
        path = emit_yaml(cell, CONFIG_DIR)
        print(f"  {path.relative_to(ROOT)}  (predictor={cell.predictor_kind}/{cell.fold_label}, "
              f"w={cell.w}, n_seeds={len(cell.seeds)}, n_lengths={len(cell.lengths)})")


def stage_generate(args) -> None:
    pred_clean = Path(args.pred_clean)
    pred_na = Path(args.pred_na)
    cells = make_smoke_cells(pred_clean, pred_na) + make_sweep_cells(pred_na)
    for cell in cells:
        cfg_path = CONFIG_DIR / f"{cell.name}.yaml"
        if not cfg_path.exists():
            print(f"SKIP {cell.name}: config missing — run write-configs first", file=sys.stderr)
            continue
        if cell_pdbs_present(cell):
            print(f"[skip] {cell.name}: all expected PDBs already exist")
            continue
        out = cell_result_dir(cell)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "steering.generate",
            "--proteina_config", PROTEINA_CFG,
            "--steering_config", str(cfg_path),
            "--lengths", *map(str, cell.lengths),
            "--seeds", *map(str, cell.seeds),
            "--output_dir", str(out),
            "--device", args.device,
            "--nsteps", str(NSTEPS),
            "--skip_unguided",
        ]
        print(f"\n[run] {cell.name}\n  $ {shlex.join(cmd)}")
        subprocess.run(cmd, check=True)


def stage_eval(args) -> None:
    from steering.property_evaluate import evaluate_directory
    pred_clean = Path(args.pred_clean)
    pred_na = Path(args.pred_na)
    cells = make_smoke_cells(pred_clean, pred_na) + make_sweep_cells(pred_na)
    for cell in cells:
        out = cell_result_dir(cell)
        guided = out / "guided"
        csv = out / "properties_guided.csv"
        if not guided.is_dir():
            print(f"[skip] {cell.name}: no guided/ — run generate first")
            continue
        if csv.exists():
            print(f"[skip] {cell.name}: properties_guided.csv exists")
            continue
        print(f"[eval] {cell.name}: real TANGO on {len(list(guided.glob('*.pdb')))} PDBs ...")
        df = evaluate_directory(guided, skip_tango=False)
        df.to_csv(csv, index=False)


def predictor_final_tango(diag_path: Path) -> float:
    steps = json.loads(diag_path.read_text())
    return float(steps[-1]["predicted_properties"]["tango"])


def per_protein_gaps(run_dir: Path, pids: Optional[list[str]] = None) -> pd.DataFrame:
    csv = run_dir / "properties_guided.csv"
    diag_dir = run_dir / "diagnostics"
    if not csv.exists() or not diag_dir.is_dir():
        return pd.DataFrame()
    real = pd.read_csv(csv).set_index("protein_id")
    rows = []
    for diag in sorted(diag_dir.glob("*_diagnostics.json")):
        pid = diag.stem.replace("_diagnostics", "")
        if pids is not None and pid not in pids:
            continue
        if pid not in real.index or "tango_total" not in real.columns:
            continue
        v = real.loc[pid, "tango_total"]
        if pd.isna(v):
            continue
        pred = predictor_final_tango(diag)
        s, L = pid.split("_")
        rows.append({"pid": pid, "seed": int(s[1:]), "length": int(L[1:]),
                     "predictor": pred, "real": float(v), "gap": pred - float(v)})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "pred_mean": float("nan"), "real_mean": float("nan"),
                "gap_mean": float("nan"), "gap_std": float("nan")}
    return {
        "n": len(df),
        "pred_mean": float(df.predictor.mean()),
        "real_mean": float(df.real.mean()),
        "gap_mean": float(df.gap.mean()),
        "gap_std": float(df.gap.std()) if len(df) > 1 else 0.0,
    }


def stage_report(args) -> None:
    smoke_pids = [f"s{s}_n300" for s in SMOKE_SEEDS]

    smoke_rows: list[dict] = []

    def add_smoke(label: str, predictor: str, fold: str, smoothing: str, run_dir: Path):
        df = per_protein_gaps(run_dir, smoke_pids)
        agg = aggregate(df)
        smoke_rows.append({"label": label, "predictor": predictor, "fold": fold,
                           "smoothing": smoothing, **agg, "source": str(run_dir.relative_to(ROOT)) if run_dir.exists() else "MISSING"})

    pred_clean = Path(args.pred_clean)
    pred_na = Path(args.pred_na)

    add_smoke("E028 ref",                 "clean",   "ens5",  "σ=0.1, K=4", REF_E028)
    add_smoke("E029 ref",                 "NA-v1",   "fold2", "off",        REF_E029)
    add_smoke("Tier 1.1 — clean f0",      "clean",   "fold0", "off",        cell_result_dir(make_smoke_cells(pred_clean, pred_na)[0]))
    add_smoke("Tier 1.1 — clean f2",      "clean",   "fold2", "off",        cell_result_dir(make_smoke_cells(pred_clean, pred_na)[1]))
    add_smoke("Tier 1.1 — clean ens5 (no smooth)", "clean", "ens5", "off",  cell_result_dir(make_smoke_cells(pred_clean, pred_na)[2]))
    for i, fid in enumerate([0, 1, 3, 4]):
        add_smoke(f"Tier 1.2 — NA-v1 f{fid}", "NA-v1", f"fold{fid}", "off",
                  cell_result_dir(make_smoke_cells(pred_clean, pred_na)[3 + i]))

    sweep_rows: list[dict] = []

    def add_sweep(label: str, predictor: str, fold: str, smoothing: str, run_dir: Path,
                  by_w: dict[int, Path]):
        for w, rdir in by_w.items():
            df = per_protein_gaps(rdir)
            agg = aggregate(df)
            sweep_rows.append({"label": label, "predictor": predictor, "fold": fold,
                               "smoothing": smoothing, "w": w, **agg,
                               "source": str(rdir.relative_to(ROOT)) if rdir.exists() else "MISSING"})

    add_sweep("E028 ref full sweep", "clean", "ens5", "σ=0.1, K=4", REF_E028,
              {w: ROOT / f"results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w{w}" for w in SWEEP_W})
    add_sweep("E032 ref full sweep", "NA-v1", "ens5", "off", REF_E032_SWEEP,
              {w: REF_E032_SWEEP / f"tango_min_w{w}" for w in SWEEP_W})
    sweep_cells = make_sweep_cells(pred_na)
    add_sweep("Tier 1.3 — NA-v1 f2 sweep", "NA-v1", "fold2", "off", RESULTS_DIR,
              {c.w: cell_result_dir(c) for c in sweep_cells})

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_summary: list[dict] = []
    for label in sweep_df.label.unique():
        sub = sweep_df[sweep_df.label == label].sort_values("w")
        if sub.empty or sub.n.iloc[0] == 0:
            continue
        base_pred = sub[sub.w == 1].pred_mean.values
        base_real = sub[sub.w == 1].real_mean.values
        if base_pred.size == 0 or base_real.size == 0:
            continue
        bp, br = base_pred[0], base_real[0]
        for _, row in sub.iterrows():
            dp = row.pred_mean - bp
            dr = row.real_mean - br
            ratio = dp / dr if abs(dr) > 1e-9 else float("inf")
            sweep_summary.append({**row.to_dict(), "delta_pred": dp, "delta_real": dr, "ratio": ratio})

    out_lines: list[str] = []

    def w(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    w("# Steering audit — unified matrix")
    w("")
    w("Metric in both tables: per-protein gap = (predictor's last-step claim of TANGO) − (real `tango_total` from TANGO binary).")
    w("Gap < 0 ⇒ predictor under-claims real TANGO (classical hacking direction).")
    w("Gap ≈ 0 ⇒ predictor is honest about real-property change.")
    w("")
    w("## Table 1 — n=4 smoke (seeds {42-45}, L=300, w=16, tango_min)")
    w("")
    w("| Cell | Predictor | Fold | Smoothing | n | pred mean | real mean | **gap mean** | gap std | source |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in smoke_rows:
        if r["n"] == 0:
            w(f"| {r['label']} | {r['predictor']} | {r['fold']} | {r['smoothing']} | 0 | — | — | — | — | `{r['source']}` |")
            continue
        w(f"| {r['label']} | {r['predictor']} | {r['fold']} | {r['smoothing']} | {r['n']} | "
          f"{r['pred_mean']:.1f} | {r['real_mean']:.1f} | **{r['gap_mean']:+.1f}** | {r['gap_std']:.1f} | "
          f"`{r['source']}` |")

    w("")
    w("## Table 2 — n=48 full sweep (seeds 42-57 × L ∈ {300,400,500}, tango_min)")
    w("")
    w("| Cell | Predictor | Fold | Smoothing | w | n | pred mean | real mean | gap mean | Δpred (vs w=1) | Δreal (vs w=1) | **Δratio** |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sweep_summary:
        if r["n"] == 0:
            continue
        ratio_str = f"{r['ratio']:.2f}×" if np.isfinite(r["ratio"]) else "n/a"
        w(f"| {r['label']} | {r['predictor']} | {r['fold']} | {r['smoothing']} | {r['w']} | {r['n']} | "
          f"{r['pred_mean']:.1f} | {r['real_mean']:.1f} | {r['gap_mean']:+.1f} | "
          f"{r['delta_pred']:+.1f} | {r['delta_real']:+.1f} | **{ratio_str}** |")

    w("")
    w("## Smoothing key")
    w("- σ=0.1, K=4 = randomised gradient smoothing (4 N(0, 0.1²) draws averaged).")
    w("- off = single deterministic gradient.")
    w("- All cells use unit-norm gradient + w-scaling (matched effective step size).")

    if args.markdown:
        Path(args.markdown).write_text("\n".join(out_lines) + "\n")
        print(f"\nReport written to {args.markdown}")


def stage_all(args) -> None:
    stage_write_configs(args)
    stage_generate(args)
    stage_eval(args)
    stage_report(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pred-clean", default=str(PRED_CLEAN_DEFAULT),
                        help=f"Dir holding fold_{{0..4}}_best.pt for the clean predictor")
    common.add_argument("--pred-na", default=str(PRED_NA_V1_DEFAULT),
                        help=f"Dir holding fold_{{0..4}}_best.pt for the noise-aware-v1 predictor")

    p_wc = sub.add_parser("write-configs", parents=[common])
    p_wc.set_defaults(fn=stage_write_configs)

    p_gen = sub.add_parser("generate", parents=[common])
    p_gen.add_argument("--device", default="cuda:0")
    p_gen.set_defaults(fn=stage_generate)

    p_ev = sub.add_parser("eval", parents=[common])
    p_ev.set_defaults(fn=stage_eval)

    p_rep = sub.add_parser("report", parents=[common])
    p_rep.add_argument("--markdown", default=None,
                       help="Optional path to write the markdown report (also printed to stdout).")
    p_rep.set_defaults(fn=stage_report)

    p_all = sub.add_parser("all", parents=[common])
    p_all.add_argument("--device", default="cuda:0")
    p_all.add_argument("--markdown", default="audit_report.md")
    p_all.set_defaults(fn=stage_all)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
