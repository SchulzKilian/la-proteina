"""Steering cost audit + smart rating.

For a sweep tree containing {camsol_max,tango_min}_w{N}/guided/*.pdb cells,
this script:

  1. Auto-detects cells and their w-levels.
  2. Orchestrates the four eval pipelines on any cell missing its output
     CSV (resume-safe): property panel, scRMSD-with-MPNN-rescue,
     codesignability, pairwise-TM diversity, AA composition.
  3. Loads all per-cell CSVs and produces a smart rating per cell:
     PASS / WARN / FAIL per cost gate, plus a property-delivery verdict,
     plus an overall single-word verdict.

The rating is "smart" in that thresholds are relative to the w=1 anchor of
the same direction — so a baseline that's already low (e.g. codesign 38 %
on this generator) is calibrated as such, not measured against an absolute
"100 % designable" bar.

Usage:
    # Full pipeline: run any missing evals, then summarize.
    python script_utils/steering_cost_audit.py --tree results/noise_aware_high_w_scout

    # Skip orchestration; just rate already-computed CSVs.
    python script_utils/steering_cost_audit.py \
        --tree results/noise_aware_ensemble_sweep --summarize-only

    # Pick which evals to run.
    python script_utils/steering_cost_audit.py --tree <tree> --evals property,aa
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
UNSTEERED_PROPERTIES = ROOT / "results/generated_stratified_300_800_nsteps400/properties_generated.csv"

# Threshold table — bring these together so they're easy to retune. All
# deltas are vs the w=1 anchor of the same direction.
GATE_THRESHOLDS = {
    "codesign":      {"warn_pp": -15.0, "fail_pp": -30.0},
    "designability": {"warn_pp": -15.0, "fail_pp": -30.0},
    "diversity":     {"warn_dtm":  0.05, "fail_dtm":  0.15},
    "aa_kl":         {"warn_nats": 0.05, "fail_nats": 0.20},
    "max_aa":        {"warn_freq": 0.22, "fail_freq": 0.28},  # absolute, single AA > 22 % is WARN
    "shannon_drop":  {"warn_bits": 0.30, "fail_bits": 0.60},
    "lowcplx_rise":  {"warn":      0.05, "fail":      0.15},
    "homopoly_rise": {"warn":      2.0,  "fail":      4.0},
}
PROP_DELIVERED_SIGMA = 0.15  # |Δ| in σ-units to count as "delivered"
PROP_NULL_SIGMA = 0.05       # below this, signal is null


def detect_cells(tree: Path) -> list[tuple[Path, str, int]]:
    cells = []
    for d in sorted(tree.glob("*/guided")):
        cell = d.parent
        m = re.match(r"^(.+)_w(\d+)$", cell.name)
        if m:
            cells.append((cell, m.group(1), int(m.group(2))))
    return cells


# (real CSV column, want_sign) per steering direction.
# want_sign = +1 if steering wants the property to INCREASE, -1 if DECREASE.
DIRECTION_PROP_TARGET = {
    "camsol_max":         ("swi",                          +1),
    "tango_min":          ("tango_total",                  -1),
    "iupred_max":         ("iupred3_fraction_disordered",  +1),
    "combo_camsol_tango": ("swi",                          +1),  # multi-objective; primary = SWI
    "combo_devel4":       ("swi",                          +1),  # 4-obj: camsol+tango+sap+scmpos; primary = SWI
    "net_charge_max":     ("net_charge_ph7",               +1),
    "net_charge_min":     ("net_charge_ph7",               -1),
    # Schedule-variant sweeps (all targeting tango_total at w=32)
    "tango_min_early":    ("tango_total",                  -1),
    "tango_min_late":     ("tango_total",                  -1),
    "tango_min_wide":     ("tango_total",                  -1),
}

# True-w=0 paired-seed unsteered baselines (used when no w=1 cell in the tree).
UNSTEERED_CODESIGN_CSV = ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv"
UNSTEERED_PROPERTIES_CSV = ROOT / "results/generated_stratified_300_800_nsteps400/properties_generated.csv"


# ---- eval orchestration ----

def run_property_eval(cells: list[tuple[Path, str, int]]) -> None:
    """compute_developability per cell via steering.evaluate_samples_dir."""
    env = os.environ.copy()
    env.setdefault("TANGO_EXE", str(ROOT / "tango_x86_64_release"))
    for cell, _, _ in cells:
        out = cell / "properties_guided.csv"
        if out.exists():
            continue
        print(f"[property] {cell.name}")
        subprocess.run(
            [PY, "-m", "steering.evaluate_samples_dir",
             "--samples_dir", str(cell / "guided"),
             "--output_csv", str(out)],
            cwd=ROOT, env=env, check=True,
        )


def run_scrmsd_eval(cells: list[tuple[Path, str, int]], tree: Path,
                    seeds: list[int], lengths: list[int]) -> None:
    """scRMSD with MPNN rescue (designability) per cell. Resume-safe."""
    missing = [c.name for c, _, _ in cells if not (c / "scRMSD_guided.csv").exists()]
    if not missing:
        return
    env = os.environ.copy()
    env["OUT_BASE"] = str(tree.resolve().relative_to(ROOT))
    print(f"[scrmsd] cells={missing}")
    subprocess.run(
        [PY, "script_utils/run_scrmsd_steering.py",
         "--seeds", *map(str, seeds),
         "--lengths", *map(str, lengths),
         "--cfgs", *missing],
        cwd=ROOT, env=env, check=True,
    )


def run_codesign_eval(cells: list[tuple[Path, str, int]], tree: Path,
                      seeds: list[int], lengths: list[int]) -> None:
    """Codesignability per cell. Resume-safe."""
    missing = [c.name for c, _, _ in cells if not (c / "codesign_guided.csv").exists()]
    if not missing:
        return
    env = os.environ.copy()
    env["OUT_BASE"] = str(tree.resolve().relative_to(ROOT))
    print(f"[codesign] cells={missing}")
    subprocess.run(
        [PY, "scripts/run_codesignability_sweep.py",
         "--seeds", *map(str, seeds),
         "--lengths", *map(str, lengths),
         "--cfgs", *missing],
        cwd=ROOT, env=env, check=True,
    )


def run_diversity_eval(cells: list[tuple[Path, str, int]], tree: Path,
                       seeds: list[int], lengths: list[int]) -> None:
    """Pairwise TM-score per (cell, L). Writes diversity_pairwise_tm.csv at tree root."""
    out_csv = tree / "diversity_pairwise_tm.csv"
    if out_csv.exists():
        return
    print(f"[diversity] writing {out_csv.name}")
    # Import the helpers from the existing script rather than re-implementing.
    sys.path.insert(0, str(ROOT / "scripts"))
    from diversity_pairwise_tm import pairwise_tm, stats  # type: ignore

    rows = []
    for cell, direction, w in cells:
        for L in lengths:
            pdbs = sorted([cell / "guided" / f"s{s}_n{L}.pdb" for s in seeds
                           if (cell / "guided" / f"s{s}_n{L}.pdb").exists()])
            label = f"{direction}_w{w}_L{L}"
            print(f"  [{label}] {len(pdbs)} PDBs", flush=True)
            tms = pairwise_tm(pdbs) if len(pdbs) >= 2 else []
            row = stats(label, tms)
            row.update({"set": "steered", "direction": direction, "w": w, "L": L})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_aa_eval(tree: Path) -> None:
    """AA composition / Shannon / homopolymer / low-complexity."""
    out_dir = tree.parent / f"sequence_collapse_audit_{tree.name}"
    if (out_dir / "summary.csv").exists():
        return
    print(f"[aa_composition] writing {out_dir.name}/summary.csv")
    subprocess.run(
        [PY, "script_utils/check_sequence_collapse.py",
         "--tree", str(tree),
         "--out", str(out_dir)],
        cwd=ROOT, check=True,
    )


# ---- summary + rating ----

def codesign_rate(df: pd.DataFrame, threshold: float = 2.0) -> float:
    if df.empty or "coScRMSD_ca" not in df:
        return float("nan")
    finite = df.loc[np.isfinite(df.coScRMSD_ca), "coScRMSD_ca"]
    return float((finite <= threshold).mean()) if len(finite) else float("nan")


def designable_rate(df: pd.DataFrame, threshold: float = 2.0) -> float:
    if df.empty or "scRMSD_ca_min" not in df:
        return float("nan")
    finite = df.loc[np.isfinite(df.scRMSD_ca_min), "scRMSD_ca_min"]
    return float((finite <= threshold).mean()) if len(finite) else float("nan")


def load_cell_data(cell: Path) -> dict:
    """Read all CSVs for a cell. Missing CSVs become empty DataFrames."""
    out = {}
    for key, name in [("codesign", "codesign_guided.csv"),
                      ("scrmsd",   "scRMSD_guided.csv"),
                      ("props",    "properties_guided.csv")]:
        p = cell / name
        out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def unsteered_baseline_anchor() -> dict:
    """True-w=0 paired-seed baseline. Used when a tree has no in-tree w=1 cell
    (e.g., the iupred sweep). Codesign rates come from the matched-seed CSV
    (n=30, seeds 42-51 × L=300/400/500). Property anchor comes from the 1000-
    protein unsteered stratified panel filtered to L∈[290, 510] (n=422)."""
    out = {}
    out["codesign"] = pd.read_csv(UNSTEERED_CODESIGN_CSV) if UNSTEERED_CODESIGN_CSV.exists() else pd.DataFrame()
    if UNSTEERED_PROPERTIES_CSV.exists():
        df = pd.read_csv(UNSTEERED_PROPERTIES_CSV)
        out["props"] = df[df.sequence_length.between(290, 510)].copy() if "sequence_length" in df else df
    else:
        out["props"] = pd.DataFrame()
    out["scrmsd"] = pd.DataFrame()  # MPNN-rescue not measured for unsteered baseline
    return out


def unsteered_property_stats() -> dict:
    if not UNSTEERED_PROPERTIES.exists():
        return {}
    df = pd.read_csv(UNSTEERED_PROPERTIES)
    df = df[df.sequence_length.between(290, 510)]
    stats = {}
    for col in ["swi", "tango_total", "net_charge_ph7", "hydrophobic_patch_total_area"]:
        if col in df:
            stats[col] = (float(df[col].mean()), float(df[col].std()))
    return stats


def verdict(value: float, warn: float, fail: float, *, direction: str = "lower_is_worse") -> str:
    """direction='lower_is_worse': WARN if value<=warn, FAIL if value<=fail.
       direction='higher_is_worse': WARN if value>=warn, FAIL if value>=fail."""
    if math.isnan(value):
        return "MISSING"
    if direction == "lower_is_worse":
        if value <= fail: return "FAIL"
        if value <= warn: return "WARN"
        return "PASS"
    else:
        if value >= fail: return "FAIL"
        if value >= warn: return "WARN"
        return "PASS"


def fmt_pp(x: float) -> str:
    return f"{x*100:+.1f}pp" if not math.isnan(x) else "—"


def fmt_sigma(delta: float, sigma: float) -> tuple[float, str]:
    if sigma <= 0:
        return float("nan"), "σ=0"
    z = delta / sigma
    return z, f"{z:+.2f}σ"


def rate_cell(cell: Path, direction: str, w: int,
              anchor_cell: Path | None,
              diversity_df: pd.DataFrame,
              aa_df: pd.DataFrame,
              unsteered_stats: dict) -> dict:
    cur = load_cell_data(cell)
    if anchor_cell:
        anc = load_cell_data(anchor_cell)
        anchor_label = anchor_cell.name
    else:
        anc = unsteered_baseline_anchor()
        anchor_label = "unsteered (paired)"

    # 1. Codesignability — anchor either from anchor_cell or from unsteered paired baseline.
    cur_codesign = codesign_rate(cur["codesign"])
    anc_codesign = codesign_rate(anc["codesign"])
    d_codesign = (cur_codesign - anc_codesign) if not math.isnan(cur_codesign + anc_codesign) else float("nan")
    v_codesign = verdict(d_codesign,
                         GATE_THRESHOLDS["codesign"]["warn_pp"] / 100,
                         GATE_THRESHOLDS["codesign"]["fail_pp"] / 100,
                         direction="lower_is_worse")

    # 2. Designability (MPNN rescue)
    cur_des = designable_rate(cur["scrmsd"])
    anc_des = designable_rate(anc["scrmsd"]) if anchor_cell else cur_des
    d_des = (cur_des - anc_des) if not math.isnan(cur_des + anc_des) else float("nan")
    v_des = verdict(d_des,
                    GATE_THRESHOLDS["designability"]["warn_pp"] / 100,
                    GATE_THRESHOLDS["designability"]["fail_pp"] / 100,
                    direction="lower_is_worse")

    # 3. Diversity — mean TM across lengths
    div_sub = diversity_df[(diversity_df.direction == direction) & (diversity_df.w == w)] \
        if not diversity_df.empty else pd.DataFrame()
    div_anc = diversity_df[(diversity_df.direction == direction) & (diversity_df.w == 1)] \
        if not diversity_df.empty else pd.DataFrame()
    cur_tm = float((div_sub["mean"] * div_sub.n_pairs).sum() / max(div_sub.n_pairs.sum(), 1)) \
        if len(div_sub) else float("nan")
    anc_tm = float((div_anc["mean"] * div_anc.n_pairs).sum() / max(div_anc.n_pairs.sum(), 1)) \
        if len(div_anc) else float("nan")
    d_tm = (cur_tm - anc_tm) if not math.isnan(cur_tm + anc_tm) else float("nan")
    v_tm = verdict(d_tm,
                   GATE_THRESHOLDS["diversity"]["warn_dtm"],
                   GATE_THRESHOLDS["diversity"]["fail_dtm"],
                   direction="higher_is_worse")

    # 4. AA composition — KL vs w=1 (already computed by check_sequence_collapse.py)
    aa_row = aa_df[aa_df.cell == cell.name].iloc[0] if not aa_df.empty and (aa_df.cell == cell.name).any() else None
    if aa_row is not None:
        aa_kl = float(aa_row.kl_vs_w1)
        aa_max = float(aa_row.top_aa_freq)
        aa_shannon_delta = float(aa_row.mean_shannon - aa_df[(aa_df.direction == direction) & (aa_df.w == 1)].iloc[0].mean_shannon) \
            if (aa_df.direction == direction).any() and (aa_df.w == 1).any() else 0.0
        aa_homopoly_delta = float(aa_row.mean_longest_run - aa_df[(aa_df.direction == direction) & (aa_df.w == 1)].iloc[0].mean_longest_run) \
            if (aa_df.direction == direction).any() and (aa_df.w == 1).any() else 0.0
        aa_lowcplx_delta = float(aa_row.mean_low_complexity_frac - aa_df[(aa_df.direction == direction) & (aa_df.w == 1)].iloc[0].mean_low_complexity_frac) \
            if (aa_df.direction == direction).any() and (aa_df.w == 1).any() else 0.0
    else:
        aa_kl = aa_max = aa_shannon_delta = aa_homopoly_delta = aa_lowcplx_delta = float("nan")

    v_aa_kl = verdict(aa_kl,
                      GATE_THRESHOLDS["aa_kl"]["warn_nats"],
                      GATE_THRESHOLDS["aa_kl"]["fail_nats"],
                      direction="higher_is_worse")
    v_max_aa = verdict(aa_max,
                       GATE_THRESHOLDS["max_aa"]["warn_freq"],
                       GATE_THRESHOLDS["max_aa"]["fail_freq"],
                       direction="higher_is_worse")
    v_shannon = verdict(-aa_shannon_delta,  # bigger drop = worse
                        GATE_THRESHOLDS["shannon_drop"]["warn_bits"],
                        GATE_THRESHOLDS["shannon_drop"]["fail_bits"],
                        direction="higher_is_worse")
    v_lowcplx = verdict(aa_lowcplx_delta,
                        GATE_THRESHOLDS["lowcplx_rise"]["warn"],
                        GATE_THRESHOLDS["lowcplx_rise"]["fail"],
                        direction="higher_is_worse")
    v_homopoly = verdict(aa_homopoly_delta,
                         GATE_THRESHOLDS["homopoly_rise"]["warn"],
                         GATE_THRESHOLDS["homopoly_rise"]["fail"],
                         direction="higher_is_worse")
    seq_quality_verdicts = [v_shannon, v_lowcplx, v_homopoly]
    v_seq = ("FAIL" if "FAIL" in seq_quality_verdicts
             else "WARN" if "WARN" in seq_quality_verdicts else "PASS")

    # 5. Property delivery — compare to anchor (w=1 or unsteered baseline)
    prop_target, want_sign = DIRECTION_PROP_TARGET.get(direction, ("swi", +1))
    if not cur["props"].empty and prop_target in cur["props"]:
        cur_prop = float(cur["props"][prop_target].mean())
    else:
        cur_prop = float("nan")
    if not anc["props"].empty and prop_target in anc["props"]:
        anc_prop = float(anc["props"][prop_target].mean())
    else:
        anc_prop = cur_prop
    delta_prop = cur_prop - anc_prop
    sigma = unsteered_stats.get(prop_target, (0.0, 0.0))[1]
    z, z_label = fmt_sigma(delta_prop, sigma)
    signed_z = want_sign * z if not math.isnan(z) else float("nan")
    if math.isnan(signed_z):
        v_prop = "MISSING"
    elif abs(z) < PROP_NULL_SIGMA:
        v_prop = "NULL"
    elif signed_z >= PROP_DELIVERED_SIGMA:
        v_prop = "DELIVERED"
    elif signed_z > 0:
        v_prop = "WEAK"
    elif signed_z <= -PROP_DELIVERED_SIGMA:
        v_prop = "WRONG"
    else:
        v_prop = "WEAK_WRONG"

    # Overall verdict
    cost_gates = [v_codesign, v_des, v_tm, v_aa_kl, v_max_aa, v_seq]
    if "FAIL" in cost_gates:
        overall = "BROKEN"
    elif "WARN" in cost_gates:
        overall = "DEGRADED"
    elif v_prop == "DELIVERED":
        overall = "WORKING"
    elif v_prop == "WRONG":
        overall = "ADVERSARIAL"
    else:
        overall = "INERT"

    return {
        "cell": cell.name,
        "direction": direction,
        "w": w,
        # raw values
        "codesign_rate": cur_codesign,
        "codesign_anchor": anc_codesign,
        "d_codesign_pp": d_codesign * 100,
        "designable_rate": cur_des,
        "designable_anchor": anc_des,
        "d_designable_pp": d_des * 100,
        "diversity_tm": cur_tm,
        "diversity_anchor": anc_tm,
        "d_diversity_tm": d_tm,
        "aa_kl_vs_w1": aa_kl,
        "aa_max_freq": aa_max,
        "aa_shannon_delta": aa_shannon_delta,
        "aa_homopoly_delta": aa_homopoly_delta,
        "aa_lowcplx_delta": aa_lowcplx_delta,
        "prop_target": prop_target,
        "prop_value": cur_prop,
        "prop_anchor": anc_prop,
        "prop_delta": delta_prop,
        "prop_delta_sigma": signed_z,
        # verdicts
        "v_codesign": v_codesign,
        "v_designable": v_des,
        "v_diversity": v_tm,
        "v_aa_kl": v_aa_kl,
        "v_max_aa": v_max_aa,
        "v_seq_quality": v_seq,
        "v_property": v_prop,
        "overall": overall,
    }


def print_cell_report(row: dict) -> None:
    g = lambda v: {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]",
                   "MISSING": "[ -- ]"}.get(v, f"[{v[:4]}]")
    p = lambda v: {"DELIVERED": "[ OK ]", "WEAK": "[WEAK]", "NULL": "[NULL]",
                   "WRONG": "[WRNG]", "WEAK_WRONG": "[WW  ]",
                   "MISSING": "[ -- ]"}.get(v, f"[{v[:4]}]")
    o = {"WORKING": "✓ WORKING", "INERT": "○ INERT", "DEGRADED": "△ DEGRADED",
         "BROKEN": "✗ BROKEN", "ADVERSARIAL": "! ADVERSARIAL"}.get(row["overall"], row["overall"])

    print(f"\n=== {row['cell']} ===")
    print(f"  {g(row['v_codesign'])} Codesignability  : {row['codesign_rate']*100:5.1f}% "
          f"(anchor {row['codesign_anchor']*100:5.1f}%, Δ {fmt_pp(row['d_codesign_pp']/100)}; "
          f"fail < {GATE_THRESHOLDS['codesign']['fail_pp']}pp)")
    print(f"  {g(row['v_designable'])} Designability    : {row['designable_rate']*100:5.1f}% "
          f"(anchor {row['designable_anchor']*100:5.1f}%, Δ {fmt_pp(row['d_designable_pp']/100)}; "
          f"fail < {GATE_THRESHOLDS['designability']['fail_pp']}pp)")
    print(f"  {g(row['v_diversity'])} Diversity        : TM {row['diversity_tm']:.3f} "
          f"(anchor {row['diversity_anchor']:.3f}, Δ {row['d_diversity_tm']:+.3f}; "
          f"fail > +{GATE_THRESHOLDS['diversity']['fail_dtm']:.2f})")
    print(f"  {g(row['v_aa_kl'])} AA collapse (KL) : {row['aa_kl_vs_w1']:.4f} nats vs w=1 "
          f"(fail > {GATE_THRESHOLDS['aa_kl']['fail_nats']} nats)")
    print(f"  {g(row['v_max_aa'])} Single-AA ceiling: top-AA freq {row['aa_max_freq']*100:.1f}% "
          f"(fail > {GATE_THRESHOLDS['max_aa']['fail_freq']*100:.0f}% absolute)")
    print(f"  {g(row['v_seq_quality'])} Sequence quality : "
          f"ΔShannon {row['aa_shannon_delta']:+.2f} bits; "
          f"Δlongest_run {row['aa_homopoly_delta']:+.2f}; "
          f"Δlow_cplx {row['aa_lowcplx_delta']:+.3f}")
    print(f"  {p(row['v_property'])} Property delivery: Δ{row['prop_target']} = "
          f"{row['prop_delta']:+.3f} ({row['prop_delta_sigma']:+.2f}σ signed-toward-target; "
          f"delivered ≥ {PROP_DELIVERED_SIGMA}σ)")
    print(f"  Overall: {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 58)))
    ap.add_argument("--lengths", type=int, nargs="+", default=[300, 400, 500])
    ap.add_argument("--evals", type=str, default="property,scrmsd,codesign,diversity,aa",
                    help="Comma-separated subset of: property, scrmsd, codesign, diversity, aa.")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Skip orchestration; just rate existing CSVs.")
    args = ap.parse_args()

    cells = detect_cells(args.tree)
    if not cells:
        raise SystemExit(f"no steering cells under {args.tree}")
    print(f"Detected {len(cells)} cells under {args.tree}")
    for c, d, w in cells:
        print(f"  {d}_w{w} : {c}")

    if not args.summarize_only:
        evals = set(args.evals.split(","))
        if "property" in evals: run_property_eval(cells)
        if "aa"       in evals: run_aa_eval(args.tree)
        if "scrmsd"   in evals: run_scrmsd_eval(cells, args.tree, args.seeds, args.lengths)
        if "codesign" in evals: run_codesign_eval(cells, args.tree, args.seeds, args.lengths)
        if "diversity" in evals: run_diversity_eval(cells, args.tree, args.seeds, args.lengths)

    # Load aggregate data
    div_csv = args.tree / "diversity_pairwise_tm.csv"
    div_df = pd.read_csv(div_csv) if div_csv.exists() else pd.DataFrame()
    aa_csv = args.tree.parent / f"sequence_collapse_audit_{args.tree.name}" / "summary.csv"
    aa_df = pd.read_csv(aa_csv) if aa_csv.exists() else pd.DataFrame()
    unst = unsteered_property_stats()

    rows = []
    anchors = {d: next((c for c, dd, w in cells if dd == d and w == 1), None)
               for d in {d for _, d, _ in cells}}
    # If no w=1 in the tree, fall back to the noise_aware_ensemble_sweep w=1.
    fallback_root = ROOT / "results/noise_aware_ensemble_sweep"
    for d, anchor in list(anchors.items()):
        if anchor is None and (fallback_root / f"{d}_w1").exists():
            anchors[d] = fallback_root / f"{d}_w1"
            print(f"\n[anchor] {d}: using {anchors[d]} (no w=1 in current tree)")

    for cell, direction, w in cells:
        anchor = anchors.get(direction)
        if anchor is not None and anchor.resolve() == cell.resolve():
            anchor = None  # don't anchor a cell to itself
        rows.append(rate_cell(cell, direction, w, anchor, div_df, aa_df, unst))

    for row in rows:
        print_cell_report(row)

    df = pd.DataFrame(rows).sort_values(["direction", "w"]).reset_index(drop=True)
    out_csv = args.tree / "steering_cost_audit.csv"
    df.to_csv(out_csv, index=False)

    # Overall verdict table
    print("\n=== Summary ===")
    print(df[["cell", "overall", "v_codesign", "v_designable", "v_diversity",
              "v_aa_kl", "v_max_aa", "v_seq_quality", "v_property",
              "prop_delta_sigma", "d_codesign_pp", "d_designable_pp"]]
          .to_string(index=False,
                     formatters={"prop_delta_sigma": "{:+.2f}".format,
                                 "d_codesign_pp": "{:+.1f}".format,
                                 "d_designable_pp": "{:+.1f}".format}))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
