"""Compare predictor's final-step claim vs real (TANGO binary / dev panel) metric.

Gradient-hacking signature: predictor's mean target moves monotonically with the
steering weight w, but the real-world metric on the generated proteins does not.

Reads:
  results/steering_camsol_tango_L500_ensemble_smoothed/<config>/
      diagnostics/sX_nL_diagnostics.json   (per-protein per-step predictor calls)
      properties_guided.csv                (real metrics from compute_developability)

Writes one markdown table to stdout — one block per direction (tango_min, camsol_max).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/steering_camsol_tango_L500_ensemble_smoothed"

# predictor head name -> column in properties_guided.csv
PRED_TO_REAL = {
    "tango": "tango_total",
    "swi": "swi",
    "net_charge": "net_charge_ph7",
    "pI": "pI",
    "iupred3": "iupred3_mean",
    "shannon_entropy": "shannon_entropy",
    "hydrophobic_patch_total_area": "hydrophobic_patch_total_area",
    "hydrophobic_patch_n_large": "hydrophobic_patch_n_large",
    "sap": "sap_total",
    "scm_positive": "scm_positive",
    "scm_negative": "scm_negative",
    "rg": "radius_of_gyration",
    # "camsol_intrinsic" -> NOT computable locally (compute_developability returns NaN)
}


def per_config_summary(cfg_dir: Path, target_pred_key: str):
    csv = pd.read_csv(cfg_dir / "properties_guided.csv").set_index("protein_id")
    diag_dir = cfg_dir / "diagnostics"

    pred_finals: list[float] = []
    real_vals: list[float] = []
    matched = 0
    for jp in sorted(diag_dir.glob("*_diagnostics.json")):
        steps = json.loads(jp.read_text())
        # last step has the predictor's view of the final protein
        last = steps[-1]
        pred_props = last.get("predicted_properties", {})
        if target_pred_key not in pred_props:
            continue
        pred_finals.append(float(pred_props[target_pred_key]))

        pid = jp.stem.replace("_diagnostics", "")
        real_col = PRED_TO_REAL.get(target_pred_key)
        if pid in csv.index and real_col and real_col in csv.columns:
            v = csv.loc[pid, real_col]
            if pd.notna(v):
                real_vals.append(float(v))
                matched += 1

    pred_mean = mean(pred_finals) if pred_finals else float("nan")
    real_mean = mean(real_vals) if real_vals else float("nan")
    return {
        "n_proteins": len(pred_finals),
        "n_real_matched": matched,
        "pred_mean": pred_mean,
        "real_mean": real_mean,
    }


def gather_direction(prefix: str, target: str):
    rows = []
    for w in [1, 2, 4, 8, 16]:
        d = SWEEP / f"{prefix}_w{w}"
        if not d.exists():
            continue
        s = per_config_summary(d, target)
        s["w"] = w
        rows.append(s)
    return rows


def fmt(rows, direction: str, target: str, real_col_label: str):
    print(f"\n### {direction} (predictor target = `{target}`)")
    if real_col_label is None:
        print(f"_real-world `{target}` is not computed locally — predictor column only._")
    else:
        print(f"_real-world column = `{real_col_label}`._")
    print()
    print("| w | n | predictor mean (final step) | real mean | Δ predictor vs w=1 | Δ real vs w=1 |")
    print("|---|---|---|---|---|---|")
    pred0 = rows[0]["pred_mean"] if rows else float("nan")
    real0 = rows[0]["real_mean"] if rows else float("nan")
    for r in rows:
        dpred = r["pred_mean"] - pred0
        dreal = r["real_mean"] - real0
        print(
            f"| {r['w']} | {r['n_proteins']} | {r['pred_mean']:.3f} | "
            f"{r['real_mean']:.3f} | {dpred:+.3f} | {dreal:+.3f} |"
        )


if __name__ == "__main__":
    # tango_min_*: predictor target = "tango", real = "tango_total"
    rows = gather_direction("tango_min", "tango")
    fmt(rows, "tango_min sweep", "tango", "tango_total")

    # camsol_max_*: predictor target = "camsol_intrinsic", no real column locally
    rows = gather_direction("camsol_max", "camsol_intrinsic")
    fmt(rows, "camsol_max sweep", "camsol_intrinsic", None)

    # Bonus: does steering for camsol affect collateral real properties?
    print("\n### camsol_max sweep — collateral real-property drift")
    print("(predictor not asked to move these; reports real metric only)\n")
    print("| w | n | sap_total | tango_total | scm_positive | scm_negative | hyd_patch_area |")
    print("|---|---|---|---|---|---|---|")
    for w in [1, 2, 4, 8, 16]:
        d = SWEEP / f"camsol_max_w{w}"
        if not d.exists():
            continue
        df = pd.read_csv(d / "properties_guided.csv")
        print(
            f"| {w} | {len(df)} | "
            f"{df['sap_total'].mean():.2f} | "
            f"{df['tango_total'].mean():.1f} | "
            f"{df['scm_positive'].mean():.2f} | "
            f"{df['scm_negative'].mean():.2f} | "
            f"{df['hydrophobic_patch_total_area'].mean():.0f} |"
        )

    # Also: real tango under tango_min sweep + collateral
    print("\n### tango_min sweep — collateral real-property drift")
    print("| w | n | sap_total | tango_total | net_charge_ph7 | hyd_patch_area |")
    print("|---|---|---|---|---|---|")
    for w in [1, 2, 4, 8, 16]:
        d = SWEEP / f"tango_min_w{w}"
        if not d.exists():
            continue
        df = pd.read_csv(d / "properties_guided.csv")
        print(
            f"| {w} | {len(df)} | "
            f"{df['sap_total'].mean():.2f} | "
            f"{df['tango_total'].mean():.1f} | "
            f"{df['net_charge_ph7'].mean():.2f} | "
            f"{df['hydrophobic_patch_total_area'].mean():.0f} |"
        )
