"""Evaluate biophysical properties on guided vs. unguided generated proteins.

Usage:
    python -m steering.property_evaluate \
        --input_dir results/steering_eval/net_charge_up \
        [--skip-tango] [--skip-esm]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openfold.np.residue_constants import (
    atom_types as OPENFOLD_ATOM_TYPES,
    restypes,
    restype_1to3,
)

logger = logging.getLogger(__name__)

# Import property computation functions
from proteinfoundation.analysis.compute_developability import (
    compute_swi,
    compute_tango,
    compute_camsol,
    compute_charge_and_pI,
    compute_shannon_entropy,
    compute_radius_of_gyration,
    compute_hydrophobic_patches,
    compute_sap_scm,
    build_pdb_string,
    run_freesasa,
    residue_type_to_sequence,
)

# Try to import iupred3
try:
    iupred3_dir = os.environ.get("IUPRED3_DIR", str(Path.home() / "iupred3"))
    if iupred3_dir not in sys.path and Path(iupred3_dir).is_dir():
        sys.path.insert(0, iupred3_dir)
    import iupred3_lib
    _IUPRED_FN = lambda seq: iupred3_lib.iupred(seq, "long")[0]
except ImportError:
    _IUPRED_FN = None
    logger.warning("IUPred3 not available — iupred3 columns will be NaN")


# Properties we compute (subset of the full developability panel)
PROPERTY_COLUMNS = [
    "protein_id", "sequence_length",
    "swi",
    "tango_total", "tango_aggregation_positions",
    "net_charge_ph7", "pI",
    "iupred3_mean", "iupred3_fraction_disordered",
    "shannon_entropy",
    "hydrophobic_patch_total_area", "hydrophobic_patch_n_large",
    "sap_total", "scm_positive", "scm_negative",
    "radius_of_gyration",
]


def compute_iupred3_standalone(sequence: str):
    """Compute IUPred3 disorder scores."""
    if _IUPRED_FN is None:
        return float("nan"), float("nan")
    scores = _IUPRED_FN(sequence)
    mean_score = float(np.mean(scores))
    frac_disordered = float(np.mean(np.array(scores) > 0.5))
    return mean_score, frac_disordered


def compute_properties_from_pt(pt_path: Path, skip_tango: bool = False) -> dict:
    """Compute biophysical properties from a .pt file saved by steering/generate.py.

    The .pt file contains coords already in OpenFold atom order (no reindex needed).
    """
    row = {col: float("nan") for col in PROPERTY_COLUMNS}

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    protein_id = data["id"]
    row["protein_id"] = protein_id

    # Coordinates — already in OpenFold order
    coords = data["coords_openfold"].numpy()  # [L, 37, 3]
    coord_mask = data["coord_mask"].numpy()   # [L, 37]
    residue_type = data["residue_type"]       # [L] tensor
    residue_names = data["residues"]          # list of 3-letter strings
    sequence = data["sequence"]

    L = len(sequence)
    row["sequence_length"] = L

    ca_mask = coord_mask[:, 1].astype(bool)
    ca_coords = coords[:, 1, :]  # [L, 3]

    # --- Sequence-based properties ---
    try:
        row["swi"] = compute_swi(sequence)
    except Exception as e:
        logger.warning("[swi] %s: %s", protein_id, e)

    if not skip_tango:
        try:
            tango_exe = os.environ.get("TANGO_EXE", str(_ROOT / "tango_x86_64_release"))
            os.environ["TANGO_EXE"] = tango_exe
            row["tango_total"], row["tango_aggregation_positions"] = compute_tango(sequence, protein_id)
        except Exception as e:
            logger.warning("[tango] %s: %s", protein_id, e)

    try:
        net_charge, pI = compute_charge_and_pI(sequence)
        row["net_charge_ph7"] = net_charge
        row["pI"] = pI
    except Exception as e:
        logger.warning("[charge/pI] %s: %s", protein_id, e)

    try:
        row["iupred3_mean"], row["iupred3_fraction_disordered"] = compute_iupred3_standalone(sequence)
    except Exception as e:
        logger.warning("[iupred3] %s: %s", protein_id, e)

    try:
        row["shannon_entropy"] = compute_shannon_entropy(sequence)
    except Exception as e:
        logger.warning("[shannon] %s: %s", protein_id, e)

    # --- Structure-based properties ---
    try:
        row["radius_of_gyration"] = compute_radius_of_gyration(ca_coords, ca_mask)
    except Exception as e:
        logger.warning("[rg] %s: %s", protein_id, e)

    # FreeSASA for patches, SAP, SCM
    per_residue_sasa = None
    per_sidechain_sasa = None
    try:
        pdb_str = build_pdb_string(coords, coord_mask, residue_names)
        sasa_result, sasa_structure = run_freesasa(pdb_str)
        if sasa_result is not None:
            res_areas = sasa_result.residueAreas()
            chain_map = res_areas.get("A", {})
            per_residue_sasa = np.array(
                [chain_map[str(i + 1)].total if str(i + 1) in chain_map else 0.0
                 for i in range(L)], dtype=float)
            per_sidechain_sasa = np.array(
                [chain_map[str(i + 1)].sideChain if str(i + 1) in chain_map else 0.0
                 for i in range(L)], dtype=float)
    except Exception as e:
        logger.warning("[freesasa] %s: %s", protein_id, e)

    try:
        patch_area, n_large = compute_hydrophobic_patches(ca_coords, sequence, per_residue_sasa)
        row["hydrophobic_patch_total_area"] = patch_area
        row["hydrophobic_patch_n_large"] = n_large
    except Exception as e:
        logger.warning("[patches] %s: %s", protein_id, e)

    try:
        sap, scm_pos, scm_neg = compute_sap_scm(ca_coords, sequence, per_sidechain_sasa)
        row["sap_total"] = sap
        row["scm_positive"] = scm_pos
        row["scm_negative"] = scm_neg
    except Exception as e:
        logger.warning("[sap/scm] %s: %s", protein_id, e)

    return row


def evaluate_directory(data_dir: Path, skip_tango: bool = False) -> pd.DataFrame:
    """Compute properties for all .pt files in a directory."""
    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        logger.warning("No .pt files found in %s", data_dir)
        return pd.DataFrame(columns=PROPERTY_COLUMNS)

    rows = []
    for pt_path in pt_files:
        logger.info("  Computing properties: %s", pt_path.stem)
        row = compute_properties_from_pt(pt_path, skip_tango=skip_tango)
        rows.append(row)

    return pd.DataFrame(rows)


def build_comparison(guided_df: pd.DataFrame, unguided_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired comparison: for each protein_id, compute guided - unguided delta."""
    # Merge on protein_id
    merged = guided_df.merge(
        unguided_df, on="protein_id", suffixes=("_guided", "_unguided"),
    )

    numeric_cols = [c for c in PROPERTY_COLUMNS if c not in ("protein_id", "sequence_length")]
    rows = []
    for _, row in merged.iterrows():
        entry = {"protein_id": row["protein_id"]}
        for col in numeric_cols:
            g = row.get(f"{col}_guided", np.nan)
            u = row.get(f"{col}_unguided", np.nan)
            if not (math.isnan(g) or math.isnan(u)):
                entry[f"{col}_guided"] = g
                entry[f"{col}_unguided"] = u
                entry[f"{col}_delta"] = g - u
            else:
                entry[f"{col}_guided"] = g
                entry[f"{col}_unguided"] = u
                entry[f"{col}_delta"] = np.nan
        rows.append(entry)

    return pd.DataFrame(rows)


def build_summary(comparison_df: pd.DataFrame, objectives: list) -> pd.DataFrame:
    """Per-property summary: mean delta, std, paired t-test, fraction shifted correctly."""
    from scipy import stats as sp_stats

    numeric_cols = [c for c in PROPERTY_COLUMNS if c not in ("protein_id", "sequence_length")]

    # Build a lookup for expected direction
    obj_lookup = {}
    for obj in objectives:
        obj_lookup[obj["property"]] = obj.get("direction", "maximize")

    # Map property columns to objective properties (partial name match)
    # The 13 steering properties map to the developability columns:
    STEERING_TO_DEV = {
        "net_charge": "net_charge_ph7",
        "pI": "pI",
        "swi": "swi",
        "tango": "tango_total",
        "iupred3": "iupred3_mean",
        "iupred3_fraction_disordered": "iupred3_fraction_disordered",
        "shannon_entropy": "shannon_entropy",
        "hydrophobic_patch_total_area": "hydrophobic_patch_total_area",
        "hydrophobic_patch_n_large": "hydrophobic_patch_n_large",
        "sap": "sap_total",
        "scm_positive": "scm_positive",
        "scm_negative": "scm_negative",
        "rg": "radius_of_gyration",
    }

    rows = []
    for col in numeric_cols:
        delta_col = f"{col}_delta"
        if delta_col not in comparison_df.columns:
            continue
        deltas = comparison_df[delta_col].dropna().values
        if len(deltas) < 2:
            rows.append({"property": col, "n": len(deltas), "mean_delta": np.nan})
            continue

        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1))
        t_stat, p_val = sp_stats.ttest_1samp(deltas, 0.0)

        # Determine expected direction for this property
        expected_dir = None
        is_target = False
        for steer_name, dev_name in STEERING_TO_DEV.items():
            if dev_name == col and steer_name in obj_lookup:
                direction = obj_lookup[steer_name]
                if direction == "maximize":
                    expected_dir = "positive"
                elif direction == "minimize":
                    expected_dir = "negative"
                else:
                    is_target = True

        if expected_dir == "positive":
            frac_correct = float(np.mean(deltas > 0))
        elif expected_dir == "negative":
            frac_correct = float(np.mean(deltas < 0))
        else:
            frac_correct = np.nan

        rows.append({
            "property": col,
            "n": len(deltas),
            "mean_delta": mean_d,
            "std_delta": std_d,
            "t_stat": t_stat,
            "p_value": p_val,
            "fraction_shifted_correctly": frac_correct,
            "steered": expected_dir is not None or is_target,
        })

    return pd.DataFrame(rows)


def build_report(
    summary_df: pd.DataFrame,
    guided_df: pd.DataFrame,
    unguided_df: pd.DataFrame,
    objectives: list,
) -> str:
    """Human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("STEERING EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Objectives: {objectives}")
    lines.append(f"Proteins evaluated: {len(guided_df)}")
    lines.append("")

    # Steered properties
    steered = summary_df[summary_df["steered"] == True]
    if len(steered) > 0:
        lines.append("--- STEERED PROPERTIES ---")
        for _, row in steered.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            lines.append(
                f"  {row['property']:40s}  delta={row['mean_delta']:+.4f} +/- {row['std_delta']:.4f}  "
                f"p={row['p_value']:.4f}{sig}  frac_correct={row['fraction_shifted_correctly']:.2f}"
            )
        lines.append("")

    # Non-steered properties (collateral effects)
    non_steered = summary_df[summary_df["steered"] != True]
    if len(non_steered) > 0:
        lines.append("--- COLLATERAL EFFECTS (non-steered properties) ---")
        for _, row in non_steered.iterrows():
            if math.isnan(row.get("mean_delta", float("nan"))):
                continue
            sig = "*" if row["p_value"] < 0.05 else ""
            lines.append(
                f"  {row['property']:40s}  delta={row['mean_delta']:+.4f} +/- {row['std_delta']:.4f}  "
                f"p={row['p_value']:.4f}{sig}"
            )
        lines.append("")

    # ESMFold placeholder
    lines.append("--- DESIGNABILITY ---")
    lines.append("  ESMFold not available, skipping designability evaluation.")
    lines.append("  Install esm package and re-run to compute scRMSD metrics.")
    lines.append("")

    # Verdict
    if len(steered) > 0:
        all_sig = all(row["p_value"] < 0.05 for _, row in steered.iterrows())
        all_correct = all(
            row["fraction_shifted_correctly"] > 0.5
            for _, row in steered.iterrows()
            if not math.isnan(row["fraction_shifted_correctly"])
        )
        if all_sig and all_correct:
            verdict = "STEERING WORKS: all steered properties shifted significantly in the correct direction."
        elif all_correct:
            verdict = "STEERING DIRECTION CORRECT but not all shifts are statistically significant (need more samples)."
        else:
            verdict = "STEERING INCONCLUSIVE or FAILED. Check diagnostics."
    else:
        verdict = "No steered properties found in summary."

    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering: guided vs unguided properties")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing guided/ and unguided/ subdirs")
    parser.add_argument("--skip-tango", action="store_true",
                        help="Skip TANGO computation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_dir = Path(args.input_dir)
    guided_dir = input_dir / "guided"
    unguided_dir = input_dir / "unguided"

    if not guided_dir.exists() or not unguided_dir.exists():
        logger.error("Expected guided/ and unguided/ subdirectories in %s", input_dir)
        sys.exit(1)

    # Load run config for objectives
    run_config_path = input_dir / "run_config.yaml"
    objectives = []
    if run_config_path.exists():
        import yaml
        with open(run_config_path) as f:
            run_cfg = yaml.safe_load(f)
        objectives = run_cfg.get("steering", {}).get("objectives", [])

    # Set TANGO path
    tango_path = _ROOT / "tango_x86_64_release"
    if tango_path.exists():
        os.environ["TANGO_EXE"] = str(tango_path)

    # Compute properties
    logger.info("Computing properties for unguided proteins...")
    unguided_df = evaluate_directory(unguided_dir, skip_tango=args.skip_tango)
    unguided_df.to_csv(input_dir / "unguided_properties.csv", index=False)

    logger.info("Computing properties for guided proteins...")
    guided_df = evaluate_directory(guided_dir, skip_tango=args.skip_tango)
    guided_df.to_csv(input_dir / "guided_properties.csv", index=False)

    # Comparison
    logger.info("Building comparison...")
    comparison_df = build_comparison(guided_df, unguided_df)
    comparison_df.to_csv(input_dir / "comparison.csv", index=False)

    # Summary
    logger.info("Building summary...")
    summary_df = build_summary(comparison_df, objectives)
    summary_df.to_csv(input_dir / "summary.csv", index=False)

    # Report
    report = build_report(summary_df, guided_df, unguided_df, objectives)
    with open(input_dir / "report.txt", "w") as f:
        f.write(report)

    print(report)
    logger.info("All outputs saved to %s", input_dir)


if __name__ == "__main__":
    main()
