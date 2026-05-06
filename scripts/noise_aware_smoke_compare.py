"""Compare noise-aware-fold-2 steering vs the original 5-fold-clean ensemble at
tango_min_w16, on the same 4 seeds at L=300.

For each protein:
  - predictor's last-step claim of `tango`     (from diagnostics JSON)
  - real `tango_total` from the TANGO binary   (computed here for the new run;
    pulled from properties_guided.csv for the old run)
  - gap = predictor_claim_minus_real

Honest predictor: gap is small (~regression error). Hacked predictor: gap is
large and negative (predictor *claims* TANGO is low, real value is high).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from steering.property_evaluate import evaluate_directory

OLD = ROOT / "results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w16"
NEW = ROOT / "results/noise_aware_smoke/tango_min_w16_fold2"
UG  = ROOT / "results/universal_guidance_smoke/tango_min_w16_clean_K5"
V2  = ROOT / "results/noise_aware_smoke/tango_min_w16_fold2_v2"
V1Z = ROOT / "results/zt_smoke/tango_min_w16_v1_zt"
V2Z = ROOT / "results/zt_smoke/tango_min_w16_v2_zt"
V1E = ROOT / "results/noise_aware_smoke/tango_min_w16_v1_ensemble"
PIDS = ["s42_n300", "s43_n300", "s44_n300", "s45_n300"]


def predictor_final_tango(diag_path: Path) -> float:
    steps = json.loads(diag_path.read_text())
    return float(steps[-1]["predicted_properties"]["tango"])


def gather(run_dir: Path, real_csv: pd.DataFrame | None):
    """Return a dataframe row per pid: predictor claim, real tango, gap."""
    rows = []
    for pid in PIDS:
        pred = predictor_final_tango(run_dir / "diagnostics" / f"{pid}_diagnostics.json")
        if real_csv is not None:
            real = float(real_csv.set_index("protein_id").loc[pid, "tango_total"])
        else:
            real = float("nan")
        rows.append({"pid": pid, "predictor_tango": pred, "real_tango": real,
                     "gap": pred - real})
    return pd.DataFrame(rows)


def main():
    # OLD run: real values already in properties_guided.csv
    old_real = pd.read_csv(OLD / "properties_guided.csv")
    old_df = gather(OLD, old_real)

    # NEW run: compute real TANGO on the 4 PDBs ourselves
    new_csv_path = NEW / "properties_guided.csv"
    if new_csv_path.exists():
        new_real = pd.read_csv(new_csv_path)
    else:
        print("Computing real TANGO on noise-aware smoke (4 PDBs)...", flush=True)
        new_real = evaluate_directory(NEW / "guided", skip_tango=False)
        new_real.to_csv(new_csv_path, index=False)
    new_df = gather(NEW, new_real)

    # UG run (universal guidance K=5 + clean ensemble)
    ug_df = None
    if UG.exists():
        ug_csv_path = UG / "properties_guided.csv"
        if ug_csv_path.exists():
            ug_real = pd.read_csv(ug_csv_path)
        else:
            print("Computing real TANGO on universal-guidance smoke (4 PDBs)...", flush=True)
            ug_real = evaluate_directory(UG / "guided", skip_tango=False)
            ug_real.to_csv(ug_csv_path, index=False)
        ug_df = gather(UG, ug_real)

    # v2 noise-aware run (longer + cosine decay, fold 2 of v2)
    v2_df = None
    if V2.exists():
        v2_csv_path = V2 / "properties_guided.csv"
        if v2_csv_path.exists():
            v2_real = pd.read_csv(v2_csv_path)
        else:
            print("Computing real TANGO on v2 noise-aware smoke (4 PDBs)...", flush=True)
            v2_real = evaluate_directory(V2 / "guided", skip_tango=False)
            v2_real.to_csv(v2_csv_path, index=False)
        v2_df = gather(V2, v2_real)

    # v1 + v2 with feed_z_t_directly=true (predictor sees z_t at the real t,
    # matching its noise-aware training distribution).
    v1z_df = v2z_df = v1e_df = None
    for tag, root in [("v1z", V1Z), ("v2z", V2Z), ("v1e", V1E)]:
        if not root.exists():
            continue
        csv_path = root / "properties_guided.csv"
        if csv_path.exists():
            real = pd.read_csv(csv_path)
        else:
            print(f"Computing real TANGO on {tag} smoke (4 PDBs)...", flush=True)
            real = evaluate_directory(root / "guided", skip_tango=False)
            real.to_csv(csv_path, index=False)
        df = gather(root, real)
        if tag == "v1z":
            v1z_df = df
        elif tag == "v2z":
            v2z_df = df
        else:
            v1e_df = df

    print()
    print("### Old ensemble (5-fold clean + smoothing, K_d=1) — tango_min_w16, L=300, seeds 42-45")
    print(old_df.to_string(index=False))
    print()
    print("### v1 noise-aware fold 2 (10 epochs, constant LR) — same protein IDs")
    print(new_df.to_string(index=False))
    if ug_df is not None:
        print()
        print("### Universal guidance K_d=5 (5-fold clean ensemble + smoothing) — same protein IDs")
        print(ug_df.to_string(index=False))
    if v2_df is not None:
        print()
        print("### v2 noise-aware fold 2 (30 epochs, cosine decay) — same protein IDs")
        print(v2_df.to_string(index=False))
    if v1z_df is not None:
        print()
        print("### v1 noise-aware fold 2 + feed_z_t_directly — same protein IDs")
        print(v1z_df.to_string(index=False))
    if v2z_df is not None:
        print()
        print("### v2 noise-aware fold 2 + feed_z_t_directly — same protein IDs")
        print(v2z_df.to_string(index=False))
    if v1e_df is not None:
        print()
        print("### v1 noise-aware 5-fold ensemble (legacy x_1_est+t=1) — same protein IDs")
        print(v1e_df.to_string(index=False))
    print()
    print("### Aggregate")
    print(f"  Old        predictor mean = {old_df.predictor_tango.mean():7.1f}, "
          f"real mean = {old_df.real_tango.mean():7.1f}, mean gap = {old_df.gap.mean():+8.1f}")
    print(f"  v1 NA-f2   predictor mean = {new_df.predictor_tango.mean():7.1f}, "
          f"real mean = {new_df.real_tango.mean():7.1f}, mean gap = {new_df.gap.mean():+8.1f}")
    if ug_df is not None:
        print(f"  UG  K=5    predictor mean = {ug_df.predictor_tango.mean():7.1f}, "
              f"real mean = {ug_df.real_tango.mean():7.1f}, mean gap = {ug_df.gap.mean():+8.1f}")
    if v2_df is not None:
        print(f"  v2 NA-f2   predictor mean = {v2_df.predictor_tango.mean():7.1f}, "
              f"real mean = {v2_df.real_tango.mean():7.1f}, mean gap = {v2_df.gap.mean():+8.1f}")
    if v1z_df is not None:
        print(f"  v1 + z_t   predictor mean = {v1z_df.predictor_tango.mean():7.1f}, "
              f"real mean = {v1z_df.real_tango.mean():7.1f}, mean gap = {v1z_df.gap.mean():+8.1f}")
    if v2z_df is not None:
        print(f"  v2 + z_t   predictor mean = {v2z_df.predictor_tango.mean():7.1f}, "
              f"real mean = {v2z_df.real_tango.mean():7.1f}, mean gap = {v2z_df.gap.mean():+8.1f}")
    if v1e_df is not None:
        print(f"  v1 ens5    predictor mean = {v1e_df.predictor_tango.mean():7.1f}, "
              f"real mean = {v1e_df.real_tango.mean():7.1f}, mean gap = {v1e_df.gap.mean():+8.1f}")


if __name__ == "__main__":
    main()
