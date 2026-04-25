"""Aggregate evaluate.py per-cell CSVs into a tidy DataFrame and plot.

evaluate.py writes one CSV per --job_id at:
    inference/<EVAL_CONFIG>/../results_<EVAL_CONFIG>_<JOB_ID>.csv
We read all of them, parse (k, space, protein_id) from the pdb_path column,
and produce:
  - tidy.csv             :  one row per (protein, k, space) with all metrics
  - manifold_summary.csv :  mean/std per (k, space) over proteins
  - manifold_plot.png    :  all-atom codesignability RMSD vs k, two lines (coord vs latent)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PDB_NAME_RE = re.compile(
    r"^job_(?P<cell>\d+)_(?P<space>coord|latent)_k(?P<k>[0-9.]+)_(?P<pid>.+)\.pdb$"
)


def parse_pdb_name(path: str):
    name = Path(path).name
    m = PDB_NAME_RE.match(name)
    if not m:
        return None
    return {
        "cell_idx": int(m["cell"]),
        "space": m["space"],
        "k": float(m["k"]),
        "protein_id": m["pid"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-root", type=str,
                    default="./inference/eval_manifold_perturbation")
    ap.add_argument("--eval-config", type=str, default="eval_manifold_perturbation")
    ap.add_argument("--metric-col", type=str,
                    default="_res_co_scRMSD_all_atom_esmfold",
                    help="Codesignability all-atom RMSD column from evaluate.py.")
    args = ap.parse_args()

    inf_root = Path(args.inference_root).resolve()
    csv_dir = inf_root.parent  # evaluate.py writes results_*.csv here
    pattern = f"results_{args.eval_config}_*.csv"
    csvs = sorted(csv_dir.glob(pattern))
    if not csvs:
        print(f"No CSVs match {csv_dir}/{pattern}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(csvs)} CSVs from {csv_dir}")

    rows = []
    for csv in csvs:
        df = pd.read_csv(csv)
        if "pdb_path" not in df.columns:
            print(f"  skip {csv.name}: no pdb_path column")
            continue
        for _, r in df.iterrows():
            meta = parse_pdb_name(str(r["pdb_path"]))
            if meta is None:
                continue
            row = dict(meta)
            row["L"] = r.get("L", np.nan)
            row["pdb_path"] = r["pdb_path"]
            row["all_atom_rmsd"] = r.get(args.metric_col, np.nan)
            rows.append(row)

    if not rows:
        print("No usable rows parsed; check naming convention.", file=sys.stderr)
        sys.exit(1)

    tidy = pd.DataFrame(rows)
    # Codesignability returns -1 on processing error (per evaluate.py).
    tidy["all_atom_rmsd"] = pd.to_numeric(tidy["all_atom_rmsd"], errors="coerce")
    tidy.loc[tidy["all_atom_rmsd"] <= 0, "all_atom_rmsd"] = np.nan
    tidy.loc[~np.isfinite(tidy["all_atom_rmsd"]), "all_atom_rmsd"] = np.nan

    out_tidy = csv_dir / f"manifold_tidy_{args.eval_config}.csv"
    tidy.to_csv(out_tidy, index=False)
    print(f"Wrote tidy: {out_tidy}  ({len(tidy)} rows)")

    summary = (
        tidy.groupby(["k", "space"])["all_atom_rmsd"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .sort_values(["space", "k"])
    )
    out_sum = csv_dir / f"manifold_summary_{args.eval_config}.csv"
    summary.to_csv(out_sum, index=False)
    print(f"Wrote summary: {out_sum}")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    colors = {"coord": "#d62728", "latent": "#1f77b4"}
    for space in ("coord", "latent"):
        s = summary[summary["space"] == space].sort_values("k")
        if s.empty:
            continue
        ax.errorbar(
            s["k"], s["mean"],
            yerr=s["std"] / np.sqrt(s["count"].clip(lower=1)),
            label=f"{space} space",
            marker="o", capsize=3, color=colors[space], linewidth=1.8,
        )
    ax.set_xlabel(r"noise scale $k$  ($k\cdot\sigma$ per dimension)")
    ax.set_ylabel("ESMFold all-atom RMSD vs perturbed (Å)")
    ax.set_title("Sidechain manifold: coord vs latent perturbations\n"
                 f"({args.eval_config})")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out_png = csv_dir / f"manifold_plot_{args.eval_config}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
