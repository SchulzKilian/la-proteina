"""Convert developability_panel.csv to parquet with column names matching Part 2 config."""
import pandas as pd
from pathlib import Path

CSV_PATH = "/home/ks2218/la-proteina/developability_panel.csv"
CAMSOL_PATH = "/home/ks2218/la-proteina/CamSolpH_results.txt"
OUT_PATH = Path(__file__).parent / "data" / "properties.csv"

RENAME = {
    "pdb_id": "protein_id",
    "net_charge_ph7": "net_charge",
    "iupred3_mean": "iupred3",
    "tango_total": "tango",
    "sap_total": "sap",
    "radius_of_gyration": "rg",
    # These already match: swi, shannon_entropy, pI, scm_positive, scm_negative,
    # hydrophobic_patch_total_area, hydrophobic_patch_n_large, camsol_intrinsic
}

# Always-NaN in the developability panel; camsol now sourced from CamSolpH_results.txt.
DROP = ["canya_max_nucleation"]

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Replace the always-NaN camsol_intrinsic column with the real CamSol pH 7 scores.
    camsol = pd.read_csv(
        CAMSOL_PATH, sep="\t", usecols=["Name", "protein variant score"]
    ).rename(columns={"Name": "pdb_id", "protein variant score": "camsol_intrinsic"})
    print(f"Loaded {len(camsol)} CamSol scores")
    df = df.drop(columns=["camsol_intrinsic"]).merge(camsol, on="pdb_id", how="left")
    print(f"  CamSol coverage: {df['camsol_intrinsic'].notna().sum()}/{len(df)}")

    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df = df.rename(columns=RENAME)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Columns: {list(df.columns)}")
