"""Convert developability_panel.csv to parquet with column names matching Part 2 config."""
import pandas as pd
from pathlib import Path

CSV_PATH = "/rds/user/ks2218/hpc-work/developability_panel.csv"
OUT_PATH = Path(__file__).parent / "data" / "properties.csv"

RENAME = {
    "pdb_id": "protein_id",
    "net_charge_ph7": "net_charge",
    "iupred3_mean": "iupred3",
    "tango_total": "tango",
    "sap_total": "sap",
    "radius_of_gyration": "rg",
    # These already match: swi, shannon_entropy, pI, scm_positive, scm_negative,
    # hydrophobic_patch_total_area, hydrophobic_patch_n_large
}

# Columns always NaN — skip them
DROP = ["camsol_intrinsic", "canya_max_nucleation"]

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df = df.rename(columns=RENAME)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Columns: {list(df.columns)}")
