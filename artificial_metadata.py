import os
import glob
import pandas as pd

# Set this to the exact filename your pipeline was trying to load
CSV_NAME = "data/df_pdb_f0.5_minlNone_maxlNone_mtNone_minoNone_maxoNone_minrNone_maxrNone_rnsrTrue_rpuTrue_rcuFalse.csv"

# 1. Find all processed .pt files
processed_dir = "data/processed"
pt_files = glob.glob(f"{processed_dir}/**/*.pt", recursive=True)

data = []
for path in pt_files:
    # Get the filename without .pt (e.g., "1abc" or "1abc_A")
    base_name = os.path.basename(path).replace(".pt", "")
    
    # Split into PDB ID and chain (if applicable)
    parts = base_name.split("_")
    pdb_id = parts[0]
    chain = parts[1] if len(parts) > 1 else "all"
    
    data.append({"pdb": pdb_id, "id": base_name, "chain": chain})

# 2. Save as DataFrame
df = pd.DataFrame(data)
df.to_csv(CSV_NAME, index=False)
print(f"Rescued metadata! Saved {len(df)} entries to {CSV_NAME}")