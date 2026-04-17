import argparse
import pandas as pd
import pathlib
from tqdm import tqdm

# CONFIG: Ensure this matches your folder name
DATASET_NAME = "pdb_train"

def clean_dataset_strict(csv_path=None):
    # 1. Setup Paths
    base_dir = pathlib.Path.cwd() / "data" / DATASET_NAME
    processed_dir = base_dir / "processed"

    # 2. Find CSV
    if csv_path is not None:
        csv_path = pathlib.Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
        if not csv_path.exists():
            print(f"❌ CSV not found: {csv_path}")
            return
    else:
        csv_files = list(base_dir.glob("df_pdb_f1*.csv"))
        if not csv_files:
            print(f"❌ No CSV found in {base_dir}")
            return
        csv_path = csv_files[0]
    
    print(f"📄 Reading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    original_len = len(df)
    
    valid_indices = []
    missing_count = 0
    
    print(f"🔍 Strictly checking {original_len} files...")

    # 3. STRICT Check (Exactly matching DataLoader logic)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying locations"):
        pdb = row["pdb"]
        chain = row["chain"] if "chain" in row and pd.notna(row["chain"]) else "all"
        
        # Construct filename
        fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
        
        # Logic: Check Shard OR Root (only these two places)
        shard = fname[0:2].lower()
        
        path_sharded = processed_dir / shard / fname
        path_root = processed_dir / fname
        
        # We only keep it if it is in the CORRECT location
        if path_sharded.exists() or path_root.exists():
            valid_indices.append(idx)
        else:
            missing_count += 1

    # 4. Save
    if missing_count > 0:
        print(f"\n⚠️  Found {missing_count} files that are in the wrong location or missing.")
        print("   Removing them from CSV to fix the crash...")
        
        df_clean = df.loc[valid_indices]
        df_clean.to_csv(csv_path, index=False)
        print(f"✅ CSV Rewritten! Size: {original_len} -> {len(df_clean)}")
    else:
        print("\n✅ No strict mismatches found.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV filename (in data/pdb_train/) or absolute path; default: first df_pdb_f1*.csv glob match")
    args = ap.parse_args()
    clean_dataset_strict(csv_path=args.csv)