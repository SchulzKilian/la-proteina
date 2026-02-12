import pandas as pd
import pathlib
from tqdm import tqdm

# CONFIG: Ensure this matches your folder name
DATASET_NAME = "pdb_train" 

def clean_dataset_strict():
    # 1. Setup Paths
    base_dir = pathlib.Path.cwd() / "data" / DATASET_NAME
    processed_dir = base_dir / "processed"
    
    # 2. Find CSV
    csv_files = list(base_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV found in {base_dir}")
        return
    csv_path = csv_files[0]
    
    print(f"📄 Reading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    original_len = len(df)
    
    valid_indices = []
    missing_count = 0
    
    print(f"🔍 Strictly checking {original_len} files in {processed_dir}...")

    # 3. STRICT Check (Exactly matching DataLoader logic)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb = row["pdb"]
        chain = row["chain"] if "chain" in row and pd.notna(row["chain"]) else "all"
        
        # Construct filename
        fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
        
        # Logic: Check Shard OR Root (only these two places)
        shard = fname[0:2].lower()
        
        path_sharded = processed_dir / shard / fname
        path_root = processed_dir / fname
        
        if path_sharded.exists() or path_root.exists():
            valid_indices.append(idx)
        else:
            missing_count += 1
            # print(f"❌ Missing strict: {fname}") # Uncomment to see filenames

    # 4. Save
    if missing_count > 0:
        print(f"\n⚠️  Found {missing_count} files that exist loosely but are missing from strict locations.")
        print("   Removing them to fix the DataLoader crash...")
        
        df_clean = df.loc[valid_indices]
        df_clean.to_csv(csv_path, index=False)
        print(f"✅ CSV Rewritten! Size: {original_len} -> {len(df_clean)}")
    else:
        print("\n✅ No strict mismatches found. (If you still crash, delete the CSV manually and re-run prepare_data.sh)")

if __name__ == "__main__":
    clean_dataset_strict()