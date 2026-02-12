import pandas as pd
import pathlib
import os
from tqdm import tqdm

# CONFIG
DATASET_NAME = "pdb_train"  # Change this if your folder is named differently

def clean_dataset():
    data_dir = pathlib.Path.cwd() / "data" / DATASET_NAME
    processed_dir = data_dir / "processed"
    
    # 1. Find CSV
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV found!")
        return
    csv_path = csv_files[0]
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    original_len = len(df)
    
    # 2. Identify missing files
    valid_indices = []
    missing_count = 0
    
    print("Checking file existence...")
    # Pre-scan directory to make it instant
    existing_files = set()
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.endswith(".pt"):
                existing_files.add(f)
                
    # Filter
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb = row["pdb"]
        chain = row["chain"] if "chain" in row and pd.notna(row["chain"]) else "all"
        fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
        
        if fname in existing_files:
            valid_indices.append(idx)
        else:
            missing_count += 1
            # print(f"Removing missing: {fname}")

    # 3. Save cleaned CSV
    if missing_count > 0:
        df_clean = df.loc[valid_indices]
        df_clean.to_csv(csv_path, index=False)
        print(f"✅ Fixed! Removed {missing_count} missing files.")
        print(f"📉 Size: {original_len} -> {len(df_clean)}")
    else:
        print("✨ No missing files found. CSV is already clean.")

if __name__ == "__main__":
    clean_dataset()