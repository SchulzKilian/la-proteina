import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

def check_dataset_fast(target_folder="pdb_train"):
    # 1. Setup Paths
    project_root = Path.cwd()
    data_dir = project_root / "data" / target_folder
    processed_dir = data_dir / "processed"

    # 2. Load the CSV (Manifest)
    csv_files = list(data_dir.glob("df_pdb_f1*.csv"))
    if not csv_files:
        print(f"❌ No CSV file found in {data_dir}")
        return
    
    csv_path = csv_files[0]
    print(f"📄 Reading Manifest: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_count = len(df)

    # 3. RAM LOAD: Scan disk ONCE
    print(f"🚀 Scanning {processed_dir} (loading into RAM)...")
    files_on_disk = set()
    
    # os.walk is efficient; it reads directory entries in blocks
    if processed_dir.exists():
        for root, _, files in os.walk(processed_dir):
            for f in files:
                if f.endswith(".pt"):
                    files_on_disk.add(f)
    else:
        print(f"❌ Processed directory does not exist: {processed_dir}")
        return

    print(f"💾 Files loaded from disk: {len(files_on_disk)}")

    # 4. In-Memory Comparison (Instant)
    print("⚡ comparing...")
    missing_files = []
    found_count = 0

    # We iterate the dataframe just to construct the expected filenames
    # logic matches pdb_data.py
    for _, row in df.iterrows():
        pdb = row["pdb"]
        chain = row["chain"] if "chain" in row and pd.notna(row["chain"]) else "all"
        
        if chain == "all":
            fname = f"{pdb}.pt"
        else:
            fname = f"{pdb}_{chain}.pt"
        
        # O(1) lookup in the set
        if fname in files_on_disk:
            found_count += 1
        else:
            missing_files.append(fname)

    # 5. Report
    print("\n" + "="*40)
    print(f"📊 FAST REPORT FOR: {target_folder}")
    print("="*40)
    print(f"✅ Expected (CSV):   {expected_count}")
    print(f"sz Found (Disk):     {found_count}")
    print(f"❌ Missing:          {len(missing_files)}")
    print("="*40)

    if missing_files:
        print("\nFirst 10 missing files:")
        for f in missing_files[:10]:
            print(f" - {f}")
        
        # Optional: Write missing to file for debugging
        with open("missing_files_list.txt", "w") as f:
            for item in missing_files:
                f.write(f"{item}\n")
        print("\n(Full list of missing files written to 'missing_files_list.txt')")

if __name__ == "__main__":
    # Allow command line arg for folder name, default to 'pdb_train'
    folder = sys.argv[1] if len(sys.argv) > 1 else "pdb_train"
    check_dataset_fast(folder)