import pandas as pd
import pathlib
import os
from tqdm import tqdm
import argparse

def check_dataset_integrity(data_dir_name):
    # Setup paths
    project_dir = pathlib.Path.cwd()
    data_dir = project_dir / "data" / data_dir_name
    processed_dir = data_dir / "processed"
    
    # Find the CSV file (usually starts with df_pdb_...)
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV file found in {data_dir}")
        return
    
    # Pick the most likely dataset file (or just the first one)
    csv_path = csv_files[0]
    print(f"📄 Reading Manifest: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"📂 Scanning Directory: {processed_dir}")
    
    missing_files = []
    found_count = 0
    
    # Iterate and check
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        pdb = row["pdb"]
        chain = row["chain"] if "chain" in row and pd.notna(row["chain"]) else "all"
        
        # Construct filename
        if chain == "all":
            fname = f"{pdb}.pt"
        else:
            fname = f"{pdb}_{chain}.pt"
            
        # Sharding logic: first 2 chars
        shard = fname[0:2].lower()
        
        # Check Paths (Sharded vs Root)
        sharded_path = processed_dir / shard / fname
        root_path = processed_dir / fname
        
        if sharded_path.exists():
            found_count += 1
        elif root_path.exists():
            found_count += 1
        else:
            missing_files.append(fname)

    # Report
    print("\n" + "="*40)
    print(f"📊 REPORT FOR: {data_dir_name}")
    print("="*40)
    print(f"✅ Total Entries in CSV: {len(df)}")
    print(f"🆗 Files Found on Disk:  {found_count}")
    print(f"❌ Files Missing:        {len(missing_files)}")
    print("="*40)

    if missing_files:
        print("\nExample Missing Files:")
        for f in missing_files[:10]:
            print(f" - {f}")
        if len(missing_files) > 10:
            print(f" ... and {len(missing_files)-10} more.")
            
        print("\n💡 SUGGESTION: If these files failed to generate, you should")
        print("   delete the CSV file and rerun 'prepare_data.sh' to regenerate")
        print("   it without these missing entries.")

if __name__ == "__main__":
    # Default to the folder name you used in the error logs
    target_folder = "pdb_train" 
    
    # You can also pass args if you have different folder names
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="pdb_train")
    # args = parser.parse_args()
    # target_folder = args.dataset

    check_dataset_integrity(target_folder)