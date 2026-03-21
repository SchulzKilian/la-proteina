import torch
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def check_and_delete(file_path):
    """Checks if a torch file is valid; deletes and returns path if corrupt."""
    try:
        # Using weights_only=True is slightly faster if your torch version supports it
        torch.load(file_path, map_location='cpu', weights_only=True)
        return None 
    except Exception:
        try:
            os.remove(file_path)
            return file_path
        except:
            return "ERROR_DELETING"

def main():
    root_dir = '/rds/user/ks2218/hpc-work/processed_latents/**/*.pt'
    
    print("Step 1: Indexing files (this can take a few minutes on RDS)...")
    files = glob.glob(root_dir, recursive=True)
    total_files = len(files)
    
    if total_files == 0:
        print("No files found. Check your path!")
        return

    print(f"Step 2: Validating {total_files} files using 16 cores...")
    
    bad_files_count = 0
    
    # We use tqdm to wrap the executor map
    with ProcessPoolExecutor(max_workers=16) as executor:
        # list() call here forces the generator to evaluate so tqdm can track it
        results = list(tqdm(executor.map(check_and_delete, files), total=total_files, unit="file"))

    # Filter results to count how many were actually deleted
    cleaned = [r for r in results if r is not None]
    
    print(f"\nCleanup Complete.")
    print(f"Total files scanned: {total_files}")
    print(f"Total files removed: {len(cleaned)}")

if __name__ == "__main__":
    main()