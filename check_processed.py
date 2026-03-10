import os
import torch
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch_geometric.data import Data

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Update this to point to your absolute processed data path on RDS
PROCESSED_DIR = "/rds/user/ks2218/hpc-work/processed"
NUM_WORKERS = cpu_count()  # Adjust as needed for your login node/interactive session

def check_file_consistency(file_path):
    """
    Worker function to check the internal dimensions of a single PyG Data object.
    Returns: (bool, str, dict) -> (is_consistent, error_msg, metadata)
    """
    try:
        # Load the file to CPU
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            data = Data(**data)

        # 1. Establish the "Ground Truth" Sequence Length (L)
        # Coordinates and their mask are the physical source of truth.
        if not hasattr(data, 'coords'):
            return False, "Missing 'coords' attribute", None
        
        L_phys = data.coords.shape[0]
        
        # 2. Define nodal attributes that MUST match L_phys
        # These are standard keys found in your process_single_pdb_file logic
        keys_to_check = [
            'coord_mask', 
            'residue_type', 
            'residue_pdb_idx', 
            'seq_pos', 
            'bfactor'
        ]
        
        mismatches = {}
        for key in keys_to_check:
            if hasattr(data, key):
                attr = getattr(data, key)
                if attr.shape[0] != L_phys:
                    mismatches[key] = attr.shape[0]
        
        if mismatches:
            msg = f"L_phys={L_phys} | Mismatches: " + ", ".join([f"{k}:{v}" for k, v in mismatches.items()])
            return False, msg, {"file": os.path.basename(file_path), "path": file_path}
            
        return True, "Consistent", None

    except Exception as e:
        return False, f"Load error: {str(e)}", {"file": os.path.basename(file_path), "path": file_path}

def main():
    print(f"Scanning directory: {PROCESSED_DIR}")
    
    # Recursively find all .pt files
    files = glob.glob(os.path.join(PROCESSED_DIR, "**", "*.pt"), recursive=True)
    total_files = len(files)
    
    if total_files == 0:
        print("No files found. Check your PROCESSED_DIR path.")
        return

    print(f"Auditing {total_files} files using {NUM_WORKERS} workers...")

    inconsistent_files = []
    
    # Use Multiprocessing Pool
    with Pool(processes=NUM_WORKERS) as pool:
        # Use imap_unordered for efficiency and tqdm for progress tracking
        results = list(tqdm(pool.imap_unordered(check_file_consistency, files), total=total_files))

    # Analyze Results
    for (is_consistent, msg, meta), file_path in zip(results, files):
        if not is_consistent:
            inconsistent_files.append(f"{file_path} -> {msg}")

    # Summary Report
    print("\n" + "="*60)
    print("CONSISTENCY AUDIT REPORT")
    print("="*60)
    print(f"Total Files Scanned: {total_files}")
    print(f"Inconsistent Files:  {len(inconsistent_files)}")
    print("="*60)

    if inconsistent_files:
        log_name = "processed_consistency_errors.txt"
        with open(log_name, "w") as f:
            for item in inconsistent_files:
                f.write(item + "\n")
        print(f"\nDetailed errors saved to: {log_name}")
        print("Recommended Action: Delete these files or fix the Graphein processing logic.")
    else:
        print("\n✅ All scanned files are structurally consistent!")

if __name__ == "__main__":
    main()