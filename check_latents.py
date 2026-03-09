import os
import torch
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Based on your symlinks and quota output
LATENT_DIR = "/rds/user/ks2218/hpc-work/processed_latents"
NUM_THREADS = 16  # Adjust based on your login node/interactive session capabilities

def check_single_file(file_path):
    """
    Worker function to check a single .pt file.
    Returns: (status, info)
    Status: 'healthy', 'mismatch', or 'error'
    """
    try:
        # Load metadata only. Note: weights_only=False is used in your project
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Check shapes as defined in your PDBDataset logic
        l_latent = data.mean.shape[0]
        l_mask = data.coord_mask.shape[0]
        
        if l_latent != l_mask:
            return 'mismatch', {
                "file": os.path.basename(file_path),
                "latent_len": l_latent,
                "mask_len": l_mask
            }
        return 'healthy', None

    except Exception as e:
        return 'error', {"file": os.path.basename(file_path), "error": str(e)}

def main():
    print(f"Scanning directory: {LATENT_DIR}")
    
    # Recursively find all .pt files
    files = glob.glob(os.path.join(LATENT_DIR, "**", "*.pt"), recursive=True)
    total_files = len(files)
    
    if total_files == 0:
        print("No .pt files found. Please check the path.")
        return

    print(f"Starting multithreaded check for {total_files} files using {NUM_THREADS} threads...")

    mismatches = []
    errors = []
    healthy_count = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(check_single_file, f): f for f in files}
        
        # Use tqdm to monitor progress as threads complete
        for future in tqdm(as_completed(futures), total=total_files, desc="Checking Latents"):
            status, info = future.result()
            
            if status == 'healthy':
                healthy_count += 1
            elif status == 'mismatch':
                mismatches.append(info)
            elif status == 'error':
                errors.append(info)

    # Print Summary Report
    print("\n" + "="*50)
    print("CONSISTENCY CHECK REPORT")
    print("="*50)
    print(f"Total Files Scanned: {total_files}")
    print(f"Healthy Files:       {healthy_count}")
    print(f"Inconsistent Files:  {len(mismatches)}")
    print(f"Errors/Unloadable:   {len(errors)}")
    print("="*50)

    if mismatches:
        print(f"\nFound {len(mismatches)} shape mismatches.")
        with open("corrupted_latents.txt", "w") as fout:
            for m in mismatches:
                fout.write(f"{m['file']}\n")
        print("List of inconsistent filenames saved to 'corrupted_latents.txt'")
    
    if errors:
        print(f"Encountered {len(errors)} errors during loading (check 'loading_errors.txt').")
        with open("loading_errors.txt", "w") as ferr:
            for e in errors:
                ferr.write(f"{e['file']}: {e['error']}\n")

if __name__ == "__main__":
    main()