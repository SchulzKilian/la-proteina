import os
import torch
import glob
from tqdm import tqdm
from multiprocessing import Pool

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path where your current 40GB of latents are stored
SOURCE_DIR = "/rds/user/ks2218/hpc-work/processed_latents"
# Path where the new, lean latents will be saved
OUTPUT_DIR = "/rds/user/ks2218/hpc-work/processed_latents_lean"
NUM_WORKERS = 32  # Use as many CPU cores as available

def prune_protein_file(paths):
    src_path, out_path = paths
    try:
        # Load from disk (CPU only)
        data = torch.load(src_path, map_location='cpu', weights_only=False)
        
        # 1. Strip coords_nm to ONLY the CA atom (index 1)
        # We keep it as [N, 1, 3] to remain compatible with logic expecting 3D tensors
        if hasattr(data, 'coords_nm'):
            if data.coords_nm.ndim == 3: # [N, 37, 3]
                data.coords_nm = data.coords_nm[:, 1, :]
            # If it's already [N, 3], we leave it alone or keep as is.
            
        # 2. Delete heavy/redundant metadata
        # aatype is redundant to residue_type; bfactor/pdb_idx are unused in training.
        keys_to_delete = ['bfactor', 'aatype', 'residue_pdb_idx', 'coords', 'residue_id']
        for key in keys_to_delete:
            if hasattr(data, key):
                delattr(data, key)
            if key in data: # Handle case if data is a dict
                del data[key]

        # 3. Save the optimized file
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(data, out_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.pt"), recursive=True)
    print(f"Found {len(files)} files to prune.")
    
    tasks = []
    for f in files:
        rel_path = os.path.relpath(f, SOURCE_DIR)
        out_f = os.path.join(OUTPUT_DIR, rel_path)
        tasks.append((f, out_f))

    print(f"Starting CPU pruning with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(prune_protein_file, tasks), total=len(tasks)))
    
    success_count = sum(results)
    print(f"Finished! Successfully pruned {success_count}/{len(files)} files.")
    print(f"New dataset is in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()