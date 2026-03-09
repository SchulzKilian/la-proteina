import os
import torch
import glob
from tqdm import tqdm

def check_latent_consistency():
    # Define the directory where your precomputed latents are stored
    # Based on your logs, this is the absolute path on your RDS
    latent_dir = "/rds/user/ks2218/hpc-work/processed_latents"
    
    print(f"Scanning directory: {latent_dir}")
    
    # Find all .pt files recursively
    files = glob.glob(os.path.join(latent_dir, "**", "*.pt"), recursive=True)
    total_files = len(files)
    
    if total_files == 0:
        print("No .pt files found. Please check the path.")
        return

    mismatches = []
    healthy_count = 0
    error_count = 0

    print(f"Checking {total_files} files for shape consistency...")

    for f in tqdm(files):
        try:
            # Load only necessary parts to save memory
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # 1. Get Latent Length (L_latent)
            # This is dimension 0 of the mean tensor
            l_latent = data.mean.shape[0]
            
            # 2. Get Mask Length (L_mask)
            # This is dimension 0 of the coord_mask
            l_mask = data.coord_mask.shape[0]
            
            if l_latent != l_mask:
                mismatches.append({
                    "file": os.path.basename(f),
                    "latent_len": l_latent,
                    "mask_len": l_mask
                })
            else:
                healthy_count += 1
                
        except Exception as e:
            print(f"\n[ERROR] Could not load {f}: {e}")
            error_count += 1

    # Print Summary Report
    print("\n" + "="*50)
    print("CONSISTENCY CHECK REPORT")
    print("="*50)
    print(f"Total Files Scanned: {total_files}")
    print(f"Healthy Files:       {healthy_count}")
    print(f"Inconsistent Files:  {len(mismatches)}")
    print(f"Corrupted/Unloadable: {error_count}")
    print("="*50)

    if mismatches:
        print("\nTOP 20 MISMATCHES (Latent Length vs Mask Length):")
        for m in mismatches[:20]:
            print(f"- {m['file']}: Latent={m['latent_len']}, Mask={m['mask_len']}")
        
        if len(mismatches) > 20:
            print(f"... and {len(mismatches) - 20} more.")
            
        # Optional: Save list of bad files to a text file for deletion
        with open("corrupted_latents.txt", "w") as fout:
            for m in mismatches:
                fout.write(f"{m['file']}\n")
        print("\nList of inconsistent filenames saved to 'corrupted_latents.txt'")
    else:
        print("\n✅ All scanned files are consistent!")

if __name__ == "__main__":
    check_latent_consistency()