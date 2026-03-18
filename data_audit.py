import os
import torch
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Update this to your processed data path
PROCESSED_DIR = "/rds/user/ks2218/hpc-work/processed" 
NUM_WORKERS = cpu_count()

def audit_protein_file(file_path):
    """
    Comprehensive auditor for a single .pt protein file.
    Checks structural dimensions, NaNs, and backbone geometry.
    """
    try:
        # Load the file
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # 1. Check for Required Attributes
        if not hasattr(data, 'coords'):
            return False, "Missing 'coords' attribute"
        
        coords = data.coords # Expected shape [L, Atoms, 3]
        if coords.ndim != 3:
            return False, f"Invalid coords shape: {coords.shape} (expected 3 dims)"

        L_phys = coords.shape[0]  # Number of residues
        N_atoms = coords.shape[1] # Number of atoms per residue

        # 2. Check Structural Consistency
        # These nodal attributes MUST match the sequence length L_phys
        keys_to_check = ['coord_mask', 'residue_type', 'residue_pdb_idx', 'seq_pos', 'bfactor']
        for key in keys_to_check:
            if hasattr(data, key):
                attr = getattr(data, key)
                if attr.shape[0] != L_phys:
                    return False, f"Length mismatch: coords={L_phys}, {key}={attr.shape[0]}"

        # 3. Check for Numerical Corruption
        if torch.isnan(coords).any():
            return False, "NaN values detected in coordinates"

        # 4. Check Backbone Geometry (Atom Order/Scrambling)
        # Standard Order: 0:N, 1:CA. Mean N-CA distance should be ~1.46 Angstroms.
        if N_atoms >= 2:
            n_ca_dist = torch.norm(coords[:, 0, :] - coords[:, 1, :], dim=-1).mean()
            if n_ca_dist < 1.0 or n_ca_dist > 2.0:
                return False, f"Broken backbone geometry: Mean N-CA dist = {n_ca_dist:.4f}A"

        # 5. Check Atom Format
        # Your verify_processed script flags if 37-atom format is already applied
        if N_atoms == 37:
            # This is a warning/info state, not necessarily a 'failure' 
            # unless your pipeline expects raw 
            pass

        return True, "Consistent"

    except Exception as e:
        return False, f"Critical Load Error: {str(e)}"

def main():
    print(f"🔍 Scanning directory: {PROCESSED_DIR}")
    files = glob.glob(os.path.join(PROCESSED_DIR, "**", "*.pt"), recursive=True)
    
    if not files:
        print("❌ No files found. Check your PROCESSED_DIR path.")
        return

    print(f"🚀 Auditing {len(files)} files using {NUM_WORKERS} workers...")
    
    error_log = []
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(audit_protein_file, files), total=len(files)))

    for file_path, (is_ok, msg) in zip(files, results):
        if not is_ok:
            error_log.append(f"{file_path} | {msg}")

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("DATA AUDIT SUMMARY")
    print("="*60)
    print(f"Total Files:     {len(files)}")
    print(f"Passed:          {len(files) - len(error_log)}")
    print(f"Failed/Warning:  {len(error_log)}")
    print("="*60)

    if error_log:
        with open("audit_errors.txt", "w") as f:
            for error in error_log:
                f.write(error + "\n")
        print(f"\n❌ Detected issues in {len(error_log)} files.")
        print("Details saved to: audit_errors.txt")
    else:
        print("\n✅ All files are structurally and geometrically consistent!")

if __name__ == "__main__":
    main()