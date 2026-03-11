import os
import torch
import glob
import traceback
from tqdm import tqdm
from torch_geometric.transforms import Compose
from torch.utils.data import Dataset, DataLoader

# Official project imports
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.transforms import (
    CoordsToNanometers, 
    CenterStructureTransform, 
    ChainBreakCountingTransform
)
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

# ==============================================================================
# CONFIGURATION - USING ABSOLUTE RDS PATHS TO AVOID HOME QUOTA ISSUES
# ==============================================================================
FORCE_RECOMPUTE = True 
NUM_WORKERS = 16  # Matches your --cpus-per-task
BATCH_SIZE = 1   

# Using absolute paths as per troubleshooting - symlinks can trigger home quota during mkdir
DATA_DIR = "/rds/user/ks2218/hpc-work/processed"
OUT_DIR = "/rds/user/ks2218/hpc-work/processed_latents"
AE_PATH = "/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"

class ProteinDataset(Dataset):
    """Dataset to handle CPU-side loading and baking in parallel workers."""
    def __init__(self, files, data_dir, out_dir, baking_pipeline):
        self.files = files
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.baking_pipeline = baking_pipeline

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        rel_path = os.path.relpath(f, self.data_dir)
        out_path = os.path.join(self.out_dir, rel_path)
        
        if os.path.exists(out_path) and not FORCE_RECOMPUTE:
            return None, out_path

        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # Determine true length L from the mask
            L_true = data.coord_mask.shape[0]
            
            for key in data.keys():
                val = data[key]
                # Check for tensors and safely get ndim
                if torch.is_tensor(val):
                    # DEBUG PRINT: Uncomment if needed, but might be too spammy
                    # print(f"[WORKER DEBUG] {rel_path} Key: {key}, Type: {type(val)}, ndim: {val.ndim}")
                    if val.ndim > 0 and val.shape[0] > L_true:
                        data[key] = val[:L_true]
                        
            # B. Standardize Atom Order (Preserving original logic)
            data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
            
            # C. Apply Baking Pipeline (Scaling/Centering/Metadata)
            data = self.baking_pipeline(data)
            
            return data, out_path
        except Exception as e:
            # THIS WILL PRINT THE EXACT LINE OF THE WORKER CRASH
            err_msg = traceback.format_exc()
            print(f"\n[WORKER FAILURE] {f}:\n{err_msg}\n")
            return None, out_path
        
def main():
    # 1. Setup paths
    assert os.path.exists(DATA_DIR), f"Source data directory missing: {DATA_DIR}"
    assert os.path.exists(AE_PATH), f"AutoEncoder checkpoint missing: {AE_PATH}"
    
    # 2. Load AutoEncoder and set to eval mode
    print(f"Loading AutoEncoder from {AE_PATH}...")
    ae = AutoEncoder.load_from_checkpoint(AE_PATH).cuda().eval()
    
    # Assert AE properties
    assert hasattr(ae, 'latent_dim'), "AutoEncoder missing 'latent_dim' property."
    print(f"AutoEncoder verified. Latent dimension: {ae.latent_dim}")
    
    # 3. Define the DETERMINISTIC "Baking" Pipeline
    baking_pipeline = Compose([
        CoordsToNanometers(),
        CenterStructureTransform(),
        ChainBreakCountingTransform(),
    ])
    
    # Recursively find source files
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.pt"), recursive=True)
    print(f"Found {len(files)} source files. Starting computation...")

    # 4. Setup DataLoader for parallel CPU processing
    dataset = ProteinDataset(files, DATA_DIR, OUT_DIR, baking_pipeline)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda x: x[0] # Single (data, path) tuple
    )

    with torch.no_grad():
        for data, out_path in tqdm(loader, desc="Precomputing"):
            if data is None:
                continue

            try:
                L_true = data.coord_mask.shape[0]
                
                for key in ['residue_pdb_idx', 'seq_pos', 'residue_type', 'bfactor', 'aatype']:
                    if hasattr(data, key):
                        val = getattr(data, key)
                        if hasattr(val, 'shape') and val.shape[0] > L_true:
                            setattr(data, key, val[:L_true])

                # D. Prepare GPU batch for Encoder
                batch = data.to('cuda', non_blocking=True)
                
                L = batch.coord_mask.shape[0]
                assert L > 0, f"[{out_path}] Empty protein detected (L=0)"

                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                na_idx = 0
                seq_mask = batch.coord_mask[:, :, na_idx].bool() 

                batch["mask_dict"] = {
                    "coords": seq_mask.unsqueeze(-1).unsqueeze(-1), 
                    "residue_type": seq_mask
                }
                batch["mask"] = seq_mask

                # E. Run Encoder to get latents
                output_enc = ae.encoder(batch)
                
                mean = output_enc["mean"][0] 
                log_scale = output_enc["log_scale"][0]
                
                print(f"\n[DEBUG] {os.path.basename(out_path)} -> True L: {L}, AE Latent Dim: {ae.latent_dim}")
                print(f"[DEBUG] {os.path.basename(out_path)} -> RAW mean shape: {mean.shape}")
                
                # --- SHAPE ALIGNMENT FIX ---
                if mean.shape[1] == L and mean.shape[0] == ae.latent_dim:
                    mean = mean.transpose(0, 1)
                    log_scale = log_scale.transpose(0, 1)
                    print(f"[DEBUG] {os.path.basename(out_path)} -> TRANSPOSED mean shape: {mean.shape}")
                
                print(f"[DEBUG] {os.path.basename(out_path)} -> Final assertion check: mean.shape[0] ({mean.shape[0]}) == L ({L})")

                # --- CRITICAL SEQUENCE LENGTH ASSERTION ---
                assert mean.shape[0] == L, \
                    f"[{out_path}] CRITICAL: Encoder output length ({mean.shape[0]}) mismatch with mask length ({L})"
                assert mean.shape[1] == ae.latent_dim, \
                    f"[{out_path}] Latent dim mismatch: expected {ae.latent_dim}, got {mean.shape[1]}"

                data.mean = mean.cpu()
                data.log_scale = log_scale.cpu()
                
                # F. Optimization: Delete unscaled Angstrom coords
                if hasattr(data, 'coords'):
                    del data.coords
                
                assert data.mean.shape[0] == data.coord_mask.shape[0], f"Final check failed: data.mean.shape={data.mean.shape}, data.coord_mask.shape={data.coord_mask.shape}"
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                # THIS WILL PRINT THE EXACT LINE OF THE GPU CRASH
                err_msg = traceback.format_exc()
                print(f"\n[GPU FAILURE] {out_path}:\n{err_msg}\n")

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()