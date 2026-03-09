import os
import torch
import glob
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
FORCE_RECOMPUTE = False 
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
        # Resolve path relative to data_dir for sharded output
        rel_path = os.path.relpath(f, self.data_dir)
        out_path = os.path.join(self.out_dir, rel_path)
        
        if os.path.exists(out_path) and not FORCE_RECOMPUTE:
            return None, out_path

        try:
            # A. Load raw Data object
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # --- ASSERTIONS: INITIAL DATA STATE ---
            assert hasattr(data, 'coords'), f"[{f}] Missing 'coords' attribute."
            assert hasattr(data, 'coord_mask'), f"[{f}] Missing 'coord_mask' attribute."
            assert hasattr(data, 'residue_type'), f"[{f}] Missing 'residue_type' attribute."
            
            L_init = data.coord_mask.shape[0]
            assert data.coords.shape[0] == L_init, f"[{f}] Coords length {data.coords.shape[0]} != Mask length {L_init}"
            assert data.residue_type.shape[0] == L_init, f"[{f}] Residue type length {data.residue_type.shape[0]} != Mask length {L_init}"

            # B. Standardize Atom Order (Preserving original logic)
            # Ensure PDB_TO_OPENFOLD indices fit the current atom count (e.g., if cif has >37 atoms)
            num_atoms = data.coords.shape[1]
            assert PDB_TO_OPENFOLD_INDEX_TENSOR.max() < num_atoms, \
                f"[{f}] PDB_TO_OPENFOLD contains index {PDB_TO_OPENFOLD_INDEX_TENSOR.max()} but coords only has {num_atoms} atoms."

            data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
            
            # --- ASSERTION: POST-SLICING ATOM DIM ---
            assert data.coords.shape[1] == 37, f"[{f}] Standardization failed: Expected 37 atoms, got {data.coords.shape[1]}"

            # C. Apply Baking Pipeline (Scaling/Centering/Metadata)
            data = self.baking_pipeline(data)
            
            # --- ASSERTIONS: POST-TRANSFORM INTEGRITY ---
            assert hasattr(data, 'coords_nm'), f"[{f}] Baking failed: 'coords_nm' attribute missing."
            assert data.coords_nm.shape[0] == L_init, f"[{f}] Transformation changed sequence length L from {L_init} to {data.coords_nm.shape[0]}"
            
            return data, out_path
        except Exception as e:
            print(f"\n[WORKER FAILURE] {f}: {e}")
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
                # D. Prepare GPU batch for Encoder
                batch = data.to('cuda', non_blocking=True)
                
                # Check sequence length before unsqueezing for batching
                L = batch.coord_mask.shape[0]
                assert L > 0, f"[{out_path}] Empty protein detected (L=0)"

                # Unsqueeze relevant keys for model expectation
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                # CA index and Mask setup
                ca_idx = 1
                assert batch.coord_mask.shape[2] == 37, "Mask does not have 37 atom channels."
                
                # Verify CA mask aligns with sequence length L
                seq_mask = batch.coord_mask[:, :, ca_idx].bool() 
                assert seq_mask.shape[1] == L, f"[{out_path}] seq_mask length {seq_mask.shape[1]} != coord_mask L {L}"

                batch["mask_dict"] = {
                    "coords": seq_mask.unsqueeze(-1).unsqueeze(-1), 
                    "residue_type": seq_mask
                }
                batch["mask"] = seq_mask

                # E. Run Encoder to get latents
                output_enc = ae.encoder(batch)
                
                # Extract results and verify shapes
                mean = output_enc["mean"][0] # Shape [L, latent_dim]
                log_scale = output_enc["log_scale"][0]
                
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
                
                # FINAL VALIDATION BEFORE SAVE
                # This ensures the check_latents.py script will never find a mismatch again.
                assert data.mean.shape[0] == data.coord_mask.shape[0], "Final check: mean and mask length diverged before save."
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[GPU FAILURE] {out_path}: {e}")

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()