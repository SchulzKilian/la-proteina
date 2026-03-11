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
# CONFIGURATION
# ==============================================================================
FORCE_RECOMPUTE = True 
NUM_WORKERS = 16  
BATCH_SIZE = 1  
DEV = True

if DEV:
    DATA_DIR = "/home/ks2218/data/pdb_train/processed"
    OUT_DIR = "/home/ks2218/data/pdb_train/processed_latents"
    AE_PATH = "/home/ks2218/checkpoints_laproteina/AE1_ucond_512.ckpt"





else:
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
                if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] > L_true:
                    data[key] = val[:L_true]
                    
            # Standardize Atom Order
            data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
            
            # Apply Baking Pipeline
            data = self.baking_pipeline(data)
            
            return data, out_path
        except Exception as e:
            print(f"\n[WORKER FAILURE] {f}: {e}")
            return None, out_path
        
def main():
    assert os.path.exists(DATA_DIR), f"Source data directory missing: {DATA_DIR}"
    assert os.path.exists(AE_PATH), f"AutoEncoder checkpoint missing: {AE_PATH}"
    
    print(f"Loading AutoEncoder from {AE_PATH}...")
    ae = AutoEncoder.load_from_checkpoint(AE_PATH).cuda().eval()
    
    assert hasattr(ae, 'latent_dim'), "AutoEncoder missing 'latent_dim' property."
    print(f"AutoEncoder verified. Latent dimension: {ae.latent_dim}")
    
    baking_pipeline = Compose([
        CoordsToNanometers(),
        CenterStructureTransform(),
        ChainBreakCountingTransform(),
    ])
    
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.pt"), recursive=True)
    print(f"Found {len(files)} source files. Starting computation...")

    dataset = ProteinDataset(files, DATA_DIR, OUT_DIR, baking_pipeline)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda x: x[0] 
    )

    with torch.no_grad():
        for data, out_path in tqdm(loader, desc="Precomputing"):
            if data is None:
                continue

            try:
                L = data.coord_mask.shape[0]
                assert L > 0, f"[{out_path}] Empty protein detected (L=0)"
                
                # Prune any remaining metadata that is longer than L
                for key in ['residue_pdb_idx', 'seq_pos', 'residue_type', 'bfactor', 'aatype']:
                    if hasattr(data, key):
                        val = getattr(data, key)
                        if hasattr(val, 'shape') and val.shape[0] > L:
                            setattr(data, key, val[:L])

                # D. Build an ISOLATED dictionary for the Encoder (prevents mutating 'data')
                encoder_input = {}
                for k in data.keys():
                    val = data[k]
                    if isinstance(val, torch.Tensor):
                        # Move to GPU and add batch dimension [1, ...]
                        encoder_input[k] = val.to('cuda', non_blocking=True).unsqueeze(0)
                    else:
                        encoder_input[k] = val

                na_idx = 0
                seq_mask = encoder_input["coord_mask"][:, :, na_idx].bool() 

                encoder_input["mask_dict"] = {
                    "coords": seq_mask.unsqueeze(-1).unsqueeze(-1), 
                    "residue_type": seq_mask
                }
                encoder_input["mask"] = seq_mask

                # E. Run Encoder to get latents
                output_enc = ae.encoder(encoder_input)
                
                # Extract first item from batch
                mean = output_enc["mean"][0] 
                log_scale = output_enc["log_scale"][0]
                
                # --- SHAPE ALIGNMENT FIX ---
                if mean.shape[1] == L and mean.shape[0] == ae.latent_dim:
                    mean = mean.transpose(0, 1)
                    log_scale = log_scale.transpose(0, 1)

                # --- CRITICAL SEQUENCE LENGTH ASSERTION ---
                assert mean.shape[0] == L, \
                    f"[{out_path}] CRITICAL: Encoder output length ({mean.shape[0]}) mismatch with mask length ({L})"
                assert mean.shape[1] == ae.latent_dim, \
                    f"[{out_path}] Latent dim mismatch: expected {ae.latent_dim}, got {mean.shape[1]}"

                # Assign latents to the CLEAN CPU data object
                data.mean = mean.cpu()
                data.log_scale = log_scale.cpu()
                
                # F. Optimization: Delete unscaled Angstrom coords
                if hasattr(data, 'coords'):
                    del data.coords
                
                # Final check: Since data was never unsqueezed, this will be strictly L == L
                assert data.mean.shape[0] == data.coord_mask.shape[0], \
                    f"Final check failed: mean {data.mean.shape} != mask {data.coord_mask.shape}"
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[GPU FAILURE] {out_path}: {e}")

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()