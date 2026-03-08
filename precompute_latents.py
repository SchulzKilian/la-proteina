import os
import torch
import glob
from tqdm import tqdm
from torch_geometric.transforms import Compose
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.transforms import CoordsToNanometers, CenterStructureTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FORCE_RECOMPUTE = False 

def main():
    # 1. Setup paths
    # Ensure these paths match your environment setup
    data_dir = "/home/ks2218/la-proteina/data/pdb_train/processed"
    out_dir = "/home/ks2218/la-proteina/data/pdb_train/processed_latents"
    ae_path = "/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"
    
    # 2. Load AutoEncoder and set to eval mode
    print(f"Loading AutoEncoder from {ae_path}...")
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    files = glob.glob(os.path.join(data_dir, "**", "*.pt"))
    existing_outputs = set(glob.glob(os.path.join(out_dir, "**", "*.pt")))
    
    print(f"Found {len(files)} source files. Starting computation...")

    with torch.no_grad():
        for f in tqdm(files, desc="Precomputing"):
            out_path = f.replace(data_dir, out_dir)
            
            if out_path in existing_outputs and not FORCE_RECOMPUTE:
                continue

            try:
                # A. Load raw Data object
                data = torch.load(f, map_location='cpu', weights_only=False)
                L = data.coords.shape[0]

                # ASSERT 1: Verify raw PDB structure shape [L, 37 atoms, 3 coords]
                assert data.coords.ndim == 3 and data.coords.shape[1] == 37, \
                    f"[{f}] Unexpected raw shape: {data.coords.shape}. Expected [L, 37, 3]."

                # B. Reorder atoms to OpenFold standard
                data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

                # C. CRITICAL FIX: Manual Centering and Scaling
                # Calculate the centroid based ONLY on valid C-Alpha atoms (index 1)
                ca_idx = 1
                ca_coords = data.coords[:, ca_idx, :]
                ca_mask = data.coord_mask[:, ca_idx].bool()
                
                if not ca_mask.any():
                    print(f"Skipping {f}: No valid CA atoms found.")
                    continue
                    
                ca_centroid = ca_coords[ca_mask].mean(dim=0)
                
                # Apply centering to ALL 37 atoms and scale to Nanometers (0.1 multiplier)
                # This ensures the local frame matches the one used during training
                data.coords = (data.coords - ca_centroid) * 0.1
                
                # Create 'coords_nm' attribute required by the Encoder/Proteina
                data.coords_nm = data.coords.clone()

                # D. Prepare the batch for the Encoder
                batch = data.clone().to('cuda')
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                # Construct mask_dict required by EncoderTransformer
                seq_mask = batch.coord_mask[:, :, ca_idx].bool() 
                batch["mask_dict"] = {
                    "coords": seq_mask.unsqueeze(-1).unsqueeze(-1), 
                    "residue_type": seq_mask
                }
                batch["mask"] = seq_mask

                # E. Run the Encoder
                output_enc = ae.encoder(batch)
                
                # F. Save Statistics (Mean and Log-Scale)
                data.mean = output_enc["mean"][0].cpu()
                data.log_scale = output_enc["log_scale"][0].cpu()
                
                # G. Optimization: Drop 37-atom coords to save disk space
                # We slice both 'coords' and 'coords_nm' to index 1 (CA)
                data.coords = data.coords[:, ca_idx, :] 
                data.coords_nm = data.coords_nm[:, ca_idx, :] 
                data.coord_mask = data.coord_mask[:, ca_idx]
                
                # Final structural check before saving
                assert data.coords.shape == (L, 3), f"[{f}] Final CA-only shape corrupted: {data.coords.shape}."
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[CRITICAL FAILURE] {f}: {e}")
                raise e

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()