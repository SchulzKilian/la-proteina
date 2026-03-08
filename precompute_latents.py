import os
import torch
import glob
from tqdm import tqdm
from torch_geometric.transforms import Compose

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

def main():
    # 1. Setup paths
    data_dir = "/home/ks2218/la-proteina/data/pdb_train/processed"
    out_dir = "/home/ks2218/la-proteina/data/pdb_train/processed_latents"
    ae_path = "/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"
    
    # 2. Load AutoEncoder and set to eval mode
    print(f"Loading AutoEncoder from {ae_path}...")
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    # 3. Define the DETERMINISTIC "Baking" Pipeline
    # These transforms are applied ONCE here so they don't run every epoch.
    baking_pipeline = Compose([
        CoordsToNanometers(),           # Converts Angstrom -> NM
        CenterStructureTransform(),     # Centers based on valid CA atoms
        ChainBreakCountingTransform(),  # Generates metadata used by Proteina
    ])
    
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
                
                # B. Standardize Atom Order (Must happen before transforms)
                data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

                # C. Apply Baking Pipeline (Scaling/Centering/Metadata)
                data = baking_pipeline(data)

                # D. Prepare GPU batch for Encoder
                batch = data.clone().to('cuda')
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                # CA index is 1 in OpenFold standard
                ca_idx = 1
                seq_mask = batch.coord_mask[:, :, ca_idx].bool() 
                
                batch["mask_dict"] = {
                    "coords": seq_mask.unsqueeze(-1).unsqueeze(-1), 
                    "residue_type": seq_mask
                }
                batch["mask"] = seq_mask

                # E. Run Encoder to get latents
                output_enc = ae.encoder(batch)
                data.mean = output_enc["mean"][0].cpu()
                data.log_scale = output_enc["log_scale"][0].cpu()
                
                # F. Optimization: Delete unscaled Angstrom coords
                # We KEEP 'coords_nm' as [L, 37, 3] for index-safety in Proteina
                if hasattr(data, 'coords'):
                    del data.coords
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[FAILURE] {f}: {e}")
                raise e

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()