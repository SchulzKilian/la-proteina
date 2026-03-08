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
NUM_WORKERS = 16  # Matches the --cpus-per-task in your .sh script
BATCH_SIZE = 1   # Keeping logic at 1 protein at a time as per original

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
        out_path = f.replace(self.data_dir, self.out_dir)
        
        # Skip logic moved here to avoid loading if not needed
        if os.path.exists(out_path) and not FORCE_RECOMPUTE:
            return None, out_path

        try:
            # A. Load raw Data object
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # B. Standardize Atom Order (Preserving original logic)
            data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

            # C. Apply Baking Pipeline (Scaling/Centering/Metadata)
            data = self.baking_pipeline(data)
            
            return data, out_path
        except Exception as e:
            print(f"\n[WORKER FAILURE] {f}: {e}")
            return None, out_path

def main():
    # 1. Setup paths
    data_dir = "/home/ks2218/la-proteina/data/pdb_train/processed"
    out_dir = "/home/ks2218/la-proteina/data/pdb_train/processed_latents"
    ae_path = "/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"
    
    # 2. Load AutoEncoder and set to eval mode
    print(f"Loading AutoEncoder from {ae_path}...")
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    # 3. Define the DETERMINISTIC "Baking" Pipeline
    baking_pipeline = Compose([
        CoordsToNanometers(),
        CenterStructureTransform(),
        ChainBreakCountingTransform(),
    ])
    
    files = glob.glob(os.path.join(data_dir, "**", "*.pt"))
    print(f"Found {len(files)} source files. Starting computation...")

    # 4. Setup DataLoader with 16 workers for parallel CPU processing
    dataset = ProteinDataset(files, data_dir, out_dir, baking_pipeline)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Speeds up CPU-to-GPU transfer
        collate_fn=lambda x: x[0] # Returns the single (data, path) tuple
    )

    with torch.no_grad():
        for data, out_path in tqdm(loader, desc="Precomputing"):
            if data is None:
                continue

            try:
                # D. Prepare GPU batch for Encoder
                # Using non_blocking=True to overlap transfer with compute
                batch = data.to('cuda', non_blocking=True)
                
                # Maintain original unsqueeze logic for the model
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                # CA index and Mask setup (Unchanged logic)
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
                if hasattr(data, 'coords'):
                    del data.coords
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[GPU FAILURE] {out_path}: {e}")

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()