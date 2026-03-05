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
FORCE_RECOMPUTE = True 

def main():
    # 1. Setup paths
    data_dir = "/home/ks2218/la-proteina/data/pdb_train/processed"
    out_dir = "/home/ks2218/la-proteina/data/pdb_train/processed_latents"
    ae_path = "/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"
    
    # 2. Load AutoEncoder and set to eval mode
    print(f"Loading AutoEncoder from {ae_path}...")
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    # Define the exact transform pipeline used in standard training
    transform = Compose([
        CoordsToNanometers(),
        CenterStructureTransform() 
    ])
    
    files = glob.glob(os.path.join(data_dir, "**", "*.pt"))
    existing_outputs = set(glob.glob(os.path.join(out_dir, "**", "*.pt")))
    
    print(f"Found {len(files)} source files. Starting computation...")

    with torch.no_grad():
        print(files)
        for f in tqdm(files, desc="Precomputing"):
            out_path = f.replace(data_dir, out_dir)
            # In precompute_latents.py inside the for loop:
            out_path = f.replace(data_dir, out_dir)
            print(f"DEBUG: Input {f} -> Output {out_path}", flush=True) # Check this in your log!
            if out_path in existing_outputs and not FORCE_RECOMPUTE:
                continue

            try:
                # A. Load raw Data object
                data = torch.load(f, map_location='cpu', weights_only=False)
                L = data.coords.shape[0]

                # ASSERT 1: Verify raw PDB structure shape [L, 37 atoms, 3 coords]
                assert data.coords.ndim == 3 and data.coords.shape[1] == 37, \
                    f"[{f}] Unexpected raw shape: {data.coords.shape}. Expected [L, 37, 3]."

                # B. CRITICAL: Reorder atoms to OpenFold standard
                data.coords = data.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

                # ASSERT 2: Verify atom slicing didn't change sequence length or dimensionality
                assert data.coords.shape == (L, 37, 3), \
                    f"[{f}] Reordering corrupted shape: {data.coords.shape}."

                # C. Apply centering and unit conversion (Angstrom -> nm)
                data = transform(data)
                
                # ASSERT 3: Verify transform created nm coordinates
                assert hasattr(data, 'coords_nm'), f"[{f}] Transform failed to create 'coords_nm'."
                # Basic sanity check: nm values for proteins are rarely > 100
                assert data.coords_nm.abs().max() < 500, f"[{f}] Coordinates look unscaled: {data.coords_nm.max()} nm."

                # D. Prepare the batch for the Encoder
                batch = data.clone().to('cuda')
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].unsqueeze(0)

                # ASSERT 4: Verify batch dimensions [Batch=1, L, ...]
                assert batch.coords.ndim == 4 and batch.coords.shape[0] == 1, \
                    f"[{f}] Batching failed: {batch.coords.shape}."

                # Construct mask_dict required by EncoderTransformer
                seq_mask = batch.coord_mask[:, :, 1].bool() 
                batch["mask_dict"] = {"coords": seq_mask.unsqueeze(-1).unsqueeze(-1)}
                batch["mask"] = seq_mask

                # E. Run the Encoder
                output_enc = ae.encoder(batch)
                
                # ASSERT 5: Verify Latent Dimensions
                # Check that we got mean/log_scale and they match the AE's latent dim (e.g. 512 or 8)
                assert "mean" in output_enc and "log_scale" in output_enc, f"[{f}] Encoder returned no stats."
                assert output_enc["mean"].shape[-1] == ae.latent_dim, \
                    f"[{f}] Latent dim mismatch: {output_enc['mean'].shape[-1]} != {ae.latent_dim}."
                assert output_enc["mean"].shape[1] == L, \
                    f"[{f}] Sequence length mismatch in latent: {output_enc['mean'].shape[1]} != {L}."

                # F. Save Statistics (Mean and Log-Scale)
                data.mean = output_enc["mean"][0].cpu()
                data.log_scale = output_enc["log_scale"][0].cpu()
                
                # G. Optimization: Drop 37-atom coords to save disk space
                # Proteina flow matching only requires C-alpha (index 1)
                data.coords = data.coords[:, 1, :] 
                data.coord_mask = data.coord_mask[:, 1]
                
                # ASSERT 6: Final structural check before saving
                assert data.coords.shape == (L, 3), f"[{f}] Final CA-only shape corrupted: {data.coords.shape}."
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                print(f"\n[CRITICAL FAILURE] {f}: {e}")
                # We raise here to "fail hard" as requested
                raise e

    print(f"\n--- Precomputation Complete ---")

if __name__ == "__main__":
    main()