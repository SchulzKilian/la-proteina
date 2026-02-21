import os
import torch
from glob import glob
from tqdm import tqdm
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.transforms import CoordsToNanometers

def main():
    # 1. Setup paths and load AE
    data_dir = "data/pdb_train/processed"
    out_dir = "data/pdb_train/processed_latents"
    
    ae_path = "./checkpoints_laproteina/AE1_ucond_512.ckpt"
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    files = glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    transform = CoordsToNanometers()
    
    # --- MODIFICATION 2: Existing File Checker ---
    # We load all existing output paths into a set for near-instant lookup
    existing_outputs = set(glob(os.path.join(out_dir, "**", "*.pt"), recursive=True))
    
    failed_count = 0
    skipped_count = 0

    with torch.no_grad():
        for f in tqdm(files, desc="Precomputing"):
            # Determine the target output path before processing
            out_path = f.replace(data_dir, out_dir)
            
            if out_path in existing_outputs:
                skipped_count += 1
                continue

            # --- MODIFICATION 1: Try/Except Loop ---
            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                
                d_norm = transform(data.clone())
                
                coords = data.coords.unsqueeze(0).cuda()
                coords_nm = d_norm.coords_nm.unsqueeze(0).cuda()
                coord_mask = d_norm.coord_mask.unsqueeze(0).cuda()
                residue_type = d_norm.residue_type.unsqueeze(0).cuda()
                
                seq_mask = coord_mask[:, :, 1].bool() 

                batch = {
                    "coords": coords,
                    "coords_nm": coords_nm,
                    "coord_mask": coord_mask,
                    "residue_type": residue_type,
                    "mask": seq_mask,
                    "mask_dict": {
                        "residue_type": seq_mask,
                        "coords": seq_mask.unsqueeze(-1).unsqueeze(-1) 
                    }
                }
                
                out = ae.encoder(batch)
                
                data.mean = out["mean"][0].cpu()
                data.log_scale = out["log_scale"][0].cpu()
                
                # Keep only C-alpha (index 1) to save RAM
                data.coords = data.coords[:, 1, :] 
                data.coord_mask = data.coord_mask[:, 1]
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)

            except Exception as e:
                # Catching specific errors like EOFError (corrupt files) or CUDA OOM
                print(f"\n[Error] Failed to process {f}: {e}")
                failed_count += 1

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed: {len(files) - failed_count - skipped_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed to process:      {failed_count}")

if __name__ == "__main__":
    main()