import os
import torch
from glob import glob
from tqdm import tqdm
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.transforms import CoordsToNanometers

def main():
    # 1. Setup paths and load AE
    data_dir = "data/pdb_train/processed"                  # Point to your actual processed dir
    out_dir = "data/pdb_train/processed_latents"
    
    ae_path = "./checkpoints_laproteina/AE1_ucond_512.ckpt"
    ae = AutoEncoder.load_from_checkpoint(ae_path).cuda().eval()
    
    files = glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    transform = CoordsToNanometers() # AutoEncoder needs coords_nm
    
    with torch.no_grad():
        for f in tqdm(files, desc="Precomputing"):
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            d_norm = transform(data.clone())
            
            # 1. Base tensors with batch dim added
            coords = data.coords.unsqueeze(0).cuda()
            coords_nm = d_norm.coords_nm.unsqueeze(0).cuda()
            coord_mask = d_norm.coord_mask.unsqueeze(0).cuda()
            residue_type = d_norm.residue_type.unsqueeze(0).cuda()
            
            # 2. Sequence mask (Extract just the CA atom presence: [1, n])
            seq_mask = coord_mask[:, :, 1].bool() 

            # 3. Assemble dummy batch
            batch = {
                "coords": coords,
                "coords_nm": coords_nm,
                "coord_mask": coord_mask,
                "residue_type": residue_type,
                "mask": seq_mask,
                "mask_dict": {
                    "residue_type": seq_mask,
                    "coords": coord_mask.unsqueeze(-1).unsqueeze(-1)
                }
            }
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            
            # Encode
            out = ae.encoder(batch)
            
            # Add latents directly to the PyG Data object
            data.mean = out["mean"][0].cpu()
            data.log_scale = out["log_scale"][0].cpu()
            
            # ELEGANT RAM SAVING: Discard the 37 atoms, keep only C-alpha (index 1)
            # This makes the object tiny, allowing in_memory=True to hold 10x more proteins
            data.coords = data.coords[:, 1, :] 
            data.coord_mask = data.coord_mask[:, 1]
            
            # Save
            out_path = f.replace("processed", "processed_latents")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(data, out_path)

if __name__ == "__main__":
    main()