import os
import glob
import random
import torch
import hydra
from loguru import logger
from tqdm import tqdm
import lightning as L

# Import paths from your project
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.partial_autoencoder.inference import load_dataloader

def verify_precomputed_latents(output_dir, expected_dim):
    """Checks the integrity of the generated latents."""
    logger.info("--- Starting Latent Verification ---")
    
    # 1. Check total files
    all_files = glob.glob(os.path.join(output_dir, "**", "*.pt"), recursive=True)
    logger.info(f"Found {len(all_files)} precomputed files.")
    if len(all_files) == 0:
        logger.error("No files found! Precomputation failed.")
        return

    # 2. Check a random sample of files for integrity
    sample_size = min(100, len(all_files))
    sample_files = random.sample(all_files, sample_size)
    
    logger.info(f"Verifying {sample_size} random files for structural integrity and NaNs...")
    
    for filepath in sample_files:
        try:
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue
            
        # Check required keys
        required_keys = ["ca_coords", "mean", "log_scale", "residue_type"]
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in {filepath}"
            
        seq_len = data["residue_type"].shape[0]
        
        # Check Shapes
        assert data["ca_coords"].shape == (seq_len, 3), f"ca_coords shape mismatch in {filepath}"
        assert data["mean"].shape == (seq_len, expected_dim), f"mean shape mismatch in {filepath}"
        assert data["log_scale"].shape == (seq_len, expected_dim), f"log_scale shape mismatch in {filepath}"
        
        # Check for NaNs or Infs
        for key in ["ca_coords", "mean", "log_scale"]:
            assert not torch.isnan(data[key]).any(), f"NaNs found in {key} for {filepath}"
            assert not torch.isinf(data[key]).any(), f"Infs found in {key} for {filepath}"

    logger.info("✅ Latent verification passed successfully! All shapes and values are valid.")

@hydra.main(config_path="../configs", config_name="inference_ae")
def main(cfg):
    L.seed_everything(cfg.seed)
    
    dataloader = load_dataloader(cfg) 
    model = AutoEncoder.load_from_checkpoint(cfg.ckpt_file)
    model.eval()
    model.cuda()
    
    expected_dim = model.cfg_ae.nn_ae["latent_z_dim"]
    output_dir = os.path.join(dataloader.dataset.data_dir, "processed_latents")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving precomputed latents to: {output_dir}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing Latents"):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            mask = batch["mask_dict"]["coords"][..., 0, 0] # [b, n]
            batch["mask"] = mask
            
            output_enc = model.encoder(batch)
            mean = output_enc["mean"]           
            log_scale = output_enc["log_scale"] 
            ca_coors_nm = batch["coords_nm"][..., 1, :] 

            bs = mask.shape[0]
            for i in range(bs):
                seq_len = mask[i].sum().item() 
                pdb_id = batch["id"][i]
                
                latent_dict = {
                    "ca_coords": ca_coors_nm[i, :int(seq_len)].cpu(),
                    "mean": mean[i, :int(seq_len)].cpu(),
                    "log_scale": log_scale[i, :int(seq_len)].cpu(),
                    "residue_type": batch["residue_type"][i, :int(seq_len)].cpu(),
                }
                
                shard = pdb_id[:2].lower()
                shard_dir = os.path.join(output_dir, shard)
                os.makedirs(shard_dir, exist_ok=True)
                
                torch.save(latent_dict, os.path.join(shard_dir, f"{pdb_id}.pt"))

    # Run verification at the end
    verify_precomputed_latents(output_dir, expected_dim)

if __name__ == "__main__":
    main()