import sys
import os
import hydra
from omegaconf import OmegaConf

# Add root to path so imports work
sys.path.insert(0, os.path.abspath("."))

@hydra.main(version_base=None, config_path="configs", config_name="training_local_latents")
def main(cfg):
    print("🚀 Starting Data Preparation on CPU...")
    
    # 1. Force CPU-friendly settings
    cfg.dataset.datamodule.num_workers = 4  # Use all your CPU cores (adjust as needed)
    
    # 2. Instantiate ONLY the DataModule (skips Model/Trainer)
    print(f"Loading DataModule config...")
    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    
    # 3. Run the download/processing
    print("Running prepare_data()...")
    datamodule.prepare_data()
    
    print("✅ Data preparation complete! The files are ready in the processed folder.")

if __name__ == "__main__":
    # This prevents some multiprocessing errors on specific setups
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()