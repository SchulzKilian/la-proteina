import sys
import os
import hydra
from omegaconf import OmegaConf
import ssl
import wget


# 1. Global SSL Fix
ssl._create_default_https_context = ssl._create_unverified_context

# --- MONKEYPATCH SECTION (Must be before hydra.utils.instantiate) ---
import graphein.ml.datasets.pdb_data

# Fix A: CATH URL
NEW_CATH_URL = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
graphein.ml.datasets.pdb_data.CATH_ID_CATH_CODE_URL = NEW_CATH_URL

def fixed_download_cath(self):
    self.cath_id_cath_code_url = NEW_CATH_URL
    target_path = self.root_dir / "cath-b-newest-all.txt"
    if not target_path.exists():
        print(f"[Patch] Downloading CATH map to: {target_path}")
        wget.download(self.cath_id_cath_code_url, out=str(target_path))
    else:
        print("[Patch] CATH file exists.")

# Fix B: Ligand Map IndexError
def robust_parse_ligand_map(self):
    path = self.root_dir / "ligand_map.txt"
    if not path.exists(): return {}
    ligand_map = {}
    with open(path, "r") as f:
        for line in f:
            params = line.strip().split("\t")
            if len(params) > 1: # This prevents the IndexError
                ligand_map[params[0]] = params[1:]
    return ligand_map

# Apply patches
graphein.ml.datasets.pdb_data.PDBManager._download_cath_id_cath_code_map = fixed_download_cath
graphein.ml.datasets.pdb_data.PDBManager._parse_ligand_map = robust_parse_ligand_map
# --------------------------------------------------------------------

sys.path.insert(0, os.path.abspath("."))

@hydra.main(version_base=None, config_path="configs", config_name="training_local_latents")
def main(cfg):
    print("🚀 Starting Data Preparation on CPU...")
    
    # 1. Force CPU-friendly settings
    cfg.dataset.datamodule.num_workers = 1  # Use all your CPU cores (adjust as needed)
    
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