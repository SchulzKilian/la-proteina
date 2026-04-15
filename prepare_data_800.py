"""
Standalone script to download and process PDB structures up to 800 residues.

This runs only the data preparation step (download + process to .pt files),
without starting any training. Resume-safe: skips already-processed proteins.

Usage:
    python prepare_data_800.py

After this completes, run precompute_latents.py to encode the new proteins.
"""
import os
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
root = os.path.abspath(".")
sys.path.insert(0, root)

# Monkeypatches from train.py (needed for graphein)
import wget
import graphein.ml.datasets.pdb_data

NEW_CATH_URL = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
graphein.ml.datasets.pdb_data.CATH_ID_CATH_CODE_URL = NEW_CATH_URL

def fixed_download_cath(self):
    self.cath_id_cath_code_url = NEW_CATH_URL
    target_path = self.root_dir / "cath-b-newest-all.txt"
    if not target_path.exists():
        print(f"[Patch] Downloading CATH data...")
        try:
            wget.download(self.cath_id_cath_code_url, out=str(target_path))
        except Exception as e:
            print(f"\n[Patch] Python download failed: {e}")
            import subprocess
            subprocess.run(["curl", "-o", str(target_path), self.cath_id_cath_code_url], check=True)

def robust_parse_ligand_map(self):
    path = self.root_dir / "ligand_map.txt"
    if not path.exists():
        return {}
    ligand_map = {}
    with open(path, "r") as f:
        for line in f:
            params = line.strip().split("\t")
            if len(params) > 1:
                ligand_map[params[0]] = params[1:]
    return ligand_map

graphein.ml.datasets.pdb_data.PDBManager._parse_ligand_map = robust_parse_ligand_map
graphein.ml.datasets.pdb_data.PDBManager._download_cath_id_cath_code_map = fixed_download_cath

import hydra
from dotenv import load_dotenv
from loguru import logger


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="training_local_latents",
)
def main(cfg_exp):
    load_dotenv()

    # Override to use the 800-residue dataset config
    # (or pass dataset=pdb/pdb_train_ucond_800 on the command line)
    cfg_data = cfg_exp.dataset

    # Force non-latent mode so it goes through download + process path
    cfg_data.datamodule.use_precomputed_latents = False

    datamodule = hydra.utils.instantiate(cfg_data.datamodule)

    logger.info(f"Data dir: {datamodule.data_dir}")
    logger.info(f"Raw dir:  {datamodule.raw_dir}")
    logger.info(f"Processed dir: {datamodule.processed_dir}")

    # This triggers: metadata query -> download missing CIFs -> process to .pt
    logger.info("Starting prepare_data()...")
    datamodule.prepare_data()
    logger.info("Done! Now run precompute_latents.py with the new CSV.")


if __name__ == "__main__":
    main()
