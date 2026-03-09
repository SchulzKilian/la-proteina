import os
import glob
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pathlib

# 1. Load your exact configuration
config = OmegaConf.load("configs/dataset/pdb/pdb_train_ucond.yaml")

# Set the data path (this matches what your job.sh does)
data_path = pathlib.Path("data/pdb_train")

# 2. Instantiate the exact DataSelector from your codebase
selector = instantiate(config.datamodule.dataselector, data_dir=str(data_path))

# 3. Create the full dataset metadata from Graphein (Downloads sequences, needs internet)
print("Fetching full PDB metadata from Graphein...")
df_full = selector.create_dataset()

# 4. Scan your processed folder for existing latents/graphs
print("Scanning processed directory for remaining files...")
processed_dir = data_path / "processed_latents"
pt_files = glob.glob(str(processed_dir / "**" / "*.pt"), recursive=True)

# Extract just the PDB IDs from your .pt files (e.g., '1abc_A.pt' -> '1abc')
existing_pdbs = {os.path.basename(p).split('_')[0].replace(".pt", "") for p in pt_files}

# 5. Filter the dataframe to ONLY include what you have in the processed folder
df_filtered = df_full[df_full["pdb"].isin(existing_pdbs)]

# 6. Generate the EXACT filename your pipeline expects using its own logic
def get_file_identifier(ds):
    return (
        f"df_pdb_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}_mt{ds.molecule_type}"
        f"_et{''.join(ds.experiment_types) if ds.experiment_types else ''}"
        f"_mino{ds.oligomeric_min}_maxo{ds.oligomeric_max}"
        f"_minr{ds.best_resolution}_maxr{ds.worst_resolution}"
        f"_hl{''.join(ds.has_ligands) if ds.has_ligands else ''}"
        f"_rl{''.join(ds.remove_ligands) if ds.remove_ligands else ''}"
        f"_rnsr{ds.remove_non_standard_residues}_rpu{ds.remove_pdb_unavailable}"
        f"_l{''.join(ds.labels) if ds.labels else ''}"
        f"_rcu{ds.remove_cath_unavailable}"
        f"_latents"
    )

csv_name = get_file_identifier(selector) + ".csv"
csv_path = data_path / csv_name

# 7. Save the fully populated CSV
df_filtered.to_csv(csv_path, index=False)
print(f"Success! Saved {len(df_filtered)} sequences to {csv_path}")