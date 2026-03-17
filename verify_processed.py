import torch
import os
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

def verify_atoms(file_path):
    data = torch.load(file_path)
    coords = data.coords # Shape [L, N_atoms, 3]
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"Number of atoms per residue: {coords.shape[1]}")
    
    if coords.shape[1] == 37:
        print("ALERT: Data is ALREADY in 37-atom format.")
        print("Applying PDB_TO_OPENFOLD_INDEX_TENSOR in __getitem__ will SCRAMBLE these atoms.")
    elif coords.shape[1] > 37:
        print("Data is in raw/extended format. Slicing in __getitem__ is likely required.")
    
    # Check Backbone Geometry (N to CA distance should be ~1.46 Angstroms)
    # OpenFold/Standard Order: 0:N, 1:CA, 2:C, 3:O
    n_ca_dist = torch.norm(coords[:, 0, :] - coords[:, 1, :], dim=-1).mean()
    print(f"Mean N-CA distance: {n_ca_dist:.4f} A")
    
    if n_ca_dist < 1.0 or n_ca_dist > 2.0:
        print("CRITICAL: Backbone geometry is broken. Your atom order is wrong!")

# Run on a sample from your sharded structure
verify_atoms("/data/pdb_train/processed/9z/9zA.pt")