"""Reconstruct .pdb files from the .pt files in the sweep cells.

The .pt files store coords_openfold[L,37,3], residue_type[L], coord_mask[L,37].
Use proteinfoundation.utils.pdb_utils.write_prot_to_pdb with no_indexing=True
to write atom37 PDBs that match what the steering generate.py originally wrote.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

SWEEP = ROOT / "results/noise_aware_ensemble_sweep"

def main():
    n_done = 0
    for cell_dir in sorted(SWEEP.glob("*/guided")):
        for pt_path in sorted(cell_dir.glob("*.pt")):
            pdb_path = pt_path.with_suffix(".pdb")
            if pdb_path.exists():
                continue
            d = torch.load(pt_path, weights_only=False)
            coords = d["coords_openfold"].numpy()                  # [L, 37, 3]
            aatype = d["residue_type"].numpy().astype(np.int32)    # [L]
            write_prot_to_pdb(
                prot_pos=coords,
                file_path=str(pdb_path),
                aatype=aatype,
                overwrite=True,
                no_indexing=True,
            )
            n_done += 1
            if n_done % 50 == 0:
                print(f"  wrote {n_done} PDBs ...", flush=True)
    print(f"Done. Wrote {n_done} PDBs total.")

if __name__ == "__main__":
    main()
