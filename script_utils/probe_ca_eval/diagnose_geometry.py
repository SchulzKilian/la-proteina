"""Geometry sanity check for the CA-only eval pipeline.

For each native PDB:
  1. Read native backbone (N, CA, C, O) and report N-CA-C angles, |N-CA|, |CA-C|.
  2. Strip to CA only, run ca_to_backbone_atom37 (the function generate.py uses
     in CA-only mode), and report the same statistics on the reconstructed
     backbone.

Native angles should be ~111 deg (sp3-tetrahedral). Reconstructed angles should
also be ~111 deg if the reconstruction is correct. If reconstruction places N
and C purely along consecutive-CA directions, the angle will be near 180 deg.
"""
import os
import sys
import numpy as np

ROOT = "/home/ks2218/la-proteina"
sys.path.insert(0, ROOT)

import torch
from proteinfoundation.utils.pdb_utils import load_pdb
from proteinfoundation.utils.coors_utils import ca_to_backbone_atom37


def angle_deg(v1, v2):
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    cos = np.clip((v1 * v2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def report(name, atom37, mask=None):
    n = atom37.shape[0]
    N, CA, C = atom37[:, 0, :], atom37[:, 1, :], atom37[:, 2, :]
    if mask is None:
        ok = np.ones(n, dtype=bool)
    else:
        ok = mask[:, 0].astype(bool) & mask[:, 1].astype(bool) & mask[:, 2].astype(bool)
    nca = np.linalg.norm(N[ok] - CA[ok], axis=-1)
    cac = np.linalg.norm(C[ok] - CA[ok], axis=-1)
    angles = angle_deg(N[ok] - CA[ok], C[ok] - CA[ok])
    print(f"=== {name} (L={n}, kept={ok.sum()}) ===")
    print(f"  |N-CA|  mean={nca.mean():.3f}  std={nca.std():.3f}  Å  (ideal 1.459)")
    print(f"  |CA-C|  mean={cac.mean():.3f}  std={cac.std():.3f}  Å  (ideal 1.525)")
    print(f"  N-CA-C  mean={angles.mean():.2f}  std={angles.std():.2f} deg  (ideal ~111)")


def main():
    pdb_paths = [
        f"{ROOT}/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb",
        f"{ROOT}/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb",
    ]
    for p in pdb_paths:
        prot = load_pdb(p)
        atom37 = prot.atom_positions  # [N, 37, 3]
        mask = prot.atom_mask  # [N, 37]
        # Keep only residues with all 3 backbone atoms present.
        bb_ok = (mask[:, 0] > 0.5) & (mask[:, 1] > 0.5) & (mask[:, 2] > 0.5)
        atom37 = atom37[bb_ok]
        mask = mask[bb_ok]
        report(f"NATIVE  {os.path.basename(p)}", atom37, mask=mask)

        # Strip to CA only, then reconstruct via the function used in CA-only
        # generation
        ca = torch.from_numpy(atom37[:, 1, :]).float()  # [N, 3] in Å
        # ca_to_backbone_atom37 expects Å (it's the *_ang version)
        recon = ca_to_backbone_atom37(ca).numpy()  # [N, 37, 3] in Å
        report(f"RECON   {os.path.basename(p)}", recon)
        print()


if __name__ == "__main__":
    main()
