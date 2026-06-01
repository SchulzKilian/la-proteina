"""Build the cheap throttle priors from real training latents.

Outputs steering/throttle_priors/priors.pt with:
  - rama_logp [21, NB, NB]  : per-AA log p(phi, psi) on a uniform torus grid
                              (OpenFold restype order 0..19 + 20=X). Smoothed +
                              floored so disallowed cells get a finite, low logp.
  - aa_background [21]       : natural AA frequency (for the AA-prior throttle KL).
  - nbins (int), edges note  : phi,psi in [-pi, pi], bin = floor((a+pi)/(2pi)*NB).

These are READ-ONLY lookups used by the forward-only steering throttle (it scales
already-computed guidance, so the prior need NOT be differentiable). torsion order
from atom37_to_torsion_angles is [omega, phi, psi, chi1-4]; we use phi=idx1, psi=idx2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# openfold is vendored at the repo root (not pip-installed); ensure it's importable
# regardless of the cwd / how this script is launched.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# OpenFold torsion transform (same path cbm_dataset uses).
from openfold.data.data_transforms import atom37_to_torsion_angles

N_AA = 21
PHI_IDX, PSI_IDX = 1, 2


def _torsions(coords_nm, coord_mask, residue_type):
    protein = {
        "aatype": residue_type.long().clamp(max=20).unsqueeze(0),
        "all_atom_positions": coords_nm.float().unsqueeze(0),
        "all_atom_mask": coord_mask.float().unsqueeze(0),
    }
    out = atom37_to_torsion_angles()(protein)
    return out["torsion_angles_sin_cos"][0].float(), out["torsion_angles_mask"][0].float()


def _circular_blur(hist, sigma_bins):
    """Gaussian blur on the (phi, psi) torus via separable circular convolution."""
    nb = hist.shape[-1]
    r = int(np.ceil(3 * sigma_bins))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_bins) ** 2)
    k = k / k.sum()
    out = hist.astype(np.float64)
    for axis in (-2, -1):
        out = sum(np.roll(out, shift, axis=axis) * k[i] for i, shift in enumerate(range(-r, r + 1)))
    return out


def main():
    nb = 36
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    latent_root = Path("data/pdb_train/processed_latents")
    out_dir = Path("steering/throttle_priors")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in latent_root.rglob("*.pt"):
        files.append(p)
        if len(files) >= n_files:
            break
    print(f"[build] using {len(files)} latent files from {latent_root}", flush=True)

    hist = np.zeros((N_AA, nb, nb), dtype=np.float64)   # per-AA (phi, psi) counts
    aa_counts = np.zeros(N_AA, dtype=np.float64)
    n_res = 0
    for i, f in enumerate(files):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            g = (lambda k: d[k] if isinstance(d, dict) else getattr(d, k))
            rt = torch.as_tensor(g("residue_type")).long().clamp(max=20)
            sc, tm = _torsions(torch.as_tensor(g("coords_nm")),
                               torch.as_tensor(g("coord_mask")), rt)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f.name}: {e}", flush=True)
            continue
        phi = torch.atan2(sc[:, PHI_IDX, 0], sc[:, PHI_IDX, 1])
        psi = torch.atan2(sc[:, PSI_IDX, 0], sc[:, PSI_IDX, 1])
        valid = (tm[:, PHI_IDX] > 0.5) & (tm[:, PSI_IDX] > 0.5)
        bi = (((phi + np.pi) / (2 * np.pi)) * nb).long().clamp(0, nb - 1)
        bj = (((psi + np.pi) / (2 * np.pi)) * nb).long().clamp(0, nb - 1)
        for a, x, y, v in zip(rt.tolist(), bi.tolist(), bj.tolist(), valid.tolist()):
            aa_counts[a] += 1
            if v:
                hist[a, x, y] += 1.0
        n_res += rt.shape[0]
        if (i + 1) % 500 == 0:
            print(f"[build] {i+1}/{len(files)} files, {n_res} residues", flush=True)

    # Per-AA smoothed log-density. Blur on the torus, add a small pseudocount,
    # normalise to a proper density per AA, take log, floor at a low value so the
    # disallowed regions give a finite (large-negative) logp rather than -inf.
    rama_logp = np.full((N_AA, nb, nb), -20.0, dtype=np.float32)
    for a in range(N_AA):
        if hist[a].sum() < 50:        # too few examples (rare/X) -> uniform prior
            rama_logp[a] = np.log(1.0 / (nb * nb)).astype(np.float32)
            continue
        sm = _circular_blur(hist[a], sigma_bins=1.0)
        sm = sm + sm.sum() * 1e-4 / (nb * nb)   # pseudocount (0.01% of mass, uniform)
        dens = sm / sm.sum()
        lp = np.log(dens).astype(np.float32)
        rama_logp[a] = np.maximum(lp, -20.0)

    aa_background = (aa_counts / max(aa_counts.sum(), 1.0)).astype(np.float32)

    out = {
        "rama_logp": torch.from_numpy(rama_logp),       # [21, nb, nb]
        "aa_background": torch.from_numpy(aa_background),  # [21]
        "nbins": nb,
        "phi_idx": PHI_IDX, "psi_idx": PSI_IDX,
        "n_files": len(files), "n_residues": n_res,
        "note": "phi,psi in [-pi,pi]; bin=floor((a+pi)/(2pi)*nb); restype 0..19,20=X",
    }
    out_path = out_dir / "priors.pt"
    torch.save(out, out_path)
    # quick sanity: alpha (phi~-60,psi~-45) and beta (phi~-120,psi~135) for ALA(0)
    def cell(phi_deg, psi_deg):
        x = int(((np.radians(phi_deg) + np.pi) / (2 * np.pi)) * nb)
        y = int(((np.radians(psi_deg) + np.pi) / (2 * np.pi)) * nb)
        return x, y
    ax, ay = cell(-63, -43); bx, by = cell(-120, 135); fx, fy = cell(60, 60)
    print(f"[build] wrote {out_path}  ({n_res} residues, {len(files)} files)", flush=True)
    print(f"[build] ALA logp: alpha={rama_logp[0,ax,ay]:.2f} beta={rama_logp[0,bx,by]:.2f} "
          f"disallowed(+60,+60)={rama_logp[0,fx,fy]:.2f}", flush=True)
    print(f"[build] GLY(7) logp at disallowed(+60,+60)={rama_logp[7,fx,fy]:.2f} "
          f"(should be HIGHER than ALA — Gly is permissive)", flush=True)
    print(f"[build] aa_background (0..19): {np.round(aa_background[:20],3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
