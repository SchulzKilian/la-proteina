"""Build the Cα pseudo-Ramachandran density for the geometric throttle's `rama` proxy.

Output steering/throttle_priors/ca_pseudo_rama.pt:
  - logp [n_theta, n_tau]  : log p(theta, tau) over Cα pseudo-bond-angle theta in [0,pi]
                             and pseudo-dihedral tau in [-pi,pi]. Smoothed + floored.
  - n_theta, n_tau (int)
Computed from real training Cα (coords_nm[:,1,:], OpenFold CA = atom index 1). No N/C,
no decode. AA-agnostic geometric manifold prior (it guards backbone conformation, not
sequence). Used forward-only as a throttle proxy, so it need not be differentiable.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from steering.geometry import ca_pseudo_torsions  # noqa: E402


def _circular_blur_tau(hist, sigma):
    """Gaussian blur: circular along tau (axis -1), reflective along theta (axis -2)."""
    nt = hist.shape[-1]
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    out = hist.astype(np.float64)
    # tau: circular
    out = sum(np.roll(out, sh, axis=-1) * k[i] for i, sh in enumerate(range(-r, r + 1)))
    # theta: reflective (pad-reflect then convolve)
    padded = np.pad(out, ((r, r), (0, 0)), mode="reflect")
    out = sum(padded[i:i + out.shape[0]] * k[i] for i in range(2 * r + 1))
    return out


def main():
    n_theta = n_tau = 36
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    latent_root = Path("data/pdb_train/processed_latents")
    out_dir = Path("steering/throttle_priors"); out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in latent_root.rglob("*.pt"):
        files.append(p)
        if len(files) >= n_files:
            break
    print(f"[build] {len(files)} latent files", flush=True)

    hist = np.zeros((n_theta, n_tau), dtype=np.float64)
    n_res = 0
    for i, f in enumerate(files):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            g = (lambda k: d[k] if isinstance(d, dict) else getattr(d, k))
            coords = torch.as_tensor(g("coords_nm")).float()      # [L,37,3]
            cmask = torch.as_tensor(g("coord_mask"))              # [L,37]
            ca = coords[:, 1, :].unsqueeze(0)                     # [1,L,3]
            m = (cmask[:, 1] > 0.5).unsqueeze(0)                  # [1,L]
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f.name}: {e}", flush=True); continue
        theta, tau, valid = ca_pseudo_torsions(ca, m)
        theta, tau, valid = theta[0], tau[0], valid[0]
        if valid.sum() == 0:
            continue
        th = theta[valid].numpy(); tu = tau[valid].numpy()
        ti = np.clip((th / math.pi * n_theta).astype(int), 0, n_theta - 1)
        ui = np.clip(((tu + math.pi) / (2 * math.pi) * n_tau).astype(int), 0, n_tau - 1)
        for a, b in zip(ti, ui):
            hist[a, b] += 1.0
        n_res += int(valid.sum())
        if (i + 1) % 500 == 0:
            print(f"[build] {i+1}/{len(files)}  {n_res} interior residues", flush=True)

    sm = _circular_blur_tau(hist, sigma=1.0)
    sm = sm + sm.sum() * 1e-4 / (n_theta * n_tau)   # uniform pseudocount
    dens = sm / sm.sum()
    logp = np.maximum(np.log(dens), -20.0).astype(np.float32)

    out = {"logp": torch.from_numpy(logp), "n_theta": n_theta, "n_tau": n_tau,
           "n_files": len(files), "n_residues": n_res,
           "note": "theta in [0,pi] (CA pseudo-bond-angle), tau in [-pi,pi] (CA pseudo-dihedral); CA=atom1"}
    path = out_dir / "ca_pseudo_rama.pt"
    torch.save(out, path)

    # sanity: helix (theta~89 deg, tau~50 deg) and sheet (theta~120, tau~-170/+170) should be high-density
    def cell(theta_deg, tau_deg):
        a = int(math.radians(theta_deg) / math.pi * n_theta)
        b = int((math.radians(tau_deg) + math.pi) / (2 * math.pi) * n_tau)
        return logp[min(a, n_theta - 1), min(b, n_tau - 1)]
    print(f"[build] wrote {path}  ({n_res} residues, {len(files)} files)", flush=True)
    print(f"[build] logp  helix(89,50)={cell(89,50):.2f}  sheet(120,-170)={cell(120,-170):.2f} "
          f"sheet(120,170)={cell(120,170):.2f}  forbidden(20,0)={cell(20,0):.2f}", flush=True)


if __name__ == "__main__":
    main()
