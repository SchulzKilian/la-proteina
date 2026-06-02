"""Closed-form, differentiable Cα-space geometry for look-ahead guidance.

Everything here operates on Cα coordinates of shape ``[B, L, 3]`` with a
boolean residue mask ``[B, L]`` (True = real residue) and returns one scalar
per protein (shape ``[B]``). All functions are plain PyTorch / autograd, use
``torch.cdist`` for pair distances, and run on a single device — no custom
kernels.

UNITS. During La-Proteina sampling the ``bb_ca`` channel is in **nanometres**
(``process_batch`` reads ``coords_nm``). So the geometric constants here default
to nm: Cα–Cα bond ≈ 0.38 nm, Cα clash radius ≈ 0.40 nm, contact threshold
≈ 0.80 nm. If you ever feed Ångström coordinates, scale these ×10.

The proxy P_geom = L_bond + lambda_clash * L_clash is the manifold-validity
score the look-ahead controller evaluates at both the base and the guided
one-shot clean estimate; the throttle reacts to the *increase* P(guided) -
P(base), never the absolute value.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _nres(mask: Tensor) -> Tensor:
    """Number of real residues per protein, clamped to avoid 0/0. [B]"""
    return mask.float().sum(-1).clamp(min=1.0)


def _masked_center(c: Tensor, mask: Tensor) -> Tensor:
    """Mask-aware centroid of Cα coords. c [B,L,3], mask [B,L] -> [B,3]."""
    m = mask.float().unsqueeze(-1)  # [B,L,1]
    return (c * m).sum(1) / _nres(mask).unsqueeze(-1)


def _seq_sep(L: int, device) -> Tensor:
    """|i - j| sequence separation matrix. [L,L]."""
    idx = torch.arange(L, device=device)
    return (idx[None, :] - idx[:, None]).abs()


def _pair_validity(mask: Tensor, min_sep: int, upper_only: bool = True) -> Tensor:
    """Valid (i,j) pairs: both residues real, sequence separation >= min_sep,
    optionally restricted to the strict upper triangle to avoid double counting.

    Returns float mask [B,L,L].
    """
    B, L = mask.shape
    m = mask.float()
    pair = m[:, :, None] * m[:, None, :]  # [B,L,L]
    sep = _seq_sep(L, mask.device)  # [L,L]
    sep_ok = (sep >= min_sep).float().unsqueeze(0)  # [1,L,L]
    valid = pair * sep_ok
    if upper_only:
        triu = torch.triu(torch.ones(L, L, device=mask.device), diagonal=1).unsqueeze(0)
        valid = valid * triu
    return valid


# --------------------------------------------------------------------------- #
# Geometric proxy terms
# --------------------------------------------------------------------------- #
def bond_term(
    c: Tensor,
    mask: Tensor,
    target: float = 0.38,
    eps: float = 1e-6,
) -> Tensor:
    """Mean squared deviation of consecutive Cα distances from ``target`` (nm).

    eps is added inside the sqrt so the distance gradient stays finite when two
    consecutive Cα happen to coincide. Returns [B].
    """
    diff = c[:, 1:, :] - c[:, :-1, :]  # [B,L-1,3]
    d = torch.sqrt((diff ** 2).sum(-1) + eps)  # [B,L-1]
    bmask = mask[:, 1:].float() * mask[:, :-1].float()  # [B,L-1]
    nbond = bmask.sum(-1).clamp(min=1.0)
    return ((d - target) ** 2 * bmask).sum(-1) / nbond  # [B]


def clash_term(
    c: Tensor,
    mask: Tensor,
    d0: float = 0.40,
    min_sep: int = 2,
) -> Tensor:
    """Flat-bottom one-sided quadratic clash penalty over non-adjacent Cα pairs.

    sum over valid (i<j, |i-j|>=min_sep) of max(0, d0 - ||c_i - c_j||)^2.
    Pairs with d >= d0 contribute exactly zero (flat bottom), so this only
    penalises actual interpenetration. Returns [B].
    """
    d = torch.cdist(c, c)  # [B,L,L]
    valid = _pair_validity(mask, min_sep=min_sep, upper_only=True)  # [B,L,L]
    viol = torch.relu(d0 - d) ** 2 * valid
    return viol.sum(dim=(1, 2))  # [B]


def p_geom(
    c: Tensor,
    mask: Tensor,
    *,
    bond_target: float = 0.38,
    clash_radius: float = 0.40,
    lambda_clash: float = 1.0,
    clash_min_sep: int = 2,
    eps: float = 1e-6,
) -> Tensor:
    """P_geom = L_bond + lambda_clash * L_clash. Returns [B] (per protein)."""
    lb = bond_term(c, mask, target=bond_target, eps=eps)
    lc = clash_term(c, mask, d0=clash_radius, min_sep=clash_min_sep)
    return lb + lambda_clash * lc


def p_geom_perresidue(
    c: Tensor,
    mask: Tensor,
    *,
    bond_target: float = 0.38,
    clash_radius: float = 0.40,
    lambda_clash: float = 1.0,
    clash_min_sep: int = 2,
    eps: float = 1e-6,
) -> Tensor:
    """Per-residue bond+clash load. Returns [B, L] (higher = worse local geometry
    at that residue). Bond deviation of each Cα-Cα bond is assigned to BOTH its
    endpoints; clash load of residue i = sum over non-adjacent j of relu(d0-d_ij)^2.
    Per-protein sum of this ~ p_geom (up to the bond double-count / mean-vs-sum)."""
    B, L, _ = c.shape
    mf = mask.float()
    diff = c[:, 1:, :] - c[:, :-1, :]
    d = torch.sqrt((diff ** 2).sum(-1) + eps)            # [B,L-1]
    bdev = (d - bond_target) ** 2 * (mf[:, 1:] * mf[:, :-1])  # [B,L-1]
    bond_res = torch.zeros(B, L, device=c.device, dtype=c.dtype)
    bond_res[:, :-1] = bond_res[:, :-1] + bdev
    bond_res[:, 1:] = bond_res[:, 1:] + bdev
    dist = torch.cdist(c, c)                              # [B,L,L]
    ar = torch.arange(L, device=c.device)
    sep_ok = (ar[None, :] - ar[:, None]).abs() >= clash_min_sep   # [L,L]
    pv = sep_ok[None] & mask[:, None, :].bool() & mask[:, :, None].bool()
    clash_res = (torch.relu(clash_radius - dist) ** 2 * pv.float()).sum(-1)  # [B,L]
    return (bond_res + lambda_clash * clash_res) * mf     # [B,L]


def ca_pseudo_torsions(c: Tensor, mask: Tensor, eps: float = 1e-8):
    """Cα-only backbone descriptors at each interior residue i (needs i-1..i+2):
      theta_i = pseudo-bond-angle at CA_i  (CA_{i-1}, CA_i, CA_{i+1})       in [0, pi]
      tau_i   = pseudo-dihedral            (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2}) in [-pi, pi]
    Returns (theta [B,L], tau [B,L], valid [B,L]) aligned at i; positions without a
    full 4-Cα window (or with any padded residue in it) are marked invalid.
    No N/C atoms needed — computable from the Cα channel alone (no decode).
    """
    B, L, _ = c.shape
    p0, p1, p2, p3 = c[:, :-3], c[:, 1:-2], c[:, 2:-1], c[:, 3:]  # each [B, L-3, 3]
    # pseudo-bond-angle at p1
    v1 = p0 - p1
    v2 = p2 - p1
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)
    theta = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))                 # [B,L-3]
    # pseudo-dihedral over (p0,p1,p2,p3)
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    m1 = torch.cross(n1, b1n, dim=-1)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    tau = torch.atan2(y, x)                                              # [B,L-3]
    # valid where all four residues are real
    mf = mask.float()
    win = (mf[:, :-3] * mf[:, 1:-2] * mf[:, 2:-1] * mf[:, 3:]) > 0.5     # [B,L-3]
    # pad back to length L (align at index i = position 1.. ; we key by the window start)
    pad = (0, 0)
    theta_f = torch.zeros(B, L, device=c.device, dtype=c.dtype)
    tau_f = torch.zeros(B, L, device=c.device, dtype=c.dtype)
    valid_f = torch.zeros(B, L, device=c.device, dtype=torch.bool)
    theta_f[:, : L - 3] = theta
    tau_f[:, : L - 3] = tau
    valid_f[:, : L - 3] = win
    return theta_f, tau_f, valid_f


def p_pseudo_rama(
    c: Tensor,
    mask: Tensor,
    logp_table: Tensor,
    *,
    n_theta: int,
    n_tau: int,
    eps: float = 1e-8,
) -> Tensor:
    """Per-protein Cα pseudo-Ramachandran energy: mean over interior residues of
    -log p(theta_i, tau_i) under a precomputed [n_theta, n_tau] density table.
    Higher = more off the Cα backbone manifold. Length-normalised. Returns [B].

    BILINEAR interpolation (theta clamped, tau circular) — NOT nearest-bin: the
    throttle needs the proxy to respond to the small per-step Cα perturbation, and
    a nearest-bin lookup quantises that to exactly zero (p_guided==p_base)."""
    import math
    theta, tau, valid = ca_pseudo_torsions(c, mask, eps=eps)            # [B,L] each
    tbl = logp_table.to(c.device)
    # fractional (center-aligned) bin coordinates
    fi = (theta / math.pi) * n_theta - 0.5                              # theta: [0,pi]
    fj = ((tau + math.pi) / (2 * math.pi)) * n_tau - 0.5               # tau: [-pi,pi]
    i0 = torch.floor(fi); wi = (fi - i0)
    j0 = torch.floor(fj); wj = (fj - j0)
    i0 = i0.long().clamp(0, n_theta - 1)
    i1 = (i0 + 1).clamp(0, n_theta - 1)                                 # theta: clamp at edges
    j0 = j0.long() % n_tau
    j1 = (j0 + 1) % n_tau                                               # tau: circular
    flat = tbl.reshape(-1)
    def g(ii, jj): return flat[ii * n_tau + jj]
    lp = (g(i0, j0) * (1 - wi) * (1 - wj) + g(i0, j1) * (1 - wi) * wj
          + g(i1, j0) * wi * (1 - wj) + g(i1, j1) * wi * wj)            # [B,L]
    e = -lp * valid.float()
    n = valid.float().sum(-1).clamp(min=1.0)
    return e.sum(-1) / n                                                 # [B]


def p_pseudo_rama_perresidue(
    c: Tensor,
    mask: Tensor,
    logp_table: Tensor,
    *,
    n_theta: int,
    n_tau: int,
    eps: float = 1e-8,
) -> Tensor:
    """Per-residue Cα pseudo-Ramachandran energy -log p(theta_i, tau_i). [B,L].
    Same bilinear lookup as p_pseudo_rama, returned per residue (not pooled)."""
    import math
    theta, tau, valid = ca_pseudo_torsions(c, mask, eps=eps)
    tbl = logp_table.to(c.device)
    fi = (theta / math.pi) * n_theta - 0.5
    fj = ((tau + math.pi) / (2 * math.pi)) * n_tau - 0.5
    i0 = torch.floor(fi); wi = (fi - i0)
    j0 = torch.floor(fj); wj = (fj - j0)
    i0 = i0.long().clamp(0, n_theta - 1); i1 = (i0 + 1).clamp(0, n_theta - 1)
    j0 = j0.long() % n_tau; j1 = (j0 + 1) % n_tau
    flat = tbl.reshape(-1)
    def g(ii, jj): return flat[ii * n_tau + jj]
    lp = (g(i0, j0) * (1 - wi) * (1 - wj) + g(i0, j1) * (1 - wi) * wj
          + g(i1, j0) * wi * (1 - wj) + g(i1, j1) * wi * wj)
    return (-lp) * valid.float()                                        # [B,L]


# --------------------------------------------------------------------------- #
# Steering target properties (all differentiable on Cα coords)
# --------------------------------------------------------------------------- #
def radius_of_gyration(c: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Rg = sqrt( (1/N) sum_i ||c_i - c_mean||^2 ). nm. Returns [B]."""
    center = _masked_center(c, mask)  # [B,3]
    d2 = ((c - center[:, None, :]) ** 2).sum(-1)  # [B,L]
    rg2 = (d2 * mask.float()).sum(-1) / _nres(mask)  # [B]
    return torch.sqrt(rg2 + eps)


def end_to_end(c: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Distance between the first and last real Cα (nm). Returns [B].

    Assumes right-padding (the convention used by the generation path): the
    first real residue is index 0 and the last is index ``nres-1``.
    """
    B, L, _ = c.shape
    last = (mask.float().sum(-1).long() - 1).clamp(min=0)  # [B]
    first = torch.zeros_like(last)
    bidx = torch.arange(B, device=c.device)
    c_first = c[bidx, first]  # [B,3]
    c_last = c[bidx, last]  # [B,3]
    return torch.sqrt(((c_last - c_first) ** 2).sum(-1) + eps)  # [B]


def soft_contact_order(
    c: Tensor,
    mask: Tensor,
    threshold: float = 0.80,
    temperature: float = 0.05,
    min_sep: int = 3,
) -> Tensor:
    """Differentiable relative contact order.

    Hard contacts (d < threshold) have zero gradient, so the contact indicator
    is softened to sigmoid((threshold - d) / temperature). Relative CO is

        CO = ( sum_{i<j} |i-j| * contact_ij ) / ( N * sum_{i<j} contact_ij )

    with N = number of residues. Larger CO ⇒ more long-range contacts ⇒ more
    topologically complex fold. Returns [B].
    """
    d = torch.cdist(c, c)  # [B,L,L]
    valid = _pair_validity(mask, min_sep=min_sep, upper_only=True)  # [B,L,L]
    contact = torch.sigmoid((threshold - d) / temperature) * valid  # [B,L,L]
    sep = _seq_sep(c.shape[1], c.device).float().unsqueeze(0)  # [1,L,L]
    weighted = (contact * sep).sum(dim=(1, 2))  # [B]
    ncontact = contact.sum(dim=(1, 2)).clamp(min=1e-6)  # [B]
    return weighted / (_nres(mask) * ncontact)  # [B]


def asphericity(c: Tensor, mask: Tensor) -> Tensor:
    """Asphericity from the gyration tensor: λ3 - 0.5(λ1 + λ2), λ1<=λ2<=λ3.

    0 for a perfectly spherical mass distribution, positive for elongated.
    Returns [B] (units nm^2).
    """
    center = _masked_center(c, mask)  # [B,3]
    cc = (c - center[:, None, :]) * mask.float().unsqueeze(-1)  # [B,L,3]
    S = torch.einsum("bli,blj->bij", cc, cc) / _nres(mask)[:, None, None]  # [B,3,3]
    evals = torch.linalg.eigvalsh(S)  # ascending [B,3]
    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]
    return l3 - 0.5 * (l1 + l2)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# name -> (callable, set of accepted per-property kwargs). The guide passes the
# relevant config knobs (e.g. contact-order threshold) when calling.
TARGET_FNS = {
    "rg": radius_of_gyration,
    "contact_order": soft_contact_order,
    "e2e": end_to_end,
    "asphericity": asphericity,
}


def validate_target(name: str) -> None:
    if name not in TARGET_FNS:
        raise ValueError(
            f"Unknown steering target '{name}'. Valid: {list(TARGET_FNS.keys())}"
        )
