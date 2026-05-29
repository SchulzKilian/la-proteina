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
