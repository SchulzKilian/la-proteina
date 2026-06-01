"""Forward-only steering throttle (a BLOCKER on guidance, not an objective).

The latent SteeringGuide computes a guidance velocity `g` from a property
objective (e.g. tango_min). This throttle damps `g` wherever taking the step
would push the *cheap concept reconstruction* off-manifold, using the CBM's
first half (g1: latent -> AA logits + torsion angles). It never backprops; it
only multiplies the already-computed guidance by a factor s in (0, 1], exactly
like GeometricLookaheadGuide. So the priors need not be differentiable.

One-sided look-ahead (only *worsening* is penalised):
    z1_base   = z_t + (1 - t) * v_theta            (clean estimate, no guidance)
    z1_guided = z1_base + (1 - t) * guidance        (with guidance)
    P_*       = proxy(g1(z1_*))                      (rama energy / composition KL)
    dP        = P_guided - P_base                    (>0 means guidance made it worse)
    s         = exp(-beta * relu(dP))  in (0, 1]
    guidance <- s * guidance

Two proxy types (the two versions we compare):
  * "rama"     : PER-RESIDUE. P_i = -E_{aa~softmax}[ log p(phi_i, psi_i | aa) ].
                 AA-conditional Ramachandran energy. Blocks putting a residue
                 into a backbone conformation it cannot adopt. s is per-residue
                 so it vetoes specific positions, not the whole step.
  * "aa_prior" : PER-PROTEIN. P = KL( mean-composition || natural background ).
                 Blocks steering that drives the global AA composition toward
                 degeneracy (low-complexity / homopolymer), which rama cannot see.

Config block (under steering:):
    throttle:
      type: none | rama | aa_prior
      beta: 1.0            # damping sharpness in s = exp(-beta * relu(dP))
      priors_path: steering/throttle_priors/priors.pt
      t_floor: null        # if set, disable throttle (s=1) for t < t_floor
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class SteeringThrottle:
    def __init__(self, config: dict, predictor, feed_z_t_directly: bool):
        tcfg = (config.get("throttle", {}) or {})
        self.type = tcfg.get("type", "none")
        self.enabled = self.type not in (None, "none")
        self.beta = float(tcfg.get("beta", 1.0))
        self.t_floor = tcfg.get("t_floor", None)
        self.feed_z_t_directly = feed_z_t_directly
        self.predictor = predictor

        if not self.enabled:
            return
        if self.type not in ("rama", "aa_prior"):
            raise ValueError(f"Unknown throttle.type {self.type!r}")
        if predictor is None or not hasattr(predictor, "models"):
            raise ValueError("throttle requires a loaded CBM SteeringPredictor")
        # g1 lives on the CBM model(s). Single fold uses models[0]; an ensemble
        # is averaged at the bottleneck.
        self._models = [m for m in predictor.models]
        for m in self._models:
            if not hasattr(m, "encode_bottleneck"):
                raise ValueError("throttle requires a CBM predictor (encode_bottleneck missing)")

        device = predictor.device
        priors_path = tcfg.get("priors_path", "steering/throttle_priors/priors.pt")
        priors = torch.load(priors_path, map_location=device, weights_only=False)
        self.nbins = int(priors["nbins"])
        self.phi_idx = int(priors.get("phi_idx", 1))
        self.psi_idx = int(priors.get("psi_idx", 2))
        self.rama_logp = priors["rama_logp"].to(device)        # [21, nb, nb]
        self.aa_background = priors["aa_background"].to(device)  # [21]
        self._device = device

    # ------------------------------------------------------------------ #
    def _g1(self, z1, mask, t_input):
        """Mean (aa_probs, torsions) across folds from g1. no_grad."""
        aa_list, tors_list = [], []
        for m in self._models:
            t_emb = m.t_embed(t_input)
            aa_logits, tors = m.encode_bottleneck(z1, mask, t_emb)
            aa_list.append(F.softmax(aa_logits, dim=-1))
            tors_list.append(tors)
        aa = torch.stack(aa_list, 0).mean(0)      # [B,L,21]
        tors = torch.stack(tors_list, 0).mean(0)  # [B,L,7,2]
        return aa, tors

    def _rama_energy(self, aa_probs, tors):
        """Per-residue AA-conditional rama energy P_i = -E_aa[log p(phi,psi|aa)]. [B,L]."""
        phi = torch.atan2(tors[:, :, self.phi_idx, 0], tors[:, :, self.phi_idx, 1])
        psi = torch.atan2(tors[:, :, self.psi_idx, 0], tors[:, :, self.psi_idx, 1])
        nb = self.nbins
        bi = (((phi + math.pi) / (2 * math.pi)) * nb).long().clamp(0, nb - 1)
        bj = (((psi + math.pi) / (2 * math.pi)) * nb).long().clamp(0, nb - 1)
        lp = self.rama_logp[:, bi, bj].permute(1, 2, 0)   # [B,L,21]
        e_logp = (aa_probs * lp).sum(-1)                   # [B,L]
        return -e_logp

    def _aa_kl(self, aa_probs, mask):
        """Per-protein KL(mean-composition || background). [B]."""
        mf = mask.unsqueeze(-1).float()
        comp = (aa_probs * mf).sum(1) / mf.sum(1).clamp(min=1.0)  # [B,21]
        eps = 1e-8
        kl = (comp * (torch.log(comp + eps) - torch.log(self.aa_background + eps))).sum(-1)
        return kl  # [B]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply(self, guidance, z_t, v_theta, t_scalar, mask):
        """Damp `guidance` by the look-ahead throttle. Returns (guidance, diag)."""
        if not self.enabled:
            return guidance, {"throttle": "none"}
        if self.t_floor is not None and t_scalar < float(self.t_floor):
            return guidance, {"throttle": self.type, "skipped_t_floor": True, "s_mean": 1.0}

        B = z_t.shape[0]
        if self.feed_z_t_directly:
            z1_base = z_t
            t_input = torch.full((B,), float(t_scalar), device=z_t.device)
        else:
            z1_base = z_t + (1.0 - t_scalar) * v_theta
            t_input = torch.ones(B, device=z_t.device)
        z1_guided = z1_base + (1.0 - t_scalar) * guidance

        aa_b, tors_b = self._g1(z1_base, mask, t_input)
        aa_g, tors_g = self._g1(z1_guided, mask, t_input)

        if self.type == "rama":
            P_b = self._rama_energy(aa_b, tors_b)      # [B,L]
            P_g = self._rama_energy(aa_g, tors_g)
            dP = (P_g - P_b)
            s = torch.exp(-self.beta * F.relu(dP))     # [B,L] in (0,1]
            mf = mask.float()
            s = s * mf + (1.0 - mf)                    # s=1 on padded positions
            guidance = guidance * s.unsqueeze(-1)
            valid = mf.bool()
            s_valid = s[valid]
            dP_valid = dP[valid]
            diag = {
                "throttle": "rama", "beta": self.beta,
                "s_mean": s_valid.mean().item(),
                "s_min": s_valid.min().item(),
                "frac_throttled": (s_valid < 0.99).float().mean().item(),
                "dP_mean": dP_valid.mean().item(),
                "dP_max": dP_valid.max().item(),
            }
        else:  # aa_prior
            P_b = self._aa_kl(aa_b, mask)              # [B]
            P_g = self._aa_kl(aa_g, mask)
            dP = (P_g - P_b)
            s = torch.exp(-self.beta * F.relu(dP))     # [B]
            guidance = guidance * s.view(B, 1, 1)
            diag = {
                "throttle": "aa_prior", "beta": self.beta,
                "s_mean": s.mean().item(),
                "s_min": s.min().item(),
                "frac_throttled": (s < 0.99).float().mean().item(),
                "dP_mean": dP.mean().item(),
                "dP_max": dP.max().item(),
                "kl_base": P_b.mean().item(),
                "kl_guided": P_g.mean().item(),
            }
        return guidance, diag
