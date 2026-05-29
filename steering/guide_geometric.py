"""GeometricLookaheadGuide — look-ahead classifier guidance on the Cα channel.

This guide steers a *structural* target (radius of gyration, contact order, …)
computed in closed form on Cα coordinates — there is NO learned predictor. It
is API-compatible with the existing `SteeringGuide` / `SteeringGuideCoords`
call site in `product_space_flow_matcher.full_simulation`: it accepts the same
`guide(...)` signature and only consumes the `bb_ca` / `v_bb_ca` / `t_bb_ca`
kwargs (the latent `z_t` / `v_theta` / `t_scalar` are ignored).

Method
------
Under the conditional-OT path the one-shot clean estimate is
    x1 = x_t + (1 - t) * v_theta(x_t, t).
For the CA channel this is `x1_ca = bb_ca + (1 - t) * v_bb_ca`, already in Cα
space (no decode). Because `v` is detached, ∂x1/∂x_t = I, so the gradient of a
target evaluated at x1 *is* the gradient w.r.t. x_t — no model backprop.

Steering direction (per protein, unit-normalised):
    g = ∇_{x1} objective(x1)            (objective is the thing we ascend)

Look-ahead throttle (counterfactual / attribution form):
    x1_base   = bb_ca + (1 - t) * v_bb_ca
    x1_guided = x1_base + (1 - t) * λ0 * g       (= x_t + (1-t)(v + λ0 g))
    ΔP        = P_geom(x1_guided) - P_geom(x1_base)
    s         = f(max(0, ΔP))  ∈ (0, 1]          (one-sided on the *increase*)
    λ_eff     = s * λ0
    guidance  = λ_eff * g        (added to nn_out["bb_ca"]["v"])

Only the *worsening* of geometry counts: if the base one-shot estimate is
already bad (e.g. small t), both base and guided are bad and ΔP ≈ 0, so the
controller correctly does not intervene. Absolute geometry never enters s.

Ablation modes (config `mode`):
    * "baseline"               s ≡ 1 (plain classifier guidance, no look-ahead)
    * "lookahead_gated"        s = 1 if max(0,ΔP) <= gate_threshold else 0
    * "lookahead_proportional" s = f(max(0,ΔP)), f a saturating map
`f_map` ∈ {"exp", "reciprocal", "logistic"} with sharpness `beta`. Optional
`t_floor`: for t < t_floor the throttle is disabled (s ≡ 1) as a backup.

Throttle signal (config `proxy_type`) — what P measures in ΔP = P_guided − P_base:
    * "geometric"        P = bond+clash on the candidate Cα estimate (default).
    * "score_algebraic"  P = ||model score implied by the velocity AT x_t||,
                         base from v_theta, guided from v_theta + λ0·g. The
                         relation s=(t·v−x_t)/(1−t) is imported from the sampler
                         (rdn_flow_matcher.vf_to_score); 0 extra model forwards.
    * "score_faithful"   P = ||model score AT the clean estimate||, the model
                         re-evaluated at x1_base / x1_guided via flow_step_fn_bb_ca
                         (+2 forwards/step). Holds the other modes at their current
                         noisy state and uses the current bb_ca t.
The score proxies detect guidance pushing the state into a higher-score (lower-
density, off-manifold) region; the geometric proxy detects local bond/clash strain.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from steering.schedules import get_schedule_fn
from steering import geometry as geom
# Read the velocity<->score relation from the codebase — do NOT re-derive it.
# s(x_t,t) = (t*v - x_t)/(1-t)  (scale_ref=1 in the implementation).
from proteinfoundation.flow_matching.rdn_flow_matcher import vf_to_score


class GeometricLookaheadGuide:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        # This guide is meaningful only on the Cα channel.
        self.channel = config.get("channel", "bb_ca")
        self.log_diagnostics = config.get("log_diagnostics", True)
        self._diagnostics: List[dict] = []

        if not self.enabled:
            self.schedule_fn = None
            return

        if self.channel != "bb_ca":
            raise ValueError(
                f"GeometricLookaheadGuide only supports channel='bb_ca', got {self.channel!r}"
            )

        # Objectives: list of {property, direction, target_value, weight}.
        self.objectives = config.get("objectives", [])
        if not self.objectives:
            raise ValueError("GeometricLookaheadGuide requires at least one objective")
        for obj in self.objectives:
            geom.validate_target(obj["property"])

        # λ0(t): the unthrottled guidance weight schedule. w_max plays the role
        # of λ0 at the schedule plateau.
        self.schedule_fn = get_schedule_fn(config.get("schedule", {
            "type": "linear_ramp", "w_max": 32.0, "t_start": 0.3, "t_end": 0.8, "t_stop": 0.9,
        }))

        self.gradient_norm = config.get("gradient_norm", "unit")
        self.gradient_clip = float(config.get("gradient_clip", 0.0))

        # --- Throttle proxy selector ---
        # "geometric"       : P = bond+clash on the candidate Cα point (default).
        # "score_algebraic" : P = ||model score implied by the velocity AT x_t||,
        #                     base from v_theta, guided from v_theta + λ0·g.
        #                     Zero extra model forwards (cheap, convention A).
        # "score_faithful"  : P = ||model score AT the clean estimate||, with the
        #                     model re-evaluated at x1_base / x1_guided via
        #                     flow_step_fn_bb_ca. +2 model forwards/step (B).
        self.proxy_type = config.get("proxy_type", "geometric")
        if self.proxy_type not in ("geometric", "score_algebraic", "score_faithful"):
            raise ValueError(f"Unknown proxy_type {self.proxy_type!r}")
        # Fixed near-clean probe time for convention B (x̂₁ is treated as a
        # near-clean object, NOT as a sample at the current noisy t).
        self.t_p = float(config.get("t_p", 0.9))
        # Stage-1 calibration: when True (and mode='baseline'), every step also
        # logs m_base/m_guided/ΔP/r for ALL proxies (geom, A, B at each t_p in
        # calibration_t_p) along the same trajectory — no throttle is applied.
        self.calibration_probe = bool(config.get("calibration_probe", False))
        self.calibration_t_p = [float(x) for x in config.get("calibration_t_p", [0.85, 0.9, 0.95])]

        # --- Geometric proxy knobs (nm units) ---
        proxy = config.get("proxy", {}) or {}
        self.bond_target = float(proxy.get("bond_target_nm", 0.38))
        self.clash_radius = float(proxy.get("clash_radius_nm", 0.40))
        self.lambda_clash = float(proxy.get("lambda_clash", 1.0))
        self.clash_min_sep = int(proxy.get("clash_min_sep", 2))
        self.proxy_eps = float(proxy.get("eps", 1e-6))

        # --- Contact-order knobs (only used if a contact_order objective is set) ---
        co = config.get("contact_order", {}) or {}
        self.co_threshold = float(co.get("threshold_nm", 0.80))
        self.co_temperature = float(co.get("temperature_nm", 0.05))
        self.co_min_sep = int(co.get("min_seq_sep", 3))

        # --- Throttle ---
        self.mode = config.get("mode", "lookahead_proportional")
        if self.mode not in ("baseline", "lookahead_gated", "lookahead_proportional"):
            raise ValueError(f"Unknown steering mode {self.mode!r}")
        self.f_map = config.get("f_map", "exp")
        if self.f_map not in ("exp", "reciprocal", "logistic"):
            raise ValueError(f"Unknown f_map {self.f_map!r}")
        self.beta = float(config.get("beta", 1.0))
        self.gate_threshold = float(config.get("gate_threshold", 0.0))
        self.t_floor = config.get("t_floor", None)

    # ----------------------------------------------------------------------- #
    @property
    def diagnostics(self) -> List[dict]:
        return self._diagnostics

    def reset_diagnostics(self) -> None:
        self._diagnostics = []

    # ----------------------------------------------------------------------- #
    def _compute_property(self, name: str, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dispatch a target property by name. Returns [B]."""
        if name == "contact_order":
            return geom.soft_contact_order(
                c, mask, threshold=self.co_threshold,
                temperature=self.co_temperature, min_sep=self.co_min_sep,
            )
        return geom.TARGET_FNS[name](c, mask)

    def _objective(self, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scalar-per-protein objective to ASCEND. Returns [B]."""
        B = c.shape[0]
        obj = torch.zeros(B, device=c.device)
        for cfg in self.objectives:
            name = cfg["property"]
            direction = cfg.get("direction", "target")
            weight = float(cfg.get("weight", 1.0))
            val = self._compute_property(name, c, mask)  # [B]
            if direction == "target":
                target = float(cfg["target_value"])
                obj = obj - weight * (val - target) ** 2
            elif direction == "maximize":
                obj = obj + weight * val
            elif direction == "minimize":
                obj = obj - weight * val
            else:
                raise ValueError(f"Unknown direction {direction!r}")
        return obj

    def _p_geom(self, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return geom.p_geom(
            c, mask,
            bond_target=self.bond_target,
            clash_radius=self.clash_radius,
            lambda_clash=self.lambda_clash,
            clash_min_sep=self.clash_min_sep,
            eps=self.proxy_eps,
        )

    def _score_mag(
        self, c: torch.Tensor, v: torch.Tensor, t: float, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-protein RMS magnitude (over valid Cα coords) of the model score
        implied at point `c` with velocity `v` at time `t`. Length-normalised so
        a single β transfers across L. Returns [B]."""
        B = c.shape[0]
        t_vec = torch.full((B,), float(t), device=c.device, dtype=c.dtype)
        s = vf_to_score(c, v, t_vec)              # [B,L,3]  s=(t·v−c)/(1−t)
        s = s * mask.unsqueeze(-1).float()
        sq = (s ** 2).sum(dim=(1, 2))             # [B]
        n = mask.sum(dim=1).clamp(min=1).float() * 3.0
        return torch.sqrt(sq / n)                 # [B] RMS per coordinate

    @staticmethod
    def _pack(pb: torch.Tensor, pg: torch.Tensor, eps: float = 1e-8) -> dict:
        """Calibration record for one proxy (batch element 0). r is the scale-free
        relative degradation relu(ΔP)/m_base so a single β can transfer across
        proxies that live on different absolute scales."""
        m_base = float(pb[0].item())
        m_guided = float(pg[0].item())
        dP = m_guided - m_base
        r = max(0.0, dP) / max(m_base, eps)
        return {"m_base": m_base, "m_guided": m_guided, "dP": dP, "r": r}

    def _calibration_proxies(
        self, c0, v_ca, x1_base, x1_guided, t, g, lam0, m, flow_step_fn_bb_ca
    ) -> dict:
        """Compute m_base/m_guided/ΔP/r for geom + A + B(@each t_p) along the
        SAME (baseline) trajectory. B costs +2 model forwards per t_p."""
        out = {}
        out["geom"] = self._pack(self._p_geom(x1_base, m), self._p_geom(x1_guided, m))
        out["A"] = self._pack(
            self._score_mag(c0, v_ca, t, m),
            self._score_mag(c0, v_ca + lam0 * g, t, m),
        )
        if flow_step_fn_bb_ca is not None:
            for tp in self.calibration_t_p:
                vb = flow_step_fn_bb_ca(x1_base, tp)
                vg = flow_step_fn_bb_ca(x1_guided, tp)
                if vb is None or vg is None:
                    continue
                out[f"B_tp{tp:g}"] = self._pack(
                    self._score_mag(x1_base, vb.detach(), tp, m),
                    self._score_mag(x1_guided, vg.detach(), tp, m),
                )
        return out

    def _f(self, dp_pos: torch.Tensor) -> torch.Tensor:
        """Saturating map (0, 1], f(0)=1, decreasing. dp_pos >= 0, shape [B]."""
        if self.f_map == "exp":
            return torch.exp(-self.beta * dp_pos)
        if self.f_map == "reciprocal":
            return 1.0 / (1.0 + self.beta * dp_pos)
        # logistic: 2*sigmoid(-beta*x), equals 1 at x=0, ->0 as x->inf
        return 2.0 * torch.sigmoid(-self.beta * dp_pos)

    # ----------------------------------------------------------------------- #
    def guide(
        self,
        z_t: Optional[torch.Tensor] = None,
        v_theta: Optional[torch.Tensor] = None,
        t_scalar: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        flow_step_fn=None,
        flow_step_fn_bb_ca=None,
        bb_ca: Optional[torch.Tensor] = None,
        v_bb_ca: Optional[torch.Tensor] = None,
        t_bb_ca: Optional[float] = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        if bb_ca is None:
            raise ValueError(
                "GeometricLookaheadGuide requires bb_ca (the Cα state) to be plumbed "
                "through from product_space_flow_matcher."
            )

        # CA-channel time (correct (1-t) extrapolation); fall back to t_scalar.
        t = float(t_bb_ca) if t_bb_ca is not None else float(t_scalar)

        if not self.enabled:
            zero = torch.zeros_like(bb_ca)
            diag = {"t": t, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        w = self.schedule_fn(t)  # λ0 base for this step
        if w == 0.0:
            zero = torch.zeros_like(bb_ca)
            diag = {"t": t, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        c0 = bb_ca.detach()
        v_ca = v_bb_ca.detach() if v_bb_ca is not None else torch.zeros_like(c0)
        m = mask
        B = c0.shape[0]

        # Base one-shot clean Cα estimate.
        x1_base = c0 + (1.0 - t) * v_ca  # [B,L,3]

        # --- Steering gradient g = ∇ objective(x1_base) ---
        with torch.enable_grad():
            xc = x1_base.detach().clone().requires_grad_(True)
            obj = self._objective(xc, m)  # [B]
            obj.sum().backward()
            g_raw = xc.grad  # [B,L,3]

        g_raw = g_raw.detach() * m.unsqueeze(-1).float()
        raw_norms = g_raw.reshape(B, -1).norm(dim=-1)  # [B]

        if self.gradient_clip > 0:
            for b in range(B):
                nb = raw_norms[b].item()
                if nb > self.gradient_clip:
                    g_raw[b] = g_raw[b] * (self.gradient_clip / nb)

        g = g_raw
        if self.gradient_norm == "unit":
            g = g.clone()
            for b in range(B):
                nb = g[b].reshape(-1).norm().item()
                if nb > 1e-10:
                    g[b] = g[b] / nb

        lam0 = w  # scalar

        # --- Counterfactual proxy + throttle ---
        with torch.no_grad():
            x1_guided = x1_base + (1.0 - t) * lam0 * g  # [B,L,3]

            if self.proxy_type == "geometric":
                # Bond+clash geometry of the candidate clean Cα estimates.
                p_base = self._p_geom(x1_base, m)      # [B]
                p_guided = self._p_geom(x1_guided, m)  # [B]
            elif self.proxy_type == "score_algebraic":
                # Convention A: model score implied by the velocity AT x_t. The
                # guided "velocity" is v_theta + λ0·g (the unthrottled push). No
                # extra model forward — vf_to_score is linear in v.
                p_base = self._score_mag(c0, v_ca, t, m)             # [B]
                p_guided = self._score_mag(c0, v_ca + lam0 * g, t, m)  # [B]
            else:  # "score_faithful" — convention B
                if flow_step_fn_bb_ca is None:
                    raise ValueError(
                        "proxy_type='score_faithful' needs flow_step_fn_bb_ca "
                        "plumbed from product_space_flow_matcher (re-evaluates the "
                        "model's Cα velocity at the clean estimates)."
                    )
                # Probe x̂₁ as the near-clean object it is: re-evaluate the model
                # AND convert vf→score at a FIXED near-clean t_p (default 0.9, so
                # 1−t_p=0.1 — well off the t→1 boundary, well-conditioned). Probing
                # at the current noisy t would read a probe-level artifact and
                # collapse B toward A. Other modes stay frozen at their noisy state
                # (single-channel question — we steer only bb_ca).
                tp = self.t_p
                v_at_base = flow_step_fn_bb_ca(x1_base, tp).detach()
                v_at_guided = flow_step_fn_bb_ca(x1_guided, tp).detach()
                p_base = self._score_mag(x1_base, v_at_base, tp, m)       # [B]
                p_guided = self._score_mag(x1_guided, v_at_guided, tp, m)  # [B]

            dP = p_guided - p_base  # [B]
            dP_pos = torch.relu(dP)  # [B]

            if self.mode == "baseline":
                s = torch.ones(B, device=c0.device)
            elif self.mode == "lookahead_gated":
                s = (dP_pos <= self.gate_threshold).float()
            else:  # lookahead_proportional
                s = self._f(dP_pos)

            # Optional time floor: below t_floor disable the throttle (full step).
            if self.t_floor is not None and t < float(self.t_floor):
                s = torch.ones(B, device=c0.device)

            lam_eff = s * lam0  # [B]
            guidance = lam_eff[:, None, None] * g  # [B,L,3]

        diag = None
        if self.log_diagnostics:
            with torch.no_grad():
                prop_base = {
                    cfg["property"]: self._compute_property(cfg["property"], x1_base, m)[0].item()
                    for cfg in self.objectives
                }
            _pb0 = float(p_base[0].item())
            diag = {
                "t": t,
                "proxy_type": self.proxy_type,
                "t_p": self.t_p,
                "lambda0": float(lam0),
                "lambda_eff": float(lam_eff[0].item()),
                "s": float(s[0].item()),
                "p_base": _pb0,
                "p_guided": float(p_guided[0].item()),
                "delta_p": float(dP[0].item()),
                "r": max(0.0, float(dP[0].item())) / max(_pb0, 1e-8),
                "grad_norm_raw": float(raw_norms[0].item()),
                "predicted_properties": prop_base,
            }
            # Stage-1 calibration: log all proxies along this (baseline) trajectory.
            if self.calibration_probe:
                cal = self._calibration_proxies(
                    c0, v_ca, x1_base, x1_guided, t, g, lam0, m, flow_step_fn_bb_ca
                )
                for pname, rec in cal.items():
                    for k, val in rec.items():
                        diag[f"cal_{pname}_{k}"] = val
            self._diagnostics.append(diag)

        return guidance, diag
