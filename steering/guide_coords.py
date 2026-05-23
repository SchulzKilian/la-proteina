"""SteeringGuideCoords: gradient guidance with a CA-conditioned predictor.

API-compatible with `steering.guide.SteeringGuide`. The caller passes the
current `bb_ca` state and the model's predicted `v_bb_ca` velocity in addition
to the latent state — those are kwargs on `guide()`, so the existing latent-only
guide still works against the same call site (it just ignores them via
`**_kwargs`).

Gradient still flows only into `local_latents` (the steering channel). `bb_ca`
is detached for the gradient pass — we don't currently steer the CA channel
itself, we only condition the predictor on it.

Predictor input modes (same convention as the latent-only guide):

  * feed_z_t_directly=False (default; clean-predictor / Tweedie path):
      x1_est  = z_t  + (1 - t) * v_local_latents
      x1_ca   = bb_ca + (1 - t) * v_bb_ca       (Tweedie-like extrapolation
                                                  along the CA flow's velocity)
      t_input = 1.0
    The CA-side extrapolation matches the latent-side one — if you trained the
    predictor on (clean z, clean ca, t=1), this is the in-distribution input.

  * feed_z_t_directly=True (noise-aware predictor path):
      x1_est  = z_t
      x1_ca   = bb_ca
      t_input = t_scalar
    The noise-aware fine-tune trained on noisy (z, ca) pairs at the SDE marginal
    for `t ∈ [t_min, t_max]`, so feeding the raw current state matches.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from steering.predictor_coords import SteeringPredictorCoords
from steering.schedules import get_schedule_fn
from steering.registry import (
    PROPERTY_TO_INDEX,
    DIRECTION_SIGN,
    validate_objective,
)


class SteeringGuideCoords:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.log_diagnostics = config.get("log_diagnostics", True)
        self.channel = config.get("channel", "local_latents")
        self._diagnostics: List[dict] = []

        if not self.enabled:
            self.predictor = None
            self.schedule_fn = None
            return

        ckpt_path = config["checkpoint"]
        if ckpt_path is None:
            raise ValueError("steering.enabled=True but steering.checkpoint is null")

        device = config.get("device", "cpu")
        self.predictor = SteeringPredictorCoords(ckpt_path, device=device)

        self.objectives = config.get("objectives", [])
        for obj in self.objectives:
            validate_objective(obj)

        self.schedule_fn = get_schedule_fn(config.get("schedule", {
            "type": "linear_ramp", "w_max": 2.0, "t_start": 0.3, "t_end": 1.0,
        }))

        self.gradient_norm = config.get("gradient_norm", "unit")
        self.gradient_clip = config.get("gradient_clip", 10.0)

        self.feed_z_t_directly = bool(config.get("feed_z_t_directly", False))

        smoothing_cfg = config.get("smoothing", {}) or {}
        self.smoothing_sigma = float(smoothing_cfg.get("sigma", 0.0))
        self.smoothing_n_samples = int(smoothing_cfg.get("n_samples", 1))
        if self.smoothing_sigma > 0 and self.smoothing_n_samples < 2:
            self.smoothing_n_samples = 2

        # K-step universal-guidance is not currently supported on the coord
        # guide. The Tweedie extrapolation already uses the model's current
        # v for both channels; adding inner Euler steps would need an inner
        # flow_step_fn that returns BOTH v_local_latents and v_bb_ca, which
        # the existing closure in product_space_flow_matcher only returns for
        # local_latents. Left as future work — refuse loudly if requested.
        self.denoising_steps = int(config.get("denoising_steps", 1))
        if self.denoising_steps != 1:
            raise NotImplementedError(
                "SteeringGuideCoords currently supports only denoising_steps=1; "
                f"got {self.denoising_steps}."
            )

    @property
    def diagnostics(self) -> List[dict]:
        return self._diagnostics

    def reset_diagnostics(self) -> None:
        self._diagnostics = []

    def guide(
        self,
        z_t: torch.Tensor,
        v_theta: torch.Tensor,
        t_scalar: float,
        mask: torch.Tensor,
        flow_step_fn=None,
        bb_ca: Optional[torch.Tensor] = None,
        v_bb_ca: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        B, L, D = z_t.shape

        if self.channel not in ("local_latents", "bb_ca"):
            raise ValueError(
                f"SteeringGuideCoords.channel must be 'local_latents' or 'bb_ca', got {self.channel!r}"
            )

        if bb_ca is None:
            raise ValueError(
                "SteeringGuideCoords requires bb_ca to be passed at guide() time "
                "(it must be plumbed through from product_space_flow_matcher)."
            )

        # Channel-shaped zero (for early exits). 'local_latents' → [B, L, 8]; 'bb_ca' → [B, L, 3].
        target_shape_ref = z_t if self.channel == "local_latents" else bb_ca

        if not self.enabled or self.predictor is None:
            zero = torch.zeros_like(target_shape_ref)
            diag = {"t": t_scalar, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        w = self.schedule_fn(t_scalar)
        if w == 0.0:
            zero = torch.zeros_like(target_shape_ref)
            diag = {"t": t_scalar, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        bb_ca_det = bb_ca.detach()
        v_ca_det = v_bb_ca.detach() if v_bb_ca is not None else torch.zeros_like(bb_ca_det)

        with torch.enable_grad():
            v_det = v_theta.detach()

            # Set up requires_grad on the steering target. The other channel is
            # passed through detached (predictor sees both as context; gradient
            # only flows through the named channel).
            if self.channel == "local_latents":
                z_t_grad = z_t.detach().clone().requires_grad_(True)
                bb_ca_in = bb_ca_det
                if self.feed_z_t_directly:
                    x1_est = z_t_grad
                    x1_ca = bb_ca_in
                    t_input = torch.full((B,), float(t_scalar), device=z_t.device)
                else:
                    x1_est = z_t_grad + (1.0 - t_scalar) * v_det
                    x1_ca = bb_ca_in + (1.0 - t_scalar) * v_ca_det
                    t_input = torch.ones(B, device=z_t.device)
                grad_target = z_t_grad
            else:  # self.channel == "bb_ca"
                bb_ca_grad = bb_ca_det.clone().requires_grad_(True)
                z_t_det = z_t.detach()
                if self.feed_z_t_directly:
                    x1_est = z_t_det
                    x1_ca = bb_ca_grad
                    t_input = torch.full((B,), float(t_scalar), device=z_t.device)
                else:
                    x1_est = z_t_det + (1.0 - t_scalar) * v_det
                    x1_ca = bb_ca_grad + (1.0 - t_scalar) * v_ca_det
                    t_input = torch.ones(B, device=z_t.device)
                grad_target = bb_ca_grad

            K = self.smoothing_n_samples if self.smoothing_sigma > 0 else 1
            objective_sum = torch.zeros(B, device=z_t.device)

            for k in range(K):
                # Smoothing noise is added to whichever input has gradient flow
                # (i.e. the steering target's denoised estimate).
                if self.smoothing_sigma > 0 and self.channel == "local_latents":
                    noise = torch.randn_like(x1_est) * self.smoothing_sigma
                    x1_in = x1_est + noise
                    x1_ca_in = x1_ca
                elif self.smoothing_sigma > 0 and self.channel == "bb_ca":
                    noise = torch.randn_like(x1_ca) * self.smoothing_sigma
                    x1_in = x1_est
                    x1_ca_in = x1_ca + noise
                else:
                    x1_in = x1_est
                    x1_ca_in = x1_ca

                preds_zscore = self.predictor.predict_with_grad(x1_in, x1_ca_in, mask, t_input)

                objective = torch.zeros(B, device=z_t.device)
                for obj_cfg in self.objectives:
                    idx = PROPERTY_TO_INDEX[obj_cfg["property"]]
                    direction = obj_cfg.get("direction", "maximize")
                    weight = obj_cfg.get("weight", 1.0)

                    if direction == "target":
                        target_raw = obj_cfg["target_value"]
                        target_z = (target_raw - self.predictor.stats.mean[idx]) / self.predictor.stats.std[idx]
                        objective = objective - weight * (preds_zscore[:, idx] - target_z) ** 2
                    elif direction == "target_range":
                        pred_z = preds_zscore[:, idx]
                        mean_i = self.predictor.stats.mean[idx]
                        std_i = self.predictor.stats.std[idx]
                        penalty = torch.zeros_like(pred_z)
                        if obj_cfg.get("target_min") is not None:
                            tmin_z = (obj_cfg["target_min"] - mean_i) / std_i
                            penalty = penalty + torch.relu(tmin_z - pred_z) ** 2
                        if obj_cfg.get("target_max") is not None:
                            tmax_z = (obj_cfg["target_max"] - mean_i) / std_i
                            penalty = penalty + torch.relu(pred_z - tmax_z) ** 2
                        objective = objective - weight * penalty
                    else:
                        sign = DIRECTION_SIGN[direction]
                        objective = objective + weight * sign * preds_zscore[:, idx]

                objective_sum = objective_sum + objective

            mean_objective = objective_sum / K
            mean_objective.sum().backward()
            raw_grad = grad_target.grad

        raw_grad = raw_grad.detach() * mask.unsqueeze(-1).float()
        raw_norms = raw_grad.reshape(B, -1).norm(dim=-1)

        if self.gradient_clip > 0:
            for b in range(B):
                norm_b = raw_norms[b].item()
                if norm_b > self.gradient_clip:
                    raw_grad[b] = raw_grad[b] * (self.gradient_clip / norm_b)

        if self.gradient_norm == "unit":
            for b in range(B):
                norm_b = raw_grad[b].reshape(-1).norm().item()
                if norm_b > 1e-10:
                    raw_grad[b] = raw_grad[b] / norm_b

        guidance = w * raw_grad
        final_norms = guidance.reshape(B, -1).norm(dim=-1)

        diag = None
        if self.log_diagnostics:
            with torch.no_grad():
                if self.feed_z_t_directly:
                    diag_z = z_t.detach()
                    diag_ca = bb_ca_det
                    diag_t = torch.full((B,), float(t_scalar), device=z_t.device)
                else:
                    diag_z = z_t.detach() + (1.0 - t_scalar) * v_theta.detach()
                    diag_ca = bb_ca_det + (1.0 - t_scalar) * v_ca_det
                    diag_t = torch.ones(B, device=z_t.device)
                preds_denorm = self.predictor.predict(diag_z, diag_ca, mask, diag_t)

            from steering.registry import PROPERTY_NAMES
            pred_dict = {
                name: preds_denorm[0, i].item()
                for i, name in enumerate(PROPERTY_NAMES)
            }

            diag = {
                "t": t_scalar,
                "w": w,
                "grad_norm_raw": raw_norms[0].item(),
                "grad_norm_final": final_norms[0].item(),
                "predicted_properties": pred_dict,
            }
            self._diagnostics.append(diag)

        return guidance, diag
