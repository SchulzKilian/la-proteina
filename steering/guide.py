"""Core guidance logic: SteeringGuide class."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from steering.predictor import SteeringPredictor
from steering.schedules import get_schedule_fn
from steering.registry import (
    PROPERTY_TO_INDEX,
    DIRECTION_SIGN,
    validate_objective,
)


class SteeringGuide:
    """Computes gradient-based guidance to add to the latent velocity field.

    Usage in the sampling loop:
        guide = SteeringGuide(config)
        ...
        # inside the step loop, after nn_out is computed:
        if guide is not None:
            guidance, diag = guide.guide(z_t, v_theta, t_scalar, mask)
            nn_out["local_latents"]["v"] = nn_out["local_latents"]["v"] + guidance
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'steering' block from the config. Must have keys:
                enabled, checkpoint, objectives, schedule, gradient_norm,
                gradient_clip, channel. Optional: log_diagnostics.
        """
        self.enabled = config.get("enabled", False)
        self.log_diagnostics = config.get("log_diagnostics", True)
        self.channel = config.get("channel", "local_latents")
        self._diagnostics: List[dict] = []

        if not self.enabled:
            self.predictor = None
            self.schedule_fn = None
            return

        # Load predictor
        ckpt_path = config["checkpoint"]
        if ckpt_path is None:
            raise ValueError("steering.enabled=True but steering.checkpoint is null")

        device = config.get("device", "cpu")
        self.predictor = SteeringPredictor(ckpt_path, device=device)

        # Parse objectives
        self.objectives = config.get("objectives", [])
        for obj in self.objectives:
            validate_objective(obj)

        # Schedule
        self.schedule_fn = get_schedule_fn(config.get("schedule", {
            "type": "linear_ramp", "w_max": 2.0, "t_start": 0.3, "t_end": 1.0,
        }))

        # Gradient handling
        self.gradient_norm = config.get("gradient_norm", "unit")
        self.gradient_clip = config.get("gradient_clip", 10.0)

        # Predictor input mode.
        # False (default, legacy): feed `x_1_est = z_t + (1-t)*v` with t=1.0.
        #   The standard reconstruction-guidance trick; appropriate for a
        #   predictor trained on CLEAN t=1 latents (it has never seen noisy z_t).
        # True (preferred for noise-aware predictors): feed z_t directly with
        #   the real `t_scalar`. Removes the (input + t) double-mismatch when
        #   the predictor was trained on (z_t, t∈[t_min, t_max]).
        self.feed_z_t_directly = bool(config.get("feed_z_t_directly", False))

        # Randomized smoothing (anti-gradient-hacking).
        # When sigma > 0, the predictor sees x1_est + N(0, sigma^2 I) and the
        # gradient is averaged over `n_samples` noise draws. Forces the
        # gradient direction to be robust to local perturbations — adversarial
        # directions tend not to be.
        smoothing_cfg = config.get("smoothing", {}) or {}
        self.smoothing_sigma = float(smoothing_cfg.get("sigma", 0.0))
        self.smoothing_n_samples = int(smoothing_cfg.get("n_samples", 1))
        if self.smoothing_sigma > 0 and self.smoothing_n_samples < 2:
            self.smoothing_n_samples = 2  # avoid 1-sample "average"

        # Universal-guidance K-step denoising: replace the one-step Tweedie
        # estimate `x_1_est = z_t + (1 - t) * v` with K iterative Euler steps
        # of the latent flow ODE from t to 1. K=1 is the original behaviour.
        # Requires the caller to pass `flow_step_fn` to .guide().
        self.denoising_steps = int(config.get("denoising_steps", 1))
        if self.denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {self.denoising_steps}")

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
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """Compute guidance gradient to add to the velocity field.

        Args:
            z_t: [B, L, 8] current latent state (from the no_grad sampling loop)
            v_theta: [B, L, 8] velocity predicted by flow model (detached)
            t_scalar: scalar float, the local_latents schedule time for this step
            mask: [B, L] bool mask
            flow_step_fn: optional callable (z_iter, t_iter) -> v_local_latents.
                Required when self.denoising_steps > 1; ignored when K=1.

        Returns:
            guidance: [B, L, 8] gradient to ADD to v_theta. Detached, safe for no_grad context.
            diag: dict of diagnostics for this step, or None if diagnostics disabled.
        """
        B, L, D = z_t.shape

        if not self.enabled or self.predictor is None:
            zero = torch.zeros_like(z_t)
            diag = {"t": t_scalar, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        w = self.schedule_fn(t_scalar)
        if w == 0.0:
            zero = torch.zeros_like(z_t)
            diag = {"t": t_scalar, "w": 0.0, "skipped": True} if self.log_diagnostics else None
            return zero, diag

        # Compute guidance gradient inside enable_grad context
        with torch.enable_grad():
            z_t_grad = z_t.detach().clone().requires_grad_(True)
            v_det = v_theta.detach()

            if self.feed_z_t_directly:
                # Noise-aware-predictor path: feed z_t at the real t. No Tweedie,
                # no K-step inner loop — the predictor was trained for exactly
                # this distribution.
                x1_est = z_t_grad
                t_input = torch.full((B,), float(t_scalar), device=z_t.device)
            else:
                # Legacy reconstruction-guidance path (clean-predictor era).
                # K=1: one-step Tweedie     x_1 ≈ z_t + (1 - t) * v(z_t, t)
                # K>1: K-step Euler integration of the latent flow ODE from t to 1,
                #      reusing the cached outer v at the first inner step.
                K_d = self.denoising_steps
                if K_d == 1 or flow_step_fn is None:
                    x1_est = z_t_grad + (1.0 - t_scalar) * v_det
                else:
                    z_iter = z_t_grad
                    t_iter = float(t_scalar)
                    dt_inner = (1.0 - t_scalar) / K_d
                    for k_step in range(K_d):
                        if k_step == 0:
                            v_iter = v_det
                        else:
                            v_iter = flow_step_fn(z_iter, t_iter)
                        z_iter = z_iter + dt_inner * v_iter
                        t_iter = t_iter + dt_inner
                    x1_est = z_iter
                t_input = torch.ones(B, device=z_t.device)

            # Decide how many predictor calls to make. Without smoothing, K=1
            # (the original single-sample path). With smoothing, K samples
            # of N(0, sigma^2) noise are added to x1_est; the objective is
            # averaged across K and a single backward pass produces the
            # smoothed gradient w.r.t. z_t_grad.
            K = self.smoothing_n_samples if self.smoothing_sigma > 0 else 1

            # Sum, then divide once at end → matches mean_obj.sum().backward()
            # but avoids materialising a [K, B] tensor.
            objective_sum = torch.zeros(B, device=z_t.device)

            for k in range(K):
                if self.smoothing_sigma > 0:
                    noise = torch.randn_like(x1_est) * self.smoothing_sigma
                    x1_in = x1_est + noise
                else:
                    x1_in = x1_est

                preds_zscore = self.predictor.predict_with_grad(x1_in, mask, t_input)  # [B, P]

                # Compute objective scalar (averaged over K outside the loop)
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

            # Sum over batch (each sample gets its own gradient)
            mean_objective.sum().backward()
            raw_grad = z_t_grad.grad  # [B, L, 8]

        # --- Post-processing (all outside enable_grad, all detached) ---
        raw_grad = raw_grad.detach()

        # Zero out masked positions
        raw_grad = raw_grad * mask.unsqueeze(-1).float()

        # Compute raw norm per protein (for diagnostics)
        raw_norms = raw_grad.reshape(B, -1).norm(dim=-1)  # [B]

        # Clip gradient norm per protein
        if self.gradient_clip > 0:
            for b in range(B):
                norm_b = raw_norms[b].item()
                if norm_b > self.gradient_clip:
                    raw_grad[b] = raw_grad[b] * (self.gradient_clip / norm_b)

        # Normalise
        if self.gradient_norm == "unit":
            for b in range(B):
                norm_b = raw_grad[b].reshape(-1).norm().item()
                if norm_b > 1e-10:
                    raw_grad[b] = raw_grad[b] / norm_b

        # Scale by schedule weight
        guidance = w * raw_grad

        # Final norms for diagnostics
        final_norms = guidance.reshape(B, -1).norm(dim=-1)  # [B]

        # Build diagnostics
        diag = None
        if self.log_diagnostics:
            # Get de-normalised predictions for interpretability — match the
            # predictor input mode used for the gradient.
            with torch.no_grad():
                if self.feed_z_t_directly:
                    diag_input = z_t.detach()
                    diag_t = torch.full((B,), float(t_scalar), device=z_t.device)
                else:
                    diag_input = z_t.detach() + (1.0 - t_scalar) * v_theta.detach()
                    diag_t = torch.ones(B, device=z_t.device)
                preds_denorm = self.predictor.predict(diag_input, mask, diag_t)

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
