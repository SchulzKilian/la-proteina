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
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """Compute guidance gradient to add to the velocity field.

        Args:
            z_t: [B, L, 8] current latent state (from the no_grad sampling loop)
            v_theta: [B, L, 8] velocity predicted by flow model (detached)
            t_scalar: scalar float, the local_latents schedule time for this step
            mask: [B, L] bool mask

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

            # Clean estimate: x1_est = z_t + (1 - t) * v
            x1_est = z_t_grad + (1.0 - t_scalar) * v_det  # [B, L, 8]

            # Predictor forward pass (z-scored output for consistent gradient scale)
            t_ones = torch.ones(B, device=z_t.device)
            preds_zscore = self.predictor.predict_with_grad(x1_est, mask, t_ones)  # [B, 13]

            # Compute objective scalar
            objective = torch.zeros(B, device=z_t.device)
            for obj_cfg in self.objectives:
                idx = PROPERTY_TO_INDEX[obj_cfg["property"]]
                direction = obj_cfg.get("direction", "maximize")
                weight = obj_cfg.get("weight", 1.0)

                if direction == "target":
                    # Minimise squared distance to target (in z-score space)
                    target_raw = obj_cfg["target_value"]
                    target_z = (target_raw - self.predictor.stats.mean[idx]) / self.predictor.stats.std[idx]
                    # Negative because we maximise the objective, and want to minimise distance
                    objective = objective - weight * (preds_zscore[:, idx] - target_z) ** 2
                elif direction == "target_range":
                    # Squared hinge: no gradient inside [target_min, target_max];
                    # penalty grows quadratically with distance outside.
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

            # Sum over batch (each sample gets its own gradient)
            objective.sum().backward()
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
            # Get de-normalised predictions for interpretability
            with torch.no_grad():
                preds_denorm = self.predictor.predict(
                    z_t.detach() + (1.0 - t_scalar) * v_theta.detach(),
                    mask, t_ones,
                )  # [B, 13]

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
