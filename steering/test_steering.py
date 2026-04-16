"""Standalone test for steering module. Runs WITHOUT the flow model.

Usage:
    python -m steering.test_steering --checkpoint <path_to_fold_0_best.pt>
    python -m steering.test_steering  # uses a dummy checkpoint for dev testing
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steering.guide import SteeringGuide
from steering.registry import PROPERTY_NAMES, PROPERTY_TO_INDEX


def create_dummy_checkpoint(path: str) -> str:
    """Create a minimal valid checkpoint with random weights for testing."""
    # Import here to keep the test self-contained
    steerability_root = ROOT / "laproteina_steerability"
    sys.path.insert(0, str(steerability_root))
    from src.multitask_predictor.model import PropertyTransformer

    model = PropertyTransformer(
        latent_dim=8, d_model=128, n_heads=4, n_layers=3,
        ffn_expansion=4, dropout=0.1, n_properties=13, max_len=1024,
    )
    stats_mean = np.random.randn(13).astype(np.float32)
    stats_std = np.abs(np.random.randn(13).astype(np.float32)) + 0.1

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "val_r2_mean": 0.0,
        "val_results": {},
        "stats_mean": stats_mean,
        "stats_std": stats_std,
    }, path)
    return path


def test_disabled():
    """Test 1: Disabled steering returns zeros."""
    print("=" * 60)
    print("TEST 1: Disabled steering returns zeros")
    config = {"enabled": False}
    guide = SteeringGuide(config)

    z_t = torch.randn(1, 100, 8)
    v = torch.randn(1, 100, 8)
    mask = torch.ones(1, 100, dtype=torch.bool)

    guidance, diag = guide.guide(z_t, v, t_scalar=0.8, mask=mask)

    assert guidance.shape == (1, 100, 8), f"Wrong shape: {guidance.shape}"
    assert (guidance == 0).all(), "Guidance should be all zeros when disabled"
    assert diag is not None and diag["skipped"], "Should report skipped"
    print("  PASSED: shape correct, all zeros, skipped=True")


def test_enabled_shape_and_nonzero(ckpt_path: str):
    """Test 2: Enabled steering returns correct shape, non-zero, respects mask."""
    print("=" * 60)
    print("TEST 2: Enabled steering — shape, non-zero, mask")
    config = {
        "enabled": True,
        "checkpoint": ckpt_path,
        "objectives": [{"property": "net_charge", "direction": "maximize", "weight": 1.0}],
        "schedule": {"type": "linear_ramp", "w_max": 2.0, "t_start": 0.3, "t_end": 1.0},
        "gradient_norm": "unit",
        "gradient_clip": 10.0,
        "channel": "local_latents",
        "log_diagnostics": True,
        "device": "cpu",
    }
    guide = SteeringGuide(config)

    B, L, D = 1, 100, 8
    z_t = torch.randn(B, L, D)
    v = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    # Mask out last 20 positions
    mask[0, 80:] = False

    guidance, diag = guide.guide(z_t, v, t_scalar=0.8, mask=mask)

    assert guidance.shape == (B, L, D), f"Wrong shape: {guidance.shape}"
    assert not (guidance == 0).all(), "Guidance should not be all zeros"

    # Masked positions must be zero
    masked_grad = guidance[0, 80:, :]
    assert (masked_grad == 0).all(), f"Masked positions not zero: norm={masked_grad.norm():.6f}"

    # Unmasked positions must be non-zero
    unmasked_grad = guidance[0, :80, :]
    assert not (unmasked_grad == 0).all(), "Unmasked positions should be non-zero"

    print(f"  Shape: {guidance.shape}")
    print(f"  Guidance norm (unmasked): {unmasked_grad.norm():.6f}")
    print(f"  Guidance norm (masked):   {masked_grad.norm():.6f}")
    print(f"  Diagnostics t={diag['t']}, w={diag['w']:.4f}")
    print(f"  Raw grad norm: {diag['grad_norm_raw']:.6f}")
    print(f"  Final grad norm: {diag['grad_norm_final']:.6f}")
    print(f"  Predicted properties: { {k: f'{v:.4f}' for k, v in diag['predicted_properties'].items()} }")
    print("  PASSED")
    return guide


def test_schedule_gating(ckpt_path: str):
    """Test 3: Guidance is zero below t_start, non-zero above."""
    print("=" * 60)
    print("TEST 3: Schedule gating (t < t_start -> zeros)")
    config = {
        "enabled": True,
        "checkpoint": ckpt_path,
        "objectives": [{"property": "net_charge", "direction": "maximize", "weight": 1.0}],
        "schedule": {"type": "linear_ramp", "w_max": 2.0, "t_start": 0.3, "t_end": 1.0},
        "gradient_norm": "unit",
        "gradient_clip": 10.0,
        "log_diagnostics": True,
        "device": "cpu",
    }
    guide = SteeringGuide(config)

    z_t = torch.randn(1, 100, 8)
    v = torch.randn(1, 100, 8)
    mask = torch.ones(1, 100, dtype=torch.bool)

    # Below t_start
    g_low, d_low = guide.guide(z_t, v, t_scalar=0.1, mask=mask)
    assert (g_low == 0).all(), "Guidance should be zero for t=0.1 < t_start=0.3"
    assert d_low["skipped"], "Should report skipped for t < t_start"

    # Above t_start
    g_high, d_high = guide.guide(z_t, v, t_scalar=0.8, mask=mask)
    assert not (g_high == 0).all(), "Guidance should be non-zero for t=0.8"

    # Check magnitude scaling: t=0.5 should have lower w than t=0.9
    guide2 = SteeringGuide(config)
    _, d_05 = guide2.guide(z_t, v, t_scalar=0.5, mask=mask)
    _, d_09 = guide2.guide(z_t, v, t_scalar=0.9, mask=mask)
    assert d_05["w"] < d_09["w"], f"w(0.5)={d_05['w']:.4f} should be < w(0.9)={d_09['w']:.4f}"

    print(f"  t=0.1: all zeros (skipped={d_low['skipped']})")
    print(f"  t=0.5: w={d_05['w']:.4f}, final norm={d_05['grad_norm_final']:.6f}")
    print(f"  t=0.9: w={d_09['w']:.4f}, final norm={d_09['grad_norm_final']:.6f}")
    print("  PASSED")


def test_gradient_direction(ckpt_path: str):
    """Test 4: For maximize, stepping along the gradient increases prediction."""
    print("=" * 60)
    print("TEST 4: Gradient direction — maximize net_charge")
    config = {
        "enabled": True,
        "checkpoint": ckpt_path,
        "objectives": [{"property": "net_charge", "direction": "maximize", "weight": 1.0}],
        "schedule": {"type": "constant", "w_max": 1.0, "t_start": 0.0, "t_end": 1.0},
        "gradient_norm": "raw",  # raw gradient to test pure direction
        "gradient_clip": 0,      # no clipping
        "log_diagnostics": True,
        "device": "cpu",
    }
    guide = SteeringGuide(config)

    torch.manual_seed(42)
    z_t = torch.randn(1, 100, 8)
    v = torch.randn(1, 100, 8)
    mask = torch.ones(1, 100, dtype=torch.bool)

    guidance, diag = guide.guide(z_t, v, t_scalar=0.8, mask=mask)

    # Compute x1_est and x1_est + eps * gradient
    x1_base = z_t + (1.0 - 0.8) * v
    eps = 0.01
    x1_steered = x1_base + eps * guidance

    # Predict on both
    with torch.no_grad():
        t_ones = torch.ones(1)
        pred_base = guide.predictor.predict(x1_base, mask, t_ones)      # [1, 13]
        pred_steered = guide.predictor.predict(x1_steered, mask, t_ones)  # [1, 13]

    nc_idx = PROPERTY_TO_INDEX["net_charge"]
    nc_base = pred_base[0, nc_idx].item()
    nc_steered = pred_steered[0, nc_idx].item()
    delta = nc_steered - nc_base

    print(f"  net_charge (base):    {nc_base:.6f}")
    print(f"  net_charge (steered): {nc_steered:.6f}")
    print(f"  delta:                {delta:.6f}")
    assert delta > 0, f"Gradient direction wrong: delta={delta:.6f} should be positive"
    print("  PASSED: stepping along gradient INCREASES net_charge")

    # Also test minimize direction
    config_min = {**config, "objectives": [{"property": "net_charge", "direction": "minimize", "weight": 1.0}]}
    guide_min = SteeringGuide(config_min)
    guidance_min, _ = guide_min.guide(z_t, v, t_scalar=0.8, mask=mask)
    x1_steered_min = x1_base + eps * guidance_min
    with torch.no_grad():
        pred_steered_min = guide_min.predictor.predict(x1_steered_min, mask, t_ones)
    nc_steered_min = pred_steered_min[0, nc_idx].item()
    delta_min = nc_steered_min - nc_base
    print(f"  [minimize] delta: {delta_min:.6f}")
    assert delta_min < 0, f"Minimize direction wrong: delta={delta_min:.6f} should be negative"
    print("  PASSED: minimize direction correct")


def test_detached_output(ckpt_path: str):
    """Test 5: Guidance output has no grad_fn (safe for no_grad context)."""
    print("=" * 60)
    print("TEST 5: Guidance output is detached")
    config = {
        "enabled": True,
        "checkpoint": ckpt_path,
        "objectives": [{"property": "net_charge", "direction": "maximize", "weight": 1.0}],
        "schedule": {"type": "constant", "w_max": 1.0, "t_start": 0.0, "t_end": 1.0},
        "gradient_norm": "unit",
        "gradient_clip": 10.0,
        "log_diagnostics": False,
        "device": "cpu",
    }
    guide = SteeringGuide(config)
    z_t = torch.randn(1, 50, 8)
    v = torch.randn(1, 50, 8)
    mask = torch.ones(1, 50, dtype=torch.bool)

    guidance, _ = guide.guide(z_t, v, t_scalar=0.8, mask=mask)
    assert guidance.grad_fn is None, f"Guidance has grad_fn: {guidance.grad_fn}"
    assert not guidance.requires_grad, "Guidance should not require grad"
    print("  PASSED: no grad_fn, requires_grad=False")


def main():
    parser = argparse.ArgumentParser(description="Standalone steering module test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained predictor checkpoint. If not provided, creates a dummy.")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    tmp_dir = None
    if ckpt_path is None:
        tmp_dir = tempfile.mkdtemp()
        ckpt_path = str(Path(tmp_dir) / "dummy_fold_0_best.pt")
        print(f"No checkpoint provided — creating dummy at {ckpt_path}")
        create_dummy_checkpoint(ckpt_path)
        print("(Gradient direction test still valid: random weights have consistent gradients)\n")

    test_disabled()
    test_enabled_shape_and_nonzero(ckpt_path)
    test_schedule_gating(ckpt_path)
    test_gradient_direction(ckpt_path)
    test_detached_output(ckpt_path)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Cleanup
    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
