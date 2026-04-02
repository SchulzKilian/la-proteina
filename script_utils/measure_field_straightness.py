"""
Measures the intrinsic curvature of the learned vector field, independent of
sampling schedule. Uses a fine uniform grid (many steps) so the per-step
displacement profile accurately reflects the continuous-time field.

This is the right tool for ablating *why* certain sampling schedules work:
  1. Run this once to get the field curvature profile.
  2. The ideal schedule allocates more steps to t-bins with high displacement.
  3. Compare that ideal to uniform/power/log — the gap explains performance differences.

Usage:
    python script_utils/measure_field_straightness.py \\
        --ckpt_dir ./checkpoints_laproteina \\
        --ckpt_name LD3_ucond_notri_800.ckpt \\
        --ae_ckpt_name AE2_ucond_800.ckpt \\
        --nres 400 --nsteps 500 --nsamples 8

Straightness ratio:
    R = Σ_residues ||x_T - x_0|| / Σ_steps Σ_residues ||x_{t+dt} - x_t||
    R=1.0 → perfectly straight ODE (one step would suffice).
    R<1.0 → curved; particles moved sideways and corrected, wasting displacement.
    The ratio is a scalar summary. The per-step bar chart shows WHERE the curvature is.
"""

import argparse
import json
import os
import sys
from functools import partial

import torch

root = os.path.abspath(".")
sys.path.insert(0, root)

from proteinfoundation.proteina import Proteina


CKPT_DIR_DEFAULT    = "./checkpoints_laproteina"
DEFAULT_CKPT_NAME   = "LD1_ucond_notri_512.ckpt"
DEFAULT_AE_CKPT_NAME = "AE1_ucond_512.ckpt"


def build_sampling_model_args(data_modes):
    """
    Plain ODE (vf, no noise injection) with uniform schedule.
    Uniform is the only schedule that maps steps linearly to t, so the
    per-step displacement directly reflects the field magnitude at that t.
    """
    return {
        dm: {
            "schedule": {"mode": "uniform", "p": 1.0},
            "gt": {"mode": "1-t/t", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "vf",
                "sc_scale_noise": 0.0,
                "sc_scale_score": 0.0,
                "t_lim_ode": 0.99,
                "t_lim_ode_below": 0.01,
                "center_every_step": dm == "bb_ca",
            },
        }
        for dm in data_modes
    }


def run_simulation(model, nsamples, nres, nsteps, device):
    batch = {
        "nsamples": nsamples,
        "nres": nres,
        "mask": torch.ones(nsamples, nres, dtype=torch.bool, device=device),
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
    }
    fn_predict = partial(model.predict_for_sampling, n_recycle=0)
    with torch.no_grad():
        _, extra_info = model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict,
            nsteps=nsteps,
            nsamples=nsamples,
            n=nres,
            self_cond=False,
            sampling_model_args=build_sampling_model_args(model.fm.data_modes),
            device=device,
            measure_straightness=True,  # n_bins=None → one entry per step
        )
    return extra_info["straightness"]


def print_report(metrics, data_modes, nsteps):
    for dm in data_modes:
        ratio = metrics.get(f"{dm}/straightness_ratio", float("nan"))
        var   = metrics.get(f"{dm}/x1_pred_variance",   float("nan"))

        step_vals = [v for k, v in sorted(metrics.items())
                     if k.startswith(f"{dm}/step_length_bin_")]

        print(f"\n{'='*64}")
        print(f"  {dm}")
        print(f"{'='*64}")
        print(f"  straightness_ratio : {ratio:.4f}  (1.0 = perfectly straight)")
        print(f"  x1_pred_variance   : {var:.6f}  (low = model commits to endpoint early)")
        print(f"  nsteps             : {nsteps}")

        if step_vals:
            mean_v = sum(step_vals) / len(step_vals)
            max_v  = max(step_vals) or 1.0
            print(f"  per-step displacement  mean={mean_v:.4f}  min={min(step_vals):.4f}  max={max(step_vals):.4f}")

            # ASCII bar at full resolution (compress to 64 chars)
            n_chars = 64
            bucket = max(1, len(step_vals) // n_chars)
            bucketed = [
                sum(step_vals[i:i+bucket]) / bucket
                for i in range(0, len(step_vals), bucket)
            ]
            bar = "".join(
                "█" if v >= 0.75 * max_v else
                "▓" if v >= 0.50 * max_v else
                "░" if v >= 0.25 * max_v else
                "·"
                for v in bucketed
            )
            print(f"  t=0 {'─'*3}> t=1")
            print(f"  |{bar}|")
            print()

            # Find the t-range covering 80% of total displacement
            total = sum(step_vals)
            cumsum = 0.0
            t_start = t_end = None
            for i, v in enumerate(step_vals):
                cumsum += v
                if t_start is None and cumsum >= 0.1 * total:
                    t_start = i / len(step_vals)
                if t_end is None and cumsum >= 0.9 * total:
                    t_end = i / len(step_vals)
                    break
            print(f"  80% of displacement happens in t ∈ [{t_start:.2f}, {t_end:.2f}]")
            print(f"  → ideal schedule should concentrate steps in that interval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",     default=CKPT_DIR_DEFAULT)
    parser.add_argument("--ckpt_name",    default=DEFAULT_CKPT_NAME)
    parser.add_argument("--ae_ckpt_name", default=None,
                        help="AE filename (omit for CA-only models)")
    parser.add_argument("--nsamples",     type=int, default=8,
                        help="Batch size. More = lower-variance estimates.")
    parser.add_argument("--nres",         type=int, default=100,
                        help="Protein length.")
    parser.add_argument("--nsteps",       type=int, default=500,
                        help="ODE steps for the curvature profile. 500+ gives a "
                             "smooth histogram. NFEs = nsteps (one NN call per step).")
    parser.add_argument("--out_json",     default=None)
    args = parser.parse_args()

    ckpt_path    = os.path.join(args.ckpt_dir, args.ckpt_name)
    ae_ckpt_path = os.path.join(args.ckpt_dir, args.ae_ckpt_name) if args.ae_ckpt_name else None

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    if ae_ckpt_path:
        assert os.path.exists(ae_ckpt_path), f"AE checkpoint not found: {ae_ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.ckpt_name} on {device} ...")
    model = Proteina.load_from_checkpoint(
        ckpt_path, strict=False, map_location=device,
        autoencoder_ckpt_path=ae_ckpt_path,
    ).to(device).eval()

    print(f"Data modes : {model.fm.data_modes}")
    print(f"nsamples={args.nsamples}  nres={args.nres}  nsteps={args.nsteps}")
    print(f"NFEs (NN calls): {args.nsteps}  (one per Euler step, batch of {args.nsamples})")

    metrics = run_simulation(model, args.nsamples, args.nres, args.nsteps, device)
    print_report(metrics, model.fm.data_modes, args.nsteps)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nFull per-step data saved to {args.out_json}")


if __name__ == "__main__":
    main()
