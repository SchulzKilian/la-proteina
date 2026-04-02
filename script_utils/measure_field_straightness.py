"""
Downloads La Proteina weights if missing, then measures vector field straightness
across multiple sampling schedules.

Usage:
    python script_utils/measure_field_straightness.py
    python script_utils/measure_field_straightness.py --nsteps 200 --nsamples 16
    python script_utils/measure_field_straightness.py --ckpt_dir /custom/path --out_json out.json

What is the straightness ratio?
--------------------------------
In flow matching we transport noise x_0 to data x_T by following the ODE
    dx = v(x, t) dt  from t=0 to t=1.

If the learned vector field were perfectly optimal (i.e. OT-FM with perfectly matched
couplings), every particle would travel in a straight line from x_0 to x_T, needing
only a single Euler step. In practice the field has curvature: the model "changes its
mind" about where x_1 is as t advances, causing the trajectory to curve.

The straightness ratio measures this per trajectory:

    R = straight-line distance / total path length
      = Σ_residues ||x_T_i - x_0_i||  /  Σ_steps Σ_residues ||x_{t+dt}_i - x_t_i||

R = 1.0 → perfectly straight, one Euler step would suffice.
R < 1.0 → curved; the denominator is longer than the numerator because the path
           "wasted" displacement going sideways then correcting.

How the per-step length histogram helps ablate schedules:
---------------------------------------------------------
Each step contributes one bar to the histogram. If the field is straight in [0, 0.3]
but curved in [0.3, 0.6], a uniform schedule wastes half its budget in the easy region.
A power schedule with p < 1 densifies early steps; p > 1 densifies late steps.
The ideal schedule makes the histogram flat (equal displacement per step).
"""

import argparse
import json
import os
import subprocess
import sys
from functools import partial

import torch

root = os.path.abspath(".")
sys.path.insert(0, root)

from proteinfoundation.proteina import Proteina


# ---------------------------------------------------------------------------
# Checkpoint URLs (NVIDIA NGC)
# ---------------------------------------------------------------------------
CKPT_DIR_DEFAULT = "./checkpoints_laproteina"

# Default checkpoint filenames — overridable via CLI flags
DEFAULT_CKPT_NAME    = "LD1_ucond_notri_512.ckpt"
DEFAULT_AE_CKPT_NAME = "AE1_ucond_512.ckpt"


# ---------------------------------------------------------------------------
# Schedules to compare
# ---------------------------------------------------------------------------
SCHEDULES = {
    "uniform":    {"mode": "uniform", "p": 1.0},
    "power_0.5":  {"mode": "power",   "p": 0.5},   # more steps early (near t=0)
    "power_2.0":  {"mode": "power",   "p": 2.0},   # more steps late  (near t=1)
    "log_2.0":    {"mode": "log",     "p": 2.0},   # aggressive densification near t=0
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_if_missing(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  already present: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading -> {dest}")
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", url, "-o", dest],
        check=True,
    )


def build_sampling_model_args(data_modes, schedule_cfg):
    """
    Constructs sampling_model_args matching the structure expected by full_simulation.
    Uses `vf` (plain ODE) to avoid gt schedule noise confounding the curvature measurement.
    """
    return {
        dm: {
            "schedule": schedule_cfg,
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


def measure_schedule(model, schedule_cfg, nsamples, nres, nsteps, device):
    batch = {
        "nsamples": nsamples,
        "nres": nres,
        "mask": torch.ones(nsamples, nres, dtype=torch.bool, device=device),
        # tell the model not to use folding / inv-folding features
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
    }
    sampling_model_args = build_sampling_model_args(model.fm.data_modes, schedule_cfg)
    fn_predict = partial(model.predict_for_sampling, n_recycle=0)

    with torch.no_grad():
        _, extra_info = model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict,
            nsteps=nsteps,
            nsamples=nsamples,
            n=nres,
            self_cond=False,
            sampling_model_args=sampling_model_args,
            device=device,
            measure_straightness=True,
            # n_bins=None → one bin per step (full resolution)
        )

    return extra_info["straightness"]


def print_results(schedule_name, metrics, data_modes):
    print(f"\n{'='*60}")
    print(f"Schedule: {schedule_name}")
    print(f"{'='*60}")
    for dm in data_modes:
        ratio = metrics.get(f"{dm}/straightness_ratio", float("nan"))
        var   = metrics.get(f"{dm}/x1_pred_variance",   float("nan"))
        print(f"  [{dm}] straightness_ratio = {ratio:.4f}  |  x1_pred_variance = {var:.6f}")

        # Collect per-step lengths
        step_vals = sorted(
            [(k, v) for k, v in metrics.items()
             if k.startswith(f"{dm}/step_length_bin_")],
            key=lambda kv: kv[0],
        )
        if step_vals:
            vals = [v for _, v in step_vals]
            print(f"  [{dm}] per-step lengths (mean={sum(vals)/len(vals):.4f}, "
                  f"min={min(vals):.4f}, max={max(vals):.4f})")
            # Show a compact ASCII bar chart (20 chars wide)
            max_v = max(vals) or 1.0
            n_buckets = min(40, len(vals))
            bucket = max(1, len(vals) // n_buckets)
            bucketed = [sum(vals[i:i+bucket]) / bucket for i in range(0, len(vals), bucket)]
            bar = "".join(
                "█" if v >= 0.75 * max_v else
                "▓" if v >= 0.5  * max_v else
                "░" if v >= 0.25 * max_v else
                "·"
                for v in bucketed
            )
            print(f"  [{dm}] t=0→1: {bar}")


def main():
    parser = argparse.ArgumentParser(description="Measure vector field straightness")
    parser.add_argument("--ckpt_dir",     default=CKPT_DIR_DEFAULT,    help="Checkpoint directory")
    parser.add_argument("--ckpt_name",    default=DEFAULT_CKPT_NAME,    help="Main model filename inside ckpt_dir")
    parser.add_argument("--ae_ckpt_name", default=None,                 help="AE filename inside ckpt_dir (omit for CA-only models)")
    parser.add_argument("--nsamples",     type=int, default=8,          help="Samples per schedule")
    parser.add_argument("--nres",         type=int, default=100,        help="Protein length to evaluate")
    parser.add_argument("--nsteps",       type=int, default=200,        help="ODE steps (more = finer histogram)")
    parser.add_argument("--out_json",     default=None,                 help="Path to save full results JSON")
    parser.add_argument("--skip_download", action="store_true",         help="Skip auto-download of default LD1/AE1 (always skipped when --ckpt_name is set explicitly)")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ae_ckpt_path = os.path.join(args.ckpt_dir, args.ae_ckpt_name) if args.ae_ckpt_name else None

    # Auto-download only when using the defaults and not suppressed
    using_defaults = (args.ckpt_name == DEFAULT_CKPT_NAME)
    if using_defaults and not args.skip_download:
        print("Checking default weights (LD1 + AE1)...")
        download_if_missing(
            "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ld1_ucond_notri_512.ckpt/1.0/files?redirect=true&path=LD1_ucond_notri_512.ckpt",
            ckpt_path,
        )
        if ae_ckpt_path:
            download_if_missing(
                "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt",
                ae_ckpt_path,
            )

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    if ae_ckpt_path:
        assert os.path.exists(ae_ckpt_path), f"AE checkpoint not found: {ae_ckpt_path}"

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {args.ckpt_name} on {device} ...")
    model = Proteina.load_from_checkpoint(
        ckpt_path,
        strict=False,
        map_location=device,
        autoencoder_ckpt_path=ae_ckpt_path,
    )
    model = model.to(device)
    model.eval()
    print(f"Data modes: {model.fm.data_modes}")
    print(f"Running {len(SCHEDULES)} schedules × {args.nsamples} samples × nres={args.nres} × nsteps={args.nsteps}")

    # 3. Measure each schedule
    results = {}
    for name, sched_cfg in SCHEDULES.items():
        print(f"\nMeasuring schedule '{name}' ...")
        metrics = measure_schedule(
            model=model,
            schedule_cfg=sched_cfg,
            nsamples=args.nsamples,
            nres=args.nres,
            nsteps=args.nsteps,
            device=device,
        )
        results[name] = metrics
        print_results(name, metrics, model.fm.data_modes)

    # 4. Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY: straightness_ratio (higher = straighter trajectories)")
    print(f"{'='*60}")
    for name, metrics in results.items():
        for dm in model.fm.data_modes:
            r = metrics.get(f"{dm}/straightness_ratio", float("nan"))
            v = metrics.get(f"{dm}/x1_pred_variance",   float("nan"))
            print(f"  {name:15s}  {dm}: ratio={r:.4f}  x1_var={v:.6f}")

    # 5. Save
    if args.out_json:
        # step_length_bin_* values are per-step floats — fine to JSON-serialize directly
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {args.out_json}")


if __name__ == "__main__":
    main()
