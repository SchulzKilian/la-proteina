"""
Reads a straightness JSON and finds the optimal power_with_middle_bump parameters
for each data mode by minimising the expected Euler integration error.

Theory
------
For a 1st-order ODE integrator (Euler), the local error at step i is O(dt_i^2 * ||x''||).
Given a fixed number of steps N, the error-optimal schedule allocates dt(t) ∝ ||x''(t)||^(1/3)
(classic equidistribution principle).

This script:
  1. Loads per-step ||v|| (step_length) and κ (local_curvature) from the JSON.
  2. Reconstructs ||x''|| = sqrt( (κ||v||)^2 + (d||v||/dt)^2 ).
  3. Computes the target step density  ρ*(t) ∝ ||x''(t)||^(1/3).
  4. For each mode, optimises the schedule parameters so the schedule's
     step density  ρ(t) = 1/dt(t)  best matches ρ* in an L2 sense.
  5. Reports parameters and the expected error ratio E_optimised / E_baseline.

Usage
-----
    python script_utils/optimise_bump.py \\
        --json checkpoints_laproteina/straightness_ld3.json \\
        [--nsteps 400] \\
        [--p1_bbca 2.0] [--p1_latents 0.5]
"""

import argparse
import json
import sys

import numpy as np
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Handle old (nested) vs new (flat) format
    if any("/" in k for k in data):
        return data
    return next(iter(data.values()))


def extract_bins(metrics: dict, dm: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_values, values), skipping the t=0 outlier bin."""
    prefix = f"{dm}/{metric}_bin_"
    pairs = [
        (float(k.split("_t")[-1]), v)
        for k, v in metrics.items()
        if k.startswith(prefix)
    ]
    pairs.sort(key=lambda kv: kv[0])
    t = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    return t[1:], v[1:]   # skip first bin (t≈0 outlier)


# ---------------------------------------------------------------------------
# Error density
# ---------------------------------------------------------------------------

def euler_error_density(t_sl, sl, t_lc, lc) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, ||x''||) on the step_length grid.

    centripetal component  = κ · ||v||      (path bending)
    tangential  component  = d||v||/dt      (speed change)
    total                  = sqrt(centripetal^2 + tangential^2)
    """
    n = min(len(sl), len(lc))
    sl = sl[:n]; lc = lc[:n]; t = t_sl[:n]

    centripetal = np.where(sl > 0, lc * sl, 0.0)

    d_sl = np.empty(n)
    d_sl[0]    = sl[1]  - sl[0]
    d_sl[-1]   = sl[-1] - sl[-2]
    d_sl[1:-1] = 0.5 * (sl[2:] - sl[:-2])

    total = np.sqrt(centripetal**2 + d_sl**2)
    return t, total


def optimal_density(error: np.ndarray) -> np.ndarray:
    """
    Equidistribution principle: allocate dt ∝ error^(-1/3),
    so step density ρ = 1/dt ∝ error^(1/3).
    Returned normalised so it sums to 1 (treat as a probability distribution).
    """
    rho = error ** (1.0 / 3.0)
    rho = np.maximum(rho, 0.0)
    s = rho.sum()
    return rho / s if s > 0 else rho


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def make_schedule(mode: str, nsteps: int, p1: float,
                  bump_mu: float = 0.45, bump_sigma: float = 0.08,
                  bump_eps: float = 0.10) -> np.ndarray:
    """Returns t-values array of length nsteps+1, matching product_space_flow_matcher."""
    u = np.linspace(0.0, 1.0, nsteps + 1)

    if mode == "power":
        return u ** p1

    elif mode == "log":
        t = 1.0 - np.logspace(-p1, 0, nsteps + 1)[::-1].copy()
        t = t - t.min()
        t = t / t.max()
        return t

    elif mode in ("power_with_middle_bump", "log_with_middle_bump"):
        if mode == "power_with_middle_bump":
            F_base = u ** p1
        else:
            t_log = 1.0 - np.logspace(-p1, 0, nsteps + 1)[::-1].copy()
            t_log = t_log - t_log.min()
            F_base = t_log / t_log.max()

        raw_bump = np.exp(-((u - bump_mu) ** 2) / (2.0 * bump_sigma ** 2))
        bump = raw_bump - ((1 - u) * raw_bump[0] + u * raw_bump[-1])
        F_unnorm = F_base + bump_eps * bump

        if not np.all(np.diff(F_unnorm) >= -1e-10):
            return None   # non-monotone; caller treats as invalid

        F = (F_unnorm - F_unnorm[0]) / (F_unnorm[-1] - F_unnorm[0])
        return F

    else:
        raise ValueError(f"Unknown mode: {mode}")


def schedule_density(ts: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """
    Given ODE timesteps ts (length nsteps+1), compute step density on the
    same grid as t_target: for each bin in t_target count how many step
    midpoints land there, then normalise.
    """
    t_mids = 0.5 * (ts[:-1] + ts[1:])
    n = len(t_target)
    # Map each midpoint to the nearest bin in t_target
    counts = np.zeros(n)
    indices = np.searchsorted(t_target, t_mids, side="right") - 1
    indices = np.clip(indices, 0, n - 1)
    np.add.at(counts, indices, 1)
    s = counts.sum()
    return counts / s if s > 0 else counts


# ---------------------------------------------------------------------------
# Expected error given a schedule and an error density
# ---------------------------------------------------------------------------

def expected_error(ts: np.ndarray, t_target: np.ndarray,
                   error: np.ndarray) -> float:
    """
    E = Σ_i  error(t_i) * dt_i^2   (Euler local truncation error, summed)
    Interpolates the continuous error onto the step midpoints.
    """
    dt = np.diff(ts)
    t_mids = 0.5 * (ts[:-1] + ts[1:])
    err_at_mids = np.interp(t_mids, t_target, error)
    return float(np.sum(err_at_mids * dt ** 2))


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def optimise_bump(mode_base: str, p1: float, nsteps: int,
                  t_target: np.ndarray, error: np.ndarray,
                  rho_star: np.ndarray,
                  verbose: bool = True) -> dict:
    """
    Optimise bump_mu, bump_sigma, bump_eps to minimise expected Euler error.
    Falls back to L2 density matching if expected_error is flat (e.g. very smooth field).

    Returns dict with keys: mode, p1, bump_mu, bump_sigma, bump_eps,
                             error_baseline, error_optimised, error_ratio.
    """
    mode_bump = mode_base.replace("power", "power_with_middle_bump") \
                         .replace("log",   "log_with_middle_bump")
    if "with_middle_bump" not in mode_bump:
        mode_bump = mode_base + "_with_middle_bump"

    # Baseline (no bump)
    ts_base = make_schedule(mode_base, nsteps, p1)
    E_base  = expected_error(ts_base, t_target, error)

    def objective(params):
        mu, sigma, eps = params
        if sigma < 0.01 or eps < 0.0:
            return 1e9
        ts = make_schedule(mode_bump, nsteps, p1,
                           bump_mu=mu, bump_sigma=sigma, bump_eps=eps)
        if ts is None:
            return 1e9   # non-monotone
        return expected_error(ts, t_target, error)

    # --- global search with differential_evolution ---
    bounds = [
        (0.20, 0.80),   # bump_mu
        (0.03, 0.25),   # bump_sigma
        (0.01, 0.80),   # bump_eps
    ]
    result_global = differential_evolution(
        objective, bounds,
        seed=42, maxiter=300, tol=1e-6,
        popsize=15, polish=True,
    )
    mu_opt, sigma_opt, eps_opt = result_global.x

    ts_opt = make_schedule(mode_bump, nsteps, p1,
                           bump_mu=mu_opt, bump_sigma=sigma_opt, bump_eps=eps_opt)
    E_opt  = expected_error(ts_opt, t_target, error)

    if verbose:
        print(f"    baseline  E = {E_base:.6f}")
        print(f"    optimised E = {E_opt:.6f}  ({100*(1-E_opt/E_base):.1f}% reduction)")

    return {
        "mode_base":       mode_base,
        "mode_bump":       mode_bump,
        "p1":              p1,
        "bump_mu":         float(mu_opt),
        "bump_sigma":      float(sigma_opt),
        "bump_eps":        float(eps_opt),
        "error_baseline":  E_base,
        "error_optimised": E_opt,
        "error_ratio":     E_opt / E_base,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_results(dm: str, res: dict, t_target: np.ndarray, error: np.ndarray,
                  nsteps: int) -> None:
    print()
    print("=" * 64)
    print(f"  {dm}")
    print("=" * 64)

    # Where is 80% of the error?
    total = error.sum()
    cumsum = np.cumsum(error)
    t_lo = t_target[np.searchsorted(cumsum, 0.10 * total)]
    t_hi = t_target[np.searchsorted(cumsum, 0.90 * total)]
    print(f"  80% of Euler error lives in  t ∈ [{t_lo:.2f}, {t_hi:.2f}]")
    print(f"  Peak error at               t ≈ {t_target[np.argmax(error)]:.2f}")
    print()
    print(f"  Baseline schedule : {res['mode_base']}  p1={res['p1']}")
    print(f"  Optimised schedule: {res['mode_bump']}")
    print()
    print(f"    p1        = {res['p1']}")
    print(f"    bump_mu   = {res['bump_mu']:.4f}")
    print(f"    bump_sigma= {res['bump_sigma']:.4f}")
    print(f"    bump_eps  = {res['bump_eps']:.4f}")
    print()
    print(f"  Expected error: {res['error_baseline']:.6f}  →  {res['error_optimised']:.6f}")
    print(f"  Improvement   : {100*(1 - res['error_ratio']):.1f}%")
    print()

    # Config snippet
    print("  ── Config snippet ──────────────────────────────────")
    print(f"  schedule:")
    print(f"    mode: {res['mode_bump']}")
    print(f"    p: {res['p1']}")
    print(f"    bump_mu: {res['bump_mu']:.4f}")
    print(f"    bump_sigma: {res['bump_sigma']:.4f}")
    print(f"    bump_eps: {res['bump_eps']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json",         required=True, help="Path to straightness JSON")
    parser.add_argument("--nsteps",       type=int,   default=400,
                        help="ODE step budget (default: 400)")
    parser.add_argument("--p1_bbca",      type=float, default=2.0,
                        help="Base p1 for bb_ca log schedule (default: 2.0)")
    parser.add_argument("--p1_latents",   type=float, default=0.5,
                        help="Base p1 for local_latents power schedule (default: 0.5)")
    parser.add_argument("--mode_bbca",    default="log",
                        help="Base schedule mode for bb_ca (default: log)")
    parser.add_argument("--mode_latents", default="power",
                        help="Base schedule mode for local_latents (default: power)")
    args = parser.parse_args()

    metrics = load_json(args.json)

    data_modes = sorted(set(
        k.split("/")[0] for k in metrics
        if "/step_length_bin_" in k
    ))
    print(f"Data modes found: {data_modes}")
    print(f"Step budget: {args.nsteps}")

    mode_cfg = {
        "bb_ca":          (args.mode_bbca,    args.p1_bbca),
        "local_latents":  (args.mode_latents, args.p1_latents),
    }

    for dm in data_modes:
        t_sl, sl = extract_bins(metrics, dm, "step_length")
        t_lc, lc = extract_bins(metrics, dm, "local_curvature")

        if len(sl) == 0 or len(lc) == 0:
            print(f"\n[{dm}] Missing data — skipping.")
            continue

        t_err, err = euler_error_density(t_sl, sl, t_lc, lc)
        rho_star   = optimal_density(err)

        mode_base, p1 = mode_cfg.get(dm, ("power", 0.5))
        print(f"\n[{dm}]  base={mode_base}  p1={p1}  — optimising bump parameters ...")

        res = optimise_bump(mode_base, p1, args.nsteps,
                            t_err, err, rho_star, verbose=True)
        print_results(dm, res, t_err, err, args.nsteps)


if __name__ == "__main__":
    main()
