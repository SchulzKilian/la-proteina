"""
Fit optimal bump parameters (mu, sigma) by comparing the current sampling
schedule density to the optimal density target = ||x''||^(1/3).

Method
------
1. Reconstruct ||x''|| from the JSON (centripetal + tangential components).
2. Compute target density  rho*(t) = ||x''||^(1/3), normalised.
3. Compute current schedule density rho(t) analytically:
     power(p):  rho(t) = 1 / (p * t^((p-1)/p))   (normalised)
     log(p):    rho(t) = 1 / (ln(10)*p * (1-t))   (normalised)
4. Fit a Gaussian to the positive part of  gap = rho* - rho.
   The Gaussian centre is mu; its std is sigma.

Usage
-----
    python script_utils/fit_bump_params.py \
        --json checkpoints_laproteina/straightness_ld3.json \
        [--p1_bbca 2.0] [--p1_latents 0.5]
"""

import argparse
import json

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Load + extract
# ---------------------------------------------------------------------------

def load_bins(path: str, dm: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    # handle nested vs flat format
    if not any("/" in k for k in data):
        data = next(iter(data.values()))

    prefix = f"{dm}/{metric}_bin_"
    pairs = sorted(
        [(float(k.split("_t")[-1]), v) for k, v in data.items() if k.startswith(prefix)]
    )
    t = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    return t[1:], v[1:]   # skip t=0 outlier


def euler_error(t_sl, sl, t_lc, lc) -> tuple[np.ndarray, np.ndarray]:
    """||x''|| = sqrt( (kappa*||v||)^2 + (d||v||/dt)^2 ) on the step_length grid."""
    n = min(len(sl), len(lc))
    sl, lc, t = sl[:n], lc[:n], t_sl[:n]
    centripetal = np.where(sl > 0, lc * sl, 0.0)
    tangential  = np.gradient(sl)
    return t, np.sqrt(centripetal**2 + tangential**2)


# ---------------------------------------------------------------------------
# Schedule densities (analytical, in t-space)
# ---------------------------------------------------------------------------

def density_power(t: np.ndarray, p: float, eps: float = 1e-6) -> np.ndarray:
    """
    F(u) = u^p  =>  t = u^p  =>  u = t^(1/p)
    dt/du = p * u^(p-1)
    rho(t) = du/dt = 1 / (p * u^(p-1)) = 1 / (p * t^((p-1)/p))
    """
    rho = 1.0 / (p * np.maximum(t, eps) ** ((p - 1) / p))
    rho = np.maximum(rho, 0.0)
    return rho / np.trapz(rho, t)


def density_log(t: np.ndarray, p: float, eps: float = 1e-6) -> np.ndarray:
    """
    log schedule: t = 1 - 10^(-p*u)  (approx, after normalisation)
    dt/du = ln(10)*p * 10^(-p*u)  =>  rho(t) proportional to 1/(1-t)
    """
    rho = 1.0 / (np.log(10) * p * np.maximum(1.0 - t, eps))
    rho = np.maximum(rho, 0.0)
    return rho / np.trapz(rho, t)


# ---------------------------------------------------------------------------
# Gaussian fit
# ---------------------------------------------------------------------------

def gaussian(t, A, mu, sigma):
    return A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def fit_gap_gaussian(t: np.ndarray, gap: np.ndarray) -> tuple[float, float, float]:
    mask = gap > 0
    if mask.sum() < 5:
        raise RuntimeError("Gap has fewer than 5 positive points — no bump needed.")
    # Initial guess: peak of the positive gap
    mu0 = float(t[mask][np.argmax(gap[mask])])
    p0 = [gap[mask].max(), mu0, 0.10]
    bounds = (
        [0.0,  0.0, 0.01],   # lower: A>0, mu in [0,1], sigma>0.01
        [np.inf, 1.0, 0.50], # upper
    )
    popt, _ = curve_fit(gaussian, t[mask], gap[mask], p0=p0,
                        bounds=bounds, maxfev=20_000)
    A, mu, sigma = popt
    return float(A), float(mu), float(abs(sigma))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json",       required=True)
    parser.add_argument("--p1_bbca",    type=float, default=2.0)
    parser.add_argument("--p1_latents", type=float, default=2.0)
    args = parser.parse_args()

    configs = {
        "bb_ca":         ("log",   args.p1_bbca),
        "local_latents": ("power", args.p1_latents),   # p=2.0 per inference_base.yaml
    }

    for dm, (mode, p1) in configs.items():
        print(f"\n{'='*60}")
        print(f"  {dm}   [{mode}  p1={p1}]")
        print(f"{'='*60}")

        t_sl, sl = load_bins(args.json, dm, "step_length")
        t_lc, lc = load_bins(args.json, dm, "local_curvature")
        t_full, err_full = euler_error(t_sl, sl, t_lc, lc)

        # lc on the same grid as err
        lc_full = np.interp(t_full, t_sl[:len(lc)], lc)

        print(f"  ||x''|| peak (global) at t = {t_full[np.argmax(err_full)]:.3f}")
        print(f"  bend angle peak (global) at t = {t_full[np.argmax(lc_full)]:.3f}")
        print()

        fits = [
            ("||x''|| (row 3)", err_full,  0.45, 0.85),
            ("bend angle (row 2)", lc_full, 0.30, 0.65),
        ]

        for label, signal, win_lo, win_hi in fits:
            win = (t_full >= win_lo) & (t_full <= win_hi)
            t   = t_full[win]
            sig = signal[win]

            if mode == "power":
                current = density_power(t, p1)
            elif mode == "log":
                current = density_log(t, p1)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            target = sig ** (1.0 / 3.0)
            target = target / np.trapz(target, t)

            gap = target - current
            pos_mass = float(np.trapz(np.maximum(gap, 0), t))

            print(f"  ── {label}   window=[{win_lo}, {win_hi}] ──")
            print(f"     gap range: [{gap.min():.3f}, {gap.max():.3f}]  "
                  f"positive mass: {pos_mass:.3f}")

            try:
                A, mu_fit, sigma_fit = fit_gap_gaussian(t, gap)
                # Check if fit drifted to an edge (within 10% of window width)
                margin = 0.10 * (win_hi - win_lo)
                edge_warn = ""
                if mu_fit < win_lo + margin:
                    edge_warn = "  *** fit hit LEFT edge — result unreliable ***"
                elif mu_fit > win_hi - margin:
                    edge_warn = "  *** fit hit RIGHT edge — result unreliable ***"
                print(f"     mu={mu_fit:.4f}   sigma={sigma_fit:.4f}   A={A:.4f}{edge_warn}")
            except (RuntimeError, ValueError) as e:
                print(f"     Fit failed: {e}")
            print()


if __name__ == "__main__":
    main()
