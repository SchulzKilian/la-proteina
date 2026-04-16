"""Flow-matching noising: linear interpolation between noise and clean data.

Convention (matches La-Proteina codebase):
    z_t = (1 - t) * noise + t * z_clean

So t=0 is pure noise, t=1 is clean data.

WARNING: The steering predictor training uses t in [0.3, 0.8]. It has NOT
been verified whether that range uses the same t-convention as this code.
Every output from this module logs the t-convention prominently. If results
look inverted, check whether the steering code uses 1-t.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def make_noised_latents(
    z_clean: np.ndarray,
    t: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate noised latent samples at interpolation time t.

    Parameters
    ----------
    z_clean : np.ndarray
        Clean latent vectors, shape ``[L, D]`` float32.
    t : float
        Interpolation time in [0, 1]. t=0 is pure noise, t=1 is clean.
    n_samples : int
        Number of independent noise draws. For t=1.0, returns n_samples
        copies of z_clean (all identical).
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noised latents, shape ``[n_samples, L, D]`` float32.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")

    L, D = z_clean.shape

    if t == 1.0:
        # Clean data — no noise
        return np.broadcast_to(z_clean[np.newaxis], (n_samples, L, D)).copy()

    # Draw independent noise for each sample
    noise = rng.standard_normal((n_samples, L, D)).astype(np.float32)

    # Linear interpolation: z_t = (1-t)*noise + t*z_clean
    z_t = (1.0 - t) * noise + t * z_clean[np.newaxis]

    return z_t.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    z = rng.standard_normal((50, 8)).astype(np.float32)

    for t_val in [0.0, 0.3, 0.5, 0.8, 1.0]:
        noised = make_noised_latents(z, t_val, n_samples=4, rng=rng)
        print(f"t={t_val:.1f}: shape={noised.shape}, "
              f"mean_norm={np.linalg.norm(noised, axis=-1).mean():.3f}")
