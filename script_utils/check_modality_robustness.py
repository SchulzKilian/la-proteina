#!/usr/bin/env python3
"""Modality robustness check for Finding 9 sub-claim (b).

Replicates the kde_modality function from compare_properties.py
(scipy gaussian_kde + Scott bandwidth + 5%-of-peak prominence filter)
across three populations:

  1. PDB at full n=56,008
  2. AFDB at n=5,000 (the E026 reference set)
  3. PDB subsampled to AFDB's n (20 replicates, no replacement)

For each of shannon_entropy and swi, prints the detected mode count
under each population. (3) tells you whether the PDB modality is robust
to subsampling or just a high-n KDE detection effect.

Outcome (run 2026-05-03):
  shannon_entropy: PDB full=2; AFDB=1; PDB@5K all 20 replicates = 2.
                   → real population difference, not n artifact.
  swi:             PDB full=2; AFDB=1; PDB@5K all 20 replicates = 1.
                   → PDB bimodality is high-n only; no real signal at n=5K.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


def kde_modality(x: np.ndarray, grid: int = 512, prominence_frac: float = 0.05) -> int:
    """Identical to compare_properties.kde_modality."""
    x = x[np.isfinite(x)]
    if x.size < 5 or np.allclose(x, x[0]):
        return 1
    try:
        kde = stats.gaussian_kde(x)
    except (np.linalg.LinAlgError, ValueError):
        return 1
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, grid)
    ys = kde(xs)
    peaks, _ = find_peaks(ys, prominence=prominence_frac * ys.max())
    return max(1, len(peaks))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    pdb_csv = repo / "laproteina_steerability/data/properties.csv"
    afdb_csv = repo / "data/afdb_ref/properties_afdb_refschema.csv"

    pdb = pd.read_csv(pdb_csv)
    afdb = pd.read_csv(afdb_csv)
    pdb = pdb[(pdb["sequence_length"] >= 300) & (pdb["sequence_length"] <= 800)]
    afdb = afdb[(afdb["sequence_length"] >= 300) & (afdb["sequence_length"] <= 800)]
    print(f"PDB n (length-filtered):  {len(pdb)}")
    print(f"AFDB n (length-filtered): {len(afdb)}")
    print()

    rng = np.random.default_rng(0)
    n_replicates = 20

    for prop in ("shannon_entropy", "swi"):
        pdb_arr = pdb[prop].dropna().to_numpy()
        afdb_arr = afdb[prop].dropna().to_numpy()
        pdb_full_modes = kde_modality(pdb_arr)
        afdb_modes = kde_modality(afdb_arr)
        sub_modes = []
        for _ in range(n_replicates):
            sub = rng.choice(pdb_arr, size=len(afdb_arr), replace=False)
            sub_modes.append(kde_modality(sub))
        cnt = Counter(sub_modes)
        print(f"{prop}:")
        print(f"  PDB full   (n={len(pdb_arr)})  modes = {pdb_full_modes}; "
              f"mean={pdb_arr.mean():.4f}  sd={pdb_arr.std():.4f}")
        print(f"  AFDB       (n={len(afdb_arr)})   modes = {afdb_modes}; "
              f"mean={afdb_arr.mean():.4f}  sd={afdb_arr.std():.4f}")
        print(f"  PDB @ n={len(afdb_arr)} ({n_replicates} replicates): {dict(cnt)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
