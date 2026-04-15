"""Shared plotting style and utilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl


def apply_style() -> None:
    """Apply a clean, publication-ready matplotlib style.

    Call once at the start of a run. All subsequent figures inherit these
    settings.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


# Standard colormaps
CMAP_DIVERGING = "RdBu_r"
CMAP_SEQUENTIAL = "viridis"
CMAP_CORRELATION = "coolwarm"
