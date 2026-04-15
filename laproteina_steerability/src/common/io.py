"""I/O utilities: save figures, tables, and run metadata."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def save_figure(
    fig: plt.Figure,
    name: str,
    figures_dir: str | Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save a matplotlib figure to disk in one or more formats.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    name : str
        Base filename without extension.
    figures_dir : str or Path
        Directory for output figures.
    dpi : int
        Resolution for raster formats.
    formats : list[str]
        File extensions to save (default: ["png", "pdf"]).

    Returns
    -------
    list[Path]
        Paths to all saved files.
    """
    if formats is None:
        formats = ["png", "pdf"]
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = figures_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", p)
        paths.append(p)
    return paths


def save_table(
    df: pd.DataFrame,
    name: str,
    tables_dir: str | Path,
    fmt: str = "csv",
    include_index: bool = False,
) -> Path:
    """Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    name : str
        Base filename without extension.
    tables_dir : str or Path
        Directory for output tables.
    fmt : str
        "csv" or "parquet".
    include_index : bool
        If True, write the DataFrame index as a column.

    Returns
    -------
    Path
        Path to saved file.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    p = tables_dir / f"{name}.{fmt}"
    if fmt == "csv":
        df.to_csv(p, index=include_index)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    logger.info("Saved table: %s", p)
    return p


def _get_git_hash() -> str | None:
    """Return current git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def write_run_metadata(
    output_dir: str | Path,
    config: dict[str, Any],
    dataset_size: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write run metadata JSON.

    Parameters
    ----------
    output_dir : str or Path
        Directory for the metadata file.
    config : dict
        The full config dict used for this run.
    dataset_size : int
        Number of proteins loaded.
    extra : dict, optional
        Additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the written metadata file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
        "dataset_size": dataset_size,
        "config": config,
    }
    if extra:
        meta.update(extra)
    p = output_dir / "run_metadata.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Saved run metadata: %s", p)
    return p
