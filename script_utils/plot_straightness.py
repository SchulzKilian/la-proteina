"""
Plots field curvature from a straightness JSON, and overlays sampling schedule densities.

Usage:
    python script_utils/plot_straightness.py \
        --json ./checkpoints_laproteina/straightness_ld3.json \
        --out  ./checkpoints_laproteina/straightness_ld3.pdf

Two figures are produced:

  Figure 1 — Field curvature profile (from the uniform run)
    One panel per data mode (bb_ca, local_latents).
    X-axis: time t ∈ [0, 1].
    Y-axis: mean per-residue displacement per step — i.e. ||v(x_t, t)|| · dt,
            which on a uniform grid is directly proportional to field magnitude.
    Interpretation: tall bars = the field is doing a lot of work there.

  Figure 2 — Curvature vs schedule density comparison
    Same field curvature profile (grey fill) but overlaid with the step density
    of several sampling schedules (coloured lines).
    Step density of a schedule at time t = 1 / dt(t), normalised so the area = 1.
    The ideal schedule would have its density curve perfectly matching the curvature shape.
    Gaps between a schedule's density and the curvature peak = wasted / missing steps.
"""

import argparse
import json
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath("."))


# ---------------------------------------------------------------------------
# Schedule density helpers (mirrors product_space_flow_matcher.get_schedule)
# ---------------------------------------------------------------------------

def get_schedule(mode: str, nsteps: int, p: float = 1.0) -> np.ndarray:
    """Returns t-values [nsteps+1] for a given schedule."""
    s = np.linspace(0, 1, nsteps + 1)
    if mode == "uniform":
        return s
    elif mode == "power":
        return s ** p
    elif mode == "log":
        t = 1.0 - np.logspace(-p, 0, nsteps + 1)[::-1]
        t = t - t.min()
        t = t / t.max()
        return t
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")


def schedule_density(mode: str, nsteps: int, p: float = 1.0,
                     t_grid: np.ndarray = None) -> tuple:
    """
    Returns (t_midpoints, density) where density[i] = 1/dt_i, normalised
    so the curve integrates to 1 over [0, 1].
    This is the step density: how many steps per unit time at each t.
    """
    ts = get_schedule(mode, nsteps, p)
    dt = np.diff(ts)                        # [nsteps]
    t_mid = 0.5 * (ts[:-1] + ts[1:])       # midpoint of each step
    density = 1.0 / np.clip(dt, 1e-9, None)
    density = density / (density * dt).sum()  # normalise: ∫ density dt = 1
    return t_mid, density


SCHEDULES = [
    ("uniform",  "uniform", 1.0,  "#555555", "-"),
    ("power 0.5 (dense early)", "power",   0.5,  "#e07b39", "--"),
    ("power 2.0 (dense late)",  "power",   2.0,  "#3a86e0", "--"),
    ("log 2.0",  "log",     2.0,  "#2ca02c", "-."),
]


# ---------------------------------------------------------------------------
# Parse JSON
# ---------------------------------------------------------------------------

def load_field_curvature(path: str) -> dict:
    """
    Loads the JSON and returns per-mode per-step displacement arrays.
    Handles both old format (dict of schedules) and new format (flat dict).
    """
    with open(path) as f:
        data = json.load(f)

    # New format: flat dict with keys like "bb_ca/step_length_bin_00_t0.000"
    if any("/" in k for k in data):
        return {"uniform": data}

    # Old format: {"uniform": {...}, "power_0.5": {...}, ...}
    return data


def extract_steps(metrics: dict, dm: str) -> tuple:
    """Returns (t_values, step_lengths) arrays for a given data mode."""
    keys = sorted(
        [(k, v) for k, v in metrics.items() if k.startswith(f"{dm}/step_length_bin_")],
        key=lambda kv: kv[0]
    )
    if not keys:
        return np.array([]), np.array([])
    t_vals = np.array([float(k.split("_t")[-1]) for k, _ in keys])
    lengths = np.array([v for _, v in keys])
    return t_vals, lengths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curvature(ax, t_vals, lengths, dm, color="#4e79a7", title=None):
    ax.fill_between(t_vals, lengths, alpha=0.35, color=color)
    ax.plot(t_vals, lengths, color=color, linewidth=0.8)
    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("mean displacement per step\n(∝ ||v(x_t,t)||·dt)", fontsize=9)
    ax.set_title(title or dm, fontsize=12)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # Annotate 80% mass interval
    total = lengths.sum()
    if total > 0:
        cumsum = np.cumsum(lengths)
        i_lo = np.searchsorted(cumsum, 0.10 * total)
        i_hi = np.searchsorted(cumsum, 0.90 * total)
        t_lo = t_vals[min(i_lo, len(t_vals)-1)]
        t_hi = t_vals[min(i_hi, len(t_vals)-1)]
        ax.axvspan(t_lo, t_hi, alpha=0.12, color="red",
                   label=f"80% of displacement\nt∈[{t_lo:.2f},{t_hi:.2f}]")
        ax.legend(fontsize=8, loc="upper right")


def plot_comparison(ax, t_vals, lengths, dm, nsteps=500):
    """Field curvature (grey fill) + schedule density lines."""
    # Normalise field curvature to integrate to 1 for fair comparison
    dt_uniform = t_vals[1] - t_vals[0] if len(t_vals) > 1 else 1.0
    norm = (lengths * dt_uniform).sum()
    lengths_norm = lengths / max(norm, 1e-9)

    ax.fill_between(t_vals, lengths_norm, alpha=0.2, color="grey",
                    label="field curvature (ideal density)")
    ax.plot(t_vals, lengths_norm, color="grey", linewidth=1.0, linestyle="-")

    for label, mode, p, color, ls in SCHEDULES:
        t_mid, density = schedule_density(mode, nsteps, p)
        ax.plot(t_mid, density, color=color, linestyle=ls,
                linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("normalised density", fontsize=9)
    ax.set_title(f"{dm} — schedule density vs field curvature", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,  help="Path to straightness JSON")
    parser.add_argument("--out",  default=None,   help="Output PDF/PNG path (default: next to JSON)")
    parser.add_argument("--nsteps_sched", type=int, default=500,
                        help="Resolution for schedule density curves")
    args = parser.parse_args()

    out_path = args.out or args.json.replace(".json", ".pdf")

    all_data = load_field_curvature(args.json)
    # Use the uniform run
    metrics = all_data.get("uniform", next(iter(all_data.values())))

    data_modes = sorted(set(
        k.split("/")[0] for k in metrics
        if "/step_length_bin_" in k
    ))

    if not data_modes:
        print("No per-step data found in JSON. Re-run with --out_json to regenerate.")
        sys.exit(1)

    n_modes = len(data_modes)
    colors = ["#4e79a7", "#e05c5c"]

    # ---- Figure 1: field curvature profiles ----
    fig1, axes1 = plt.subplots(1, n_modes, figsize=(6 * n_modes, 4),
                               constrained_layout=True)
    if n_modes == 1:
        axes1 = [axes1]
    fig1.suptitle("Vector field curvature profile (uniform ODE, schedule-agnostic)",
                  fontsize=13)

    for i, dm in enumerate(data_modes):
        t_vals, lengths = extract_steps(metrics, dm)
        plot_curvature(axes1[i], t_vals, lengths, dm, color=colors[i % len(colors)])

    # ---- Figure 2: curvature vs schedule density ----
    fig2, axes2 = plt.subplots(1, n_modes, figsize=(6 * n_modes, 4),
                               constrained_layout=True)
    if n_modes == 1:
        axes2 = [axes2]
    fig2.suptitle("Where should schedules focus? Field curvature vs schedule density",
                  fontsize=13)

    for i, dm in enumerate(data_modes):
        t_vals, lengths = extract_steps(metrics, dm)
        plot_comparison(axes2[i], t_vals, lengths, dm, nsteps=args.nsteps_sched)

    # ---- Save ----
    from matplotlib.backends.backend_pdf import PdfPages
    if out_path.endswith(".pdf"):
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig1, dpi=150)
            pdf.savefig(fig2, dpi=150)
    else:
        # Save as two pngs
        base = out_path.rsplit(".", 1)[0]
        fig1.savefig(f"{base}_curvature.png", dpi=150)
        fig2.savefig(f"{base}_schedule_vs_curvature.png", dpi=150)

    print(f"Saved to {out_path}")
    plt.close("all")


if __name__ == "__main__":
    main()
