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
# Schedule helpers (matches product_space_flow_matcher.get_schedule exactly)
# ---------------------------------------------------------------------------

def get_schedule(mode: str, nsteps: int, *, p1: float = None,
                 eps: float = 1e-5) -> np.ndarray:
    """Returns t-values [nsteps+1].  Mirrors the torch implementation."""
    if mode == "uniform":
        return np.linspace(0.0, 1.0, nsteps + 1)
    elif mode == "power":
        t = np.linspace(0.0, 1.0, nsteps + 1)
        return t ** p1
    elif mode == "log":
        # 10^(-p1) … 10^0, then flip → dense at small values
        t = 1.0 - np.logspace(-p1, 0, nsteps + 1)[::-1].copy()
        t = t - t.min()
        t = t / t.max()
        return t
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")


def coarsen_bins(t_vals: np.ndarray, lengths: np.ndarray,
                 bin_width: float = 0.025) -> tuple:
    """
    Aggregate fine-grained curvature bins into wider bins of size bin_width.
    Returns (bin_centres, mean_lengths).
    """
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(len(centres))
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (t_vals >= lo) & (t_vals < hi)
        means[j] = lengths[mask].mean() if mask.any() else 0.0
    return centres, means


def count_steps_per_bin(bin_edges: np.ndarray,
                        schedule_ts: np.ndarray) -> np.ndarray:
    """
    For each bin defined by bin_edges, count how many ODE step midpoints fall inside.
    """
    t_mids = 0.5 * (schedule_ts[:-1] + schedule_ts[1:])
    counts, _ = np.histogram(t_mids, bins=bin_edges)
    return counts.astype(float)


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


def extract_bins(metrics: dict, dm: str, metric: str) -> tuple:
    """
    Returns (t_values, values) for a given data mode and metric prefix.
    metric: 'step_length' or 'local_curvature'
    Skips the first bin (t=0 outlier for step_length).
    """
    prefix = f"{dm}/{metric}_bin_"
    pairs = [
        (float(k.split("_t")[-1]), v)
        for k, v in metrics.items()
        if k.startswith(prefix)
    ]
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort(key=lambda kv: kv[0])  # sort by float t, not key string
    t_vals = np.array([t for t, _ in pairs])
    values = np.array([v for _, v in pairs])
    return t_vals[1:], values[1:]  # skip first bin (t=0 outlier)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curvature(ax, t_vals, lengths, dm, color="#4e79a7", title=None,
                   schedule_ts=None, schedule_label=None, bin_width=0.025):
    """
    Coarsen to bin_width bins, normalise both curvature and schedule step
    counts to [0, 1], and plot on a single shared y-axis.
    """
    centres, curv = coarsen_bins(t_vals, lengths, bin_width)
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)

    # Normalise curvature
    curv_norm = curv / curv.max() if curv.max() > 0 else curv

    ax.fill_between(centres, curv_norm, alpha=0.40, color=color)
    ax.plot(centres, curv_norm, color=color, linewidth=1.2, label="curvature (norm)")

    if schedule_ts is not None:
        counts = count_steps_per_bin(edges, schedule_ts)
        counts_norm = counts / counts.max() if counts.max() > 0 else counts
        ax.bar(centres, counts_norm, width=bin_width * 0.85, alpha=0.35,
               color="#555555", align="center",
               label=schedule_label or "schedule steps / bin (norm)")

    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("normalised value [0, 1]", fontsize=9)
    ax.set_title(title or dm, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,  help="Path to straightness JSON")
    parser.add_argument("--out",  default=None,   help="Output PDF/PNG path (default: next to JSON)")
    parser.add_argument("--nsteps", type=int, default=400,
                        help="Number of ODE steps in the schedule (default: 200)")
    parser.add_argument("--p1_bbca", type=float, default=2.0,
                        help="p1 for bb_ca log schedule (default: 2.0)")
    parser.add_argument("--p1_latents", type=float, default=0.5,
                        help="p1 for local_latents power schedule (default: 0.5)")
    parser.add_argument("--bin_width", type=float, default=0.025,
                        help="Coarse bin width in t (default: 0.025 → 40 bins)")
    # kept for compat but unused now
    parser.add_argument("--nsteps_sched", type=int, default=400)
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

    # Per-mode schedule: log for bb_ca, power for local_latents
    dm_schedules = {}
    for dm in data_modes:
        if dm == "bb_ca":
            sched_ts = get_schedule("log", args.nsteps, p1=args.p1_bbca)
            label = f"log(p1={args.p1_bbca}) n={args.nsteps}"
        else:
            sched_ts = get_schedule("power", args.nsteps, p1=args.p1_latents)
            label = f"power(p1={args.p1_latents}) n={args.nsteps}"
        dm_schedules[dm] = (sched_ts, label)

    # ---- Figure: 2 rows × n_modes cols ----
    # Row 0: field speed (step_length)
    # Row 1: local curvature (bend angle between consecutive steps)
    fig, axes = plt.subplots(2, n_modes,
                             figsize=(6 * n_modes, 8),
                             constrained_layout=True)
    axes = np.array(axes).reshape(2, n_modes)  # guarantee 2-D indexing

    row_titles = [
        "Field speed  (||v||·dt, normalised) — where is the field active?",
        "Local curvature  (mean bend angle, normalised) — where does the trajectory curve?",
    ]
    metrics_keys = ["step_length", "local_curvature"]

    for row, (mkey, row_title) in enumerate(zip(metrics_keys, row_titles)):
        for col, dm in enumerate(data_modes):
            t_vals, values = extract_bins(metrics, dm, mkey)
            sched_ts, sched_label = dm_schedules[dm]
            ax = axes[row, col]
            plot_curvature(ax, t_vals, values, dm,
                           color=colors[col % len(colors)],
                           title=f"{dm}  —  {mkey.replace('_', ' ')}",
                           schedule_ts=sched_ts,
                           schedule_label=sched_label,
                           bin_width=args.bin_width)
        # row super-label via fig.text
        fig.text(0.5, 0.97 - row * 0.5, row_title,
                 ha="center", va="top", fontsize=11, style="italic")

    # ---- Save ----
    from matplotlib.backends.backend_pdf import PdfPages
    if out_path.endswith(".pdf"):
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, dpi=150)
    else:
        base = out_path.rsplit(".", 1)[0]
        fig.savefig(f"{base}_straightness.png", dpi=150)

    print(f"Saved to {out_path}")
    plt.close("all")


if __name__ == "__main__":
    main()
