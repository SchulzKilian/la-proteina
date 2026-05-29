"""
Compare the trained `rel_seq_sep` "embedding" between a sparse and a dense
CA-only checkpoint.

`rel_seq_sep` is NOT a learnable nn.Embedding. It is a one-hot binning of the
integer sequence offset (i - j) into `seq_sep_dim=127` buckets, concatenated
with the other pair features, and fed into a single Linear projection inside
the `PairReprBuilder` (see proteinfoundation/nn/feature_factory.py:1379,2014).

Since rel_seq_sep is feature 0 in `feats_pair_repr` for both the canonical
dense (configs/nn/ca_only_score_nn_160M.yaml) and the sparse variant
(configs/nn/ca_only_sparse_160M.yaml), the trained "embedding for offset k"
is exactly column k of `linear_out.weight[:, 0:127]` (shape [256, 127]).
Transposed, E has shape [127, 256] with row k = the projection vector that
the one-hot-bucket-k signal adds to the pair representation.

Offset mapping (with seq_sep_dim=127, bin_limits = linspace(-62.5, 62.5, 126)):
  row 0        : saturation bin, offset <= -63
  rows 1..125  : integer offset (row - 63)         (i.e. row 63 == offset 0)
  row 126      : saturation bin, offset >= +63

Five plots produced (saved to notes/figures/rel_seq_sep/):
  (a) norm_per_offset.png       - ||E[k]||_2 vs signed offset
  (b) local_smoothness.png      - cos(E[k], E[k+1]) vs offset
  (c) symmetry.png              - cos(E[k], E[-k]) vs |offset|
  (d) diff_heatmap.png          - heatmap of (E_sparse - E_dense)
  (e) dist_from_init.png        - ||E_x[k] - E_init[k]||_2 (if init available)

This is pure weight inspection. No model is instantiated, no inference is run.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR = REPO_ROOT / "notes" / "figures" / "rel_seq_sep"

# rel_seq_sep is feature 0 in both ca_only_score_nn_160M.yaml and
# ca_only_sparse_160M.yaml, so column slice is fixed.
WEIGHT_KEY = "nn.pair_repr_builder.init_repr_factory.linear_out.weight"
SEQ_SEP_DIM = 127  # both configs


def load_rel_seq_sep_embedding(ckpt_path: Path):
    """Return (E [seq_sep_dim, pair_repr_dim], meta dict)."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    if WEIGHT_KEY not in state:
        raise KeyError(
            f"{WEIGHT_KEY} not in {ckpt_path}. "
            f"Available keys touching pair_repr_builder: "
            f"{[k for k in state if 'pair_repr_builder' in k]}"
        )
    W = state[WEIGHT_KEY].float()  # [pair_repr_dim, sum_feat_dims]
    pair_repr_dim, total = W.shape
    assert total >= SEQ_SEP_DIM, (
        f"linear_out has only {total} input dims, < seq_sep_dim={SEQ_SEP_DIM}"
    )
    # rel_seq_sep is feature 0 -> first 127 columns
    E = W[:, :SEQ_SEP_DIM].T.contiguous()  # [127, 256]
    meta = {
        "path": str(ckpt_path),
        "global_step": int(ckpt.get("global_step", -1)),
        "epoch": int(ckpt.get("epoch", -1)),
        "pair_repr_dim": int(pair_repr_dim),
        "total_pair_feat_dim": int(total),
        "seq_sep_dim": int(SEQ_SEP_DIM),
    }
    return E, meta


def init_embedding_like(E_ref: torch.Tensor, total_in_dim: int, seed: int = 0):
    """
    Reproduce nn.Linear's default kaiming_uniform_(a=sqrt(5)) init for the
    full [pair_repr_dim, total_in_dim] weight, then slice the first 127 cols.
    PyTorch default for nn.Linear (no bias): U(-bound, +bound) with
    bound = 1 / sqrt(fan_in) (kaiming_uniform with a=sqrt(5), gain=sqrt(2/(1+5))).
    """
    pair_repr_dim, seq_sep_dim = E_ref.shape[1], E_ref.shape[0]
    g = torch.Generator().manual_seed(seed)
    W_init = torch.empty(pair_repr_dim, total_in_dim)
    torch.nn.init.kaiming_uniform_(W_init, a=5 ** 0.5, generator=g)
    return W_init[:, :seq_sep_dim].T.contiguous()  # [127, 256]


# ---------------- metrics ----------------

def norms(E):
    return torch.linalg.vector_norm(E, dim=1)  # [seq_sep_dim]


def local_cosine(E):
    a = E[:-1]
    b = E[1:]
    return torch.nn.functional.cosine_similarity(a, b, dim=1)  # [seq_sep_dim-1]


def symmetric_cosine(E):
    """
    cos(E[63 + d], E[63 - d]) for d in 0..63.
    Row 63 is offset 0; rows 1..125 are signed integer offsets; rows 0/126 are
    saturation bins. We restrict to d in 1..62 (within the integer-offset
    region, excluding saturation).
    """
    center = (E.shape[0] - 1) // 2  # 63
    d_max = center  # 63
    out = []
    for d in range(0, d_max + 1):
        if center - d < 0 or center + d >= E.shape[0]:
            continue
        c = torch.nn.functional.cosine_similarity(
            E[center + d].unsqueeze(0), E[center - d].unsqueeze(0)
        ).item()
        out.append((d, c))
    return out  # list of (|d|, cos)


def dist_per_offset(E, E_ref):
    return torch.linalg.vector_norm(E - E_ref, dim=1)  # [seq_sep_dim]


# ---------------- offset axis ----------------

def signed_offset_axis(seq_sep_dim=SEQ_SEP_DIM):
    """
    Maps row index -> signed integer offset for plotting.
    rows 1..125 -> -62..62
    rows 0 and 126 -> saturation bins; plotted at -63 and +63 with a marker.
    """
    axis = np.arange(seq_sep_dim) - (seq_sep_dim - 1) // 2  # -63..63
    return axis


# ---------------- plotting ----------------

def plot_norm_per_offset(E_sparse, E_dense, E_init, out_path):
    x = signed_offset_axis()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, norms(E_dense).numpy(), label="dense", color="C0", lw=1.5)
    ax.plot(x, norms(E_sparse).numpy(), label="sparse K=40", color="C3", lw=1.5)
    if E_init is not None:
        ax.plot(
            x,
            norms(E_init).numpy(),
            label="init (default Linear)",
            color="0.5",
            lw=1.0,
            ls="--",
        )
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("signed sequence offset (row 0 / 126 = saturation bins at ±63)")
    ax.set_ylabel("||E[k]||_2")
    ax.set_title("(a) Per-offset L2 norm of rel_seq_sep projection vectors")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_local_smoothness(E_sparse, E_dense, E_init, out_path):
    # cos(E[k], E[k+1]) plotted against (signed offset at row k)
    x = signed_offset_axis()[:-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, local_cosine(E_dense).numpy(), label="dense", color="C0", lw=1.5)
    ax.plot(x, local_cosine(E_sparse).numpy(), label="sparse K=40", color="C3", lw=1.5)
    if E_init is not None:
        ax.plot(
            x,
            local_cosine(E_init).numpy(),
            label="init",
            color="0.5",
            lw=1.0,
            ls="--",
        )
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("signed offset (row k)")
    ax.set_ylabel("cos( E[k], E[k+1] )")
    ax.set_title("(b) Local smoothness — high = embedding varies smoothly in offset")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_symmetry(E_sparse, E_dense, E_init, out_path):
    sd = symmetric_cosine(E_dense)
    ss = symmetric_cosine(E_sparse)
    si = symmetric_cosine(E_init) if E_init is not None else None
    xs = [d for d, _ in sd]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, [c for _, c in sd], label="dense", color="C0", lw=1.5)
    ax.plot(xs, [c for _, c in ss], label="sparse K=40", color="C3", lw=1.5)
    if si is not None:
        ax.plot(xs, [c for _, c in si], label="init", color="0.5", lw=1.0, ls="--")
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("|offset| d  (cos between E[+d] and E[-d])")
    ax.set_ylabel("cos( E[+d], E[-d] )")
    ax.set_title("(c) Directional symmetry of the offset embedding")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_diff_heatmap(E_sparse, E_dense, out_path):
    diff = (E_sparse - E_dense).numpy()
    vmax = float(np.abs(diff).max())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    extent = [0, diff.shape[1], -63.5, 63.5]  # y axis is signed offset
    im = ax.imshow(
        diff,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("embedding dim (0..pair_repr_dim-1)")
    ax.set_ylabel("signed sequence offset")
    ax.set_title("(d) E_sparse - E_dense  (diverging colormap, centred at 0)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("element value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dist_from_init(E_sparse, E_dense, E_init, out_path):
    x = signed_offset_axis()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, dist_per_offset(E_dense, E_init).numpy(),
            label="||E_dense - E_init||", color="C0", lw=1.5)
    ax.plot(x, dist_per_offset(E_sparse, E_init).numpy(),
            label="||E_sparse - E_init||", color="C3", lw=1.5)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("signed sequence offset")
    ax.set_ylabel("L2 distance to init")
    ax.set_title("(e) How far each offset has moved from default-Linear init")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------- summary stats ----------------

def divergence_summary(E_sparse, E_dense):
    """Per-offset L2 norm of (sparse - dense), broken by offset region."""
    x = signed_offset_axis()
    delta = torch.linalg.vector_norm(E_sparse - E_dense, dim=1).numpy()
    regions = {
        "all (rows 0..126)": (np.full_like(x, True, dtype=bool)),
        "near-zero |off|<=4": np.abs(x) <= 4,
        "mid 5<=|off|<=20": (np.abs(x) >= 5) & (np.abs(x) <= 20),
        "large 21<=|off|<=62": (np.abs(x) >= 21) & (np.abs(x) <= 62),
        "saturation |off|>=63": np.abs(x) >= 63,
        "positive offsets (>0, not sat)": (x > 0) & (x < 63),
        "negative offsets (<0, not sat)": (x < 0) & (x > -63),
    }
    out = {}
    for name, mask in regions.items():
        if mask.sum() == 0:
            continue
        out[name] = {
            "mean_delta": float(delta[mask].mean()),
            "max_delta": float(delta[mask].max()),
            "argmax_signed_offset": int(x[mask][int(np.argmax(delta[mask]))]),
        }
    return out


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sparse-ckpt",
        default=str(REPO_ROOT / "sparse_K40_step1259.ckpt"),
        help="Path to sparse K=40 plain checkpoint",
    )
    parser.add_argument(
        "--dense-ckpt",
        default=str(REPO_ROOT / "baseline_wd0.05_step2646.ckpt"),
        help="Path to dense canonical checkpoint",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument(
        "--init-seed", type=int, default=0,
        help="Seed for reproducing default Linear init for E_init.",
    )
    parser.add_argument(
        "--no-init", action="store_true",
        help="Skip the init reference and the (e) plot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] dense ckpt:  {args.dense_ckpt}")
    E_dense, meta_dense = load_rel_seq_sep_embedding(Path(args.dense_ckpt))
    print(f"       step={meta_dense['global_step']}  epoch={meta_dense['epoch']}  "
          f"E shape={tuple(E_dense.shape)}  total_feat_dim={meta_dense['total_pair_feat_dim']}")

    print(f"[load] sparse ckpt: {args.sparse_ckpt}")
    E_sparse, meta_sparse = load_rel_seq_sep_embedding(Path(args.sparse_ckpt))
    print(f"       step={meta_sparse['global_step']}  epoch={meta_sparse['epoch']}  "
          f"E shape={tuple(E_sparse.shape)}  total_feat_dim={meta_sparse['total_pair_feat_dim']}")

    assert E_dense.shape == E_sparse.shape, (
        f"Shape mismatch: dense {E_dense.shape} vs sparse {E_sparse.shape}"
    )
    assert (
        meta_dense["total_pair_feat_dim"] == meta_sparse["total_pair_feat_dim"]
    ), "Pair feature dim disagrees — feature order may not match; aborting."

    if args.no_init:
        E_init = None
        print("[init] skipped (--no-init)")
    else:
        E_init = init_embedding_like(
            E_dense, total_in_dim=meta_dense["total_pair_feat_dim"], seed=args.init_seed
        )
        print(f"[init] reconstructed default-Linear init (seed={args.init_seed}), "
              f"shape={tuple(E_init.shape)}")

    # offset convention sanity print
    print("\n[offset convention]")
    print("  seq_sep_dim = 127, bin_limits = linspace(-62.5, 62.5, 126)")
    print("  row   0      -> saturation, offset <= -63")
    print("  row   k=1..125 -> integer offset (k - 63), so row 63 == offset 0")
    print("  row 126      -> saturation, offset >= +63")
    print("  signed offset axis used in plots: arange(127) - 63 -> -63..63 "
          "(endpoints are the saturation bins)\n")

    # plots
    paths = {
        "norm_per_offset": out_dir / "norm_per_offset.png",
        "local_smoothness": out_dir / "local_smoothness.png",
        "symmetry": out_dir / "symmetry.png",
        "diff_heatmap": out_dir / "diff_heatmap.png",
    }
    plot_norm_per_offset(E_sparse, E_dense, E_init, paths["norm_per_offset"])
    plot_local_smoothness(E_sparse, E_dense, E_init, paths["local_smoothness"])
    plot_symmetry(E_sparse, E_dense, E_init, paths["symmetry"])
    plot_diff_heatmap(E_sparse, E_dense, paths["diff_heatmap"])
    if E_init is not None:
        paths["dist_from_init"] = out_dir / "dist_from_init.png"
        plot_dist_from_init(E_sparse, E_dense, E_init, paths["dist_from_init"])

    # divergence summary
    summary = divergence_summary(E_sparse, E_dense)
    print("[divergence regions]  ||E_sparse - E_dense||_2 per row, grouped:")
    for name, stats in summary.items():
        print(f"  {name:34s} mean={stats['mean_delta']:.4f}  "
              f"max={stats['max_delta']:.4f}  argmax_off={stats['argmax_signed_offset']:+d}")

    # dump machine-readable summary alongside the plots
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dense": meta_dense,
                "sparse": meta_sparse,
                "weight_key": WEIGHT_KEY,
                "seq_sep_dim": SEQ_SEP_DIM,
                "divergence_by_region": summary,
                "init_seed": None if args.no_init else args.init_seed,
            },
            indent=2,
        )
    )

    print(f"\n[saved plots]")
    for k, p in paths.items():
        print(f"  {k:24s} -> {p.relative_to(REPO_ROOT)}")
    print(f"  summary                  -> {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
