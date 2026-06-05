# P38_v1: Sparse-attention K-neighbor "three-color halo" schematic.
# Visualizes: the SALAD-style K=40 sparse neighbor scheme used by the CA-only
#   sparse-attention variant. For ONE query residue on a folded synthetic CA
#   trace, draws the three neighbor groups that make up its K=40 attention set:
#   16 sequential (offsets +/-8), 8 nearest-in-3D spatial, 16 random sampled
#   with probability proportional to 1/d^3 (distance-biased, NOT uniform).
#   Self is included as slot 0 of the sequential group (highlighted query).
# DATA: synthetic, deterministic. A protein-like CA trace is built as a gently-
#   confined worm-like chain (np.random.default_rng, most-compact of several
#   seeds); the neighbor list is computed with the exact group sizes/rule from
#   sparse_neighbors.py. No external files.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Consistent group palette (shared with P38_v2) ---
C_SEQ = "#3b6fb6"   # sequential (blue)
C_SPA = "#e08214"   # spatial    (amber)
C_RND = "#1b9e77"   # random 1/d^3 (teal)
C_BB = "#b8b8b8"    # backbone / unselected residues
C_QRY = "#d62728"   # query residue


def build_ca_trace(n_target, seed=0):
    """Deterministic protein-like CA trace: a gently-confined worm-like chain.
    Constant CA-CA step + strong directional persistence keep the backbone
    locally smooth (so a query's sequential +/-8 neighbors form a coherent local
    arc along the chain), while a mild centering force makes the chain fold back
    into a globular shape (so spatial contacts and 1/d^3 random edges genuinely
    cross the fold). Of a few seeds, keep the most square-ish (least anisotropic)
    so the PCA view is a clean blob, not a rod."""
    step, R0 = 3.8, 10.5
    best, best_score = None, np.inf
    for s in range(seed, seed + 14):
        rng = np.random.default_rng(s)
        pos = np.zeros((n_target, 3))
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        for k in range(1, n_target):
            d = 0.86 * d + rng.normal(scale=0.42, size=3)      # persistence + noise
            d -= 0.30 * pos[k - 1] / R0                         # mild centering pull
            d /= np.linalg.norm(d)
            pos[k] = pos[k - 1] + step * d
        pos -= pos.mean(0)
        sv = np.linalg.svd(pos, compute_uv=False)
        rad = np.linalg.norm(pos, axis=1)
        rg = np.sqrt((rad ** 2).mean())
        # anisotropy + tail penalty (max radius >> radius-of-gyration => a tail)
        score = sv[0] / sv[1] + 0.9 * (rad.max() / rg)
        if score < best_score:
            best_score, best = score, pos
    return best


def neighbor_groups(i, xyz, n_seq=8, n_spatial=8, n_random=16, seed=0):
    """Return (seq_idx, spatial_idx, random_idx) for query i, following the
    K=40 scheme: 2*n_seq sequential (+/-n_seq), n_spatial nearest-3D among the
    rest, n_random sampled from the remaining pool with prob proportional to
    1/d^3. Self (i) is implicit slot-0 of the sequential group."""
    rng = np.random.default_rng(seed + i)
    N = len(xyz)
    d = np.linalg.norm(xyz - xyz[i], axis=1)
    seq = [j for j in range(N) if 0 < abs(j - i) <= n_seq]
    used = set(seq) | {i}
    rest = np.array([j for j in range(N) if j not in used])
    spat = rest[np.argsort(d[rest])[:n_spatial]]
    used |= set(spat.tolist())
    pool = np.array([j for j in range(N) if j not in used])
    w = 1.0 / np.clip(d[pool], 1e-6, None) ** 3
    w = w / w.sum()
    k = min(n_random, len(pool))
    rnd = rng.choice(pool, size=k, replace=False, p=w)
    return np.array(seq), np.array(spat), np.array(rnd)


def project2d(xyz):
    """Top-2 principal axes (the blob is near-isotropic, so this gives a clean,
    well-spread 2D view of the fold)."""
    c = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


# --- Build geometry; query = most-contacted residue in the middle of the chain
#     (well inside the fold, so all three neighbor groups are well populated). ---
xyz = build_ca_trace(76, seed=0)
N = len(xyz)
lo, hi = int(0.30 * N), int(0.70 * N)
contacts = np.array([
    int(np.sum((np.linalg.norm(xyz - xyz[i], axis=1) < 9.0)
               & (np.abs(np.arange(N) - i) > 8)))
    for i in range(N)])
mid = np.arange(lo, hi)
i_q = int(mid[np.argmax(contacts[mid])])

seq, spat, rnd = neighbor_groups(i_q, xyz, seed=0)
P = project2d(xyz)
q = P[i_q]

# --- Plot ---
fig, ax = plt.subplots(figsize=figsize(0.74, 0.98))
ax.plot(P[:, 0], P[:, 1], "-", color=C_BB, lw=0.8, zorder=1)
ax.scatter(P[:, 0], P[:, 1], s=12, color=C_BB, zorder=2, linewidths=0)

# edges (under nodes), colored by group
for j in rnd:
    ax.plot([q[0], P[j, 0]], [q[1], P[j, 1]], "-", color=C_RND,
            lw=0.9, alpha=0.55, zorder=1.4)
for j in spat:
    ax.plot([q[0], P[j, 0]], [q[1], P[j, 1]], "-", color=C_SPA,
            lw=1.1, zorder=1.6)
for j in seq:
    ax.plot([q[0], P[j, 0]], [q[1], P[j, 1]], "-", color=C_SEQ,
            lw=1.1, zorder=1.5)

# recolor neighbor node dots by group
ax.scatter(P[rnd, 0], P[rnd, 1], s=26, color=C_RND, zorder=3, linewidths=0)
ax.scatter(P[spat, 0], P[spat, 1], s=26, color=C_SPA, zorder=3, linewidths=0)
ax.scatter(P[seq, 0], P[seq, 1], s=26, color=C_SEQ, zorder=3, linewidths=0)

# query residue: distinct ringed star (self / slot 0)
ax.scatter(q[0], q[1], s=240, marker="*", color=C_QRY,
           edgecolors="black", linewidths=0.6, zorder=5)
ax.scatter(q[0], q[1], s=430, marker="o", facecolors="none",
           edgecolors=C_QRY, linewidths=1.0, zorder=4)
ax.annotate(r"self (slot 0)", xy=(q[0], q[1]),
            xytext=(q[0] + 3.2, q[1] + 3.4), fontsize=9,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

handles = [
    Line2D([0], [0], color=C_SEQ, marker="o", lw=1.1, markersize=5,
           label=r"Sequential ($\pm 8$): 16"),
    Line2D([0], [0], color=C_SPA, marker="o", lw=1.1, markersize=5,
           label=r"Spatial (nearest-3D): 8"),
    Line2D([0], [0], color=C_RND, marker="o", lw=0.9, markersize=5,
           label=r"Random ($\propto 1/d^{3}$): 16"),
    Line2D([0], [0], color=C_QRY, marker="*", lw=0, markersize=9,
           label=r"Query (self, slot 0)"),
]
ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
          fontsize=8, handletextpad=0.5, borderpad=0.5,
          bbox_to_anchor=(-0.02, -0.02))

# Short title — the per-group counts live in the legend, so the title need not
# carry them (the long form clipped the axes).
ax.set_title(r"$K{=}40$ sparse neighbors of one residue")
ax.set_aspect("equal")
ax.set_axis_off()
fig.tight_layout()

fig.savefig(Path(__file__).with_suffix(".pdf"))
