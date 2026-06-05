# P38_v2: Dense N x N vs sparse K-band attention masks.
# Visualizes: why the SALAD-style K=40 sparse-attention variant is O(N K)
#   rather than O(N^2). (a) dense attention attends to every (i,j) pair;
#   (b) sparse attention attends only to each row's ~40 neighbors, color-coded
#   by group -- a tight sequential band on the diagonal (width ~16), a
#   near-diagonal spatial scatter (8 nearest-3D), and a faint 1/d^3 random
#   spray reaching far off-diagonal (16). Self sits on the diagonal (slot 0).
# DATA: synthetic, deterministic. Same helix-bundle CA trace as P38_v1
#   (np.random.default_rng(0)), N=160 so K=40 is ~25% of each row. The exact
#   group sizes/rule mirror sparse_neighbors.py. No external files.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import figsize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

# --- Consistent group palette (shared with P38_v1) ---
C_SEQ = "#3b6fb6"   # sequential (blue)
C_SPA = "#e08214"   # spatial    (amber)
C_RND = "#1b9e77"   # random 1/d^3 (teal)
C_DENSE = "#4f7cb0" # uniform dense fill


def build_ca_trace(n_target, seed=0):
    """Deterministic protein-like CA trace: an antiparallel helix bundle
    whose segments fold back, so spatial/random neighbors are genuinely
    non-sequential. (Identical builder to P38_v1.)"""
    rng = np.random.default_rng(seed)
    r, rise, theta = 2.3, 1.5, 2.0 * np.pi / 3.6
    helix_len, loop_len = 16, 3
    columns = [(0.0, 0.0), (5.6, 0.6), (5.1, 5.7), (-0.4, 5.2),
               (2.6, 2.6), (8.0, 3.0), (2.8, 8.2)]
    top = rise * (helix_len - 1)
    pts = []
    direction, col_idx = 1, 0
    while len(pts) < n_target:
        cx, cy = columns[col_idx % len(columns)]
        phase = rng.uniform(0.0, 2.0 * np.pi)
        seg = []
        for k in range(helix_len):
            zz = rise * k if direction == 1 else top - rise * k
            seg.append((cx + r * np.cos(theta * k + phase),
                        cy + r * np.sin(theta * k + phase), zz))
        if pts:
            p0 = np.asarray(pts[-1]); p1 = np.asarray(seg[0])
            for t in np.linspace(0.0, 1.0, loop_len + 2)[1:-1]:
                jit = rng.normal(scale=0.3, size=3)
                pts.append(tuple(p0 + t * (p1 - p0) + jit))
        pts.extend(seg)
        direction *= -1
        col_idx += 1
    return np.asarray(pts[:n_target], dtype=float)


def neighbor_groups(i, xyz, n_seq=8, n_spatial=8, n_random=16, seed=0):
    """(seq, spatial, random) indices for query i under the K=40 scheme."""
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


# --- Build geometry + integer-coded sparse mask: 0 off, 1 seq, 2 spat, 3 rnd ---
N = 160
xyz = build_ca_trace(N + 10, seed=0)[:N]
codes = np.zeros((N, N), dtype=int)
for i in range(N):
    seq, spat, rnd = neighbor_groups(i, xyz, seed=0)
    codes[i, rnd] = 3
    codes[i, spat] = 2
    codes[i, seq] = 1
    codes[i, i] = 1  # self = slot 0 of sequential group

# --- Plot ---
fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize(1.0, 0.5))

# Panel (a): dense N x N, uniform fill.
axa.imshow(np.ones((N, N)), cmap=mcolors.ListedColormap([C_DENSE]),
           origin="upper", aspect="equal", interpolation="nearest")
axa.set_title(r"(a) Dense: $N\times N$")
axa.set_xlabel(r"residue $j$")
axa.set_ylabel(r"residue $i$")
axa.set_xticks([0, N]); axa.set_yticks([0, N])

# Panel (b): sparse, color-coded by group.
cmap = mcolors.ListedColormap(["white", C_SEQ, C_SPA, C_RND])
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
axb.imshow(codes, cmap=cmap, norm=norm, origin="upper",
           aspect="equal", interpolation="nearest")
axb.set_title(r"(b) Sparse: $N\times K$, $K{=}40$")
axb.set_xlabel(r"residue $j$")
axb.set_ylabel(r"residue $i$")
axb.set_xticks([0, N]); axb.set_yticks([0, N])

# shared group legend (proxy patches) to the right of panel (b)
handles = [
    Patch(facecolor=C_SEQ, label=r"Sequential ($\pm 8$): 16"),
    Patch(facecolor=C_SPA, label=r"Spatial (nearest-3D): 8"),
    Patch(facecolor=C_RND, label=r"Random ($\propto 1/d^{3}$): 16"),
]
axb.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
           frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.5)

fig.tight_layout()
fig.subplots_adjust(wspace=0.5)  # keep the two long panel titles from touching

fig.savefig(Path(__file__).with_suffix(".pdf"))
