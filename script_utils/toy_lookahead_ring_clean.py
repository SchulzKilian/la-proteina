"""Clean ring lambda-sweep figure for the look-ahead steering toy.

Stripped-down companion to script_utils/toy_lookahead_ring_lam_sweep.py: the
FM velocity / ramp / throttle rules are byte-identical, but the figure is
simplified for the writeup -- weights on top, method names on the side, and
none of the per-cell x2=/P= annotations or the busy suptitle.

  rows  = vanilla (constant w) / lookahead (state-gated) / schedule (w(t)=(1-t)^p)
  cols  = guidance weight lambda in {0, 2, 4, 6, 8}

Out: results/toy_lookahead_throttle/ring_clean.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
OUT = "results/toy_lookahead_throttle"
os.makedirs(OUT, exist_ok=True)

SIGMA_DATA = 0.08
N_CLOUD = 600
N_SAMPLE = 1200
NSTEPS = 150
T_EPS = 1e-3
V_MAX = 4.0
BETA_DEFAULT = 20.0

LAMS = [0.0, 2.0, 4.0, 6.0, 8.0]
P_SCHED_DEFAULT = 2.0
ARMS = ["plain", "throttle", "sched"]
ARM_LABELS = {"plain": "vanilla", "throttle": "lookahead", "sched": "schedule"}


# ---------------- ring manifold (matches main toy) ----------------
def ring_cloud(n):
    th = RNG.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(th), np.sin(th)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def ring_P(x):
    return (np.linalg.norm(x, axis=-1) - 1.0) ** 2


CLOUD = ring_cloud(N_CLOUD)


def fm_velocity(x, t, cloud):
    one_m_t = max(1.0 - t, T_EPS)
    diff = x[:, None, :] - t * cloud[None, :, :]
    logw = -(diff ** 2).sum(-1) / (2 * one_m_t ** 2)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    v = (w @ cloud - x) / one_m_t
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, V_MAX / (nrm + 1e-9))


def grad_phi_top(x):
    g = np.zeros_like(x); g[:, 1] = 1.0; return g   # phi = x2 (push up)


def ramp(t, t0=0.3, t1=0.8):
    return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def sample(arm, lam0, beta=BETA_DEFAULT, p_sched=P_SCHED_DEFAULT, n=N_SAMPLE, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)); dt = 1.0 / NSTEPS
    for i in range(NSTEPS):
        t = (i + 0.5) * dt
        v = fm_velocity(x, t, CLOUD)
        if arm != "none" and lam0 > 0:
            one_m_t = max(1.0 - t, T_EPS); g = grad_phi_top(x)
            x1 = x + one_m_t * v
            if arm == "plain":
                s = np.ones(n)
            elif arm == "sched":
                s = np.full(n, one_m_t ** p_sched)
            elif arm == "throttle":
                dP = ring_P(x1 + one_m_t * lam0 * g) - ring_P(x1)
                s = np.exp(-beta * np.maximum(0.0, dP))
            wt = ramp(t) * (s * lam0)
            v = v + wt[:, None] * g
        x = x + dt * v
    return x


# ---------------- render: rows = arms, cols = lambdas ----------------
th = np.linspace(0, 2 * np.pi, 300); curve = (np.cos(th), np.sin(th))
ncol = len(LAMS)
fig, axes = plt.subplots(len(ARMS), ncol, figsize=(2.4 * ncol, 2.5 * len(ARMS)),
                         squeeze=False)
for r, arm in enumerate(ARMS):
    for c, lam in enumerate(LAMS):
        x = sample(arm if lam > 0 else "none", lam)
        ax = axes[r][c]
        ax.plot(*curve, "k-", lw=1, alpha=0.5)
        d = np.abs(np.linalg.norm(x, axis=-1) - 1.0)
        ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.35,
                   c=d, cmap="viridis", vmin=0.0, vmax=0.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.6)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"weight = {lam:g}", fontsize=13)
        if c == 0:
            ax.set_ylabel(ARM_LABELS[arm], fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/ring_clean.png", dpi=130); plt.close(fig)
print(f"wrote {OUT}/ring_clean.png")
