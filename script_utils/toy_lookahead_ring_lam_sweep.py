"""Ring-only lambda sweep for the look-ahead throttle toy (companion to
script_utils/toy_lookahead_throttle.py).

Motivation: the main toy's clouds plot fixes a single guidance strength
(lam_show=8.0), which can look "too harsh -> off manifold" without showing the
*progression*. This script sweeps lam0 across a grid and renders the ring cloud at
each strength, for the `plain` (constant weight) and `throttle` arms side by side,
so you can SEE the strength at which plain leaves the ring and whether the throttle
holds on-manifold as lam0 grows.

Everything (FM velocity, ramp, throttle rule) is byte-identical to the main toy;
this only changes the visualisation (lam sweep instead of a single value).

Out: results/toy_lookahead_throttle/ring_lam_sweep.png
     results/toy_lookahead_throttle/ring_lam_sweep_metrics.txt
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

# the lambda grid to sweep -- finer near the off-manifold onset
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
P_SCHED_DEFAULT = 2.0   # w(t)=(1-t)^p schedule shape (the schedule arm's one knob)
ARMS = ["plain", "throttle", "sched"]


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


def metrics(x):
    return x[:, 1].mean(), ring_P(x).mean()   # mean x2, mean off-manifold P


# ---------------- render: rows = arms, cols = lambdas ----------------
th = np.linspace(0, 2 * np.pi, 300); curve = (np.cos(th), np.sin(th))
ncol = len(LAMS)
fig, axes = plt.subplots(len(ARMS), ncol, figsize=(2.5 * ncol, 2.7 * len(ARMS)),
                         squeeze=False)
lines = ["=== RING lambda sweep (phi=x2, push up) ==="]
lines.append(f"cloud={N_CLOUD} samples={N_SAMPLE} nsteps={NSTEPS} V_MAX={V_MAX} beta={BETA_DEFAULT}")
lines.append(f"{'arm':>9} {'lam0':>5} {'mean_x2':>9} {'mean_P':>10}")
for r, arm in enumerate(ARMS):
    for c, lam in enumerate(LAMS):
        x = sample(arm if lam > 0 else "none", lam)
        x2, P = metrics(x)
        lines.append(f"{arm:>9} {lam:>5.1f} {x2:>9.3f} {P:>10.4f}")
        ax = axes[r][c]
        ax.plot(*curve, "k-", lw=1, alpha=0.5)
        # colour by off-manifold-ness so excursions are obvious
        d = np.abs(np.linalg.norm(x, axis=-1) - 1.0)
        ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.35,
                   c=d, cmap="viridis", vmin=0.0, vmax=0.5)
        ax.set_title(f"lam={lam:g}\nx2={x2:.2f} P={P:.3f}", fontsize=8)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.6)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            rlabel = f"sched\n(p={P_SCHED_DEFAULT:g})" if arm == "sched" else arm
            ax.set_ylabel(rlabel, fontsize=11)
fig.suptitle("Ring: guidance strength sweep -- plain (constant w) drifts off-ring as lam grows; "
             "throttle (state-gated) and sched (w(t)=(1-t)^p, p="
             f"{P_SCHED_DEFAULT:g}) hold on-manifold\n(point colour = |dist to ring|; "
             "higher x2 with low P = better)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/ring_lam_sweep.png", dpi=130); plt.close(fig)

with open(f"{OUT}/ring_lam_sweep_metrics.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT}/ring_lam_sweep.png and ring_lam_sweep_metrics.txt")
