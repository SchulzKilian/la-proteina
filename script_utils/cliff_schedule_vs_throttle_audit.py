"""Cliff counterpart of sine_schedule_vs_throttle_audit.py (E106). The cliff is
the regime where the THROTTLE beats the schedule (E104: thr 1.170 > best_w(t)
1.117 at P<=0.05, gap widens at tight ceilings). Same per-sample audit so the
winning case is shown in the identical visual language to the losing (sine) case:
plot the winning-schedule cloud next to the winning-throttle cloud, colored by
per-sample off-manifold distance P, with the distribution stats.

Cliff: segment {x1 in [-1,1], x2=0}, hard ends. Objective phi=x1 (push RIGHT).
P = (x1 - clip(x1,+-1))^2 + x2^2. The schedule must push late to drag laggards
right, which launches right-enders OFF the end (x1>1); a scalar w(t) cannot say
"brake this sample, push that one" -> the throttle wins. Reproduces E104's exact
cloud RNG stream (ring, then sine, then cliff drawn from default_rng(0)).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
OUT = "results/toy_lookahead_throttle"
os.makedirs(OUT, exist_ok=True)
SIGMA_DATA = 0.08; N_CLOUD = 600; N_SAMPLE = 1200; NSTEPS = 150
T_EPS = 1e-3; V_MAX = 4.0
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
P_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
BETA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def sine_f(u):
    return (0.7 + 0.15 * u) * np.sin(2.0 * u)


# burn ring + sine draws first to reproduce the cliff RNG stream exactly
def _ring(n):
    th = RNG.uniform(0, 2 * np.pi, n); pts = np.stack([np.cos(th), np.sin(th)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def _sine(n):
    u = RNG.uniform(-3, 3, n); pts = np.stack([u, sine_f(u)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def _cliff(n):
    u = RNG.uniform(-1.0, 1.0, n); pts = np.stack([u, np.zeros(n)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


_ = _ring(N_CLOUD); _ = _sine(N_CLOUD)
CLOUD = _cliff(N_CLOUD)


def cliff_P(x):
    cx = np.clip(x[:, 0], -1.0, 1.0)
    return (x[:, 0] - cx) ** 2 + x[:, 1] ** 2


def fm_velocity(x, t):
    one_m_t = max(1.0 - t, T_EPS)
    diff = x[:, None, :] - t * CLOUD[None, :, :]
    logw = -(diff ** 2).sum(-1) / (2 * one_m_t ** 2)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    v = (w @ CLOUD - x) / one_m_t
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, V_MAX / (nrm + 1e-9))


def ramp(t, t0=0.3, t1=0.8):
    return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def sample(arm, lam0, beta=20.0, p_sched=2.0, n=N_SAMPLE, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)); dt = 1.0 / NSTEPS
    g = np.zeros((n, 2)); g[:, 0] = 1.0           # phi = x1, push RIGHT
    for i in range(NSTEPS):
        t = (i + 0.5) * dt
        v = fm_velocity(x, t)
        if arm != "none" and lam0 > 0:
            one_m_t = max(1.0 - t, T_EPS)
            x1 = x + one_m_t * v
            if arm == "plain":
                s = np.ones(n)
            elif arm == "sched":
                s = np.full(n, one_m_t ** p_sched)
            elif arm == "throttle":
                dP = cliff_P(x1 + one_m_t * lam0 * g) - cliff_P(x1)
                s = np.exp(-beta * np.maximum(0.0, dP))
            v = v + (ramp(t) * (s * lam0))[:, None] * g
        x = x + dt * v
    return x


def pstats(x, tag):
    P = cliff_P(x); x1 = x[:, 0]; on = P < 0.02
    print(f"{tag:>22} | meanX1 {x1.mean():6.3f} | meanP {P.mean():6.4f} "
          f"medP {np.median(P):6.4f} p90 {np.quantile(P,0.9):6.4f} "
          f"| frac P>0.05 {(P>0.05).mean():4.2f} P>0.2 {(P>0.2).mean():4.2f} "
          f"| frac past end (x1>1) {(x1>1.0).mean():4.2f} "
          f"| meanX1|on {x1[on].mean():6.3f} (n_on {on.sum()})")


TAU = 0.05
best_s = (-9, None, None); best_t = (-9, None, None)
for lam in LAMS:
    if lam == 0:
        continue
    for p in P_GRID:
        x = sample("sched", lam, p_sched=p)
        if cliff_P(x).mean() <= TAU and x[:, 0].mean() > best_s[0]:
            best_s = (x[:, 0].mean(), ("sched", lam, p), x)
    for be in BETA_GRID:
        x = sample("throttle", lam, beta=be)
        if cliff_P(x).mean() <= TAU and x[:, 0].mean() > best_t[0]:
            best_t = (x[:, 0].mean(), ("throttle", lam, be), x)

print(f"=== CLIFF audit (mean-P ceiling {TAU}) ===")
print(f"winning schedule : {best_s[1]}")
print(f"winning throttle : {best_t[1]}\n")
pstats(best_s[2], f"sched {best_s[1][1:]}")
pstats(best_t[2], f"throttle {best_t[1][1:]}")

uu = np.linspace(-1, 1, 200)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
for ax, (x, lbl) in zip(axes, [
        (best_s[2], f"WINNING SCHEDULE {best_s[1][1:]}"),
        (best_t[2], f"WINNING THROTTLE {best_t[1][1:]}")]):
    P = cliff_P(x)
    ax.plot(uu, np.zeros_like(uu), "k-", lw=2.0, alpha=0.7, zorder=1)
    ax.axvline(1.0, color="r", ls="--", lw=1, alpha=0.5, zorder=1)  # the cliff edge
    sc = ax.scatter(x[:, 0], x[:, 1], c=P, s=8, alpha=0.6, cmap="inferno",
                    vmin=0, vmax=0.3, zorder=2)
    ax.set_title(f"{lbl}\nmeanX1={x[:,0].mean():.3f}  meanP={P.mean():.4f}  "
                 f"medP={np.median(P):.4f}  past-edge(x1>1)={(x[:,0]>1).mean():.2f}",
                 fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlabel("x1  (push right ->;  red dashed = cliff edge)")
axes[0].set_ylabel("x2")
fig.colorbar(sc, ax=axes, label="per-sample P (off-manifold)", fraction=0.025)
fig.suptitle("CLIFF (throttle's winning regime): schedule launches right-enders OFF "
             "the edge (x1>1, bright); throttle brakes them, climbs further within bounds",
             fontsize=11)
fig.savefig(f"{OUT}/cliff_schedule_vs_throttle_audit.png", dpi=130, bbox_inches="tight")
print(f"\nwrote {OUT}/cliff_schedule_vs_throttle_audit.png")
