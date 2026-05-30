"""Audit: on the SINE manifold, is the schedule's property win (mean x2~0.41 at
mean-P<=0.05) a genuine on-manifold redistribution onto the tall crests, or is it
a mean-P artifact -- a tail of samples flung UP and OFF the curve that inflates
mean-x2 while the bulk stays on?  Plots the winning schedule cloud (never plotted
by the original script) next to the winning throttle cloud, and prints the
per-sample P distribution (mean is gameable; median + off-manifold fraction are not).
Reuses the EXACT cloud-construction order of toy_lookahead_throttle.py so numbers match.
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


def ring_cloud(n):  # consumed first, to reproduce RNG stream exactly
    th = RNG.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(th), np.sin(th)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def sine_cloud(n):
    u = RNG.uniform(-3, 3, n)
    pts = np.stack([u, sine_f(u)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


_ = ring_cloud(N_CLOUD)            # burn the ring draw (faithful order)
CLOUD = sine_cloud(N_CLOUD)


def sine_P(x):
    return (x[:, 1] - sine_f(x[:, 0])) ** 2


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
    g = np.zeros((n, 2)); g[:, 1] = 1.0           # phi = x2, push up
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
                dP = sine_P(x1 + one_m_t * lam0 * g) - sine_P(x1)
                s = np.exp(-beta * np.maximum(0.0, dP))
            v = v + (ramp(t) * (s * lam0))[:, None] * g
        x = x + dt * v
    return x


def pstats(x, tag):
    P = sine_P(x); x2 = x[:, 1]
    on = P < 0.02                       # tight on-manifold
    print(f"{tag:>22} | meanX2 {x2.mean():6.3f} | meanP {P.mean():6.4f} "
          f"medP {np.median(P):6.4f} p90 {np.quantile(P,0.9):6.4f} "
          f"| frac P>0.05 {(P>0.05).mean():4.2f} P>0.2 {(P>0.2).mean():4.2f} "
          f"| meanX2|on-manifold {x2[on].mean():6.3f} (n_on {on.sum()})")
    return P, x2


# ---- find each method's winning operating point at mean-P <= 0.05 ----
TAU = 0.05
best_s = (-9, None); best_t = (-9, None)
for lam in LAMS:
    if lam == 0:
        continue
    for p in P_GRID:
        x = sample("sched", lam, p_sched=p)
        if sine_P(x).mean() <= TAU and x[:, 1].mean() > best_s[0]:
            best_s = (x[:, 1].mean(), ("sched", lam, p), x)
    for be in BETA_GRID:
        x = sample("throttle", lam, beta=be)
        if sine_P(x).mean() <= TAU and x[:, 1].mean() > best_t[0]:
            best_t = (x[:, 1].mean(), ("throttle", lam, be), x)

print(f"=== SINE audit (mean-P ceiling {TAU}) ===")
print(f"winning schedule : {best_s[1]}")
print(f"winning throttle : {best_t[1]}")
print()
Ps, X2s = pstats(best_s[2], f"sched {best_s[1][1:]}")
Pt, X2t = pstats(best_t[2], f"throttle {best_t[1][1:]}")

# ---- plot the two winning clouds side by side ----
uu = np.linspace(-3, 3, 400)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, (x, lbl, opp) in zip(axes, [
        (best_s[2], f"WINNING SCHEDULE {best_s[1][1:]}", best_s),
        (best_t[2], f"WINNING THROTTLE {best_t[1][1:]}", best_t)]):
    P = sine_P(x)
    ax.plot(uu, sine_f(uu), "k-", lw=1.2, alpha=0.6, zorder=1)
    sc = ax.scatter(x[:, 0], x[:, 1], c=P, s=8, alpha=0.6, cmap="inferno",
                    vmin=0, vmax=0.2, zorder=2)
    ax.set_title(f"{lbl}\nmeanX2={x[:,1].mean():.3f}  meanP={P.mean():.4f}  "
                 f"medP={np.median(P):.4f}  frac off(P>0.05)={ (P>0.05).mean():.2f}",
                 fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
fig.colorbar(sc, ax=axes, label="per-sample P (off-manifold)", fraction=0.025)
fig.suptitle("SINE: did the schedule push samples UP-and-OFF (bright tail above the curve), "
             "or genuinely onto the tall crests?", fontsize=11)
fig.savefig(f"{OUT}/sine_schedule_vs_throttle_audit.png", dpi=130,
            bbox_inches="tight")
print(f"\nwrote {OUT}/sine_schedule_vs_throttle_audit.png")
