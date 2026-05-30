"""2D toy validating the counterfactual look-ahead throttle (E097/E098) as a
task-agnostic guidance stabiliser, on TWO manifolds.

Method mirrors the protein steering exactly:
  - Flow model: the ANALYTIC flow-matching velocity for the empirical data cloud,
        v*(x,t) = (E[x1 | x_t=x] - x)/(1-t),   E[x1|x_t] = softmax over the cloud,
    norm-capped at V_MAX (a trained net is bounded; the exact field has a 1/(1-t)
    restoring singularity that makes the manifold a perfect attractor — uncapped,
    guidance can NEVER leave it and there is nothing for a throttle to fix).
  - Steered property phi(x) = x2  ("push up"). Guidance gradient g = grad phi,
    unit-normalised (== `gradient_norm: unit`).
  - On-manifold proxy P(x): squared distance to the manifold. EVALUATED ONLY,
    never differentiated; the method never sees its closed form.
  - Throttle (== E098 `lookahead_proportional`):
        x1_hat        = x + (1-t) v                     (look-ahead clean estimate)
        x1_hat_guided = x1_hat + (1-t) lam0 g           (counterfactual)
        dP = P(x1_hat_guided) - P(x1_hat);   s = exp(-beta * relu(dP)) in (0,1]
        v_guided = v + ramp(t) * s * lam0 * g
  - ramp(t): linear 0->1 on [0.3,0.8], held at 1 after (no t_stop; throttle guards
    the tail) -- identical to E098.

Two manifolds:
  RING  : unit circle. P=(||x||-1)^2. phi=x2 climbs to the single top.
          Includes the Rg-analog CONTROL phi=-||x|| (degenerate optimum at origin).
  SINE  : irregular multi-crest curve y=f(x1)=(0.7+0.15 x1) sin(2 x1), x1 in [-3,3].
          P=(x2-f(x1))^2. Crests have DIFFERENT heights, so the feasible max-x2
          depends on WHICH crest a sample sits under -> the correct damping is purely
          spatial -> no fixed w(t) can match a state-dependent throttle.

Baselines that establish non-triviality:
  plain      : s=1, sweep lam0  == "just lower the (constant) weight" (the reviewer's
               objection: less guidance trivially stays on-manifold). The plain
               lam-sweep IS that tradeoff curve; the throttle beating it = not trivial.
  best w(t)  : s=(1-t)^p, grid over (p, lam0); Pareto envelope = best fixed schedule.
  matched-budget : plain at the constant lam that spends the SAME integral
               sum dt*ramp*s*lam0 as a throttle run -> refutes "just less guidance".

Run:  /opt/conda/bin/python script_utils/toy_lookahead_throttle.py
Out:  results/toy_lookahead_throttle/{ring,sine}_{frontier,clouds}.png,
      ring_control.png, summary.txt
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
NSTEPS = 150           # 2D ODE; nsteps=400 protein rule is about scRMSD, N/A here
T_EPS = 1e-3
V_MAX = 4.0            # bounded velocity (emulate a trained model; see header)

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
P_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]   # w(t)=(1-t)^p family  (schedule's ONE knob = p)
BETA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # throttle's ONE knob = beta
BETA_DEFAULT = 20.0                  # the single untuned value the first run used
# combo = schedule x throttle (TWO shape knobs). MUST include p=0 (pure throttle),
# b=0 (pure schedule), (0,0)=plain so the grid genuinely NESTS both single levers
# -> best_combo >= max(sched,thr) by construction; interior (p>0 AND b>0) = synergy.
P_COMBO = [0.0, 1.0, 2.0, 4.0]
B_COMBO = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
COMBO_SHAPES = [(p, b) for p in P_COMBO for b in B_COMBO]


# ---------------- manifolds ----------------
def sine_f(u):
    return (0.7 + 0.15 * u) * np.sin(2.0 * u)


def ring_cloud(n):
    th = RNG.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(th), np.sin(th)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def sine_cloud(n):
    u = RNG.uniform(-3, 3, n)
    pts = np.stack([u, sine_f(u)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def ring_P(x):
    return (np.linalg.norm(x, axis=-1) - 1.0) ** 2


def sine_P(x):
    return (x[:, 1] - sine_f(x[:, 0])) ** 2


# CLIFF: a finite horizontal segment x1 in [-1,1], x2=0, with HARD ENDS (no data
# beyond +-1). Objective phi=x1 (push right). Heterogeneous danger at fixed t
# (samples spread along the segment: right-enders are at the cliff, left-enders
# have headroom) AND brake-is-correct (no taller crest past the end -> no excursion
# benefit, unlike the sine). The faithful analog of the protein clash proxy:
# per-sample, not-a-function-of-t, and not gracefully recoverable by the flow.
def cliff_cloud(n):
    u = RNG.uniform(-1.0, 1.0, n)
    pts = np.stack([u, np.zeros(n)], 1)
    return pts + RNG.normal(0, SIGMA_DATA, pts.shape)


def cliff_P(x):
    # squared distance to the segment {(u,0): u in [-1,1]} (clamp x1, then dist)
    cx = np.clip(x[:, 0], -1.0, 1.0)
    return (x[:, 0] - cx) ** 2 + x[:, 1] ** 2


MANIFOLDS = {
    "ring": dict(cloud=ring_cloud(N_CLOUD), P=ring_P, has_control=True, mode="top"),
    "sine": dict(cloud=sine_cloud(N_CLOUD), P=sine_P, has_control=False, mode="top"),
    "cliff": dict(cloud=cliff_cloud(N_CLOUD), P=cliff_P, has_control=False, mode="right"),
}


# ---------------- analytic FM velocity ----------------
def fm_velocity(x, t, cloud):
    one_m_t = max(1.0 - t, T_EPS)
    diff = x[:, None, :] - t * cloud[None, :, :]
    logw = -(diff ** 2).sum(-1) / (2 * one_m_t ** 2)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    v = (w @ cloud - x) / one_m_t
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, V_MAX / (nrm + 1e-9))


def grad_phi(x, mode):
    if mode == "top":
        g = np.zeros_like(x); g[:, 1] = 1.0; return g          # (0,1)  phi=x2
    if mode == "right":
        g = np.zeros_like(x); g[:, 0] = 1.0; return g          # (1,0)  phi=x1
    if mode == "shrink":
        return -x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def ramp(t, t0=0.3, t1=0.8):
    return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


# ---------------- sampler ----------------
def sample(mani, arm, lam0, mode="top", beta=20.0, p_sched=2.0, n=N_SAMPLE,
           seed=1, return_budget=False):
    cloud = MANIFOLDS[mani]["cloud"]; Pfn = MANIFOLDS[mani]["P"]
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)); dt = 1.0 / NSTEPS
    budget = np.zeros(n)
    for i in range(NSTEPS):
        t = (i + 0.5) * dt
        v = fm_velocity(x, t, cloud)
        if arm != "none" and lam0 > 0:
            one_m_t = max(1.0 - t, T_EPS); g = grad_phi(x, mode)
            x1 = x + one_m_t * v
            if arm == "plain":
                s = np.ones(n)
            elif arm == "sched":
                s = np.full(n, one_m_t ** p_sched)
            elif arm == "throttle":
                dP = Pfn(x1 + one_m_t * lam0 * g) - Pfn(x1)
                s = np.exp(-beta * np.maximum(0.0, dP))
            elif arm == "combo":
                # schedule AND throttle composed (orthogonal: time-envelope x state-gate)
                dP = Pfn(x1 + one_m_t * lam0 * g) - Pfn(x1)
                s = (one_m_t ** p_sched) * np.exp(-beta * np.maximum(0.0, dP))
            wt = ramp(t) * (s * lam0)
            v = v + wt[:, None] * g
            budget += dt * wt
        x = x + dt * v
    return (x, float(budget.mean())) if return_budget else x


def metrics(mani, x, mode="top"):
    if mode == "top":
        prop = x[:, 1].mean()
    elif mode == "right":
        prop = x[:, 0].mean()
    else:  # shrink
        prop = np.linalg.norm(x, axis=-1).mean()
    return prop, MANIFOLDS[mani]["P"](x).mean()


# ---------------- run ----------------
lines = []
def log(s):
    print(s); lines.append(s)

ramp_int = sum((1.0 / NSTEPS) * ramp((i + 0.5) / NSTEPS) for i in range(NSTEPS))

def run_manifold(mani):
    log(f"\n################  MANIFOLD = {mani}  ################")
    mode = MANIFOLDS[mani]["mode"]
    plabel = "x1" if mode == "right" else "x2"   # the steered coordinate
    res = {a: [] for a in ["none", "plain", "throttle"]}
    for lam in LAMS:
        for a in ["none", "plain", "throttle"]:
            if a == "none" and lam != 0.0:
                continue
            # res["throttle"] is the DEFAULT-beta throttle (beta=20), for the
            # "single untuned knob" column.
            pr, pp = metrics(mani, sample(mani, a, lam, mode=mode, beta=BETA_DEFAULT), mode)
            res[a].append((lam, pr, pp))
    # oracle frontiers: schedule sweeps (p, lam0); throttle sweeps (beta, lam0).
    # SAME degrees of freedom (one shape-knob x one strength-knob) -> fair.
    sched_pts, thr_pts, combo_pts = [], [], []
    for lam in LAMS:
        if lam == 0.0:
            continue
        for p in P_GRID:
            pr, pp = metrics(mani, sample(mani, "sched", lam, mode=mode, p_sched=p), mode)
            sched_pts.append((pr, pp, lam, p))
        for be in BETA_GRID:
            pr, pp = metrics(mani, sample(mani, "throttle", lam, mode=mode, beta=be), mode)
            thr_pts.append((pr, pp, lam, be))
        for (p, be) in COMBO_SHAPES:   # combo: schedule x throttle, both knobs
            pr, pp = metrics(mani, sample(mani, "combo", lam, mode=mode, p_sched=p, beta=be), mode)
            combo_pts.append((pr, pp, lam, (p, be)))

    log(f"phi = {plabel} (maximise)")
    log(f"{'arm':>9} {'lam0':>5} {'mean_'+plabel:>9} {'mean_P':>10}")
    for a in ["none", "plain", "throttle"]:
        for (lam, pr, pp) in res[a]:
            log(f"{a:>9} {lam:>5.1f} {pr:>9.3f} {pp:>10.4f}")

    log(f"\nNON-TRIVIALITY: best mean_{plabel} achievable under on-manifold ceiling P<=tau")
    log("  oracle: best_w(t) sweeps p; best_thr sweeps beta; best_combo sweeps (p,beta)")
    log(f"{'P<=tau':>8} {'plain':>8} {'best_w(t)':>10} {'best_thr':>9} {'best_combo':>11} {'combo argmax':>18}")
    def bx(records, tau):
        c = [r[1] for r in records if r[2] <= tau]; return max(c) if c else float("nan")
    def bx_pts(pts, tau):
        c = [(pr, l, k) for (pr, pp, l, k) in pts if pp <= tau]
        return max(c) if c else (float("nan"), float("nan"), float("nan"))
    for tau in [0.01, 0.02, 0.05, 0.10]:
        bw = bx_pts(sched_pts, tau)[0]
        bt = bx_pts(thr_pts, tau)[0]
        bc, bc_lam, bc_k = bx_pts(combo_pts, tau)
        arg = f"p={bc_k[0]:g},b={bc_k[1]:g},lam={bc_lam:g}" if bc == bc else "-"
        log(f"{tau:>8.2f} {bx(res['plain'], tau):>8.3f} "
            f"{bw:>10.3f} {bt:>9.3f} {bc:>11.3f} {arg:>18}")

    log("\nMATCHED-BUDGET (refutes 'throttle just applies less guidance'):")
    log(f"{'lam0':>5} {'arm':>9} {'budget':>8} {'mean_'+plabel:>9} {'mean_P':>10}")
    for lam in [4.0, 8.0, 16.0]:
        xt, b = sample(mani, "throttle", lam, mode=mode, return_budget=True)
        prt, ppt = metrics(mani, xt, mode)
        lam_m = b / ramp_int
        xp, bp = sample(mani, "plain", lam_m, mode=mode, return_budget=True)
        prp, ppp = metrics(mani, xp, mode)
        log(f"{lam:>5.1f} {'throttle':>9} {b:>8.3f} {prt:>9.3f} {ppt:>10.4f}")
        log(f"{'':>5} {'plain@==':>9} {bp:>8.3f} {prp:>9.3f} {ppp:>10.4f}   (const lam0={lam_m:.2f})")

    # frontier plot
    fig, ax = plt.subplots(figsize=(6.2, 5))
    def front(pts):
        a = sorted(pts, key=lambda r: r[2]); return [r[2] for r in a], [r[1] for r in a]
    px, py = front(res["plain"]); ax.plot(px, py, "o-", color="tab:red", label="plain (s=1) = lower constant weight")
    tx, ty = front(res["throttle"]); ax.plot(tx, ty, "s:", color="tab:green", alpha=0.5, label="throttle (untuned beta=20)")
    def envelope(pts):
        eP, eX, best = [], [], -1e9
        for pr, pp, *_ in sorted(pts, key=lambda r: r[1]):
            if pr > best:
                best = pr; eP.append(pp); eX.append(pr)
        return eP, eX
    eP, eX = envelope(sched_pts)
    ax.plot(eP, eX, "^--", color="tab:blue", alpha=0.8, label="best fixed w(t)=(1-t)^p (oracle over p)")
    teP, teX = envelope(thr_pts)
    ax.plot(teP, teX, "D-", color="tab:green", label="best throttle (oracle over beta)")
    ax.scatter([res["none"][0][2]], [res["none"][0][1]], marker="*", s=180, color="k", zorder=5, label="no guidance")
    ax.set_xlabel("mean P (off-manifold ->)"); ax.set_ylabel(f"mean {plabel} (achieved property, up=better)")
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_title(f"{mani}: property vs on-manifold-ness (up-left dominates)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/{mani}_frontier.png", dpi=130); plt.close(fig)

    # clouds
    lam_show = 8.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cloud = MANIFOLDS[mani]["cloud"]
    if mani == "ring":
        th = np.linspace(0, 2 * np.pi, 300); curve = (np.cos(th), np.sin(th))
    elif mani == "sine":
        uu = np.linspace(-3, 3, 300); curve = (uu, sine_f(uu))
    else:  # cliff segment
        uu = np.linspace(-1, 1, 300); curve = (uu, np.zeros_like(uu))
    for ax, (a, ttl) in zip(axes, [("none", "no guidance"), ("plain", f"plain lam={lam_show}"), ("throttle", f"throttle lam={lam_show}")]):
        xx = sample(mani, a, 0.0 if a == "none" else lam_show, mode=mode)
        ax.plot(*curve, "k-", lw=1, alpha=0.5)
        ax.scatter(xx[:, 0], xx[:, 1], s=4, alpha=0.3, color="tab:purple")
        pr, pp = metrics(mani, xx, mode)
        ax.set_title(f"{ttl}\n{plabel}={pr:.2f} P={pp:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.suptitle(f"{mani}: phi={plabel} -> plain flies off-manifold, throttle climbs and stops on it")
    fig.tight_layout(); fig.savefig(f"{OUT}/{mani}_clouds.png", dpi=130); plt.close(fig)

    # control (ring only)
    if MANIFOLDS[mani]["has_control"]:
        log("\nCONTROL phi=-||x|| (degenerate optimum at origin; Rg-minimise analog)")
        log(f"{'arm':>9} {'lam0':>5} {'mean_r':>9} {'mean_P':>10}")
        ctrl = {a: [] for a in ["plain", "throttle"]}
        for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
            for a in ["plain", "throttle"]:
                pr, pp = metrics(mani, sample(mani, a, lam, mode="shrink"), "shrink")
                ctrl[a].append((lam, pr, pp))
                log(f"{a:>9} {lam:>5.1f} {pr:>9.3f} {pp:>10.4f}")
        fig, ax = plt.subplots(figsize=(6, 5))
        for a, c, m in [("plain", "tab:red", "o"), ("throttle", "tab:green", "s")]:
            ax.plot([r[2] for r in ctrl[a]], [r[1] for r in ctrl[a]], m + "-", color=c, label=a)
        ax.set_xlabel("mean P = mean (||x||-1)^2"); ax.set_ylabel("mean ||x|| (lower=more shrunk)")
        ax.set_title("CONTROL phi=-||x||: plain collapses off-manifold;\nthrottle stays on ring but cannot shrink")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(f"{OUT}/{mani}_control.png", dpi=130); plt.close(fig)

    return sched_pts, thr_pts, combo_pts


# ---------------- committed-shape (FAIR) comparison ----------------
# The oracle tables let each method pick its shape knob (schedule p / throttle beta)
# per-manifold per-ceiling, with hindsight -- not deployable. The FAIR comparison
# fixes ONE shape value for each method across ALL manifolds (only lambda, the
# strength knob you always set per-run, is swept), then asks how much of each
# method's own per-manifold oracle it retains. A schedule shape that wins the sine
# is wrong for the cliff; the throttle's single look-ahead rule self-adapts.
def best_at(pts, shape_val, tau):
    c = [pr for (pr, pp, lam, sh) in pts if sh == shape_val and pp <= tau]
    return max(c) if c else float("nan")


def oracle_at(pts, tau):
    c = [pr for (pr, pp, lam, sh) in pts if pp <= tau]
    return max(c) if c else float("nan")


def commit_shape(all_pts, shape_grid, tau):
    """Pick the single shape value maximising the AVERAGE over manifolds of
    best_at(shape)/oracle (fraction of that manifold's oracle retained)."""
    best_sh, best_score = None, -1.0
    for sh in shape_grid:
        fr = []
        for pts in all_pts.values():
            orc = oracle_at(pts, tau); val = best_at(pts, sh, tau)
            if orc == orc and orc > 1e-9 and val == val:
                fr.append(val / orc)
        if fr and sum(fr) / len(fr) > best_score:
            best_score, best_sh = sum(fr) / len(fr), sh
    return best_sh, best_score


log("=== TOY: look-ahead throttle on three manifolds ===")
log(f"cloud={N_CLOUD} samples={N_SAMPLE} nsteps={NSTEPS} sigma_data={SIGMA_DATA} V_MAX={V_MAX}")
ALL_SCHED, ALL_THR, ALL_COMBO = {}, {}, {}
for m in ["ring", "sine", "cliff"]:
    sp, tp, cp = run_manifold(m)
    ALL_SCHED[m], ALL_THR[m], ALL_COMBO[m] = sp, tp, cp

# fair committed-shape comparison at a representative production-like ceiling
TAU_FAIR = 0.05
p_star, p_score = commit_shape(ALL_SCHED, P_GRID, TAU_FAIR)
b_star, b_score = commit_shape(ALL_THR, BETA_GRID, TAU_FAIR)
c_star, c_score = commit_shape(ALL_COMBO, COMBO_SHAPES, TAU_FAIR)
log("\n################  COMMITTED-SHAPE FAIR COMPARISON  ################")
log(f"shape fixed ACROSS all manifolds, only lambda swept; ceiling P<={TAU_FAIR}")
log(f"committed p* = {p_star:g} (avg frac-of-oracle {p_score:.3f}); "
    f"beta* = {b_star:g} (avg frac {b_score:.3f}); "
    f"combo (p,b)* = {c_star} (avg frac {c_score:.3f})")
log("ABSOLUTE property at committed shape per manifold (oracle in parens):")
log(f"{'manifold':>9} | {'sched@p*':>9} {'thr@b*':>8} {'combo@*':>9} | "
    f"{'orc_sch':>8} {'orc_thr':>8} {'orc_combo':>9}")
agg = {k: [] for k in ["sched", "thr", "combo"]}
for m in ["ring", "sine", "cliff"]:
    osd, otr, oco = (oracle_at(ALL_SCHED[m], TAU_FAIR), oracle_at(ALL_THR[m], TAU_FAIR),
                     oracle_at(ALL_COMBO[m], TAU_FAIR))
    ss = best_at(ALL_SCHED[m], p_star, TAU_FAIR)
    ts = best_at(ALL_THR[m], b_star, TAU_FAIR)
    cs = best_at(ALL_COMBO[m], c_star, TAU_FAIR)
    log(f"{m:>9} | {ss:>9.3f} {ts:>8.3f} {cs:>9.3f} | {osd:>8.3f} {otr:>8.3f} {oco:>9.3f}")
    if osd > 1e-9: agg["sched"].append(ss / osd)
    if otr > 1e-9: agg["thr"].append(ts / otr)
    if oco > 1e-9: agg["combo"].append(cs / oco)
def _m(v): return sum(v) / len(v) if v else float("nan")
log(f"{'avgfrac':>9} | {_m(agg['sched']):>9.3f} {_m(agg['thr']):>8.3f} {_m(agg['combo']):>9.3f} | "
    f"{'(1.000)':>8} {'(1.000)':>8} {'(1.000)':>9}")
log("READ: combo nests both (beta->0 = schedule, p->0 = throttle); its oracle should")
log(">= max(sched,thr) per manifold. Question = does ONE committed (p,beta) win across all.")

with open(f"{OUT}/summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nwrote {OUT}/[ring,sine]_*.png and summary.txt")
