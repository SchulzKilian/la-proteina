"""Stage-1 calibration report for the score-magnitude look-ahead throttle.

Reads the per-step diagnostics dumped by run_geom_lookahead_calibration and, per
proxy {geom, A, B@each t_p} and per target, reports the ΔP and r distributions
overall and in early/mid/late t-bins, plus B's sensitivity to t_p. It then
proposes a STARTING β per proxy so the throttle s=f(·) maps usefully into (0,1]:
    median active step -> s≈0.7,   90th-pct active step -> s≈0.2.
β is reported for both the relative input r (scale-free, transfers across proxies)
and the absolute input ΔP (what the current guide code consumes).

    python -m steering.analyze_geom_lookahead_calibration

NOTE: Stage 1 has no designability eval, so it CANNOT know whether the active
(ΔP>0) steps are the steps where designability is actually at risk. Treat every β
here as a starting value only — the real β is decided in Stage 2 against the
designability frontier.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
CALIB = _ROOT / "results" / "geom_lookahead_calib"

# s = exp(-β·x);  s=0.7 -> β·x=0.357 ;  s=0.2 -> β·x=1.609
LN_07 = -math.log(0.7)   # 0.3567
LN_02 = -math.log(0.2)   # 1.6094

T_BINS = [("early", 0.30, 0.467), ("mid", 0.467, 0.633), ("late", 0.633, 0.801)]


def _proxy_names(diag_step: dict) -> list[str]:
    names = set()
    for k in diag_step:
        m = re.match(r"cal_(.+)_r$", k)
        if m:
            names.add(m.group(1))
    # order: geom, A, then B by t_p
    def key(n):
        if n == "geom": return (0, 0.0)
        if n == "A": return (1, 0.0)
        mm = re.match(r"B_tp([\d.]+)", n)
        return (2, float(mm.group(1)) if mm else 9.0)
    return sorted(names, key=key)


def _collect(steps, proxy):
    """Return arrays (t, dP, r) over all logged steps for one proxy."""
    t, dP, r = [], [], []
    for d in steps:
        if d.get("skipped"):
            continue
        rk = f"cal_{proxy}_r"
        if rk not in d:
            continue
        t.append(d["t"]); dP.append(d[f"cal_{proxy}_dP"]); r.append(d[rk])
    return np.array(t), np.array(dP), np.array(r)


def _q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def _summary(arr_active):
    if len(arr_active) == 0:
        return dict(n=0, med=float("nan"), p90=float("nan"), p99=float("nan"), mx=float("nan"))
    return dict(n=len(arr_active), med=float(np.median(arr_active)),
                p90=_q(arr_active, 0.90), p99=_q(arr_active, 0.99), mx=float(arr_active.max()))


def _beta(med, p90):
    b_med = LN_07 / med if med and med > 0 else float("nan")
    b_p90 = LN_02 / p90 if p90 and p90 > 0 else float("nan")
    return b_med, b_p90


def main():
    targets = sorted(p.name for p in CALIB.glob("*") if p.is_dir())
    if not targets:
        print(f"No calibration diagnostics under {CALIB}")
        return

    # Load all steps per target.
    per_target_steps: dict[str, list] = {}
    for tgt in targets:
        steps = []
        for jf in sorted((CALIB / tgt).glob("*_diagnostics.json")):
            steps.extend(json.load(open(jf)))
        per_target_steps[tgt] = steps

    sample = next((s for st in per_target_steps.values() for s in st
                   if not s.get("skipped") and any(k.startswith("cal_") for k in s)), None)
    if sample is None:
        print("No calibration steps found (cal_* keys absent).")
        return
    proxies = _proxy_names(sample)

    print("=" * 78)
    print("STAGE-1 CALIBRATION REPORT  (score-magnitude look-ahead throttle)")
    print("β is a STARTING value only — Stage 1 has no designability eval.")
    print("=" * 78)

    # --- Per target × proxy: active fraction, r & ΔP distributions ---
    for tgt in targets:
        steps = per_target_steps[tgt]
        n_struct = len(list((CALIB / tgt).glob("*_diagnostics.json")))
        lam = next((s.get("lambda0") for s in steps if not s.get("skipped")), None)
        print(f"\n### target={tgt}   λ(plateau)≈{lam}   ({n_struct} trajectories) ###")
        print(f"  {'proxy':10s} {'m_base':>9s} {'%active':>8s} "
              f"{'r_med':>8s} {'r_p90':>8s} {'r_max':>8s} "
              f"{'dP_med':>9s} {'dP_p90':>9s}   β_r(med→.7 / p90→.2)")
        for px in proxies:
            t, dP, r = _collect(steps, px)
            if len(r) == 0:
                continue
            mb = float(np.median([d[f"cal_{px}_m_base"] for d in steps
                                  if not d.get("skipped") and f"cal_{px}_m_base" in d]))
            active = r > 0
            frac = float(active.mean())
            sr = _summary(r[active]); sdp = _summary(dP[dP > 0])
            b_med, b_p90 = _beta(sr["med"], sr["p90"])
            print(f"  {px:10s} {mb:9.4f} {frac*100:7.1f}% "
                  f"{sr['med']:8.4f} {sr['p90']:8.4f} {sr['mx']:8.4f} "
                  f"{sdp['med']:9.4f} {sdp['p90']:9.4f}   {b_med:7.2f} / {b_p90:7.2f}")

    # --- t-band breakdown (where does guidance degrade the proxy?) ---
    print("\n" + "-" * 78)
    print("Active fraction & median r by t-band (steering window 0.3–0.8):")
    for tgt in targets:
        steps = per_target_steps[tgt]
        print(f"\n  target={tgt}")
        print(f"    {'proxy':10s} " + "".join(f"{lab+' %act':>13s}" for lab, _, _ in T_BINS)
              + "   |   " + "".join(f"{lab+' r_med':>13s}" for lab, _, _ in T_BINS))
        for px in proxies:
            t, dP, r = _collect(steps, px)
            if len(r) == 0:
                continue
            acts, meds = [], []
            for _, lo, hi in T_BINS:
                inb = (t >= lo) & (t < hi)
                rb = r[inb]; ab = rb > 0
                acts.append(f"{(ab.mean()*100 if len(rb) else float('nan')):11.1f}%")
                meds.append(f"{(np.median(rb[ab]) if ab.any() else float('nan')):13.4f}")
            print(f"    {px:10s} " + "".join(f"{a:>13s}" for a in acts)
                  + "   |   " + "".join(meds))

    # --- B's t_p sensitivity (instability = OOD-ceiling signature) ---
    bpx = [p for p in proxies if p.startswith("B_tp")]
    if len(bpx) >= 2:
        print("\n" + "-" * 78)
        print("Convention-B sensitivity to probe time t_p:")
        for tgt in targets:
            steps = per_target_steps[tgt]
            meds = {}
            for px in bpx:
                _, _, r = _collect(steps, px)
                act = r[r > 0]
                meds[px] = float(np.median(act)) if len(act) else float("nan")
            vals = [v for v in meds.values() if v == v and v > 0]
            spread = (max(vals) / min(vals)) if len(vals) >= 2 and min(vals) > 0 else float("nan")
            flag = "  <-- WILD (OOD-ceiling signature)" if spread == spread and spread > 3.0 else ""
            line = "  ".join(f"{px.replace('B_','')}:r_med={meds[px]:.4f}" for px in bpx)
            print(f"  {tgt:14s} {line}   spread×={spread:.2f}{flag}")

    # --- Pooled β proposal per proxy (across targets) ---
    print("\n" + "=" * 78)
    print("PROPOSED STARTING β PER PROXY  (pooled over targets; s=exp(-β·x))")
    print(f"  {'proxy':10s} {'input':>8s} {'med':>9s} {'p90':>9s}   {'β(med→.7)':>11s} {'β(p90→.2)':>11s}  note")
    for px in proxies:
        rr, dd = [], []
        for tgt in targets:
            _, dP, r = _collect(per_target_steps[tgt], px)
            rr.append(r[r > 0]); dd.append(dP[dP > 0])
        r_act = np.concatenate(rr) if rr else np.array([])
        d_act = np.concatenate(dd) if dd else np.array([])
        for label, arr in (("r", r_act), ("ΔP", d_act)):
            s = _summary(arr)
            b_med, b_p90 = _beta(s["med"], s["p90"])
            ratio = (b_p90 / b_med) if (b_med and b_med == b_med and b_med > 0) else float("nan")
            note = "heavy tail (med/p90 β disagree >3×)" if ratio == ratio and (ratio > 3 or ratio < 1/3) else ""
            print(f"  {px:10s} {label:>8s} {s['med']:9.4f} {s['p90']:9.4f}   "
                  f"{b_med:11.3f} {b_p90:11.3f}  {note}")
    print("=" * 78)
    print("Reminder: pick β in Stage 2 against the designability frontier, not here.")


if __name__ == "__main__":
    main()
