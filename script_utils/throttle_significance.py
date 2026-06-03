#!/usr/bin/env python3
"""Significance of the w16 throttle 'above-frontier' effect.

Per-sample (deliveredCO, designable) from PDB CA + scRMSD CSV. Three tests:
  T1  same-w rescue:  throttle_w16 designability vs baseline_w16        (Fisher exact)
  T2  matched-CO:     baseline samples in throttle's CO band vs throttle (Fisher exact)
  T3  frontier shift: logistic  designable ~ CO + is_throttle           (Wald p on is_throttle)
Bootstrap 95% CI on each throttle cell's designability fraction.
"""
import csv, glob, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

ROOT = Path("results/geom_lookahead_sweep")
CONTACT_A, MINSEP, DESIG = 8.0, 3, 2.0
LFILTER = sys.argv[1] if len(sys.argv) > 1 else "300"
COBAND = 0.015   # +/- CO window for matched-CO test

def ca(p):
    xs=[(float(l[30:38]),float(l[38:46]),float(l[46:54])) for l in open(p)
        if l.startswith("ATOM") and l[12:16].strip()=="CA"]
    return np.asarray(xs,float)
def co(c):
    L=c.shape[0]; d=np.linalg.norm(c[:,None]-c[None],axis=-1)
    iu=np.triu_indices(L,k=MINSEP); sep=(iu[1]-iu[0]).astype(float); ct=d[iu]<CONTACT_A; n=ct.sum()
    return 0.0 if n==0 else float((sep*ct).sum()/(L*n))
def scr(cell):
    f=ROOT/cell/"scRMSD_guided.csv"; o={}
    if f.exists():
        for r in csv.DictReader(open(f)):
            try:o[r["protein_id"]]=float(r["scRMSD_ca_min"])
            except:o[r["protein_id"]]=float("inf")
    return o
def lmatch(pid):
    if LFILTER=="all":return True
    m=re.search(r"_n(\d+)$",pid); return bool(m) and m.group(1)==LFILTER
def samples(cell):
    s=scr(cell); out=[]
    for pdb in sorted((ROOT/cell/"guided").glob("s*_n*.pdb")):
        pid=pdb.stem
        if pid not in s or not lmatch(pid):continue
        c=ca(pdb)
        if c.shape[0]<5:continue
        out.append((co(c), 1 if s[pid]<DESIG else 0))
    return out

def boot_ci(d, B=5000):
    d=np.asarray(d); n=len(d)
    if n==0:return (float("nan"),float("nan"))
    idx=np.random.RandomState(0).randint(0,n,(B,n))
    fr=d[idx].mean(1); return float(np.percentile(fr,2.5)),float(np.percentile(fr,97.5))

base_cells=sorted([p.name for p in ROOT.glob("contact_order_baseline_w*")])
base=[]
for c in base_cells: base+=samples(c)
base=np.array(base) if base else np.zeros((0,2))
print(f"[L={LFILTER}]  baseline pool n={len(base)} from {len(base_cells)} cells; CO band ±{COBAND}\n")

for thr in ["contact_order_geometricres_w16","contact_order_ramares_w16","contact_order_perres_w16"]:
    S=samples(thr)
    if not S: print(f"{thr}: no data\n"); continue
    S=np.array(S); tco=S[:,0].mean(); tdes=S[:,1]
    lo,hi=boot_ci(tdes)
    print(f"=== {thr} ===")
    print(f"  throttle: n={len(S)}  designable={tdes.mean():.1%} (95%CI {lo:.0%}-{hi:.0%})  meanCO={tco:.4f}")

    # T1 same-w rescue vs baseline_w16
    bw16=samples("contact_order_baseline_w16")
    if bw16:
        b=np.array(bw16)[:,1]
        tab=[[int(tdes.sum()),int(len(tdes)-tdes.sum())],[int(b.sum()),int(len(b)-b.sum())]]
        _,p=fisher_exact(tab)
        print(f"  T1 same-w (vs baseline_w16 {b.mean():.1%},n={len(b)}): Fisher p={p:.2g}")

    # T2 matched-CO: baseline samples within band of throttle's mean CO
    if len(base):
        m=np.abs(base[:,0]-tco)<=COBAND; bm=base[m,1]
        if len(bm)>=3:
            tab=[[int(tdes.sum()),int(len(tdes)-tdes.sum())],[int(bm.sum()),int(len(bm)-bm.sum())]]
            _,p=fisher_exact(tab)
            print(f"  T2 matched-CO (baseline in {tco-COBAND:.3f}-{tco+COBAND:.3f}: {bm.mean():.1%},n={len(bm)}): Fisher p={p:.2g}")
        else:
            print(f"  T2 matched-CO: only n={len(bm)} baseline samples in band (need >=3)")

    # T3 logistic designable ~ CO + is_throttle  (pool baseline + this throttle)
    if len(base):
        X0=np.column_stack([base[:,0], np.zeros(len(base))])
        X1=np.column_stack([S[:,0],   np.ones(len(S))])
        X=np.vstack([X0,X1]); y=np.concatenate([base[:,1],tdes])
        Xd=np.column_stack([np.ones(len(X)),X])  # intercept, CO, throttle
        # Newton-Raphson logistic
        b=np.zeros(3)
        for _ in range(50):
            eta=Xd@b; pmu=1/(1+np.exp(-eta)); W=pmu*(1-pmu)
            g=Xd.T@(y-pmu); H=Xd.T@(Xd*W[:,None])+1e-6*np.eye(3)
            step=np.linalg.solve(H,g); b+=step
            if np.abs(step).max()<1e-8:break
        cov=np.linalg.inv(Xd.T@(Xd*( (1/(1+np.exp(-(Xd@b))))*(1-1/(1+np.exp(-(Xd@b)))) )[:,None])+1e-6*np.eye(3))
        se=np.sqrt(np.diag(cov))[2]; z=b[2]/se
        from math import erf,sqrt
        p=2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
        print(f"  T3 logistic (designable~CO+throttle): throttle coef={b[2]:+.2f} (OR={np.exp(b[2]):.2g}), z={z:.2f}, p={p:.2g}")
    print()
