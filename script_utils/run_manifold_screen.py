#!/usr/bin/env python3
"""Within-w ceiling screen for the off-manifold panel. At FIXED w64, does any
latent-geometry signal predict codesign — and at which t?

Per protein: per-step manifold panel (from diagnostics json) + codesign label
(reused from b0/codesign_burial.csv, identical seeds => identical structures).
Outputs: (a) per-aggregator within-w AUC for each signal, (b) signal × t-bin AUC
heatmap (the 'critical window'). AUC>0.5: higher signal => more collapse.
"""
import json, glob, csv, re, sys
from pathlib import Path
import numpy as np

DIAG = Path("results/manifold_panel_probe/w64/diagnostics")
CODES_SRC = "results/burial_camsol_probe/b0/codesign_burial.csv"
SIGNALS = ["vel_disagreement","vel_norm_guided","guidance_flow_cos","z1_jump",
           "prior_drift","z1_norm","latent_extremity"]

def auc(score, label):
    # label=1 -> collapsed (co>=2). AUC of score predicting collapse.
    score=np.asarray(score,float); label=np.asarray(label,int)
    ok=~np.isnan(score); score=score[ok]; label=label[ok]
    n1=label.sum(); n0=len(label)-n1
    if n1<3 or n0<3: return float("nan"), len(label)
    order=np.argsort(score); ranks=np.empty(len(score),float); ranks[order]=np.arange(1,len(score)+1)
    a=(ranks[label==1].sum()-n1*(n1+1)/2)/(n1*n0)
    return float(a), len(label)

def main():
    codes={}
    for r in csv.DictReader(open(CODES_SRC)):
        try: codes[r["pid"]]=0 if float(r["co"])<2 else 1   # 1 = collapsed
        except: pass
    rows=[]  # per protein: dict(pid, collapsed, steps=[{t,sig...}])
    for jf in sorted(DIAG.glob("*_diagnostics.json")):
        pid=jf.stem.replace("_diagnostics","")
        if pid not in codes: continue
        d=json.load(open(jf)); steps=d if isinstance(d,list) else d.get("diagnostics",[])
        man=[s["manifold"] for s in steps if isinstance(s,dict) and "manifold" in s]
        if man: rows.append(dict(pid=pid, collapsed=codes[pid], man=man))
    if not rows: print("no joined data yet"); return
    y=np.array([r["collapsed"] for r in rows])
    print(f"proteins: {len(rows)}  collapsed(co>=2): {int(y.sum())}  designable: {int((y==0).sum())}  (w64)\n")

    print("=== (a) within-w AUC by AGGREGATOR (AUC>0.5 => signal predicts collapse) ===")
    print(f"{'signal':20s} {'mean':>7} {'max':>7} {'late(.6-.9)':>11} {'early(.3-.5)':>12}")
    def agg(r,sig,lo=None,hi=None,fn=np.mean):
        vs=[m.get(sig,np.nan) for m in r["man"] if (lo is None or lo<=m["t"]<=hi)]
        vs=[v for v in vs if v==v]
        return fn(vs) if vs else np.nan
    best=[]
    for sig in SIGNALS:
        cols={}
        cols["mean"]=auc([agg(r,sig,fn=np.mean) for r in rows],y)[0]
        cols["max"]=auc([agg(r,sig,fn=np.max) for r in rows],y)[0]
        cols["late"]=auc([agg(r,sig,0.6,0.9) for r in rows],y)[0]
        cols["early"]=auc([agg(r,sig,0.3,0.5) for r in rows],y)[0]
        print(f"{sig:20s} {cols['mean']:>7.2f} {cols['max']:>7.2f} {cols['late']:>11.2f} {cols['early']:>12.2f}")
        for k,v in cols.items():
            if v==v: best.append((abs(v-0.5),sig,k,v))
    best.sort(reverse=True)
    print("\ntop separators:", ", ".join(f"{s}/{k}={v:.2f}" for _,s,k,v in best[:4]))

    print("\n=== (b) signal × t-bin AUC (the critical window) ===")
    tbins=[(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90)]
    hdr="".join(f"{f'{lo:.1f}-{hi:.1f}':>9}" for lo,hi in tbins)
    print(f"{'signal':20s}{hdr}")
    for sig in SIGNALS:
        line=f"{sig:20s}"
        for lo,hi in tbins:
            vals=[agg(r,sig,lo,hi) for r in rows]
            a,_=auc(vals,y); line+=f"{a:>9.2f}" if a==a else f"{'—':>9}"
        print(line)
    print("\nAUC~0.5 everywhere => no within-w signal (capstone null). |AUC-0.5|>~0.15 in a t-window => exploitable, t-localized.")

if __name__=="__main__": main()
