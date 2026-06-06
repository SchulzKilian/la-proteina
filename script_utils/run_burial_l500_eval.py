#!/usr/bin/env python3
"""L500 unlock test: does the burial throttle revive designable camsol steering at
L=500 (where longer proteins have bigger, more-separable cores)? Direct A/B per w:
b025 (burial) vs b0 (no throttle), same w & length. Codesign via ESMFold use_pdb_seq.
Modes:  fold <dir...>  (populate codesign_l500.csv, parallelizable)  |  (default) compare.
"""
import csv, glob, os, sys, shutil
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L',
        'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,
    'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':0.0}
DESIG=2.0
PROBE=Path("results/burial_camsol_probe")

def surf_core(pdb):
    seq=[];ca=[]
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip()=="CA":
            seq.append(AA3TO1.get(l[17:20].strip(),'X')); ca.append((float(l[30:38]),float(l[38:46]),float(l[46:54])))
    ca=np.array(ca)
    if len(seq)<5: return np.nan,np.nan
    d=np.linalg.norm(ca[:,None]-ca[None],axis=-1); nbr=(d<10).sum(1)-1; ex=nbr<np.median(nbr); k=np.array([KD[c] for c in seq])
    return float(k[ex].mean()), float(k[~ex].mean())

def fold_dir(scRMSD, d, tmp):
    d=Path(d); g=d/"guided"
    if not g.exists(): return
    csvf=d/"codesign_l500.csv"; done=set()
    if csvf.exists():
        for r in csv.DictReader(open(csvf)): done.add(r["pid"])
    fo=open(csvf,"a",newline=""); w=csv.writer(fo)
    if csvf.stat().st_size==0: w.writerow(["pid","co","surfkd","corekd"]); fo.flush()
    for pdb in sorted(g.glob("s*_n*.pdb")):
        if pdb.stem in done: continue
        s,c=surf_core(pdb)
        td=tmp/pdb.stem; shutil.rmtree(td,ignore_errors=True); td.mkdir(parents=True,exist_ok=True)
        try:
            r=scRMSD(pdb_file_path=str(pdb),tmp_path=str(td),num_seq_per_target=1,use_pdb_seq=True,
                     rmsd_modes=["ca"],folding_models=["esmfold"],keep_outputs=True,ret_min=False)
            rm=r["ca"]["esmfold"]; co=rm[0] if rm else float("inf")
        except Exception: co=float("inf")
        finally: shutil.rmtree(td,ignore_errors=True)
        w.writerow([pdb.stem,f"{co:.4f}",f"{s:.4f}",f"{c:.4f}"]); fo.flush()
        print(f"  [{d.name}] {pdb.stem} co={co:.2f}")
    fo.close()

def read(d):
    f=Path(d)/"codesign_l500.csv"
    if not f.exists(): return None
    co=[];sk=[];ck=[]
    for r in csv.DictReader(open(f)): co.append(float(r["co"])); sk.append(float(r["surfkd"])); ck.append(float(r["corekd"]))
    return np.array(co),np.array(sk),np.array(ck)

def compare():
    print("=== L500 UNLOCK TEST: burial throttle (b025) vs no-throttle (b0), per w ===")
    print(f"{'w':>4} {'cell':5s} {'n':>3} {'codes%':>7} {'surfKD':>7} {'coreKD':>7} {'des_surfKD':>10}")
    for W in (32,64):
        cells={}
        for tag in ("b0","b025"):
            r=read(PROBE/f"L500_w{W}_{tag}"); cells[tag]=r
            if r is None: print(f"{W:>4} {tag:5s}  (no data)"); continue
            co,sk,ck=r; des=co<DESIG
            dsk=np.nanmean(sk[des]) if des.any() else float('nan')
            print(f"{W:>4} {tag:5s} {len(co):>3} {des.mean():>6.0%} {np.nanmean(sk):>7.3f} {np.nanmean(ck):>7.3f} {dsk:>10.3f}")
        if cells["b0"] is not None and cells["b025"] is not None:
            c0=cells["b0"][0]; c1=cells["b025"][0]
            d0=int((c0<DESIG).sum()); d1=int((c1<DESIG).sum())
            p=fisher_exact([[d1,len(c1)-d1],[d0,len(c0)-d0]])[1]
            print(f"     -> w{W} throttle {d1}/{len(c1)}={d1/len(c1):.0%} vs no-throttle {d0}/{len(c0)}={d0/len(c0):.0%}  Fisher p={p:.3g}\n")
    print("UNLOCK = b025 codesign >> b0 codesign at L500 (baseline near-dead, throttle revives).")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="fold":
        sys.path.insert(0,os.getcwd())
        from proteinfoundation.metrics.designability import scRMSD
        tmp=Path("tmp/burial_l500"); tmp.mkdir(parents=True,exist_ok=True)
        for d in sys.argv[2:]: print(f"[fold] {d}"); fold_dir(scRMSD,d,tmp)
        print("[fold] done")
    else:
        compare()
