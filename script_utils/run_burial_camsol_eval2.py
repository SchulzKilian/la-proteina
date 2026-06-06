#!/usr/bin/env python3
"""Rigorous burial-throttle eval: codesign vs delivered SURFACE-KD, with the
matched-delivery test done against ACTUAL baseline samples in-band (Fisher) —
NO chord interpolation. Surface-KD = mean Kyte-Doolittle over EXPOSED residues
(the honest camsol delivery proxy; whole-seq KD is confounded because the throttle
keeps buried hydrophobics). Folds throttle cells (resume-safe), reuses on-disk
baseline codesign. Auto-discovers throttle cells under results/burial_camsol_probe/.
"""
import csv, glob, os, sys, shutil
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L',
        'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,
    'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':0.0}
DESIG=2.0; BAND=0.15
PROBE=Path("results/burial_camsol_probe")
BASE={w:Path(f"results/noise_aware_high_w_scout/camsol_max_w{w}") for w in (32,48,64,128)}

def parse(pdb):
    seq=[];ca=[]
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip()=="CA":
            seq.append(AA3TO1.get(l[17:20].strip(),'X')); ca.append((float(l[30:38]),float(l[38:46]),float(l[46:54])))
    return seq,np.array(ca)
def surf_core(pdb):
    seq,ca=parse(pdb)
    if len(seq)<5: return np.nan,np.nan
    d=np.linalg.norm(ca[:,None]-ca[None],axis=-1); nbr=(d<10).sum(1)-1
    exposed=nbr<np.median(nbr); k=np.array([KD[c] for c in seq])
    return float(k[exposed].mean()), float(k[~exposed].mean())

def codesign_fold(scRMSD,pdb,tmp):
    td=tmp/pdb.stem; shutil.rmtree(td,ignore_errors=True); td.mkdir(parents=True,exist_ok=True)
    try:
        r=scRMSD(pdb_file_path=str(pdb),tmp_path=str(td),num_seq_per_target=1,use_pdb_seq=True,
                 rmsd_modes=["ca"],folding_models=["esmfold"],keep_outputs=True,ret_min=False)
        rm=r["ca"]["esmfold"]; co=rm[0] if rm else float("inf")
    except Exception: co=float("inf")
    finally: shutil.rmtree(td,ignore_errors=True)
    return co

def baseline_pool():
    co=[];sk=[]
    for w,d in BASE.items():
        cs={r["protein_id"]:float(r["coScRMSD_ca"]) for r in csv.DictReader(open(d/"codesign_guided.csv"))}
        for pdb in sorted((d/"guided").glob("s*_n*.pdb")):
            if pdb.stem in cs:
                s,_=surf_core(pdb); co.append(cs[pdb.stem]); sk.append(s)
    return np.array(co),np.array(sk)

def throttle_cell(scRMSD,d,tmp):
    g=d/"guided"
    if not g.exists(): return None
    csvf=d/"codesign_burial.csv"; done={}
    if csvf.exists():
        for r in csv.DictReader(open(csvf)): done[r["pid"]]=(float(r["co"]),float(r.get("surfkd","nan")))
    fo=open(csvf,"a",newline=""); w=csv.writer(fo)
    if csvf.stat().st_size==0: w.writerow(["pid","co","surfkd","corekd"]); fo.flush()
    co=[];sk=[]
    for pdb in sorted(g.glob("s*_n*.pdb")):
        pid=pdb.stem; s,c=surf_core(pdb)
        if pid in done and not np.isnan(done[pid][1]): cval=done[pid][0]
        else:
            cval=codesign_fold(scRMSD,pdb,tmp); w.writerow([pid,f"{cval:.4f}",f"{s:.4f}",f"{c:.4f}"]); fo.flush()
        co.append(cval); sk.append(s)
    fo.close()
    return np.array(co),np.array(sk)

def main():
    sys.path.insert(0,os.getcwd())
    from proteinfoundation.metrics.designability import scRMSD
    tmp=Path("tmp/burial_camsol2"); tmp.mkdir(parents=True,exist_ok=True)
    bco,bsk=baseline_pool()
    print("baseline frontier (codesign vs surfKD):")
    for w,d in BASE.items():
        cs={r["protein_id"]:float(r["coScRMSD_ca"]) for r in csv.DictReader(open(d/"codesign_guided.csv"))}
        co=[];sk=[]
        for pdb in sorted((d/"guided").glob("s*_n*.pdb")):
            if pdb.stem in cs: s,_=surf_core(pdb); co.append(cs[pdb.stem]); sk.append(s)
        co=np.array(co)
        print(f"  baseline w{w:<3} n={len(co)} codes={ (co<DESIG).mean():.0%} surfKD={np.nanmean(sk):.3f}")
    cells=sorted([Path(p) for p in glob.glob(str(PROBE/"*")) if Path(p).is_dir() and (Path(p)/"guided").exists()
                  and not Path(p).name.startswith("SMOKE")])
    print(f"\n{'throttle cell':16s} {'n':>3} {'codes%':>7} {'surfKD':>7} | matched-band baseline (Fisher)")
    for d in cells:
        res=throttle_cell(scRMSD,d,tmp)
        if res is None: continue
        co,sk=res; m=np.nanmean(sk); tdes=int((co<DESIG).sum()); tn=len(co)
        band=np.abs(bsk-m)<=BAND; bb=bco[band]; bdes=int((bb<DESIG).sum()); bn=len(bb)
        if bn>=4:
            _,p=fisher_exact([[tdes,tn-tdes],[bdes,bn-bdes]]); d2=tdes/tn-bdes/bn
            flag="ABOVE*" if (p<0.05 and d2>0) else ("above(ns)" if d2>0.10 else "on-frontier")
            print(f"{d.name:16s} {tn:>3} {tdes/tn:>6.0%} {m:>7.3f} | base {bdes}/{bn}={bdes/bn:.0%} Δ={d2:+.0%} p={p:.3g} {flag}")
        else:
            print(f"{d.name:16s} {tn:>3} {tdes/tn:>6.0%} {m:>7.3f} | only {bn} baseline in band")
    print("\n* ABOVE = significantly above the no-throttle frontier at matched delivered surface-solubility.")

def fold_only(dirs):
    """Fold codesign for the given throttle dirs only (parallelizable across GPUs)."""
    sys.path.insert(0,os.getcwd())
    from proteinfoundation.metrics.designability import scRMSD
    tmp=Path("tmp/burial_camsol2"); tmp.mkdir(parents=True,exist_ok=True)
    for d in dirs:
        d=Path(d)
        if (d/"guided").exists():
            print(f"[fold] {d.name}"); throttle_cell(scRMSD,d,tmp)
    print("[fold] done")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="fold":
        fold_only(sys.argv[2:])
    else:
        main()
