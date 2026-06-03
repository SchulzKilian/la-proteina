#!/usr/bin/env python3
"""Matched-seed %-improved per developability target, single- vs multi-objective runs.

Pairing on (seed, length) vs the unguided baseline sanity_unsteered_seed42_45/unguided.
CamSol = external pH-7 (consolidated CSV); TANGO/SAP/SCM/SWI = in-house panels.
Favorable: camsol ↑, swi ↑, tango ↓, sap ↓, scm_positive ↓.
Only the STEERED target(s) of each run are reported (the thing the run was optimizing).
"""
import csv, re, math, statistics as st, os

ROOT="/home/ks2218/la-proteina"
BASE_DEVEL=f"{ROOT}/results/sanity_unsteered_seed42_45/unguided/properties_unguided.csv"
CONSOL=f"{ROOT}/results/camsol_ph7_full_2026_05_27.csv"
SIGN={"camsol":+1,"swi":+1,"tango_total":-1,"sap_total":-1,"scm_positive":-1}
NICE={"camsol":"CamSol↑","swi":"SWI↑","tango_total":"TANGO↓","sap_total":"SAP↓","scm_positive":"SCM⁺↓"}

def cell(o):
    m=re.search(r's(\d+)_n(\d+)',o); return (int(m.group(1)),int(m.group(2))) if m else None
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

base={}
for r in csv.DictReader(open(BASE_DEVEL)):
    c=cell(r['protein_id'])
    if c: base[c]=r
cam_base={cell(r['orig_id']):float(r['camsol_ph7']) for r in csv.DictReader(open(CONSOL)) if r['subrun']=='sanity_seed42_45_u'}
CONSOL_ROWS=list(csv.DictReader(open(CONSOL)))

def prop_improved(steered_csv, col, camsol_subrun, parent_filter):
    rows=list(csv.DictReader(open(steered_csv)))
    diffs=[]
    if col=="camsol":
        cam_steer={cell(r['orig_id']):float(r['camsol_ph7']) for r in CONSOL_ROWS
                   if r['subrun']==camsol_subrun and (parent_filter in r['source_path'])}
        for c in cam_steer:
            if c in cam_base: diffs.append(cam_steer[c]-cam_base[c])
    else:
        for r in rows:
            c=cell(r['protein_id'])
            if c in base and base[c].get(col,'') not in ('','None') and r.get(col,'') not in ('','None'):
                diffs.append(SIGN[col]*(float(r[col])-float(base[c][col])))
    return diffs

# (label, props_csv, camsol_subrun, parent_filter, [targets], w)
RUNS=[
 ("camsol_max  SINGLE  w32", "noise_aware_high_w_scout/camsol_max_w32", "camsol_max_w32","noise_aware",["camsol"]),
 ("tango_min   SINGLE  w32", "noise_aware_high_w_scout/tango_min_w32",  "tango_min_w32", "noise_aware",["tango_total"]),
 ("tango_min   SINGLE  w48", "noise_aware_high_w_scout/tango_min_w48",  "tango_min_w48", "noise_aware",["tango_total"]),
 ("combo_camsol_tango  w32", "combo_camsol_tango_scout/combo_camsol_tango_w32","combo_camsol_tango_w32","combo_camsol_tango_scout",["camsol","tango_total"]),
 ("combo_devel4 MULTI  w32", "combo_devel4_scout/combo_devel4_w32","combo_devel4_w32","combo_devel4_scout",["camsol","tango_total","sap_total","scm_positive"]),
]

print(f"{'run':>26} {'target':>9} {'improved':>9} {'%':>5} {'CI95lo':>7} {'meanΔ(fav)':>11}")
for label,subpath,camsub,pfilt,targets in RUNS:
    pc=f"{ROOT}/results/{subpath}/properties_guided.csv"
    if not os.path.exists(pc):
        print(f"{label:>26}  (missing {subpath})"); continue
    for col in targets:
        d=prop_improved(pc,col,camsub,pfilt)
        if not d: print(f"{label:>26} {NICE[col]:>9}  (no data)"); continue
        k=sum(1 for x in d if x>0); n=len(d); lo,hi=wilson(k,n)
        print(f"{label:>26} {NICE[col]:>9} {f'{k}/{n}':>9} {100*k/n:>4.0f}% {100*lo:>6.0f}% {st.mean(d):>+11.3f}")
