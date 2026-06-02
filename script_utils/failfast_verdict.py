#!/usr/bin/env python3
"""Fail-fast verdict for the per-residue CO throttle.

Delivered contact_order is read straight from PDB CA atoms (pure numpy, no torch import,
instant). Designability (scRMSD_ca_min < 2.0) is read from each cell's scRMSD_guided.csv.
Throttle cells (geom/rama w16) are overlaid on the on-disk no-throttle baseline curve
(baseline_w4/8/16/32). Verdict: does a throttle point sit ABOVE the baseline
designability-vs-delivered-CO curve at matched delivered CO, or on/below it?

CO minimize: lower delivered CO = more steering effect. Unguided rawCO ~0.104.
"""
import csv, glob, re, sys
from pathlib import Path
import numpy as np

ROOT = Path("results/geom_lookahead_sweep")
CONTACT_A, MINSEP, DESIG = 8.0, 3, 2.0
LFILTER = sys.argv[1] if len(sys.argv) > 1 else "all"   # "300", "400", or "all"

def ca_from_pdb(p):
    xs = []
    for line in open(p):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(xs, float)

def contact_order(ca):
    L = ca.shape[0]
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    iu = np.triu_indices(L, k=MINSEP); sep = (iu[1] - iu[0]).astype(float)
    ct = d[iu] < CONTACT_A; nc = ct.sum()
    return 0.0 if nc == 0 else float((sep * ct).sum() / (L * nc))

def load_scr(cell):
    f = ROOT / cell / "scRMSD_guided.csv"; out = {}
    if not f.exists(): return out
    for r in csv.DictReader(open(f)):
        try: out[r["protein_id"]] = float(r["scRMSD_ca_min"])
        except: out[r["protein_id"]] = float("inf")
    return out

def lmatch(pid):
    if LFILTER == "all": return True
    m = re.search(r"_n(\d+)$", pid)
    return bool(m) and m.group(1) == LFILTER

def cell_stats(cell):
    scr = load_scr(cell); rows = []
    for pdb in sorted((ROOT / cell / "guided").glob("s*_n*.pdb")):
        pid = pdb.stem
        if pid not in scr or not lmatch(pid): continue
        ca = ca_from_pdb(pdb)
        if ca.shape[0] < 5: continue
        rows.append((pid, contact_order(ca), scr[pid]))
    if not rows: return None
    co = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
    des = int((s < DESIG).sum())
    return dict(n=len(rows), des=des, desfrac=des / len(rows),
                meanCO=float(co.mean()), medCO=float(np.median(co)))

def wkey(c):
    m = re.search(r"_w(\d+)$", c); return int(m.group(1)) if m else 0
BASE = sorted([p.name for p in ROOT.glob("contact_order_baseline_w*")], key=wkey)
THR = sorted([p.name for p in ROOT.glob("contact_order_*res_w*")], key=lambda c: (c.split("res_")[0], wkey(c)))
CELLS = BASE + THR
N_BASE = len(BASE)
print(f"[L filter = {LFILTER}]  (lower delivered CO = more steering; unguided ~0.104)")
print(f"{'cell':34s} {'n':>3} {'des':>4} {'desfrac':>8} {'meanCO':>8} {'medCO':>8}")
stats = {}
for c in CELLS:
    st = cell_stats(c); stats[c] = st
    if st:
        print(f"{c:34s} {st['n']:>3} {st['des']:>4} {st['desfrac']:>7.1%} {st['meanCO']:>8.4f} {st['medCO']:>8.4f}")
    else:
        print(f"{c:34s}  (no data yet)")

# Baseline curve (delivered CO -> designability), sorted by delivered CO ascending.
base = [(stats[c]["meanCO"], stats[c]["desfrac"]) for c in BASE if stats[c]]
base.sort()
def baseline_at(co):
    if not base: return None
    xs = [b[0] for b in base]; ys = [b[1] for b in base]
    return float(np.interp(co, xs, ys))   # linear interp of baseline frontier

print("\n=== VERDICT (throttle vs no-throttle baseline at matched delivered CO) ===")
for c in THR:
    st = stats[c]
    if not st: print(f"{c}: no data yet"); continue
    b = baseline_at(st["meanCO"])
    if b is None: print(f"{c}: no baseline yet"); continue
    delta = st["desfrac"] - b
    flag = "ABOVE frontier (throttle helps)" if delta > 0.10 else \
           ("ON frontier (= w-reduction, dead)" if abs(delta) <= 0.10 else "BELOW frontier (hurts)")
    print(f"{c}: {st['desfrac']:.1%} @ deliveredCO={st['meanCO']:.4f} (n={st['n']}) | "
          f"baseline@sameCO≈{b:.1%} | Δ={delta:+.1%} -> {flag}")
