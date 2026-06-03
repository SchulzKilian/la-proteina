#!/usr/bin/env python3
"""Overnight throttle-viability probe across MANY steered properties.

THE QUESTION: a per-residue (burial-gated) throttle can only beat plain
w-reduction if the property-driving change and the designability-breaking change
are SEPARABLE. Structural props (Rg/CO): identical channel -> dead (tonight).
TANGO: collocate at buried core -> dead (E114-add-6). UNTESTED at breadth:
the other surface/sequence props.

Reference-free design (no seed-matched unguided sequence needed):

  PHASE 1 (CPU, fast, broad) -- separability GATE.
    Per property set & w cell, per guided sample compute the burden of
    "damaging-when-buried" residues (per-property DRIVER set) and correlate
    WITHIN-w with NON-codesignability (coScRMSD_ca >= 2.0). corr>0 & AUC>~0.65
    => damage localizes to buried driver residues => a burial throttle has signal.
    corr~0 / AUC~0.5 => collapse not buried-localized => throttle == w-reduction.

  PHASE 2 (GPU) -- designability-RECOVERY validation.
    High-w (most-collapsed) cells only -- the throttle implicitly lowers guidance,
    so the test must live where steering is strong. Per sample, revert the BURIED
    driver residues (per-property revert rule) and REFOLD (ESMFold codesign). High
    recovery => buried residues were the damage => throttle viable. Low recovery =>
    global/compositional collapse => dead.

Driver / revert semantics by property family:
  charge-add (net_charge_max +K/R, net_charge_min +D/E): steering injects charges;
      damage if buried in the core. Driver = the injected charge; revert buried->L.
  core-polarize (camsol_max, hydpatch_min, sap_min): steering strips hydrophobics
      to clean the surface; damage = a buried position left POLAR (cores need
      hydrophobics). Driver = polar residues; revert buried polar->L.
  aggregation (tango_min): CONTROL, known buried-collocated dead. Driver=agg-prone.
  structural (rg_min): CONTROL, no sequence driver.

Burial = per-CA neighbor count within 10 A; buried := above the per-protein median.
Robust: per-sample try/except, atomic resume-safe CSV, Phase 1 needs no torch.
"""
import csv, glob, os, sys, shutil, re
from pathlib import Path
import numpy as np

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
HYDRO = set("AVLIMFWCY")
POLAR = set("STNQDEKRHGP")          # "should-be-hydrophobic-if-buried" candidates
AGG   = set("VILFYW")
BURIAL_RADIUS = 10.0
DESIG = 2.0
OUT = Path("results/throttle_viability"); OUT.mkdir(parents=True, exist_ok=True)

# name -> dict(glob, driver(set), family, p2cells(list of w to refold), label)
# ORDER MATTERS for phase2: gate-POSITIVE polarize props first so the decisive
# folds land first within the 8h budget; dead controls (1 high-w cell each) last.
SETS = {
  "camsol_max":     dict(glob="results/noise_aware_high_w_scout/camsol_max_w*",
        driver=POLAR, family="polarize", p2=[64,128], label="strip surface hydrophobics; revert buried polar->L  [GATE+]"),
  "hydpatch_min":   dict(glob="results/hydpatch_min_sweep/hydpatch_min_w*",
        driver=POLAR, family="polarize", p2=[64,128], label="shrink hydrophobic patches; revert buried polar->L  [GATE+]"),
  "net_charge_max": dict(glob="results/net_charge_max_sweep/net_charge_max_w*",
        driver=set("KR"), family="charge+", p2=[128], label="inject +charge; revert buried K/R->L  [GATE-: dead]"),
  "net_charge_min": dict(glob="results/net_charge_min_scout/net_charge_min_w*",
        driver=set("DE"), family="charge-", p2=[128], label="inject -charge; revert buried D/E->L"),
  "tango_min":      dict(glob="results/noise_aware_high_w_scout/tango_min_w*",
        driver=AGG, family="aggregation", p2=[128], label="CONTROL: known buried-collocated dead"),
  "iupred_max":     dict(glob="results/iupred_max_scout/iupred_max_w*",
        driver=POLAR, family="polarize", p2=[32], label="maximize disorder (gate+ at low w; caveat objective fights folding)"),
  "sap_min":        dict(glob="results/coords_na_sap_coord_only_w*",
        driver=HYDRO, family="coord", p2=[], label="coord-channel; throttle==structural (dead tonight)"),
  "rg_min":         dict(glob="results/coords_na_rg_min_coord_only_w*",
        driver=set(), family="structural", p2=[], label="structural CONTROL"),
}

def parse_residues(pdb):
    out = []
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            out.append((AA3TO1.get(l[17:20].strip(), 'X'),
                        (float(l[30:38]), float(l[38:46]), float(l[46:54]))))
    return out

def burial_mask(ca):
    ca = np.asarray(ca, float)
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    nbr = (d < BURIAL_RADIUS).sum(1) - 1
    return nbr >= np.median(nbr)

def net_charge(seq):
    return sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)

def wkey(p):
    m = re.search(r"_w(\d+)$", str(p)); return int(m.group(1)) if m else 0

def load_codesign(cell):
    f = Path(cell) / "codesign_guided.csv"; o = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            try: o[r["protein_id"]] = float(r["coScRMSD_ca"])
            except: pass
    return o

# ---------------- PHASE 1 ----------------
def phase1():
    print("="*82); print("PHASE 1 -- buried-driver separability gate (within-w)"); print("="*82)
    rows = []
    for name, S in SETS.items():
        cells = sorted(glob.glob(S["glob"]), key=wkey)
        if not cells: print(f"\n[{name}] no cells ({S['glob']})"); continue
        print(f"\n### {name}  [{S['family']}]  {S['label']}")
        print(f"{'w':>4} {'n':>4} {'codes%':>7} {'burden_des':>11} {'burden_non':>11} {'corr':>7} {'AUC':>6}")
        for cell in cells:
            cs = load_codesign(cell); gdir = Path(cell)/"guided"
            if not cs or not gdir.exists(): continue
            bur, non = [], []
            for pdb in sorted(gdir.glob("s*_n*.pdb")):
                pid = pdb.stem
                if pid not in cs: continue
                try:
                    res = parse_residues(pdb)
                    if len(res) < 10: continue
                    seq = "".join(r[0] for r in res); ca = [r[1] for r in res]
                    bm = burial_mask(ca)
                    if S["driver"]:
                        di = np.array([c in S["driver"] for c in seq]); nd = di.sum()
                        burden = float((di & bm).sum()/nd) if nd else 0.0
                    else:
                        burden = float(bm.mean())
                    bur.append(burden); non.append(1 if cs[pid] >= DESIG else 0)
                except Exception:
                    continue
            if len(bur) < 6: continue
            b = np.array(bur); y = np.array(non)
            frac = (y == 0).mean()
            bd = b[y==0].mean() if (y==0).any() else float('nan')
            bn = b[y==1].mean() if (y==1).any() else float('nan')
            corr = float(np.corrcoef(b, y)[0,1]) if b.std()>0 and y.std()>0 else float('nan')
            auc = float('nan')
            if y.std() > 0:
                order = np.argsort(b); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(b)+1)
                n1 = (y==1).sum(); n0 = (y==0).sum()
                auc = (ranks[y==1].sum() - n1*(n1+1)/2)/(n1*n0)
            print(f"{wkey(cell):>4} {len(y):>4} {frac:>6.0%} {bd:>11.3f} {bn:>11.3f} {corr:>7.2f} {auc:>6.2f}")
            rows.append((name, S["family"], wkey(cell), len(y), frac, bd, bn, corr, auc))
    with open(OUT/"phase1_gate.csv", "w", newline="") as f:
        csv.writer(f).writerows([["set","family","w","n","codes_frac","burden_des","burden_non","corr","auc"]]+rows)
    print(f"\n[phase1] wrote {OUT/'phase1_gate.csv'}")
    print("GATE: corr>0 & AUC>~0.65 within a w => buried drivers localize damage => throttle has signal.")
    print("      corr~0 / AUC~0.5 => not buried-localized => throttle == w-reduction (dead).")

# ---------------- PHASE 2 ----------------
def write_reverted_pdb(src, target_ordinals, revert_to3, dst):
    ordinal = -1; prev = None
    with open(src) as fi, open(dst, "w") as fo:
        for l in fi:
            if l.startswith(("ATOM","HETATM")):
                key = (l[21], l[22:27])
                if key != prev: ordinal += 1; prev = key
                if ordinal in target_ordinals: l = l[:17] + f"{revert_to3:>3}" + l[20:]
            fo.write(l)

def _fold(scRMSD, tmp_root, tag, src, target_ordinals):
    """write reverted pdb (ordinals->LEU), fold its sequence, return coScRMSD_ca."""
    rp = tmp_root/f"{tag}_rev.pdb"; write_reverted_pdb(src, set(target_ordinals), "LEU", rp)
    td = tmp_root/f"{tag}_f"; shutil.rmtree(td, ignore_errors=True); td.mkdir(parents=True, exist_ok=True)
    r = scRMSD(pdb_file_path=str(rp), tmp_path=str(td), num_seq_per_target=1, use_pdb_seq=True,
               rmsd_modes=["ca"], folding_models=["esmfold"], keep_outputs=True, ret_min=False)
    rm = r["ca"]["esmfold"]; co = rm[0] if rm else float("inf")
    shutil.rmtree(td, ignore_errors=True); rp.unlink(missing_ok=True)
    return co

def phase2(device="cuda:0"):
    import time
    BUDGET_S = float(os.environ.get("THROTTLE_BUDGET_S", 27000))   # ~7.5h default; guard against >8h
    t0 = time.time()
    print("="*82); print(f"PHASE 2 -- buried-driver revert + refold recovery (GPU) | budget {BUDGET_S/3600:.1f}h"); print("="*82)
    sys.path.insert(0, os.getcwd())
    from proteinfoundation.metrics.designability import scRMSD
    out_csv = OUT/"phase2_recovery.csv"
    done = set()
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv)): done.add((r["set"], r["cell"], r["protein_id"]))
    write_header = not out_csv.exists()
    tmp_root = Path("tmp/throttle_viability"); tmp_root.mkdir(parents=True, exist_ok=True)
    fo = open(out_csv, "a", newline=""); w = csv.writer(fo)
    if write_header:
        w.writerow(["set","family","cell","protein_id","L","co_before","co_after_driver","co_after_randctrl",
                    "charge_before","charge_after","charge_retained","n_drivers","n_buried_reverted",
                    "recovered_driver","recovered_randctrl","stayed_designable"]); fo.flush()
    n_fold = 0; stopped = False
    for name, S in SETS.items():
        if stopped: break
        if not S["p2"]: continue
        cells = {wkey(c): c for c in glob.glob(S["glob"])}
        for wv in S["p2"]:
            if stopped: break
            cell = cells.get(wv)
            if not cell: print(f"  [skip] {name} w{wv} not found"); continue
            cs = load_codesign(cell); gdir = Path(cell)/"guided"
            if not gdir.exists(): continue
            pdbs = sorted(gdir.glob("s*_n*.pdb")); cn = Path(cell).name
            is_pol = (S["family"] == "polarize")
            print(f"\n[{name} {cn}] {len(pdbs)} samples; codesign {sum(v<DESIG for v in cs.values())}/{len(cs)}; "
                  f"{'polar-revert + RANDOM-matched control' if is_pol else 'driver-revert only'}")
            for pi, pdb in enumerate(pdbs):
                pid = pdb.stem
                if (name, cn, pid) in done: continue
                if time.time() - t0 > BUDGET_S:
                    print(f"\n[budget] {BUDGET_S/3600:.1f}h reached after {n_fold} folds -- stopping cleanly."); stopped = True; break
                try:
                    res = parse_residues(pdb)
                    seq = "".join(r[0] for r in res); ca = [r[1] for r in res]; L = len(seq)
                    bm = burial_mask(ca); buried_idx = [o for o in range(L) if bm[o]]
                    tgt = [o for o in buried_idx if seq[o] in S["driver"]]
                    n_drv = sum(c in S["driver"] for c in seq)
                    ch_b = net_charge(seq)
                    seq_rev = list(seq)
                    for o in tgt: seq_rev[o] = "L"
                    ch_a = net_charge("".join(seq_rev))
                    co_b = cs.get(pid, float("nan"))
                    # driver-revert fold
                    if not tgt:
                        co_drv = co_b
                    else:
                        co_drv = _fold(scRMSD, tmp_root, f"{name}_{cn}_{pid}_drv", pdb, tgt); n_fold += 1
                    # matched-random control (polarize only): revert |tgt| RANDOM buried positions
                    co_rnd = float("nan")
                    if is_pol and tgt:
                        rng = np.random.RandomState(1000 + pi)
                        pool = buried_idx if len(buried_idx) >= len(tgt) else buried_idx
                        rand_tgt = list(rng.choice(buried_idx, size=min(len(tgt), len(buried_idx)), replace=False))
                        co_rnd = _fold(scRMSD, tmp_root, f"{name}_{cn}_{pid}_rnd", pdb, rand_tgt); n_fold += 1
                    rec_d = int(co_b >= DESIG and co_drv < DESIG)
                    rec_r = int(co_b >= DESIG and (co_rnd == co_rnd) and co_rnd < DESIG)  # co_rnd==co_rnd: not NaN
                    stay = int(co_b < DESIG and co_drv < DESIG)
                    ret = (ch_a/ch_b) if ch_b else float("nan")
                    w.writerow([name, S["family"], cn, pid, L, f"{co_b:.4f}", f"{co_drv:.4f}",
                                (f"{co_rnd:.4f}" if co_rnd==co_rnd else "nan"), ch_b, ch_a, f"{ret:.3f}",
                                n_drv, len(tgt), rec_d, rec_r, stay]); fo.flush()
                    tag = "REC" if rec_d else ""
                    extra = f" rand {co_rnd:.2f}" if co_rnd==co_rnd else ""
                    print(f"  {pid}: co {co_b:.2f}->drv {co_drv:.2f}{extra}  buriedDrv={len(tgt)}/{n_drv}  {tag}")
                except Exception as e:
                    print(f"  {pid}: ERROR {e}"); continue
    fo.close()
    print(f"\n=== PHASE 2 SUMMARY ({n_fold} folds, {(time.time()-t0)/3600:.1f}h) ===")
    import collections
    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(out_csv)): agg[(r["set"], r["cell"])].append(r)
    print(f"{'set':15s} {'cell':22s} {'fam':11s} {'collapsed':>9} {'rec_drv':>8} {'rec_rnd':>8}")
    for (name, cn), rs in sorted(agg.items()):
        fam = rs[0]["family"]
        coll = [r for r in rs if float(r["co_before"]) >= DESIG]
        rd = sum(int(r["recovered_driver"]) for r in coll)
        rr = sum(int(r["recovered_randctrl"]) for r in coll)
        rdp = f"{rd}/{len(coll)}={rd/max(1,len(coll)):.0%}"
        rrp = (f"{rr}/{len(coll)}={rr/max(1,len(coll)):.0%}" if fam=="polarize" else "-")
        print(f"{name:15s} {cn:22s} {fam:11s} {len(coll):>9} {rdp:>8} {rrp:>8}")
    print("\nINTERPRET: rec_drv >> rec_rnd  => buried DRIVER residues specifically were the damage => burial throttle")
    print("           VIABLE. rec_drv ~ rec_rnd (both high) => trivial poly-Leu-core artifact, not driver-specific.")
    print("           rec_drv ~ 0 => global/compositional collapse => throttle DEAD. Bar to pursue: rec_drv>20% AND >rec_rnd.")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all","phase1"): phase1()
    if mode in ("all","phase2"): phase2()
    print("\nDONE.")
