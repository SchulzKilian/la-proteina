#!/usr/bin/env python3
"""Eval for the burial-throttle camsol probe: is the throttle ABOVE the
no-throttle codesign-vs-delivered frontier, or just w-reduction?

Per cell: codesign rate (ESMFold use_pdb_seq, coScRMSD_ca<2.0), a TOOL-FREE
delivery proxy = mean Kyte-Doolittle hydrophobicity (camsol_max lowers it;
lower = more steered toward solubility), and the throttle s_mean from
diagnostics. Throttle cells (b010/b025 @ w64) are overlaid on the EXISTING
no-throttle baseline frontier (noise_aware_high_w camsol_max w32/48/64/128).
VERDICT: at matched delivered hydrophobicity, is throttle codesign > baseline?
"""
import csv, glob, os, sys, shutil
from pathlib import Path
import numpy as np

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
          'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
      'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':0.0}
DESIG = 2.0
PROBE = Path("results/burial_camsol_probe")

THROTTLE_CELLS = {  # label -> guided dir
    "b0 (no throttle) w64":  PROBE/"b0",
    "b010 (burial .10) w64": PROBE/"b010",
    "b025 (burial .25) w64": PROBE/"b025",
}
BASELINE_CELLS = {  # existing no-throttle frontier
    f"baseline w{w}": Path(f"results/noise_aware_high_w_scout/camsol_max_w{w}")
    for w in (32, 48, 64, 128)
}

def seq_kd(pdb):
    s = [AA3TO1.get(l[17:20].strip(),'X') for l in open(pdb) if l.startswith("ATOM") and l[12:16].strip()=="CA"]
    if not s: return None, 0
    return float(np.mean([KD[c] for c in s])), len(s)

def codesign_one(scRMSD, pdb, tmp_root):
    td = tmp_root/pdb.stem; shutil.rmtree(td, ignore_errors=True); td.mkdir(parents=True, exist_ok=True)
    try:
        r = scRMSD(pdb_file_path=str(pdb), tmp_path=str(td), num_seq_per_target=1, use_pdb_seq=True,
                   rmsd_modes=["ca"], folding_models=["esmfold"], keep_outputs=True, ret_min=False)
        rm = r["ca"]["esmfold"]; co = rm[0] if rm else float("inf")
    except Exception as e:
        co = float("inf")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return co

def s_mean_of(cell_dir):
    import json
    vals = []
    for j in (cell_dir/"diagnostics").glob("*_diagnostics.json") if (cell_dir/"diagnostics").exists() else []:
        try:
            d = json.load(open(j))
            steps = d if isinstance(d, list) else d.get("diagnostics", d.get("steering_diagnostics", []))
            for st in (steps if isinstance(steps, list) else []):
                th = st.get("throttle") if isinstance(st, dict) else None
                if isinstance(th, dict) and "s_mean" in th: vals.append(th["s_mean"])
        except Exception: pass
    return float(np.mean(vals)) if vals else float("nan")

def eval_cell(scRMSD, label, gdir, tmp_root):
    guided = gdir/"guided" if (gdir/"guided").exists() else gdir
    pdbs = sorted(guided.glob("s*_n*.pdb"))
    if not pdbs: return None
    csvf = gdir/"codesign_burial.csv"
    done = {}
    # reuse existing on-disk codesign (baseline cells already have it -> no refold)
    pre = gdir/"codesign_guided.csv"
    if pre.exists():
        for r in csv.DictReader(open(pre)):
            try: done[r["protein_id"]] = float(r["coScRMSD_ca"])
            except Exception: pass
    if csvf.exists():
        for r in csv.DictReader(open(csvf)): done[r["pid"]] = float(r["co"])
    rows = []
    fo = open(csvf, "a", newline=""); w = csv.writer(fo)
    if csvf.stat().st_size == 0: w.writerow(["pid","co","kd"]); fo.flush()
    for pdb in pdbs:
        pid = pdb.stem; kd,_ = seq_kd(pdb)
        if pid in done: co = done[pid]
        else:
            co = codesign_one(scRMSD, pdb, tmp_root)
            w.writerow([pid, f"{co:.4f}", f"{kd:.4f}"]); fo.flush()
        rows.append((co, kd))
    fo.close()
    co = np.array([r[0] for r in rows]); kd = np.array([r[1] for r in rows])
    return dict(n=len(rows), codes=int((co<DESIG).sum()), codesfrac=float((co<DESIG).mean()),
                medco=float(np.median(co)), kd=float(kd.mean()), smean=s_mean_of(gdir))

def main():
    sys.path.insert(0, os.getcwd())
    from proteinfoundation.metrics.designability import scRMSD
    tmp_root = Path("tmp/burial_camsol"); tmp_root.mkdir(parents=True, exist_ok=True)
    allcells = {**BASELINE_CELLS, **THROTTLE_CELLS}
    stats = {}
    print(f"{'cell':26s} {'n':>3} {'codes%':>7} {'medCO':>7} {'meanKD':>7} {'s_mean':>7}")
    for label, d in allcells.items():
        st = eval_cell(scRMSD, label, d, tmp_root); stats[label] = st
        if st: print(f"{label:26s} {st['n']:>3} {st['codesfrac']:>6.0%} {st['medco']:>7.2f} {st['kd']:>7.3f} {st['smean']:>7.3f}")
        else:  print(f"{label:26s}  (no data yet)")
    # frontier: baseline codesfrac vs delivered KD (lower KD = more steered)
    base = [(stats[l]["kd"], stats[l]["codesfrac"]) for l in BASELINE_CELLS if stats.get(l)]
    base.sort()  # ascending KD
    def base_at(kd):
        if len(base)<2: return None
        xs=[b[0] for b in base]; ys=[b[1] for b in base]
        return float(np.interp(kd, xs, ys))
    print("\n=== VERDICT (throttle vs no-throttle baseline at matched delivered KD-hydrophobicity) ===")
    print("(camsol_max lowers KD; lower meanKD = more steering delivered)")
    for l in THROTTLE_CELLS:
        st = stats.get(l)
        if not st: print(f"{l}: no data"); continue
        b = base_at(st["kd"]);
        if b is None: print(f"{l}: codes {st['codesfrac']:.0%} @ KD {st['kd']:.3f} (no baseline curve)"); continue
        d = st["codesfrac"]-b
        flag = "ABOVE frontier (throttle helps)" if d>0.10 else ("ON frontier (= w-reduction)" if abs(d)<=0.10 else "BELOW")
        print(f"{l}: codes {st['codesfrac']:.0%} @ KD {st['kd']:.3f} | baseline@sameKD≈{b:.0%} | Δ={d:+.0%} -> {flag}")
    print("\nNOTE: KD is a tool-free delivery proxy (camsol_intrinsic is always-NaN in the dev panel).")

if __name__ == "__main__":
    main()
