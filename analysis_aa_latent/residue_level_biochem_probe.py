"""
Residue-level biochemical-axis probe (large-n, quotable).

The AA-centroid probe (E114) is intrinsically n=20 (only 20 amino acids). This is
the large-n companion: predict each RESIDUE's biochemical property (a function of
its amino-acid identity) from its per-residue 8-d latent, with a held-out-PROTEIN
train/test split (no within-protein leakage) and a bootstrap-over-test-proteins
95% CI on test R^2.

Interpretation note: at the residue level the property is identity-determined, so
this R^2 measures how linearly the latent lets you *read off* the property per
residue (ceiling set by the 99.4% identity-decodability), which is a different and
larger-n question than "do the 20 AA centroids align with the property axis".
Both are reported.

CPU only.
"""
import glob, random
import numpy as np
import torch
from sklearn.linear_model import RidgeCV

LATENT_DIR = "data/pdb_train/processed_latents_300_800"
N_PROTEINS = 4000
TEST_FRAC = 0.25
N_BOOT = 300
SEED = 42

RESTYPES = ["A","R","N","D","C","Q","E","G","H","I",
            "L","K","M","F","P","S","T","W","Y","V"]
KD_HYDRO = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
            "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
            "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
VOLUME = {"A":88.6,"R":173.4,"N":114.1,"D":111.1,"C":108.5,"Q":143.8,"E":138.4,
          "G":60.1,"H":153.2,"I":166.7,"L":166.7,"K":168.6,"M":162.9,"F":189.9,
          "P":112.7,"S":89.0,"T":116.1,"W":227.8,"Y":193.6,"V":140.0}
NET_CHARGE = {"D":-1.0,"E":-1.0,"K":1.0,"R":1.0,"H":0.1}
HELIX_DDG = {"A":0.00,"L":0.21,"R":0.21,"M":0.24,"K":0.26,"Q":0.39,"E":0.40,
             "I":0.41,"W":0.49,"S":0.50,"Y":0.53,"F":0.54,"H":0.61,"V":0.61,
             "N":0.65,"T":0.66,"C":0.68,"D":0.69,"G":1.00,"P":3.16}

def prop_lut(d, default=0.0):
    return np.array([d.get(aa, default) for aa in RESTYPES], dtype=np.float64)

def main():
    files = sorted(glob.glob(f"{LATENT_DIR}/*/*.pt"))
    random.seed(SEED); random.shuffle(files); files = files[:N_PROTEINS]
    print(f"[load] {len(files)} proteins from {LATENT_DIR}", flush=True)

    Xs, aas, pids = [], [], []
    for i, f in enumerate(files):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        g = (lambda k: d[k]) if isinstance(d, dict) else (lambda k: getattr(d, k))
        m = np.asarray(g("mean"), dtype=np.float32)
        rt = np.asarray(g("residue_type")).astype(int)
        v = rt < 20
        m, rt = m[v], rt[v]
        Xs.append(m); aas.append(rt); pids.append(np.full(len(rt), i, dtype=np.int32))
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    X = np.concatenate(Xs); aa = np.concatenate(aas); pid = np.concatenate(pids)
    print(f"[data] {X.shape[0]} residues, {len(set(pid))} proteins", flush=True)

    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(pid.tolist())))
    test_p = set(rng.choice(uniq, size=int(len(uniq)*TEST_FRAC), replace=False).tolist())
    test_mask = np.array([p in test_p for p in pid])
    tr, te = ~test_mask, test_mask
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    Xtr = (X[tr]-mu)/sd; Xte = (X[te]-mu)/sd
    pid_te = pid[te]
    n_tr_p = len(set(pid[tr].tolist())); n_te_p = len(test_p)
    print(f"[split] train {Xtr.shape[0]} res / {n_tr_p} prot | test {Xte.shape[0]} res / {n_te_p} prot\n")

    print(f"{'axis':20s} {'test R2':>9s}   {'95% CI (bootstrap over test proteins)':>38s}")
    axes = {"KD_hydrophobicity": prop_lut(KD_HYDRO),
            "residue_volume":    prop_lut(VOLUME),
            "net_charge":        prop_lut(NET_CHARGE),
            "helix_propensity":  prop_lut(HELIX_DDG)}
    te_pids = np.array(sorted(test_p))
    # group test-residue indices by protein for fast bootstrap
    idx_by_p = {p: np.where(pid_te == p)[0] for p in te_pids}

    for name, lut in axes.items():
        ytr = lut[aa[tr]]; yte = lut[aa[te]]
        r = RidgeCV(alphas=[0.01,0.1,1,10,100]).fit(Xtr, ytr - ytr.mean())
        pred = r.predict(Xte) + ytr.mean()
        ss_res = np.sum((yte-pred)**2); ss_tot = np.sum((yte-yte.mean())**2)
        r2 = 1 - ss_res/ss_tot
        # bootstrap over test proteins
        boot = []
        for _ in range(N_BOOT):
            samp = rng.choice(te_pids, size=len(te_pids), replace=True)
            ii = np.concatenate([idx_by_p[p] for p in samp])
            yy = yte[ii]; pp = pred[ii]
            sr = np.sum((yy-pp)**2); st = np.sum((yy-yy.mean())**2)
            boot.append(1 - sr/st)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{name:20s} {r2:>9.3f}   [{lo:.3f}, {hi:.3f}]")

    # AA-centroid (n=20) numbers on the same data, for continuity
    print("\n[AA-centroid n=20, same data] in-sample multi-dim R2 (context only):")
    cen = np.zeros((20,8))
    for k in range(20):
        cen[k] = X[aa==k].mean(0)
    cen = (cen-mu)/sd
    for name, lut in axes.items():
        from numpy.linalg import lstsq
        A = np.column_stack([cen, np.ones(20)]); y = lut - lut.mean()
        w,_,_,_ = lstsq(A, y, rcond=None); p = A@w
        r2 = 1 - np.sum((y-p)**2)/np.sum(y**2)
        print(f"  {name:20s} R2 = {r2:+.2f}")

if __name__ == "__main__":
    main()
