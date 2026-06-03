"""
AA-latent clustering: does La-Proteina's 8-dim per-residue latent organize the
20 amino acids by biochemistry?

For a subsample of cached latents we compute the mean 8-d latent vector per amino
acid, then (a) hierarchically cluster the 20 AAs, (b) report each AA's nearest
latent neighbour, and (c) probe how much of four biochemical axes
(Kyte-Doolittle hydrophobicity, residue volume, net charge, helix propensity)
the latent geometry recovers, via leave-one-out ridge (n=20) and best single-dim
Pearson. Also reports nearest-neighbour class purity for hydrophobic / polar /
charged / aromatic / special, and for the user's specific asks: beta-branched,
hydroxyl, helix-former groupings.

CPU only. No model loaded.
"""
import glob, random, sys
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut

LATENT_DIR = "data/pdb_train/processed_latents_300_800"
N_PROTEINS = 2000
SEED = 42

# OpenFold restype order (alphabetical by 3-letter), index 0..19, 20 = X
RESTYPES = ["A","R","N","D","C","Q","E","G","H","I",
            "L","K","M","F","P","S","T","W","Y","V"]

# --- biochemical reference scalars (standard literature values) ---
KD_HYDRO = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
            "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
            "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
VOLUME = {"A":88.6,"R":173.4,"N":114.1,"D":111.1,"C":108.5,"Q":143.8,"E":138.4,
          "G":60.1,"H":153.2,"I":166.7,"L":166.7,"K":168.6,"M":162.9,"F":189.9,
          "P":112.7,"S":89.0,"T":116.1,"W":227.8,"Y":193.6,"V":140.0}
NET_CHARGE = {"D":-1.0,"E":-1.0,"K":1.0,"R":1.0,"H":0.1}  # default 0
# Pace & Scholtz helix propensity (kcal/mol; lower = stronger helix former)
HELIX_DDG = {"A":0.00,"L":0.21,"R":0.21,"M":0.24,"K":0.26,"Q":0.39,"E":0.40,
             "I":0.41,"W":0.49,"S":0.50,"Y":0.53,"F":0.54,"H":0.61,"V":0.61,
             "N":0.65,"T":0.66,"C":0.68,"D":0.69,"G":1.00,"P":3.16}

# qualitative classes
CLASS = {"A":"hydrophobic","V":"hydrophobic","L":"hydrophobic","I":"hydrophobic",
         "M":"hydrophobic","F":"aromatic","W":"aromatic","Y":"aromatic","H":"charged_pos",
         "C":"special","G":"special","P":"special",
         "S":"polar","T":"polar","N":"polar","Q":"polar",
         "D":"charged_neg","E":"charged_neg","K":"charged_pos","R":"charged_pos"}
BETA_BRANCHED = {"V","I","T"}
HYDROXYL = {"S","T","Y"}
HELIX_FORMER = {"A","L","M","E","Q","K","R"}  # classic strong helix formers

def load_files():
    files = sorted(glob.glob(f"{LATENT_DIR}/*/*.pt"))
    random.seed(SEED)
    random.shuffle(files)
    return files[:N_PROTEINS]

def main():
    files = load_files()
    print(f"[load] sampling {len(files)} proteins from {LATENT_DIR}", flush=True)
    D = 8
    sums = np.zeros((20, D)); counts = np.zeros(20)
    all_resid = []  # for global std
    n_res_total = 0
    for i, f in enumerate(files):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            continue
        g = (lambda k: d[k]) if isinstance(d, dict) else (lambda k: getattr(d, k))
        m = np.asarray(g("mean"), dtype=np.float64)           # [L,8]
        rt = np.asarray(g("residue_type")).astype(int)         # [L]
        valid = rt < 20
        m, rt = m[valid], rt[valid]
        np.add.at(sums, rt, m)
        np.add.at(counts, rt, 1)
        n_res_total += len(rt)
        if len(all_resid) < 200000:  # cap memory for global std
            all_resid.append(m)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(files)} proteins, {n_res_total} residues", flush=True)

    mean_vec = sums / counts[:, None]                          # [20,8]
    glob_std = np.concatenate(all_resid, 0).std(0)             # [8]
    Z = mean_vec / glob_std[None, :]                           # z-scored per dim

    print(f"\n[counts] residues per AA (total {int(counts.sum())}):")
    print("  " + "  ".join(f"{RESTYPES[i]}:{int(counts[i])}" for i in range(20)))

    # --- nearest latent neighbour per AA (z-scored Euclidean) ---
    Dmat = squareform(pdist(Z, metric="euclidean"))
    np.fill_diagonal(Dmat, np.inf)
    print("\n[nearest latent neighbour]  AA -> closest AA (same biochem class?)")
    nn_same = 0
    for i, aa in enumerate(RESTYPES):
        j = int(np.argmin(Dmat[i]))
        same = CLASS[aa] == CLASS[RESTYPES[j]]
        nn_same += same
        print(f"  {aa} ({CLASS[aa]:11s}) -> {RESTYPES[j]} ({CLASS[RESTYPES[j]]:11s}) {'SAME' if same else ''}")
    print(f"  nearest-neighbour class purity: {nn_same}/20 = {nn_same/20:.0%}")

    # --- hierarchical clustering (Ward on z-scored vectors) ---
    Lk = linkage(Z, method="ward")
    print("\n[hierarchical clustering] flat clusters:")
    for k in (4, 5, 6):
        lab = fcluster(Lk, k, criterion="maxclust")
        groups = {}
        for aa, c in zip(RESTYPES, lab):
            groups.setdefault(int(c), []).append(aa)
        print(f"  k={k}: " + " | ".join("{" + ",".join(sorted(v)) + "}" for v in groups.values()))

    # save dendrogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Lk, labels=RESTYPES, ax=ax, color_threshold=0.7*max(Lk[:,2]))
        ax.set_title("Amino acids clustered by mean 8-d La-Proteina latent (Ward, z-scored)")
        ax.set_ylabel("Ward distance")
        fig.tight_layout(); fig.savefig("analysis_aa_latent/aa_dendrogram.png", dpi=150)
        print("  saved analysis_aa_latent/aa_dendrogram.png")
    except Exception as e:
        print("  (dendrogram fig skipped:", e, ")")

    # --- biochemical-axis probes ---
    print("\n[biochemical axis recovery]  leave-one-out ridge R^2 (n=20) + best |single-dim Pearson|")
    def vec_for(prop, default=0.0):
        return np.array([prop.get(aa, default) for aa in RESTYPES], dtype=np.float64)
    axes = {"KD_hydrophobicity": vec_for(KD_HYDRO),
            "residue_volume":    vec_for(VOLUME),
            "net_charge":        vec_for(NET_CHARGE, 0.0),
            "helix_propensity":  vec_for(HELIX_DDG)}
    X = Z  # 20 x 8
    for name, y in axes.items():
        yc = (y - y.mean())
        # LOO ridge
        loo = LeaveOneOut(); preds = np.zeros(20)
        for tr, te in loo.split(X):
            r = RidgeCV(alphas=[0.01,0.1,1,10,100]).fit(X[tr], yc[tr])
            preds[te] = r.predict(X[te])
        ss_res = np.sum((yc - preds)**2); ss_tot = np.sum(yc**2)
        loo_r2 = 1 - ss_res/ss_tot
        # best single-dim Pearson
        pear = [abs(np.corrcoef(X[:,dd], y)[0,1]) for dd in range(8)]
        bd = int(np.argmax(pear))
        print(f"  {name:20s}  LOO-R2 = {loo_r2:+.2f}   best dim = {bd} (|r|={pear[bd]:.2f})")

    # --- specific user groupings: are they latent clusters? (NN purity within group) ---
    print("\n[user-asked groupings]  fraction whose nearest latent neighbour is in the same group")
    for gname, gset in [("beta_branched(V,I,T)", BETA_BRANCHED),
                        ("hydroxyl(S,T,Y)", HYDROXYL),
                        ("helix_formers(A,L,M,E,Q,K,R)", HELIX_FORMER),
                        ("aromatic(F,Y,W,H)", {"F","Y","W","H"})]:
        idx = [RESTYPES.index(a) for a in gset]
        hit = 0
        for i in idx:
            j = int(np.argmin(Dmat[i]))
            hit += RESTYPES[j] in gset
        print(f"  {gname:32s} {hit}/{len(idx)}")

    # save matrix
    np.savetxt("analysis_aa_latent/mean_latent_per_aa.csv",
               np.column_stack([np.arange(20), mean_vec]),
               header="aa_idx," + ",".join(f"dim{d}" for d in range(8)),
               delimiter=",", comments="")
    print("\n  saved analysis_aa_latent/mean_latent_per_aa.csv")

if __name__ == "__main__":
    main()
