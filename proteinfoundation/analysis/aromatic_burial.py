#!/usr/bin/env python3
"""Aromatic burial analysis: gen vs PDB. RSA via FreeSASA + Tien et al. 2013 max ASA.

Inputs may be either .pdb (read directly) or .pt PyG Data with the latent-cache
keys `coords_nm` (L,37,3, nanometres, OpenFold atom order), `coord_mask` (L,37),
`residue_type` (L,). Multi-chain: residues from all chains are pooled.
Bootstrap resamples PROTEINS (1000 resamples), not residues.

Example:
    python proteinfoundation/analysis/aromatic_burial.py \\
        --gen-dir inference/inference_ucond_notri \\
        --ref-dir data/pdb_train/processed_latents_300_800 \\
        --out-dir results/aromatic_burial \\
        --n-ref-sample 1000
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import freesasa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from openfold.np.residue_constants import atom_types as OF_ATOMS, restypes, restype_1to3

freesasa.setVerbosity(freesasa.silent)
warnings.filterwarnings("ignore")

# Tien et al. 2013 theoretical max ASA (Å²).
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
AA_LIST = list(MAX_ASA.keys())
AA_IDX = {a: i for i, a in enumerate(AA_LIST)}
THREE_TO_ONE = {restype_1to3[a]: a for a in restypes}
AROMATIC = ["W", "F", "Y", "H"]

N_BOOT = 1000
N_RSA_BINS = 20
RSA_EDGES = np.linspace(0.0, 1.0, N_RSA_BINS + 1)
RSA_CENTERS = 0.5 * (RSA_EDGES[:-1] + RSA_EDGES[1:])


# ---------- I/O ----------

def build_pdb_str(coords_nm: np.ndarray, coord_mask: np.ndarray, residue_type: np.ndarray) -> str:
    """coords in nm → Å, OpenFold-37 atom order. Skip masked atoms; skip unknown AAs."""
    lines = []
    serial = 1
    for ri in range(len(residue_type)):
        rt = int(residue_type[ri])
        if rt >= 20:
            continue
        aa3 = restype_1to3[restypes[rt]]
        for ai, aname in enumerate(OF_ATOMS):
            if not coord_mask[ri, ai]:
                continue
            x, y, z = (coords_nm[ri, ai] * 10.0).tolist()
            label = aname.ljust(4) if len(aname) < 4 else aname[:4]
            lines.append(
                f"ATOM  {serial:5d} {label} {aa3} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
            serial += 1
    lines.append("END\n")
    return "".join(lines)


def sasa_residues(pdb_path: str):
    """Return list of (aa3letter, total_sasa) over all chains, or None on failure."""
    try:
        s = freesasa.Structure(pdb_path)
        r = freesasa.calc(s)
        ra = r.residueAreas()
    except Exception:
        return None
    out = []
    for ch in ra.values():
        for area in ch.values():
            out.append((area.residueType, float(area.total)))
    return out or None


def process_file(path: Path):
    """Return (aas, rsa) arrays for valid residues, or None."""
    if path.suffix == ".pt":
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
            cn = d["coords_nm"].numpy()
            cm = d["coord_mask"].numpy().astype(bool)
            rt = d["residue_type"].numpy()
            pdb_str = build_pdb_str(cn, cm, rt)
        except Exception:
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            fh.write(pdb_str)
            tmp = fh.name
        try:
            sasa = sasa_residues(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
    elif path.suffix == ".pdb":
        sasa = sasa_residues(str(path))
    else:
        return None
    if not sasa:
        return None
    aas, rsas = [], []
    for aa3, total in sasa:
        aa1 = THREE_TO_ONE.get(aa3.upper())
        if aa1 is None:
            continue
        rsa = total / MAX_ASA[aa1]
        rsas.append(min(max(rsa, 0.0), 1.5))
        aas.append(aa1)
    if not aas:
        return None
    return np.array(aas), np.array(rsas)


def discover_files(directory: Path, sample_n: int | None, seed: int):
    paths = sorted(list(directory.rglob("*.pdb")) + list(directory.rglob("*.pt")))
    if sample_n is not None and len(paths) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=sample_n, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def load_set(directory: Path, sample_n: int | None, seed: int, label: str):
    paths = discover_files(directory, sample_n, seed)
    print(f"[{label}] discovered {len(paths)} files (after sampling) under {directory}")
    proteins, failed = [], 0
    for i, p in enumerate(paths, 1):
        out = process_file(p)
        if out is None:
            failed += 1
        else:
            proteins.append((p.stem, *out))
        if i % 200 == 0:
            print(f"  [{label}] processed {i}/{len(paths)}  (failed so far: {failed})")
    return proteins, failed, len(paths)


def length_match(gen_proteins, ref_proteins, target_n: int, rng,
                 bin_width: int = 50):
    """Resample ref to match gen's length distribution.
    Per length bin: sample (without replacement when possible) a count
    proportional to the gen frequency in that bin, totalling ~target_n.
    """
    gen_lens = np.array([len(p[1]) for p in gen_proteins])
    ref_lens = np.array([len(p[1]) for p in ref_proteins])
    lo = min(gen_lens.min(), ref_lens.min())
    hi = max(gen_lens.max(), ref_lens.max())
    edges = np.arange(lo, hi + bin_width + 1, bin_width)
    gen_hist, _ = np.histogram(gen_lens, bins=edges)
    if gen_hist.sum() == 0:
        return ref_proteins[:target_n]
    gen_freq = gen_hist / gen_hist.sum()
    ref_bin = np.clip(np.digitize(ref_lens, edges) - 1, 0, len(edges) - 2)
    matched = []
    for b in range(len(edges) - 1):
        n_target = int(round(target_n * gen_freq[b]))
        if n_target == 0:
            continue
        candidates = np.where(ref_bin == b)[0]
        if len(candidates) == 0:
            continue
        replace = n_target > len(candidates)
        chosen = rng.choice(candidates, size=n_target, replace=replace)
        matched.extend(ref_proteins[i] for i in chosen)
    return matched


# ---------- features + bootstrap ----------

def bin_burial(rsa: np.ndarray) -> np.ndarray:
    return np.where(rsa < 0.20, 0, np.where(rsa < 0.50, 1, 2))


def per_protein_features(proteins):
    P = len(proteins)
    n_res = np.zeros(P, dtype=np.int64)
    n_per_aa = np.zeros((P, 20), dtype=np.int64)
    n_per_bin = np.zeros((P, 3), dtype=np.int64)
    n_per_aa_per_bin = np.zeros((P, 20, 3), dtype=np.int64)
    n_per_rsa = np.zeros((P, N_RSA_BINS), dtype=np.int64)
    n_per_aa_per_rsa = np.zeros((P, 20, N_RSA_BINS), dtype=np.int64)
    lens = np.zeros(P, dtype=np.int64)
    for p, (_, aas, rsa) in enumerate(proteins):
        bins = bin_burial(rsa)
        rsa_bin = np.clip(np.digitize(rsa, RSA_EDGES) - 1, 0, N_RSA_BINS - 1)
        aa_i = np.array([AA_IDX[a] for a in aas])
        lens[p] = n_res[p] = len(aas)
        np.add.at(n_per_aa[p], aa_i, 1)
        np.add.at(n_per_bin[p], bins, 1)
        np.add.at(n_per_aa_per_bin[p], (aa_i, bins), 1)
        np.add.at(n_per_rsa[p], rsa_bin, 1)
        np.add.at(n_per_aa_per_rsa[p], (aa_i, rsa_bin), 1)
    return {
        "n_res": n_res, "lens": lens,
        "n_per_aa": n_per_aa, "n_per_bin": n_per_bin,
        "n_per_aa_per_bin": n_per_aa_per_bin,
        "n_per_rsa": n_per_rsa, "n_per_aa_per_rsa": n_per_aa_per_rsa,
    }


def aa_indices(residues):
    return np.array([AA_IDX[r] for r in residues])


def bootstrap(feat, stat_fn, rng, n_boot=N_BOOT):
    P = len(feat["n_res"])
    samples = [stat_fn(feat, rng.integers(0, P, size=P)) for _ in range(n_boot)]
    arr = np.asarray(samples, dtype=float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return np.nanmean(arr, axis=0), lo, hi


def _safe_div(num, den):
    return num / den if den > 0 else np.nan


def stat_overall(residues):
    ai = aa_indices(residues)
    def fn(feat, idx):
        return _safe_div(feat["n_per_aa"][idx][:, ai].sum(), feat["n_res"][idx].sum())
    return fn


def stat_in_bin(residues, bin_id):
    ai = aa_indices(residues)
    def fn(feat, idx):
        return _safe_div(feat["n_per_aa_per_bin"][idx][:, ai, bin_id].sum(),
                         feat["n_per_bin"][idx, bin_id].sum())
    return fn


def stat_burial_ratio(residues):
    ai = aa_indices(residues)
    def fn(feat, idx):
        f_b = _safe_div(feat["n_per_aa_per_bin"][idx][:, ai, 0].sum(), feat["n_per_bin"][idx, 0].sum())
        f_e = _safe_div(feat["n_per_aa_per_bin"][idx][:, ai, 2].sum(), feat["n_per_bin"][idx, 2].sum())
        if not np.isfinite(f_b) or not np.isfinite(f_e) or f_e == 0:
            return np.nan
        return f_b / f_e
    return fn


def stat_curve(residues):
    ai = aa_indices(residues)
    def fn(feat, idx):
        num = feat["n_per_aa_per_rsa"][idx][:, ai, :].sum(axis=(0, 1))
        den = feat["n_per_rsa"][idx].sum(axis=0)
        out = np.full(N_RSA_BINS, np.nan)
        m = den > 0
        out[m] = num[m] / den[m]
        return out
    return fn


def summarize(feat, label, rng):
    rows = []
    cols = [("overall", stat_overall),
            ("buried", lambda r: stat_in_bin(r, 0)),
            ("intermediate", lambda r: stat_in_bin(r, 1)),
            ("exposed", lambda r: stat_in_bin(r, 2)),
            ("ratio_bur_exp", stat_burial_ratio)]
    for residues, name in [(["W"], "W"), (["F"], "F"), (["Y"], "Y"), (["H"], "H"),
                            (AROMATIC, "Aromatic")]:
        row = {"set": label, "residue": name}
        for col, factory in cols:
            m, lo, hi = bootstrap(feat, factory(residues), rng)
            row[col], row[f"{col}_lo"], row[f"{col}_hi"] = m, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curve(curves, ax, title, ylabel="P(aromatic)"):
    for label, (mean, lo, hi), color in curves:
        ax.plot(RSA_CENTERS, mean, label=label, color=color, lw=1.5)
        ax.fill_between(RSA_CENTERS, lo, hi, alpha=0.25, color=color, lw=0)
    ax.axvline(0.20, color="k", ls=":", alpha=0.4)
    ax.axvline(0.50, color="k", ls=":", alpha=0.4)
    ax.set_xlabel("Relative SASA")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-ref-sample", type=int, default=1000,
                    help="Random subsample of ref dir (uniform). Set 0/None to keep all.")
    ap.add_argument("--max-gen", type=int, default=None,
                    help="Optional cap on gen set size.")
    ap.add_argument("--ref-oversample-factor", type=int, default=3,
                    help="How many times more refs to draw than --n-ref-sample, "
                         "before length-matching down to n-ref-sample.")
    ap.add_argument("--no-length-match", action="store_true",
                    help="Skip the post-hoc length-match resample of ref.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_ref = None if (args.n_ref_sample is None or args.n_ref_sample <= 0) else args.n_ref_sample
    gen, gen_failed, gen_total = load_set(args.gen_dir, args.max_gen, args.seed, "gen")
    # Oversample the ref draw so we have enough candidates in each length
    # bin for the post-hoc length-match step. Final ref size ≈ n_ref.
    ref_draw = None if n_ref is None else min(args.ref_oversample_factor * n_ref, 64000)
    ref_full, ref_failed, ref_total = load_set(args.ref_dir, ref_draw, args.seed + 1, "ref")
    print(f"\nparsed: gen={len(gen)}/{gen_total} ({gen_failed} failed)   "
          f"ref={len(ref_full)}/{ref_total} ({ref_failed} failed)")
    if not gen or not ref_full:
        raise SystemExit("One set is empty after parsing — aborting.")

    if n_ref and not args.no_length_match:
        ref = length_match(gen, ref_full, target_n=n_ref, rng=rng)
        print(f"[length-match] ref resampled to {len(ref)} proteins matched to gen length distribution")
    else:
        ref = ref_full

    gen_feat = per_protein_features(gen)
    ref_feat = per_protein_features(ref)

    sanity = []
    g_n, r_n = gen_feat["n_res"].sum(), ref_feat["n_res"].sum()
    print(f"\nResidue totals: gen={g_n}, ref={r_n}")
    if max(g_n, r_n) > 2 * min(g_n, r_n):
        sanity.append(f"WARN: residue counts differ >2x (gen={g_n}, ref={r_n})")
    g_ok, r_ok = len(gen) / max(gen_total, 1), len(ref) / max(ref_total, 1)
    print(f"FreeSASA parse rate: gen={g_ok:.1%}, ref={r_ok:.1%}")
    if g_ok < 0.95: sanity.append(f"WARN: gen parse rate {g_ok:.1%} < 95%")
    if r_ok < 0.95: sanity.append(f"WARN: ref parse rate {r_ok:.1%} < 95%")
    g_l, r_l = gen_feat["lens"], ref_feat["lens"]
    print(f"Length: gen median={np.median(g_l):.0f} (IQR {np.percentile(g_l,25):.0f}-{np.percentile(g_l,75):.0f}), "
          f"ref median={np.median(r_l):.0f} (IQR {np.percentile(r_l,25):.0f}-{np.percentile(r_l,75):.0f})")
    if abs(np.median(g_l) - np.median(r_l)) > 0.1 * np.median(r_l):
        sanity.append(f"WARN: median length differs by >10% (gen={np.median(g_l):.0f}, ref={np.median(r_l):.0f})")

    df = pd.concat([summarize(gen_feat, "gen", rng), summarize(ref_feat, "ref", rng)], ignore_index=True)
    df.to_csv(args.out_dir / "aromatic_frequencies.csv", index=False)
    print("\nFrequency / burial-ratio table (95% bootstrap CIs):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBootstrapping continuous frequency curves ...")
    gen_arom = bootstrap(gen_feat, stat_curve(AROMATIC), rng)
    ref_arom = bootstrap(ref_feat, stat_curve(AROMATIC), rng)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    plot_curve([("gen", gen_arom, "C0"), ("PDB", ref_arom, "C1")], ax, "P(aromatic) vs RSA")
    fig.tight_layout()
    fig.savefig(args.out_dir / "aromatic_vs_rsa.png", dpi=150)
    fig.savefig(args.out_dir / "aromatic_vs_rsa.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, res in zip(axes.flat, AROMATIC):
        g = bootstrap(gen_feat, stat_curve([res]), rng)
        r = bootstrap(ref_feat, stat_curve([res]), rng)
        plot_curve([("gen", g, "C0"), ("PDB", r, "C1")], ax, f"P({res}) vs RSA", ylabel=f"P({res})")
    fig.tight_layout()
    fig.savefig(args.out_dir / "aromatic_by_residue.png", dpi=150)
    fig.savefig(args.out_dir / "aromatic_by_residue.pdf")
    plt.close(fig)

    print("\nBurial-targeting ratio  P(aa|buried) / P(aa|exposed):")
    flagged = []
    for name in ["W", "F", "Y", "H", "Aromatic"]:
        g = df[(df["set"] == "gen") & (df["residue"] == name)].iloc[0]
        r = df[(df["set"] == "ref") & (df["residue"] == name)].iloc[0]
        rg, rr = g["ratio_bur_exp"], r["ratio_bur_exp"]
        print(f"  {name:8s}  gen={rg:.2f} [{g['ratio_bur_exp_lo']:.2f}, {g['ratio_bur_exp_hi']:.2f}]   "
              f"ref={rr:.2f} [{r['ratio_bur_exp_lo']:.2f}, {r['ratio_bur_exp_hi']:.2f}]")
        if np.isfinite(rg) and np.isfinite(rr) and rr > 0:
            fold = rg / rr
            if fold > 1.5 or fold < 1 / 1.5:
                flagged.append(f"  {name}: gen/ref = {fold:.2f}x  (gen={rg:.2f}, ref={rr:.2f})")

    print("\nSanity warnings:")
    print("  (none)" if not sanity else "")
    for s in sanity: print(s)

    print("\nResidues with gen burial-targeting differing from PDB by >1.5x:")
    print("  (none)" if not flagged else "")
    for f in flagged: print(f)

    print(f"\nDone. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
