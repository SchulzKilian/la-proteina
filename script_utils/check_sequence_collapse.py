"""Sequence-level sanity check for the camsol/tango steering sweep.

scRMSD + pairwise TM are blind to "the joint sequence head collapsed to poly-K".
This script reads the steered FASTAs and flags cells where the sequence
distribution is degenerate vs three reference baselines: w=1 same direction,
unsteered stratified samples, and natural PDB composition.

Usage:
    python script_utils/check_sequence_collapse.py
    python script_utils/check_sequence_collapse.py --tree results/<other>
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_TREE = ROOT / "results/steering_camsol_tango_L500_nsteps400"
UNSTEERED_FASTA = ROOT / "results/generated_stratified_300_800_nsteps400/sequences.fasta"
NATURAL_CSV = ROOT / "results/aa_composition_nsteps400/stratified_vs_pdb/aa_composition.csv"
OUT_DIR = ROOT / "results/sequence_collapse_audit"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Collapse thresholds. We want to catch *steering damage*, not baseline
# properties of the joint sequence head — so most flags fire on the delta vs
# w=1 within the same direction. MAX_AA is absolute (a hard cap on any single
# residue regardless of baseline).
THRESH_MAX_AA = 0.20             # any single AA > 20% of residues, absolute
THRESH_SHANNON_DROP = 0.30       # bits drop in mean per-seq Shannon vs w=1
THRESH_LOW_COMPLEXITY_RISE = 0.05  # absolute rise in low-complexity fraction vs w=1
THRESH_HOMOPOLYMER_RISE = 2.0    # residues rise in mean longest run vs w=1
THRESH_KL_VS_W1 = 0.20           # KL(cell || w=1 same direction), nats


def read_fasta(path: Path) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    name, buf = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs.append((name, "".join(buf)))
    return seqs


def seq_from_pdb(path: Path) -> str:
    """Extract a one-letter sequence from a single-chain PDB by reading the
    residue type at each CA atom in residue-number order."""
    seen: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        resname = line[17:20].strip()
        chain = line[21:22]
        resnum = int(line[22:26])
        key = (chain, resnum)
        if key not in seen:
            seen[key] = THREE_TO_ONE.get(resname, "X")
    return "".join(seen[k] for k in sorted(seen))


def read_sequences(cell_dir: Path) -> list[tuple[str, str]]:
    """Prefer sequences_guided.fasta; fall back to extracting from guided/*.pdb."""
    fasta = cell_dir / "sequences_guided.fasta"
    if fasta.exists():
        return read_fasta(fasta)
    pdb_dir = cell_dir / "guided"
    if pdb_dir.is_dir():
        return [(p.stem, seq_from_pdb(p)) for p in sorted(pdb_dir.glob("*.pdb"))]
    raise FileNotFoundError(f"no sequences_guided.fasta or guided/*.pdb under {cell_dir}")


def composition(seq: str) -> np.ndarray:
    c = Counter(seq)
    return np.array([c.get(a, 0) for a in AA20], dtype=float)


def shannon_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def longest_run(seq: str) -> int:
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best if seq else 0


def low_complexity_fraction(seq: str, window: int = 12, bits_thresh: float = 2.0) -> float:
    """Fraction of residues that lie inside any window of `window` residues whose
    Shannon entropy is below `bits_thresh`. SEG-style heuristic; 2.0 bits is
    looser than SEG's 2.2 default to keep healthy proteins quiet."""
    if len(seq) < window:
        return 0.0
    flagged = np.zeros(len(seq), dtype=bool)
    for i in range(len(seq) - window + 1):
        if shannon_bits(composition(seq[i:i + window])) < bits_thresh:
            flagged[i:i + window] = True
    return float(flagged.mean())


def unique_kmer_fraction(seq: str, k: int = 3) -> float:
    if len(seq) < k:
        return 0.0
    kmers = {seq[i:i + k] for i in range(len(seq) - k + 1)}
    return len(kmers) / (len(seq) - k + 1)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = (p + eps) / (p.sum() + 20 * eps)
    q = (q + eps) / (q.sum() + 20 * eps)
    return float((p * np.log(p / q)).sum())


def per_seq_metrics(seqs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, s in seqs:
        c = composition(s)
        rows.append({
            "name": name,
            "length": len(s),
            "shannon_bits": shannon_bits(c),
            "max_aa_freq": float(c.max() / max(c.sum(), 1)),
            "max_aa": AA20[int(c.argmax())] if c.sum() else "-",
            "longest_run": longest_run(s),
            "low_complexity_frac": low_complexity_fraction(s),
            "unique_3mer_frac": unique_kmer_fraction(s, 3),
            "composition": c,
        })
    return pd.DataFrame(rows)


def cell_composition(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["composition"].values).sum(axis=0)


def parse_cell_name(name: str) -> tuple[str, int]:
    m = re.match(r"^(.+)_w(\d+)$", name)
    if not m:
        raise ValueError(f"unexpected cell name: {name}")
    return m.group(1), int(m.group(2))


def natural_composition() -> np.ndarray:
    """Load PDB-derived AA frequencies from the cached aa_composition.csv."""
    nat = pd.read_csv(NATURAL_CSV)[["aa", "ref_mean"]].set_index("aa")
    return np.array([nat.loc[a, "ref_mean"] for a in AA20], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nat = natural_composition()
    unsteered = cell_composition(per_seq_metrics(read_fasta(UNSTEERED_FASTA)))

    cell_dirs = sorted({p.parent for p in args.tree.glob("*/sequences_guided.fasta")}
                       | {p.parent.parent for p in args.tree.glob("*/guided/*.pdb")})
    cell_dirs = sorted(d for d in cell_dirs if re.match(r"^.+_w\d+$", d.name))
    if not cell_dirs:
        raise SystemExit(f"no steering cells (sequences_guided.fasta or guided/*.pdb) under {args.tree}")

    per_seq_rows = []
    cell_summary = []
    cell_comps: dict[str, np.ndarray] = {}

    for cd in cell_dirs:
        direction, w = parse_cell_name(cd.name)
        df = per_seq_metrics(read_sequences(cd))
        df["cell"] = cd.name
        df["direction"] = direction
        df["w"] = w
        per_seq_rows.append(df.drop(columns=["composition"]))

        comp = cell_composition(df)
        cell_comps[cd.name] = comp
        cell_summary.append({
            "cell": cd.name,
            "direction": direction,
            "w": w,
            "n_seqs": len(df),
            "mean_shannon": df["shannon_bits"].mean(),
            "mean_max_aa_freq": df["max_aa_freq"].mean(),
            "top_aa": AA20[int(comp.argmax())],
            "top_aa_freq": float(comp.max() / comp.sum()),
            "mean_longest_run": df["longest_run"].mean(),
            "mean_low_complexity_frac": df["low_complexity_frac"].mean(),
            "mean_unique_3mer_frac": df["unique_3mer_frac"].mean(),
            "kl_vs_natural": kl_div(comp, nat),
            "kl_vs_unsteered": kl_div(comp, unsteered),
        })

    summary = pd.DataFrame(cell_summary)
    # KL vs the lowest-w cell of the same direction (the in-tree anchor). If
    # only one w-level is present for a direction (e.g., iupred sweep at
    # w∈{16,32,64,128} with no w=1), KL_vs_w1 falls back to that lowest-w cell.
    directions = sorted({r.direction for r in summary.itertuples()})
    lowest_w = {d: min(r.w for r in summary.itertuples() if r.direction == d) for d in directions}
    w1_map = {d: cell_comps[f"{d}_w{lowest_w[d]}"] for d in directions
              if f"{d}_w{lowest_w[d]}" in cell_comps}
    summary["kl_vs_w1"] = [
        kl_div(cell_comps[r.cell], w1_map[r.direction]) if r.direction in w1_map else math.nan
        for r in summary.itertuples()
    ]

    # Per-direction w=1 anchors for relative flags.
    w1_metrics = {
        r.direction: r for r in summary.itertuples() if r.w == 1
    }

    def flag_row(r) -> str:
        anchor = w1_metrics.get(r.direction)
        flags = []
        if r.top_aa_freq > THRESH_MAX_AA:
            flags.append("MAX_AA")
        if anchor is not None:
            if r.mean_shannon < anchor.mean_shannon - THRESH_SHANNON_DROP:
                flags.append("SHANNON_DROP")
            if r.mean_low_complexity_frac > anchor.mean_low_complexity_frac + THRESH_LOW_COMPLEXITY_RISE:
                flags.append("LOW_COMPLEXITY_RISE")
            if r.mean_longest_run > anchor.mean_longest_run + THRESH_HOMOPOLYMER_RISE:
                flags.append("HOMOPOLYMER_RISE")
        if not math.isnan(r.kl_vs_w1) and r.kl_vs_w1 > THRESH_KL_VS_W1:
            flags.append("KL_W1")
        return ",".join(flags) or "ok"

    summary["flags"] = [flag_row(r) for r in summary.itertuples()]
    summary = summary.sort_values(["direction", "w"]).reset_index(drop=True)

    pd.concat(per_seq_rows, ignore_index=True).to_csv(args.out / "per_sequence.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)

    comp_pct = pd.DataFrame(
        {cell: comp / comp.sum() for cell, comp in cell_comps.items()},
        index=AA20,
    ).T
    comp_pct.loc["__unsteered__"] = unsteered / unsteered.sum()
    comp_pct.loc["__natural_pdb__"] = nat / nat.sum()
    comp_pct.round(4).to_csv(args.out / "composition_per_cell.csv")

    print("\n=== Per-cell sequence sanity ===")
    print(summary.to_string(
        index=False,
        formatters={
            "mean_shannon": "{:.2f}".format,
            "mean_max_aa_freq": "{:.3f}".format,
            "top_aa_freq": "{:.3f}".format,
            "mean_longest_run": "{:.2f}".format,
            "mean_low_complexity_frac": "{:.3f}".format,
            "mean_unique_3mer_frac": "{:.3f}".format,
            "kl_vs_natural": "{:.3f}".format,
            "kl_vs_unsteered": "{:.3f}".format,
            "kl_vs_w1": "{:.3f}".format,
        },
    ))
    print(f"\nReference baselines:")
    print(f"  unsteered  top AA: {AA20[int(unsteered.argmax())]}  freq={unsteered.max()/unsteered.sum():.3f}  "
          f"shannon={shannon_bits(unsteered):.2f} bits")
    print(f"  natural    top AA: {AA20[int(nat.argmax())]}  freq={nat.max()/nat.sum():.3f}  "
          f"shannon={shannon_bits(nat):.2f} bits")
    print(f"\nWrote {args.out}/summary.csv, per_sequence.csv, composition_per_cell.csv")


if __name__ == "__main__":
    main()
