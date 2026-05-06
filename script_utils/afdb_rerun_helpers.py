#!/usr/bin/env python3
"""Helpers for the AFDB-as-reference rerun (E020 + E023 + E024).

Subcommands:
  sample-afdb Stratified-sample accessions from AFDB's full accession_ids.csv
              (~214 M rows, length per row from `last_res - first_res + 1`),
              length-filtered to [LMIN, LMAX] in 50-residue bins. Random
              within each bin — diversity-balanced approximation to a
              representative AFDB sample. Writes one accession per line.
  convert     Parse AFDB v4/v6 PDBs into PyG Data .pt files matching the
              schema compute_developability.py expects (.coords + .coord_mask
              in graphein/PDB-37 atom order, .residue_type in OpenFold
              restype indices, .residues = list of 3-letter codes,
              .id = accession). Also writes a single FASTA with one record
              per .pt for downstream aa_composition.py / thermal_stability.py.
  rename-csv  Translate a compute_developability output (gen-schema) into the
              column names compare_properties.py expects on its --ref side.

Run via the laproteina_env Python (graphein, openfold, biopython, pyg).
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser
from graphein.protein.resi_atoms import ATOM_NUMBERING
from openfold.np.residue_constants import restypes
from torch_geometric.data import Data

# 3-letter -> OpenFold restype index (0-19, 20=X)
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_RESTYPE_TO_IDX = {aa: i for i, aa in enumerate(restypes)}


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: sample-afdb
# ─────────────────────────────────────────────────────────────────────────────

def cmd_sample_afdb(args):
    """Random length-stratified sample from AFDB's full accession_ids.csv.

    Single-pass stream over the CSV (or downloaded local file): for each row
    that lands in [LMIN, LMAX], assign to a 50-residue bin and run reservoir
    sampling within that bin so we end up with at most PER_BIN ids per bin
    after one pass — total ≈ PER_BIN × n_bins accessions to download.

    Reservoir sampling keeps the sample uniform within each bin without
    holding the full bin in memory; AFDB has ~214 M rows so this matters.
    """
    rng = random.Random(args.seed)
    bin_w = args.bin_width
    lo, hi = args.length_min, args.length_max
    n_bins = (hi - lo) // bin_w
    K = args.per_bin

    # Per-bin reservoirs and per-bin "items seen so far" counters.
    reservoirs: List[List[str]] = [[] for _ in range(n_bins)]
    seen_per_bin = [0] * n_bins
    n_total, n_in_range = 0, 0

    src = args.src
    opener = gzip.open if str(src).endswith(".gz") else open
    with opener(src, "rt") as fh:
        for line in fh:
            n_total += 1
            if not line:
                continue
            parts = line.rstrip().split(",")
            if len(parts) < 3:
                continue
            try:
                # accession,first_residue,last_residue,model_id,version
                first_res = int(parts[1]); last_res = int(parts[2])
            except ValueError:
                continue
            L = last_res - first_res + 1
            if L < lo or L >= hi:
                continue
            n_in_range += 1
            b = (L - lo) // bin_w
            if not (0 <= b < n_bins):
                continue
            seen_per_bin[b] += 1
            acc = parts[0]
            res = reservoirs[b]
            if len(res) < K:
                res.append(acc)
            else:
                # Reservoir sampling: replace random slot with prob K/n_seen.
                j = rng.randrange(seen_per_bin[b])
                if j < K:
                    res[j] = acc
            if n_total % 5_000_000 == 0:
                kept = sum(len(r) for r in reservoirs)
                print(f"  …{n_total:>10,} rows scanned  "
                      f"in_range={n_in_range:>9,}  reservoir_kept={kept}",
                      flush=True)

    print(f"  scanned {n_total:,} AFDB rows; {n_in_range:,} in [{lo}, {hi})",
          flush=True)
    out_ids: List[str] = []
    for b, res in enumerate(reservoirs):
        rng.shuffle(res)
        out_ids.extend(res)
        print(f"  bin {lo + b * bin_w:>4}-{lo + (b + 1) * bin_w:<4}  "
              f"pool={seen_per_bin[b]:>10,}  sampled={len(res)}", flush=True)

    rng.shuffle(out_ids)
    args.out_list.write_text("\n".join(out_ids) + "\n")
    print(f"  wrote {len(out_ids)} accessions to {args.out_list}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: convert
# ─────────────────────────────────────────────────────────────────────────────

# graphein PDB-37 atom order: ATOM_NUMBERING[atom_name] = index 0..36
_ATOM_TO_IDX = dict(ATOM_NUMBERING)
_N_ATOMS = 37


def _parse_pdb(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], str]]:
    """Read an AFDB v4 PDB.

    Returns (coords[L,37,3] in Å, mask[L,37] bool, three_letter[L], sequence)
    or None on failure (unknown residues, empty file, etc).
    """
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            struct = parser.get_structure(path.stem, str(path))
        except Exception:
            return None

    residues = []
    for model in struct:
        for chain in model:
            for res in chain:
                # Skip HETATMs / waters / non-standard
                hetflag = res.id[0]
                if hetflag != " ":
                    continue
                rname = res.get_resname().upper()
                if rname not in _THREE_TO_ONE:
                    continue
                residues.append(res)
        break  # only model 0

    if not residues:
        return None

    L = len(residues)
    coords = np.zeros((L, _N_ATOMS, 3), dtype=np.float32)
    mask = np.zeros((L, _N_ATOMS), dtype=bool)
    three_letter: List[str] = []
    seq_chars: List[str] = []

    for i, res in enumerate(residues):
        rname = res.get_resname().upper()
        three_letter.append(rname)
        seq_chars.append(_THREE_TO_ONE[rname])
        for atom in res:
            aname = atom.get_name().strip().upper()
            idx = _ATOM_TO_IDX.get(aname)
            if idx is None:
                continue
            coords[i, idx] = atom.coord
            mask[i, idx] = True

    return coords, mask, three_letter, "".join(seq_chars)


def _convert_one(args_tuple):
    pdb_path, out_pt_dir, accession = args_tuple
    parsed = _parse_pdb(pdb_path)
    if parsed is None:
        return accession, None, "parse_failed"
    coords, mask, three_letter, sequence = parsed
    L = len(sequence)
    if L < 1:
        return accession, None, "empty"

    residue_type = torch.tensor(
        [_RESTYPE_TO_IDX[aa] for aa in sequence], dtype=torch.long
    )
    data = Data(
        coords=torch.from_numpy(coords),               # PDB-37 order, Å
        coord_mask=torch.from_numpy(mask),             # PDB-37 order
        residue_type=residue_type,                     # OpenFold restype idx
        residues=three_letter,                         # list[str] 3-letter
        id=accession,
    )

    # Sharded layout: <out_pt_dir>/<acc[:2]>/<acc>.pt to keep dir sizes small.
    shard = accession[:2].lower()
    out_dir = out_pt_dir / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{accession}.pt"
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(data, tmp_path)
    tmp_path.rename(out_path)

    return accession, sequence, "ok"


def cmd_convert(args):
    pdb_paths: List[Path] = []
    if args.id_list and args.id_list.exists():
        ids = [s for s in args.id_list.read_text().splitlines() if s.strip()]
    else:
        ids = sorted(p.stem.split("-")[1] for p in args.raw_dir.glob("AF-*-F1-model_v4.pdb"))
    print(f"  candidate ids: {len(ids)}", flush=True)

    work = []
    for acc in ids:
        p = args.raw_dir / f"AF-{acc}-F1-model_v4.pdb"
        if not p.exists() or p.stat().st_size == 0:
            continue
        work.append((p, args.out_pt_dir, acc))

    print(f"  pdbs found on disk: {len(work)}", flush=True)
    args.out_pt_dir.mkdir(parents=True, exist_ok=True)

    seqs: List[Tuple[str, str]] = []
    n_ok, n_fail = 0, 0
    if args.workers <= 1:
        for w in work:
            acc, seq, status = _convert_one(w)
            if status == "ok":
                seqs.append((acc, seq))
                n_ok += 1
            else:
                n_fail += 1
    else:
        with mp.Pool(args.workers) as pool:
            for i, (acc, seq, status) in enumerate(
                pool.imap_unordered(_convert_one, work, chunksize=8), 1
            ):
                if status == "ok":
                    seqs.append((acc, seq))
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 500 == 0:
                    print(f"  …{i}/{len(work)} (ok={n_ok}, fail={n_fail})", flush=True)

    print(f"  converted {n_ok} pt files (failures: {n_fail})", flush=True)

    # Length-cap to N_FINAL by stratified-trimming each 50-residue bin.
    if args.n_final and len(seqs) > args.n_final:
        rng = random.Random(args.seed)
        bin_w = args.bin_width
        lo = args.length_min
        # Group by bin
        by_bin: dict[int, list] = defaultdict(list)
        for acc, seq in seqs:
            L = len(seq)
            if L < lo:
                continue
            b = (L - lo) // bin_w
            by_bin[b].append((acc, seq))
        # Trim each bin uniformly
        n_bins = max(by_bin.keys()) + 1
        target_per_bin = args.n_final // n_bins
        kept: List[Tuple[str, str]] = []
        for b in sorted(by_bin):
            lst = by_bin[b]
            rng.shuffle(lst)
            kept.extend(lst[:target_per_bin])
        # Top up to n_final randomly from the remainder
        remainder = [
            x for b, lst in by_bin.items() for x in lst[target_per_bin:]
        ]
        rng.shuffle(remainder)
        kept.extend(remainder[: args.n_final - len(kept)])
        seqs = kept[: args.n_final]
        print(f"  length-stratified to N={len(seqs)} (target {args.n_final})", flush=True)

        # Move kept .pt to a dedicated dir, mark others as parked
        keep_set = {acc for acc, _ in seqs}
        moved = 0
        for shard in args.out_pt_dir.iterdir():
            if not shard.is_dir():
                continue
            for ptf in shard.glob("*.pt"):
                if ptf.stem in keep_set:
                    moved += 1
                else:
                    ptf.unlink()
            try:
                shard.rmdir()  # remove empty shards
            except OSError:
                pass
        print(f"  retained {moved} .pt files in {args.out_pt_dir}", flush=True)

    # Write FASTA, sorted by accession for deterministic ordering.
    seqs.sort()
    with args.out_fasta.open("w") as fh:
        for acc, seq in seqs:
            fh.write(f">{acc}\n{seq}\n")
    print(f"  wrote FASTA with {len(seqs)} records to {args.out_fasta}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: rename_csv_for_compare
# ─────────────────────────────────────────────────────────────────────────────

def cmd_rename_csv(args):
    """Rename a compute_developability output (gen-schema) into the column
    names compare_properties.py expects on its --ref side.
    """
    import pandas as pd

    df = pd.read_csv(args.in_csv)
    rename = {
        "tango_total":         "tango",
        "net_charge_ph7":      "net_charge",
        "iupred3_mean":        "iupred3",
        "radius_of_gyration":  "rg",
        "sap_total":           "sap",
        "pdb_id":              "protein_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.to_csv(args.out_csv, index=False)
    print(f"  wrote {args.out_csv} ({len(df)} rows)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample-afdb")
    sp.add_argument("--src", type=Path, required=True,
                    help="Path to AFDB accession_ids.csv (or .gz)")
    sp.add_argument("--out-list", type=Path, required=True)
    sp.add_argument("--length-min", type=int, default=300)
    sp.add_argument("--length-max", type=int, default=800)
    sp.add_argument("--bin-width", type=int, default=50)
    sp.add_argument("--per-bin", type=int, default=1000,
                    help="Accessions to sample per 50-residue bin (oversample factor)")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_sample_afdb)

    sp = sub.add_parser("convert")
    sp.add_argument("--raw-dir", type=Path, required=True)
    sp.add_argument("--id-list", type=Path, default=None)
    sp.add_argument("--out-pt-dir", type=Path, required=True)
    sp.add_argument("--out-fasta", type=Path, required=True)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--n-final", type=int, default=5000)
    sp.add_argument("--length-min", type=int, default=300)
    sp.add_argument("--bin-width", type=int, default=50)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("rename-csv")
    sp.add_argument("--in-csv", type=Path, required=True)
    sp.add_argument("--out-csv", type=Path, required=True)
    sp.set_defaults(func=cmd_rename_csv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
