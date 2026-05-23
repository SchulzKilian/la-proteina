"""Bundle 296 sequences into one FASTA for CamSol web-server submission.

Three groups, structured headers (recoverable after CamSol returns CSV):
    un_s{seed}_n{len}    unsteered baseline (200, 20 from each of the 10
                         50-residue length bins in [300, 800))
    n32_s{seed}_n{len}   noise-aware ensemble camsol_max w=32 (48)
    n128_s{seed}_n{len}  noise-aware ensemble camsol_max w=128 (48)

CamSol's CSV uses the first whitespace-delimited token of each FASTA header as
the `Name` column. Splitting on `_` after results return recovers
(group, w, seed, length).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "camsol_submission_296.fasta"

UNSTEERED_FASTA = ROOT / "results/generated_stratified_300_800_nsteps400/sequences.fasta"
HIGH_W_BASE = ROOT / "results/noise_aware_high_w_scout"
WEIGHTS = [32, 128]
UNSTEERED_PER_BIN = 20  # 10 length bins × 20 = 200
BIN_WIDTH = 50
BIN_LO, BIN_HI = 300, 800


def parse_unsteered(path: Path, per_bin: int):
    """Yield (id, seq) for `per_bin` sequences from each 50-residue length bin.

    The 1000-sequence stratified FASTA cycles through bins (s1000 ∈ [300, 350),
    s1001 ∈ [350, 400), ... s1009 ∈ [750, 800), s1010 wraps), so a stride-based
    subsample only hits a subset of bins. Bucketing by length and taking the
    first `per_bin` per bucket guarantees per-bin coverage.
    """
    records = []
    cur_id, cur_seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id = "un_" + line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            records.append((cur_id, "".join(cur_seq)))

    bins: dict[int, list] = {}
    for rec_id, seq in records:
        L = len(seq)
        if not (BIN_LO <= L < BIN_HI):
            continue
        b = ((L - BIN_LO) // BIN_WIDTH) * BIN_WIDTH + BIN_LO
        bins.setdefault(b, []).append((rec_id, seq))

    for b in sorted(bins):
        for rec_id, seq in bins[b][:per_bin]:
            yield rec_id, seq


def parse_steered_cell(cell_dir: Path, prefix: str):
    for pt in sorted((cell_dir / "guided").glob("*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        yield f"{prefix}_{d['id']}", d["sequence"]


def write_fasta(records, out_path: Path) -> int:
    n = 0
    seen = set()
    with open(out_path, "w") as fh:
        for rec_id, seq in records:
            if rec_id in seen:
                print(f"WARNING: duplicate id {rec_id}", file=sys.stderr)
                continue
            seen.add(rec_id)
            fh.write(f">{rec_id}\n{seq}\n")
            n += 1
    return n


def main() -> None:
    counts: dict[str, int] = {}
    records = []

    g = list(parse_unsteered(UNSTEERED_FASTA, UNSTEERED_PER_BIN))
    counts["un"] = len(g)
    records.extend(g)

    for w in WEIGHTS:
        g = list(parse_steered_cell(HIGH_W_BASE / f"camsol_max_w{w}", f"n{w}"))
        counts[f"n{w}"] = len(g)
        records.extend(g)

    n_written = write_fasta(records, OUT)

    print(f"Wrote {n_written} sequences -> {OUT.relative_to(ROOT)}")
    print(f"  size: {OUT.stat().st_size / 1024:.1f} KB")
    print()
    print("Per-group counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
