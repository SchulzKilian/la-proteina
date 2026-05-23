"""Bundle 1480 sequences into one FASTA for CamSol submission.

Three groups, structured headers (recoverable after CamSol returns CSV):
    un_s{seed}_n{len}   unsteered baseline (1000)
    c{w}_s{seed}_n{len} clean+ensemble+smoothing arm (E028), 5x48
    n{w}_s{seed}_n{len} noise-aware+ensemble arm (E032), 5x48

CamSol's CSV uses the first whitespace-delimited token of each FASTA header
as the `Name` column (see CamSolpH_results.txt for the format). So splitting
on `_` after a result comes back recovers (group, w, seed, length).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "camsol_submission.fasta"

UNSTEERED_FASTA = ROOT / "results/generated_stratified_300_800_nsteps400/sequences.fasta"
NA_BASE = ROOT / "results/noise_aware_ensemble_sweep"
CLEAN_BASE = ROOT / "results/steering_camsol_tango_L500_ensemble_smoothed"
WEIGHTS = [1, 2, 4, 8, 16]


def parse_unsteered(path: Path):
    """Yield (id, seq) from a stock FASTA, prepending `un_` to each id."""
    cur_id, cur_seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur_seq)
                # >s1000_n310 length=310  ->  un_s1000_n310
                cur_id = "un_" + line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            yield cur_id, "".join(cur_seq)


def parse_steered_cell(cell_dir: Path, prefix: str):
    """Yield (id, seq) from a steering output dir's guided/*.pt files."""
    for pt in sorted((cell_dir / "guided").glob("*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        # d['id'] is e.g. "s42_n300"; d['sequence'] is the AA string
        yield f"{prefix}_{d['id']}", d["sequence"]


def write_fasta(records, out_path: Path) -> int:
    n = 0
    seen_ids = set()
    with open(out_path, "w") as fh:
        for rec_id, seq in records:
            if rec_id in seen_ids:
                print(f"WARNING: duplicate id {rec_id}, skipping", file=sys.stderr)
                continue
            seen_ids.add(rec_id)
            fh.write(f">{rec_id}\n{seq}\n")
            n += 1
    return n


def main() -> None:
    counts: dict[str, int] = {}
    records = []

    # 1000 unsteered baseline
    g = list(parse_unsteered(UNSTEERED_FASTA))
    counts["un"] = len(g)
    records.extend(g)

    # 240 NA+ensemble steered
    for w in WEIGHTS:
        g = list(parse_steered_cell(NA_BASE / f"camsol_max_w{w}", f"n{w}"))
        counts[f"n{w}"] = len(g)
        records.extend(g)

    # 240 clean+ensemble+smoothing steered
    for w in WEIGHTS:
        g = list(parse_steered_cell(CLEAN_BASE / f"camsol_max_w{w}", f"c{w}"))
        counts[f"c{w}"] = len(g)
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
