"""E101 codesignability: do the de-collapsed (higher-jitter) sequences fold?

For each SDE-jitter condition (results/sde_jitter_entropy_probe/local<X>/samples),
fold each protein's OWN joint-head sequence with ESMFold and compute Cα scRMSD
to the generated backbone (use_pdb_seq=True = codesignability, the metric that
is sensitive to what the sequence actually is — not MPNN-redesigned).

Designable threshold: coScRMSD_ca < 2.0 A. Writes codesign.csv per condition
and prints per-condition + per-length-bin rates.
"""
from __future__ import annotations
import argparse, csv, glob, os, re, shutil, sys, time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from proteinfoundation.metrics.designability import scRMSD  # noqa: E402

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sde_codesign")

LEN_RE = re.compile(r"_n(\d+)$")


def fbin(L):
    return min([300, 400, 500, 600, 700, 800], key=lambda x: abs(x - L))


def evaluate_one(pdb_path: Path, tmp_root: Path) -> float:
    name = pdb_path.stem
    tmp_dir = tmp_root / name
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    res = scRMSD(
        pdb_file_path=str(pdb_path),
        tmp_path=str(tmp_dir),
        num_seq_per_target=1,
        use_pdb_seq=True,
        rmsd_modes=["ca"],
        folding_models=["esmfold"],
        keep_outputs=True,   # keep_outputs=False is buggy: run_esmfold rmtree's
                             # the output dir before scRMSD reads it. We self-clean
                             # tmp_dir below instead (matches run_codesignability_sweep).
        ret_min=False,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    rmsds = res["ca"]["esmfold"]
    return rmsds[0] if rmsds else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sde_jitter_entropy_probe")
    args = ap.parse_args()

    tmp_root = _ROOT / "tmp" / "sde_jitter_codesign"
    tmp_root.mkdir(parents=True, exist_ok=True)

    conds = sorted(glob.glob(os.path.join(args.root, "local*")))
    summary = []
    for c in conds:
        label = os.path.basename(c)
        pdbs = sorted(glob.glob(os.path.join(c, "samples", "*.pdb")))
        out_csv = Path(c) / "codesign.csv"
        done = {}
        if out_csv.exists():
            with open(out_csv) as f:
                for r in csv.DictReader(f):
                    done[r["protein_id"]] = r
        write_header = not out_csv.exists()
        fh = open(out_csv, "a", newline="")
        w = csv.writer(fh)
        if write_header:
            w.writerow(["protein_id", "length", "coScRMSD_ca"])

        rows = []
        t0 = time.time()
        for i, p in enumerate(pdbs):
            pp = Path(p)
            name = pp.stem
            m = LEN_RE.search(name)
            L = int(m.group(1)) if m else 0
            if name in done:
                rows.append((name, L, float(done[name]["coScRMSD_ca"])))
                continue
            try:
                v = evaluate_one(pp, tmp_root)
            except Exception as e:
                log.exception("FAILED %s: %s", name, e)
                v = float("inf")
            w.writerow([name, L, f"{v:.4f}"]); fh.flush()
            rows.append((name, L, v))
            el = time.time() - t0
            log.info("[%s] %s (L=%d) coScRMSD=%.2f  (%d/%d, %.1fmin elapsed)",
                     label, name, L, v, i + 1, len(pdbs), el / 60)
        fh.close()
        summary.append((label, rows))

    # Report
    print("\n=== E101 codesignability (coScRMSD_ca < 2.0 A = designable) ===")
    print(f"{'condition':14}{'n':>4}{'designable':>12}{'rate%':>8}{'median':>9}{'mean':>9}")
    print("-" * 56)
    per_len = {}
    for label, rows in summary:
        vals = [v for _, _, v in rows]
        n = len(vals)
        des = sum(1 for v in vals if v < 2.0)
        sv = sorted(vals)
        med = sv[n // 2] if n else float("nan")
        mean = sum(v for v in vals if v != float("inf")) / max(1, sum(1 for v in vals if v != float("inf")))
        print(f"{label:14}{n:>4}{des:>12}{100*des/max(1,n):>8.1f}{med:>9.2f}{mean:>9.2f}")
        pl = defaultdict(list)
        for _, L, v in rows:
            pl[fbin(L)].append(v)
        per_len[label] = pl

    print("\n--- designable count per length bin ---")
    bins = [300, 400, 500, 600, 700, 800]
    print(f"{'condition':14}" + "".join(f"{b:>8}" for b in bins))
    for label, _ in summary:
        line = f"{label:14}"
        for b in bins:
            vs = per_len[label].get(b, [])
            line += f"{sum(1 for v in vs if v<2.0)}/{len(vs):>2}".rjust(8)
        print(line)


if __name__ == "__main__":
    main()
