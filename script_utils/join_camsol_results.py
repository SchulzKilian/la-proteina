#!/usr/bin/env python3
"""Join Vendruscolo CamSol results back onto the steering submission.

CamSol's `Name` column is the FASTA header used in the submission; the
submission index (camsol_submission_full_*.index.tsv) maps each header to
(subrun, orig_id, length, source_path). orig_id is the `protein_id` used in
each subrun's properties_guided.csv.

Note: (subrun, orig_id) is NOT globally unique (the same label was submitted
from multiple parent runs), so the canonical key is the unique `header`. We
therefore (1) write one consolidated CSV keyed by header, and (2) back-fill a
`camsol_ph7` column into each properties_guided.csv, scoped per-CSV-directory
so there is no cross-parent collision.

Idempotent: re-running refreshes the column / consolidated file in place.
Atomic writes (tmp + os.replace) so a SLURM/SSH kill can't leave a half file.
"""
import argparse, csv, os, sys, collections

REPO = "/home/ks2218/la-proteina"
SCORE_COL = "protein variant score"
OUT_COL = "camsol_ph7"


def parse_camsol(path):
    """header -> (score, pH). Handles CRLF and the per-residue profile column."""
    out = {}
    with open(path, newline="") as fh:
        r = csv.DictReader((ln.replace("\r\n", "\n") for ln in fh), delimiter="\t")
        for row in r:
            nm = row["Name"].strip()
            try:
                out[nm] = (float(row[SCORE_COL]), row.get("pH", "").strip())
            except (KeyError, ValueError):
                continue
    return out


def parse_index(path):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append(row)
    return rows


def find_props_csv(src):
    d = os.path.dirname(src)
    for cand in (d, os.path.dirname(d)):
        for name in ("properties_guided.csv", "properties.csv"):
            p = os.path.join(cand, name)
            if os.path.isfile(p):
                return p
    return None


def atomic_write_csv(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camsol", default=os.path.join(REPO, "CamSolpH_submission_full_2026_05_27.txt"))
    ap.add_argument("--index", default=os.path.join(REPO, "camsol_submission_full_2026_05_27.index.tsv"))
    ap.add_argument("--consolidated", default=os.path.join(REPO, "results", "camsol_ph7_full_2026_05_27.csv"))
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    scores = parse_camsol(args.camsol)
    idx = parse_index(args.index)
    print(f"camsol records: {len(scores)} | index records: {len(idx)}")

    # ---- 1. consolidated table (keyed by unique header) ----
    consolidated = []
    n_no_score = 0
    for row in idx:
        h = row["header"]
        sc = scores.get(h)
        if sc is None:
            n_no_score += 1
            continue
        consolidated.append({
            "header": h, "subrun": row["subrun"], "orig_id": row["orig_id"],
            "length": row["length"], "source_path": row["source_path"],
            "camsol_ph7": f"{sc[0]:.6f}", "pH": sc[1],
        })
    print(f"consolidated rows: {len(consolidated)} (index headers with no camsol score: {n_no_score})")
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.consolidated), exist_ok=True)
        atomic_write_csv(args.consolidated,
                         ["header", "subrun", "orig_id", "length", "source_path", "camsol_ph7", "pH"],
                         consolidated)
        print(f"  -> wrote {args.consolidated}")

    # ---- 2. back-fill each properties_guided.csv (scoped per CSV) ----
    # group index records by resolved properties csv path
    by_csv = collections.defaultdict(list)  # csv_path -> [(orig_id, header)]
    n_unresolved = 0
    for row in idx:
        p = find_props_csv(row["source_path"])
        if p:
            by_csv[p].append((row["orig_id"], row["header"]))
        else:
            n_unresolved += 1
    print(f"\nproperties CSVs to back-fill: {len(by_csv)} "
          f"| index records with no property CSV: {n_unresolved}")

    total_written = total_unmatched_rows = 0
    for csv_path, items in sorted(by_csv.items()):
        id2score = {}
        for orig_id, header in items:
            sc = scores.get(header)
            if sc is not None:
                id2score[orig_id] = sc[0]
        with open(csv_path, newline="") as fh:
            r = csv.DictReader(fh)
            fields = list(r.fieldnames)
            data = list(r)
        if "protein_id" not in fields:
            print(f"  SKIP (no protein_id): {os.path.relpath(csv_path, REPO)}")
            continue
        if OUT_COL not in fields:
            fields.append(OUT_COL)
        matched = 0
        for d in data:
            sc = id2score.get(d["protein_id"])
            if sc is not None:
                d[OUT_COL] = f"{sc:.6f}"; matched += 1
            else:
                d.setdefault(OUT_COL, "")
        total_written += matched
        unrows = len(data) - matched
        total_unmatched_rows += unrows
        rel = os.path.relpath(csv_path, REPO)
        print(f"  {matched:4d}/{len(data):<4d} matched  {rel}" + (f"  ({unrows} rows left blank)" if unrows else ""))
        if not args.dry_run:
            atomic_write_csv(csv_path, fields, data)

    print(f"\nback-filled {total_written} property-CSV rows "
          f"({total_unmatched_rows} existing rows had no camsol match)")
    if args.dry_run:
        print("DRY RUN — nothing written.")


if __name__ == "__main__":
    main()
