"""Build a single FASTA + TSV index for Vendruscolo CamSol submission.

Headers: <BUCKET>__<SUBRUN>__<ORIG_ID> so every CamSol result row maps back to a run.
"""
from __future__ import annotations
import os, re, sys, csv
from pathlib import Path

REPO = Path("/home/ks2218/la-proteina")
RESULTS = REPO / "results"
OUT_FASTA = REPO / "camsol_submission_full_2026_05_27.fasta"
OUT_TSV   = REPO / "camsol_submission_full_2026_05_27.index.tsv"

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O','MSE':'M',
}

def seq_from_pdb(path: Path) -> str:
    """Read first model, chain A, return 1-letter sequence in residue order. CA atoms only."""
    out = []
    seen = set()
    with open(path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = THREE_TO_ONE.get(resname, "X")
            out.append(aa)
    return "".join(out)


# ---- bucket specs ----------------------------------------------------------
# Each entry: (bucket_tag, kind, spec)
#   kind="fasta": spec=(fasta_path, subrun_tag) → headers become <bucket>__<subrun>__<orig_id>
#   kind="fasta_multi": spec=(parent_dir, glob_for_subdirs, fasta_subpath) → one subrun per match
#   kind="pdb_multi":   spec=(parent_dir, glob_for_subdirs, pdb_subpath="guided")
#   kind="pdb_flat":    spec=(pdbs_dir, subrun_tag) → all *.pdb in pdbs_dir
#
BUCKETS = []

# ---- headline CamSol/TANGO steering (FASTAs already exist) ----
BUCKETS.append(("steerCT_n400", "fasta_multi",
    ("steering_camsol_tango_L500_nsteps400", "*", "sequences_guided.fasta")))
BUCKETS.append(("steerCT_ens", "fasta_multi",
    ("steering_camsol_tango_L500_ensemble_smoothed", "*", "sequences_guided.fasta")))
BUCKETS.append(("steerCT_r1raw", "fasta_multi",
    ("steering_camsol_tango_L500", "*", "sequences_guided.fasta")))

# ---- multi-objective combos (PDBs) ----
BUCKETS.append(("combo_CT", "pdb_multi",
    ("combo_camsol_tango_scout", "combo_camsol_tango_w*", "guided")))
BUCKETS.append(("combo_devel4", "pdb_multi",
    ("combo_devel4_scout", "combo_devel4_w*", "guided")))

# ---- noise-aware (denoised recipe) ablations ----
BUCKETS.append(("noise_aware_ens", "pdb_multi",
    ("noise_aware_ensemble_sweep", "*_w*", "guided")))
BUCKETS.append(("noise_aware_highw", "pdb_multi",
    ("noise_aware_high_w_scout", "*_w*", "guided")))

# ---- fixt1 ablation (nested) ----
BUCKETS.append(("fixt1_E066_highw", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/E066_high_w", "*_w*", "guided")))
BUCKETS.append(("fixt1_E067_iupred", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/E067_iupred_max", "*_w*", "guided")))
BUCKETS.append(("fixt1_E068_combo", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/E068_combo_camsol_tango", "*_w*", "guided")))
BUCKETS.append(("fixt1_E032_ens", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/E032_ensemble_sweep", "*_w*", "guided")))
BUCKETS.append(("fixt1_ncmin", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/net_charge_min_sweep", "*_w*", "guided")))
BUCKETS.append(("fixt1_ncmax", "pdb_multi",
    ("fixt1_full_replication_2026_05_18/net_charge_max_sweep", "*_w*", "guided")))

# ---- tango_min schedule sweep ----
BUCKETS.append(("tango_sched", "pdb_multi",
    ("tango_min_schedule_scout", "tango_min_*", "guided")))

# ---- coord-only steering (off-target CamSol readouts) ----
# These dirs are FLAT: results/coords_na_tango_w<W>/guided/*.pdb
for w in (32, 48, 64, 128):
    BUCKETS.append((f"coords_tango_w{w}", "pdb_flat",
        (f"coords_na_tango_w{w}/guided", f"coords_na_tango_w{w}")))
for w in (4, 8, 16, 32, 48, 64):
    BUCKETS.append((f"coords_rgmin_co_w{w}", "pdb_flat",
        (f"coords_na_rg_min_coord_only_w{w}/guided", f"coords_na_rg_min_coord_only_w{w}")))
for w in (32, 48, 64):
    BUCKETS.append((f"coords_sap_co_w{w}", "pdb_flat",
        (f"coords_na_sap_coord_only_w{w}/guided", f"coords_na_sap_coord_only_w{w}")))
for w in (32, 48, 64):
    BUCKETS.append((f"coords_tango_co_w{w}", "pdb_flat",
        (f"coords_na_tango_coord_only_w{w}/guided", f"coords_na_tango_coord_only_w{w}")))

# ---- predictor-side scouts (other latent objectives) ----
BUCKETS.append(("iupred", "pdb_multi",
    ("iupred_max_scout", "iupred_max_w*", "guided")))
BUCKETS.append(("netcharge_max", "pdb_multi",
    ("net_charge_max_scout", "net_charge_max_w*", "guided")))
BUCKETS.append(("netcharge_min", "pdb_multi",
    ("net_charge_min_scout", "net_charge_min_w*", "guided")))

# ---- paired unsteered controls (the W=0 baselines for the steering sweep) ----
# sanity dirs store PDBs under both guided/ (12) and unguided/ (48) — include both.
BUCKETS.append(("sanity_unst_s42_45", "pdb_flat",
    ("sanity_unsteered_seed42_45/guided", "sanity_seed42_45_g")))
BUCKETS.append(("sanity_unst_s42_45_u", "pdb_flat",
    ("sanity_unsteered_seed42_45/unguided", "sanity_seed42_45_u")))
BUCKETS.append(("sanity_unst_s52_57", "pdb_flat",
    ("sanity_unsteered_seed52_57/unguided", "sanity_seed52_57_u")))

# ---- big unsteered nsteps=400 stratified panel (replaces the old 296-result baseline) ----
BUCKETS.append(("strat1000_n400", "fasta",
    ("generated_stratified_300_800_nsteps400/sequences.fasta", "stratified")))


def read_fasta(path: Path):
    """Yield (id, seq) pairs from a FASTA file. id is first whitespace-token after '>'."""
    cur_id, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        yield cur_id, "".join(cur_seq)


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main() -> int:
    records = []  # (header, seq, source_path, bucket, subrun, orig_id)
    for bucket, kind, spec in BUCKETS:
        bucket_records = 0
        if kind == "fasta":
            fasta_rel, subrun = spec
            p = RESULTS / fasta_rel
            if not p.is_file():
                print(f"  [MISS] {bucket}: {p}", file=sys.stderr); continue
            for sid, seq in read_fasta(p):
                hdr = f"{bucket}__{sanitize(subrun)}__{sanitize(sid)}"
                records.append((hdr, seq, str(p), bucket, subrun, sid))
                bucket_records += 1

        elif kind == "fasta_multi":
            parent_rel, subdir_glob, fasta_sub = spec
            parent = RESULTS / parent_rel
            if not parent.is_dir():
                print(f"  [MISS] {bucket}: {parent}", file=sys.stderr); continue
            for sub in sorted(parent.glob(subdir_glob)):
                if not sub.is_dir(): continue
                fp = sub / fasta_sub
                if not fp.is_file(): continue
                subrun = sub.name
                for sid, seq in read_fasta(fp):
                    hdr = f"{bucket}__{sanitize(subrun)}__{sanitize(sid)}"
                    records.append((hdr, seq, str(fp), bucket, subrun, sid))
                    bucket_records += 1

        elif kind == "pdb_multi":
            parent_rel, subdir_glob, pdb_sub = spec
            parent = RESULTS / parent_rel
            if not parent.is_dir():
                print(f"  [MISS] {bucket}: {parent}", file=sys.stderr); continue
            for sub in sorted(parent.glob(subdir_glob)):
                if not sub.is_dir(): continue
                pdbs_dir = sub / pdb_sub
                if not pdbs_dir.is_dir(): continue
                subrun = sub.name
                for pdb in sorted(pdbs_dir.glob("*.pdb")):
                    seq = seq_from_pdb(pdb)
                    if not seq: continue
                    sid = pdb.stem
                    hdr = f"{bucket}__{sanitize(subrun)}__{sanitize(sid)}"
                    records.append((hdr, seq, str(pdb), bucket, subrun, sid))
                    bucket_records += 1

        elif kind == "pdb_flat":
            pdbs_rel, subrun = spec
            pdbs_dir = RESULTS / pdbs_rel
            if not pdbs_dir.is_dir():
                print(f"  [MISS] {bucket}: {pdbs_dir}", file=sys.stderr); continue
            for pdb in sorted(pdbs_dir.glob("*.pdb")):
                seq = seq_from_pdb(pdb)
                if not seq: continue
                sid = pdb.stem
                hdr = f"{bucket}__{sanitize(subrun)}__{sanitize(sid)}"
                records.append((hdr, seq, str(pdb), bucket, subrun, sid))
                bucket_records += 1
        else:
            print(f"  [BAD KIND] {bucket}: {kind}", file=sys.stderr); continue

        print(f"  {bucket:<28} {bucket_records:>5} seqs")

    # Dedup by header (keep first); warn on duplicate seqs across different headers (informational).
    seen_headers = {}
    dropped_dup_header = 0
    for r in records:
        if r[0] in seen_headers:
            dropped_dup_header += 1
            continue
        seen_headers[r[0]] = r

    final = list(seen_headers.values())
    print(f"\nTotal records: {len(final)} (dropped {dropped_dup_header} dup-headers)")

    OUT_FASTA.write_text("".join(f">{h}\n{s}\n" for (h,s,_,_,_,_) in final))
    with open(OUT_TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["header","bucket","subrun","orig_id","length","source_path"])
        for (h,s,src,b,sr,sid) in final:
            w.writerow([h,b,sr,sid,len(s),src])

    print(f"Wrote {OUT_FASTA}")
    print(f"Wrote {OUT_TSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
