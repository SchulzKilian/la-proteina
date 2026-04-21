#!/usr/bin/env python
"""
Build the training metadata CSV without running structure processing.

The normal pipeline (pdb_data.py `prepare_data`) does:
  1. create_dataset()              — fast, in-memory DataFrame filters (~90s)
  2. _process_structure_data()     — slow, CIF → .pt conversion on Lustre (hours)
  3. df_data.to_csv()              — instant write

Steps 1 and 3 are the only things needed if the `.pt` shards are already
on disk. This script runs just those, and additionally intersects the
filtered chains against the existing `.pt` files so training never asks
for a chain that has no processed file.

Output: `<data_dir>/<file_identifier>.csv` — the exact filename the
DataModule expects (pdb_data.py:546). Training picks up the cached CSV
and skips `prepare_data()` entirely (line 551 early-return).

Usage examples:
  # default: reproduce the training_ca_only filter (maxl=512, res<=2.0)
  python script_utils/build_train_csv.py

  # override filter knobs
  python script_utils/build_train_csv.py --max-length 200 --worst-resolution 2.0

  # dry-run (print stats, don't write)
  python script_utils/build_train_csv.py --dry-run

Runs on login node in <3 min. No GPU, no multiprocessing on Lustre.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path so proteinfoundation imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so DATA_PATH resolves.
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Apply the graphein monkeypatches prepare_data.py applies — needed because
# PDBManager's upstream CATH URL is dead and ligand-map parsing is fragile.
import ssl  # noqa: E402
ssl._create_default_https_context = ssl._create_unverified_context

import graphein.ml.datasets.pdb_data  # noqa: E402
import wget  # noqa: E402

NEW_CATH_URL = (
    "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)
graphein.ml.datasets.pdb_data.CATH_ID_CATH_CODE_URL = NEW_CATH_URL


def _fixed_download_cath(self):
    self.cath_id_cath_code_url = NEW_CATH_URL
    target = self.root_dir / "cath-b-newest-all.txt"
    if not target.exists():
        print(f"[Patch] Downloading CATH map to: {target}")
        wget.download(NEW_CATH_URL, out=str(target))
    else:
        print("[Patch] CATH file already present.")


def _robust_parse_ligand_map(self):
    path = self.root_dir / "ligand_map.txt"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                out[parts[0]] = parts[1:]
    return out


graphein.ml.datasets.pdb_data.PDBManager._download_cath_id_cath_code_map = _fixed_download_cath
graphein.ml.datasets.pdb_data.PDBManager._parse_ligand_map = _robust_parse_ligand_map

# Now safe to import the project's selector.
from proteinfoundation.datasets.pdb_data import PDBDataSelector  # noqa: E402


# File identifier formula — must mirror PDBLightningDataModule._get_file_identifier
# (pdb_data.py:729-744) exactly, otherwise training will not find the cached CSV.
def file_identifier(ds: PDBDataSelector, use_precomputed_latents: bool = False) -> str:
    fid = (
        f"df_pdb_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}_mt{ds.molecule_type}"
        f"_et{''.join(ds.experiment_types) if ds.experiment_types else ''}"
        f"_mino{ds.oligomeric_min}_maxo{ds.oligomeric_max}"
        f"_minr{ds.best_resolution}_maxr{ds.worst_resolution}"
        f"_hl{''.join(ds.has_ligands) if ds.has_ligands else ''}"
        f"_rl{''.join(ds.remove_ligands) if ds.remove_ligands else ''}"
        f"_rnsr{ds.remove_non_standard_residues}_rpu{ds.remove_pdb_unavailable}"
        f"_l{''.join(ds.labels) if ds.labels else ''}"
        f"_rcu{ds.remove_cath_unavailable}"
    )
    if use_precomputed_latents:
        fid += "_latents"
    return fid


# Default filter — matches training_ca_only.yaml (via pdb_train_ucond.yaml
# with worst_resolution overridden to 2.0). The exclude_ids list is copied
# verbatim from configs/dataset/pdb/pdb_train_ucond.yaml:37-39.
DEFAULT_EXCLUDE_IDS = [
    "9b57", "9b5p", "9b5s", "9b5n", "9b5a", "9b5k", "9b5v", "9b59", "9b5w",
    "9b5g", "9b5i", "9b5x", "9b5d", "9b5e", "9b5t", "9b5f", "9b5o", "9b58",
    "9b5u", "9b5c", "9b5j", "9b5b", "9b5h", "9b5q", "9b5l", "9b56", "9b5m",
    "9b5l", "9b55", "9b5r", "9ij9", "9iix",
    "2ezq", "1kld", "1crr", "2ezs", "2ezr", "1vve", "7ll9_G",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="data_dir for PDBDataSelector. Defaults to $DATA_PATH/pdb_train.")
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--molecule-type", default="protein")
    p.add_argument("--worst-resolution", type=float, default=2.0)
    p.add_argument("--best-resolution", type=float, default=0.0)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Data fraction to keep. Passed to PDBDataSelector.")
    p.add_argument("--skip-pt-intersect", action="store_true",
                   help="Skip intersecting with existing .pt files on disk.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute CSV but don't write it.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the CSV if it already exists.")
    args = p.parse_args()

    # pdb_train_ucond.yaml declares `fraction: 1` (int), so the cached CSV
    # name uses "f1", not "f1.0". Match that exactly — argparse produces a
    # float, which would render "f1.0" and break the cache-hit check at
    # pdb_data.py:550. Collapse the integer case.
    if args.fraction == int(args.fraction):
        args.fraction = int(args.fraction)

    if args.data_dir is None:
        data_path = os.environ.get("DATA_PATH")
        if not data_path:
            raise RuntimeError("DATA_PATH not set and --data-dir not given.")
        data_dir = Path(data_path) / "pdb_train"
    else:
        data_dir = Path(args.data_dir)

    print(f"[+] data_dir: {data_dir}")
    print(f"[+] filter: minl={args.min_length}, maxl={args.max_length}, "
          f"maxr={args.worst_resolution}")

    selector = PDBDataSelector(
        data_dir=str(data_dir),
        fraction=args.fraction,
        molecule_type=args.molecule_type,
        experiment_types=["diffraction", "EM"],
        min_length=args.min_length,
        max_length=args.max_length,
        best_resolution=args.best_resolution,
        worst_resolution=args.worst_resolution,
        has_ligands=[],
        remove_ligands=[],
        remove_non_standard_residues=True,
        remove_pdb_unavailable=False,
        exclude_ids=DEFAULT_EXCLUDE_IDS,
    )

    csv_name = file_identifier(selector) + ".csv"
    csv_path = data_dir / csv_name
    print(f"[+] Target CSV: {csv_path}")

    if csv_path.exists() and not args.overwrite and not args.dry_run:
        print(f"[!] CSV already exists; use --overwrite to replace. Exiting.")
        return

    # Step 1 — In-memory filter, ~90s.
    print("[+] Running create_dataset() (in-memory filter)...")
    df = selector.create_dataset()
    print(f"[+] Filter result: {len(df)} chains")

    # Step 2 — Intersect with existing .pt files. This is the key step that
    # lets training skip _process_structure_data entirely: every chain in
    # the CSV is guaranteed to have a .pt on disk already.
    if not args.skip_pt_intersect:
        processed_dir = data_dir / "processed"
        print(f"[+] Scanning {processed_dir} for existing .pt files...")
        import glob
        existing = glob.glob(str(processed_dir / "**" / "*.pt"), recursive=True)
        existing_set = {os.path.basename(p) for p in existing}
        print(f"[+] Found {len(existing_set)} .pt files on disk.")

        def has_pt(row):
            pdb = row["pdb"].lower()
            chain = row.get("chain")
            # Match the fname convention process_single_pdb_file uses:
            # {pdb}_{chain}.pt when chain is a real chain id.
            if chain is None or (isinstance(chain, float) and chain != chain) or chain == "all":
                fname = f"{pdb}.pt"
            else:
                fname = f"{pdb}_{chain}.pt"
            return fname in existing_set

        before = len(df)
        df = df[df.apply(has_pt, axis=1)]
        print(f"[+] Intersected: {before} → {len(df)} chains have .pt on disk")

    if args.dry_run:
        print("[+] --dry-run: not writing CSV.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[+] Wrote {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
