#!/usr/bin/env python3
"""
compute_developability.py

Compute a 16-property biophysical panel for every protein in the corpus.
Writes results to a CSV: pdb_id → property values.

Usage:
    python compute_developability.py output.csv \
        --data-dir /rds/user/ks2218/hpc-work \
        [--filter-csv /path/to/df_pdb_*.csv] \
        [--limit 100] \
        [--workers 8]

Data layout expected:
    <data-dir>/processed/<shard>/<pdb_id>.pt   (shard = pdb_id[:2])
    <data-dir>/processed/<pdb_id>.pt           (flat fallback)

Each .pt file is a PyG Data object with:
    coords       (L, 37, 3)  — graphein/PDB atom ordering (reindexed below)
    coord_mask   (L, 37)     — bool, True = atom resolved
    residue_type (L,)        — int, index into openfold restypes (0-19, 20=unk)
    residues     [str]*L     — 3-letter residue names
    id           str         — pdb_id
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ── project imports (script lives at proteinfoundation/analysis/) ────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from openfold.np.residue_constants import (
    atom_types as OPENFOLD_ATOM_TYPES,   # list of 37 atom names in OpenFold order
    restypes,                             # 20 single-letter AA codes (A,R,N,D,...)
    restype_1to3,                         # single → three letter
)

# ── output columns ────────────────────────────────────────────────────────────
COLUMNS = [
    "pdb_id", "sequence_length", "n_resolved_residues",
    "camsol_intrinsic",
    "swi",
    "tango_total", "tango_aggregation_positions",
    "canya_max_nucleation",
    "net_charge_ph7", "pI", "pI_distance_physiological",
    "iupred3_mean", "iupred3_fraction_disordered",
    "shannon_entropy",
    "hydrophobic_patch_total_area", "hydrophobic_patch_n_large",
    "sap_total", "scm_positive", "scm_negative",
    "radius_of_gyration",
]

# ── amino acid index → single letter (openfold restypes + X) ─────────────────
IDX_TO_AA = {i: aa for i, aa in enumerate(restypes)}
IDX_TO_AA[20] = "X"

# ── SWI (Solubility-Weighted Index) per-residue propensity scores ─────────────
# Source: Bhandari et al. 2020, Table 1
# These are log-odds of each AA appearing in soluble vs insoluble NESG proteins.
# *** VERIFY THESE VALUES AGAINST TABLE 1 OF THE PAPER BEFORE PUBLICATION ***
SWI_SCORES: Dict[str, float] = {
    "A":  0.06, "R":  0.25, "N":  0.17, "D":  0.23, "C": -0.13,
    "Q":  0.16, "E":  0.26, "G":  0.05, "H":  0.11, "I": -0.15,
    "L": -0.16, "K":  0.22, "M": -0.05, "F": -0.20, "P":  0.15,
    "S":  0.14, "T":  0.09, "W": -0.19, "Y": -0.12, "V": -0.10,
    "X":  0.00,
}

# ── hydrophobicity scale for SAP (Black & Mould 1991, glycine-centred) ────────
# Source: Chennamsetty et al. 2009 (SAP paper).  Black & Mould 0–1 values
# shifted so glycine = 0: hydrophobic residues are positive, hydrophilic
# residues are negative.  This makes the positive-only SAP filter meaningful.
SAP_HYDROPHOBICITY: Dict[str, float] = {
    "A":  0.115, "R": -0.501, "N": -0.265, "D": -0.473, "C":  0.179,
    "Q": -0.250, "E": -0.458, "G":  0.000, "H": -0.336, "I":  0.442,
    "L":  0.442, "K": -0.218, "M":  0.237, "F":  0.499, "P":  0.210,
    "S": -0.142, "T": -0.051, "W":  0.377, "Y":  0.379, "V":  0.324,
    "X": -0.501,
}

# ── formal charge at pH 7.0 for SCM ──────────────────────────────────────────
# N-term: +1, C-term: -1 handled separately; here residue side-chain charges only
RESIDUE_CHARGE_PH7: Dict[str, float] = {
    "R": +1.0, "K": +1.0, "H": +0.1,   # positive (H partial at pH 7)
    "D": -1.0, "E": -1.0,               # negative
}

# ── hydrophobic residue set for patch detection ───────────────────────────────
HYDROPHOBIC_AA = {"A", "V", "L", "I", "M", "F", "W", "Y"}

# ── TANGO aggregation threshold ───────────────────────────────────────────────
TANGO_AGG_THRESHOLD = 5.0   # % aggregation to flag a residue
TANGO_MIN_SEG_LEN   = 5     # consecutive residues needed to count as a segment

# ── SAP/hydrophobic patch SASA threshold ─────────────────────────────────────
SASA_THRESHOLD_PATCH = 20.0   # Å² — minimum per-residue SASA for a patch residue
PATCH_CLUSTER_DIST   = 5.0    # Å  — Cα distance to merge into a patch
PATCH_LARGE_AREA     = 100.0  # Å² — patches above this count as "large"

SAP_SPHERE_RADIUS    = 5.0    # Å  — sphere radius for SAP neighbourhood

# ─────────────────────────────────────────────────────────────────────────────
# Worker-level globals (populated once per worker process in _worker_init)
# ─────────────────────────────────────────────────────────────────────────────
_CANYA_MODEL   = None   # loaded TF model or None if unavailable
_IUPRED_FN     = None   # callable(sequence, mode) -> list[float] or None

def _worker_init():
    """Called once per worker process to load heavy models."""
    global _CANYA_MODEL, _IUPRED_FN

    # ── CANYA ─────────────────────────────────────────────────────────────────
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from canya.canya import CANYA  # type: ignore
        _CANYA_MODEL = CANYA()
    except Exception as e:
        sys.stderr.write(f"[worker] CANYA not available: {e}\n")
        _CANYA_MODEL = None

    # ── IUPred3 ───────────────────────────────────────────────────────────────
    # iupred3 is a standalone download (not on PyPI).
    # Set IUPRED3_DIR to the directory containing iupred3_lib.py.
    # Default: ~/iupred3  (where the tarball unpacks on the cluster).
    iupred3_dir = os.environ.get(
        "IUPRED3_DIR",
        str(Path.home() / "iupred3"),
    )
    if iupred3_dir not in sys.path and Path(iupred3_dir).is_dir():
        sys.path.insert(0, iupred3_dir)
    try:
        import iupred3_lib  # type: ignore
        _iupred3_raw = iupred3_lib.iupred
        _IUPRED_FN = lambda seq: _iupred3_raw(seq, "long")[0]
    except ImportError:
        sys.stderr.write(
            f"[worker] iupred3 not available — check IUPRED3_DIR (tried: {iupred3_dir})\n"
        )
        _IUPRED_FN = None


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_protein(pt_path: Path):
    """Load a .pt file and return a PyG Data object with OpenFold-ordered coords."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Apply the PDB→OpenFold atom reindex (same as PDBDataset.__getitem__)
    if hasattr(data, "coords") and data.coords.ndim == 3:
        data.coords     = data.coords    [:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        data.coord_mask = data.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
    return data


def residue_type_to_sequence(residue_type_tensor: torch.Tensor) -> str:
    """Convert (L,) residue_type tensor → single-letter sequence string."""
    return "".join(IDX_TO_AA.get(int(idx), "X") for idx in residue_type_tensor)


def build_pdb_string(
    coords_of: np.ndarray,   # (L, 37, 3) — OpenFold-ordered, Å
    coord_mask: np.ndarray,  # (L, 37)    — bool
    residue_names: List[str],  # length L, 3-letter codes
) -> str:
    """
    Construct a minimal PDB-format string from tensor data.
    Only writes atoms that are present (coord_mask == True).
    """
    lines = []
    serial = 1
    for res_idx, res3 in enumerate(residue_names):
        res3 = res3.upper()[:3].ljust(3)
        res_num = res_idx + 1
        for atom_idx, atom_name in enumerate(OPENFOLD_ATOM_TYPES):
            if not coord_mask[res_idx, atom_idx]:
                continue
            x, y, z = coords_of[res_idx, atom_idx]
            # PDB column widths: atom name right-pad to 4 chars
            aname = atom_name.ljust(4) if len(atom_name) < 4 else atom_name[:4]
            lines.append(
                f"ATOM  {serial:5d} {aname} {res3} A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
            serial += 1
    lines.append("END\n")
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-based properties
# ─────────────────────────────────────────────────────────────────────────────

def compute_camsol(sequence: str) -> float:
    """CamSol intrinsic score — no public standalone available; returns NaN."""
    pass  # placeholder: replace with local binary call when available
    return float("nan")


def compute_swi(sequence: str) -> float:
    """
    Solubility-Weighted Index (Bhandari et al. 2020).
    Mean per-residue log-odds solubility propensity.
    """
    scores = [SWI_SCORES.get(aa, 0.0) for aa in sequence]
    return float(np.mean(scores)) if scores else float("nan")


def compute_tango(sequence: str, pdb_id: str) -> Tuple[float, int]:
    """
    Run the TANGO binary and return (total_aggregation_score, n_segments).

    TANGO writes output files into the current working directory, so we
    cd into a tmpdir for each call.

    Returns (nan, nan) if the binary is not found or fails.
    """
    # The binary on this cluster is tango_x86_64_release.
    # Override with: export TANGO_EXE=/path/to/tango_x86_64_release
    tango_exe = os.environ.get("TANGO_EXE", "tango_x86_64_release")
    jobname = (pdb_id.replace("/", "_"))[:25]  # TANGO max 25 chars

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            tango_exe, jobname,
            'ct=N', 'nt=N', 'ph=7.4', 'te=298', 'io=0.15',
            f'seq={sequence}',
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120,
                cwd=tmpdir,
            )
        except FileNotFoundError:
            sys.stderr.write(
                f"[tango] binary '{tango_exe}' not found "
                f"(set TANGO_EXE env var or add to PATH)\n"
            )
            return float("nan"), float("nan")
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[tango] timeout on {pdb_id}\n")
            return float("nan"), float("nan")

        if result.returncode != 0:
            sys.stderr.write(
                f"[tango] non-zero exit for {pdb_id}: {result.stderr.strip()[:200]}\n"
            )
            return float("nan"), float("nan")

        # This version of TANGO prints a single summary line to stdout:
        #   AGG <n_segments> AMYLO <score> TURN <score> HELIX <score> HELAGG <score> BETA <total_beta>
        # BETA = sum of per-residue beta-aggregation tendencies (our tango_total)
        # AGG  = number of aggregation-prone positions (our tango_aggregation_positions)
        stdout = result.stdout.strip()
        if not stdout:
            sys.stderr.write(f"[tango] empty output for {pdb_id}\n")
            return float("nan"), float("nan")

        fields: Dict[str, str] = {}
        parts = stdout.split()
        for i in range(0, len(parts) - 1, 2):
            fields[parts[i]] = parts[i + 1]

        try:
            total_agg = float(fields["BETA"])
            n_segments = int(float(fields["AGG"]))
        except (KeyError, ValueError) as e:
            sys.stderr.write(f"[tango] could not parse output for {pdb_id} ('{stdout}'): {e}\n")
            return float("nan"), float("nan")

        return float(total_agg), int(n_segments)


def compute_canya(sequence: str) -> float:
    """
    CANYA nucleation propensity (Lehner lab).
    Uses worker-local model if available; falls back to subprocess CLI.
    Returns max nucleation score across all windows.
    """
    # Try Python API (model loaded in _worker_init)
    if _CANYA_MODEL is not None:
        try:
            result = _CANYA_MODEL.predict(sequence)
            # result is typically a dict or array with per-position scores
            scores = result if isinstance(result, (list, np.ndarray)) else result.get("score", [])
            return float(np.max(scores)) if len(scores) > 0 else float("nan")
        except Exception as e:
            sys.stderr.write(f"[canya] model predict failed: {e}\n")

    # Subprocess fallback
    try:
        proc = subprocess.run(
            ["canya", "--sequence", sequence, "--summarize", "max"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            # Expect a single float on stdout
            return float(proc.stdout.strip().split()[-1])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        sys.stderr.write(f"[canya] subprocess failed: {e}\n")

    return float("nan")


def compute_charge_and_pI(sequence: str) -> Tuple[float, float]:
    """Net charge at pH 7.0 and isoelectric point via Biopython."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis  # type: ignore
    # Biopython rejects X residues — strip them
    clean_seq = sequence.replace("X", "")
    if not clean_seq:
        return float("nan"), float("nan")
    pa = ProteinAnalysis(clean_seq)
    charge = pa.charge_at_pH(7.0)
    pI = pa.isoelectric_point()
    return float(charge), float(pI)


def compute_iupred3(sequence: str) -> Tuple[float, float]:
    """
    IUPred3 disorder prediction.
    Returns (mean_disorder, fraction_disordered > 0.5).
    """
    if _IUPRED_FN is None:
        return float("nan"), float("nan")
    try:
        clean_seq = sequence.replace("X", "A")  # IUPred3 dislikes X
        scores = _IUPRED_FN(clean_seq)
        if scores is None or len(scores) == 0:
            return float("nan"), float("nan")
        arr = np.array(scores, dtype=float)
        return float(arr.mean()), float((arr > 0.5).mean())
    except Exception as e:
        sys.stderr.write(f"[iupred3] failed: {e}\n")
        return float("nan"), float("nan")


def compute_shannon_entropy(sequence: str) -> float:
    """Shannon entropy of amino acid composition (bits)."""
    if not sequence:
        return float("nan")
    from collections import Counter
    counts = Counter(sequence)
    n = len(sequence)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Structure-based properties
# ─────────────────────────────────────────────────────────────────────────────

def run_freesasa(pdb_string: str):
    """
    Run FreeSASA on an in-memory PDB string.
    Returns (result, structure) or (None, None) if freesasa unavailable.
    """
    try:
        import freesasa  # type: ignore
    except ImportError:
        sys.stderr.write("[freesasa] not installed\n")
        return None, None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as tmp:
        tmp.write(pdb_string)
        tmp_path = tmp.name

    try:
        structure = freesasa.Structure(tmp_path)
        result = freesasa.calc(structure)
        return result, structure
    except Exception as e:
        sys.stderr.write(f"[freesasa] calc failed: {e}\n")
        return None, None
    finally:
        os.unlink(tmp_path)


def compute_hydrophobic_patches(
    ca_coords: np.ndarray,   # (L, 3)
    sequence: str,
    per_residue_sasa: Optional[np.ndarray],  # (L,) or None
) -> Tuple[float, int]:
    """
    Identify hydrophobic patches.

    Algorithm:
    1. Select hydrophobic residues with SASA > SASA_THRESHOLD_PATCH.
    2. Build a graph where residues within PATCH_CLUSTER_DIST Å (Cα–Cα) are connected.
    3. Each connected component is a patch.
    4. Report total SASA area of all patches and number of patches >= PATCH_LARGE_AREA.
    """
    if per_residue_sasa is None:
        return float("nan"), float("nan")

    L = len(sequence)
    # Candidate residues: hydrophobic AND exposed
    candidates = [
        i for i in range(L)
        if sequence[i] in HYDROPHOBIC_AA
        and i < len(per_residue_sasa)
        and per_residue_sasa[i] > SASA_THRESHOLD_PATCH
    ]

    if not candidates:
        return 0.0, 0

    # Union-find clustering by Cα distance
    parent = {i: i for i in candidates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    cand_coords = ca_coords[candidates]
    for ii in range(len(candidates)):
        for jj in range(ii + 1, len(candidates)):
            dist = np.linalg.norm(cand_coords[ii] - cand_coords[jj])
            if dist <= PATCH_CLUSTER_DIST:
                union(candidates[ii], candidates[jj])

    # Aggregate SASA per patch
    patch_areas: Dict[int, float] = {}
    for idx in candidates:
        root = find(idx)
        patch_areas[root] = patch_areas.get(root, 0.0) + float(per_residue_sasa[idx])

    total_area = sum(patch_areas.values())
    n_large = sum(1 for area in patch_areas.values() if area >= PATCH_LARGE_AREA)

    return float(total_area), int(n_large)


def compute_sap_scm(
    ca_coords: np.ndarray,    # (L, 3)
    sequence: str,
    per_sidechain_sasa: Optional[np.ndarray],     # (L,) side-chain SASA
) -> Tuple[float, float, float]:
    """
    Compute SAP (Spatial Aggregation Propensity) and SCM (Spatial Charge Map).

    SAP implementation (Chennamsetty et al. 2009):
      For each residue i, define a neighbourhood N(i) = { j : |CA_j - CA_i| < 5 Å }.
      SAP_i = sum_{j in N(i)} (sidechain_SASA_j / max_sidechain_SASA_j) * hydrophobicity_j
      Protein SAP = sum of positive SAP_i only (negative values excluded per paper).

    SCM: same neighbourhood but weighted by formal charge instead of hydrophobicity.
    SCM_positive = sum of positive charge contributions (ARG, LYS, HIS patches).
    SCM_negative = sum of negative charge contributions (ASP, GLU patches).

    Both SAP and SCM use side-chain SASA normalised by Ala-X-Ala reference values
    (Tien et al. 2013 / Chennamsetty et al. 2009).  Formal charges live on
    side-chain tip atoms, so side-chain SASA is the appropriate weight for SCM.
    """
    # Max side-chain SASA for each residue type in an Ala-X-Ala tripeptide (Å²).
    # Source: Tien et al. 2013 (empirical, consistent with Chennamsetty 2009).
    # Glycine has no side chain so its reference is 0; its SAP contribution is
    # always zero regardless of exposure.
    AXA_SC_MAX_SASA: Dict[str, float] = {
        "A":  75.0, "R": 256.0, "N": 160.0, "D": 155.0, "C": 142.0,
        "Q": 189.0, "E": 187.0, "G":   0.0, "H": 194.0, "I": 185.0,
        "L": 183.0, "K": 218.0, "M": 203.0, "F": 218.0, "P": 141.0,
        "S": 116.0, "T": 142.0, "W": 254.0, "Y": 230.0, "V": 162.0,
        "X": 150.0,
    }

    if per_sidechain_sasa is None:
        return float("nan"), float("nan"), float("nan")

    L = len(sequence)
    sap_contributions = np.zeros(L)
    scm_contributions = np.zeros(L)

    for i in range(L):
        # Neighbours within sphere (including self)
        dists = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        neighbours = np.where(dists < SAP_SPHERE_RADIUS)[0]

        for j in neighbours:
            aa_j = sequence[j] if j < len(sequence) else "X"

            # SAP: side-chain SASA / AXA reference
            sc_sasa_j = float(per_sidechain_sasa[j]) if j < len(per_sidechain_sasa) else 0.0
            max_sc_j = AXA_SC_MAX_SASA.get(aa_j, 150.0)
            sap_frac_j = min(sc_sasa_j / max_sc_j, 1.0) if max_sc_j > 0 else 0.0

            hydro_j = SAP_HYDROPHOBICITY.get(aa_j, 0.0)
            sap_contributions[i] += sap_frac_j * hydro_j

            # SCM: side-chain SASA (formal charges live on side-chain atoms)
            charge_j = RESIDUE_CHARGE_PH7.get(aa_j, 0.0)
            scm_contributions[i] += sap_frac_j * charge_j

    # SAP: sum positive contributions only
    sap_total = float(np.sum(sap_contributions[sap_contributions > 0]))

    # SCM: split into positive and negative
    scm_positive = float(np.sum(scm_contributions[scm_contributions > 0]))
    scm_negative = float(np.sum(scm_contributions[scm_contributions < 0]))

    return sap_total, scm_positive, scm_negative


def compute_radius_of_gyration(ca_coords: np.ndarray, ca_mask: np.ndarray) -> float:
    """Rg from resolved Cα coordinates."""
    resolved = ca_coords[ca_mask]
    if len(resolved) < 2:
        return float("nan")
    center = resolved.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((resolved - center) ** 2, axis=1))))
    return rg


# ─────────────────────────────────────────────────────────────────────────────
# Main per-protein function (called in workers)
# ─────────────────────────────────────────────────────────────────────────────

def compute_properties(pt_path: Path) -> Dict:
    """
    Compute all properties for one protein.
    Returns a dict with COLUMNS as keys.
    Failed properties are NaN; errors are written to stderr.
    """
    row: Dict = {col: float("nan") for col in COLUMNS}

    try:
        data = load_protein(pt_path)
    except Exception as e:
        sys.stderr.write(f"[load] failed {pt_path}: {e}\n")
        row["pdb_id"] = pt_path.stem
        return row

    pdb_id = getattr(data, "id", pt_path.stem)
    row["pdb_id"] = pdb_id

    # ── coords and sequence ───────────────────────────────────────────────────
    coords_of  = data.coords.numpy()      # (L, 37, 3) — OpenFold order
    coord_mask = data.coord_mask.numpy()  # (L, 37)

    res_names  = list(data.residues)      # list of 3-letter strings, length L
    sequence   = residue_type_to_sequence(data.residue_type)

    L = len(sequence)
    row["sequence_length"] = L

    # Number of resolved residues: at least Cα present (OpenFold index 1)
    ca_mask = coord_mask[:, 1].astype(bool)
    row["n_resolved_residues"] = int(ca_mask.sum())

    ca_coords = coords_of[:, 1, :]   # (L, 3)

    # ── sequence-based ────────────────────────────────────────────────────────
    try:
        row["camsol_intrinsic"] = compute_camsol(sequence)
    except Exception as e:
        sys.stderr.write(f"[camsol] {pdb_id}: {e}\n")

    try:
        row["swi"] = compute_swi(sequence)
    except Exception as e:
        sys.stderr.write(f"[swi] {pdb_id}: {e}\n")

    try:
        row["tango_total"], row["tango_aggregation_positions"] = compute_tango(sequence, pdb_id)
    except Exception as e:
        sys.stderr.write(f"[tango] {pdb_id}: {e}\n")

    try:
        row["canya_max_nucleation"] = compute_canya(sequence)
    except Exception as e:
        sys.stderr.write(f"[canya] {pdb_id}: {e}\n")

    try:
        net_charge, pI = compute_charge_and_pI(sequence)
        row["net_charge_ph7"]             = net_charge
        row["pI"]                         = pI
        row["pI_distance_physiological"]  = abs(pI - 7.4) if not math.isnan(pI) else float("nan")
    except Exception as e:
        sys.stderr.write(f"[charge/pI] {pdb_id}: {e}\n")

    try:
        row["iupred3_mean"], row["iupred3_fraction_disordered"] = compute_iupred3(sequence)
    except Exception as e:
        sys.stderr.write(f"[iupred3] {pdb_id}: {e}\n")

    try:
        row["shannon_entropy"] = compute_shannon_entropy(sequence)
    except Exception as e:
        sys.stderr.write(f"[shannon] {pdb_id}: {e}\n")

    # ── structure-based ───────────────────────────────────────────────────────
    try:
        row["radius_of_gyration"] = compute_radius_of_gyration(ca_coords, ca_mask)
    except Exception as e:
        sys.stderr.write(f"[rg] {pdb_id}: {e}\n")

    # FreeSASA — shared by patches, SAP, SCM
    per_residue_sasa: Optional[np.ndarray] = None
    per_sidechain_sasa: Optional[np.ndarray] = None
    try:
        pdb_str = build_pdb_string(coords_of, coord_mask, res_names)
        sasa_result, sasa_structure = run_freesasa(pdb_str)

        if sasa_result is not None:
            # freesasa.Result.residueAreas() returns {chain: {resnum_str: Area}}
            # Area has .total (all atoms) and .sideChain (side-chain only).
            try:
                res_areas = sasa_result.residueAreas()
                # chain A, residues numbered 1..L
                chain_map = res_areas.get("A", {})
                per_residue_sasa = np.array(
                    [chain_map[str(i + 1)].total if str(i + 1) in chain_map else 0.0
                     for i in range(L)],
                    dtype=float,
                )
                per_sidechain_sasa = np.array(
                    [chain_map[str(i + 1)].sideChain if str(i + 1) in chain_map else 0.0
                     for i in range(L)],
                    dtype=float,
                )
            except Exception as e:
                sys.stderr.write(f"[freesasa residue areas] {pdb_id}: {e}\n")
    except Exception as e:
        sys.stderr.write(f"[freesasa] {pdb_id}: {e}\n")

    try:
        patch_area, n_large = compute_hydrophobic_patches(
            ca_coords, sequence, per_residue_sasa
        )
        row["hydrophobic_patch_total_area"] = patch_area
        row["hydrophobic_patch_n_large"]    = n_large
    except Exception as e:
        sys.stderr.write(f"[patches] {pdb_id}: {e}\n")

    try:
        sap, scm_pos, scm_neg = compute_sap_scm(
            ca_coords, sequence, per_sidechain_sasa
        )
        row["sap_total"]    = sap
        row["scm_positive"] = scm_pos
        row["scm_negative"] = scm_neg
    except Exception as e:
        sys.stderr.write(f"[sap/scm] {pdb_id}: {e}\n")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_pt_files(
    data_dir: Path,
    filter_csv: Optional[Path] = None,
) -> List[Path]:
    """
    Return sorted list of .pt file paths in <data_dir>/processed/.

    If filter_csv is given, restrict to the pdb/chain entries it contains
    (same logic as PDBDataset / precompute_latents.py).
    """
    processed = data_dir / "processed"

    if filter_csv is not None:
        import pandas as pd  # type: ignore
        df = pd.read_csv(filter_csv)
        paths = []
        for _, r in df.iterrows():
            pdb   = str(r["pdb"]).lower()
            chain = str(r.get("chain", "")).strip()
            fname = f"{pdb}_{chain}.pt" if chain and chain != "nan" else f"{pdb}.pt"
            shard = fname[:2].lower()
            p = processed / shard / fname
            if not p.exists():
                p = processed / fname
            if p.exists():
                paths.append(p)
        return paths

    # No filter: glob everything
    paths = list(processed.rglob("*.pt"))
    paths.sort()
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute biophysical property panel for the protein corpus."
    )
    parser.add_argument("output_csv", help="Path to write the output CSV")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_PATH", "/rds/user/ks2218/hpc-work"),
        help="Root data directory (contains processed/ subdirectory)",
    )
    parser.add_argument(
        "--filter-csv", default=None,
        help="Optional metadata CSV to restrict which proteins are processed "
             "(df_pdb_*.csv generated by PDBDataSelector)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N proteins (for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, os.cpu_count() - 1),
        help="Number of parallel worker processes (default: nCPU - 1)",
    )
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    output_csv  = Path(args.output_csv)
    filter_csv  = Path(args.filter_csv) if args.filter_csv else None
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── discover files ────────────────────────────────────────────────────────
    print(f"Discovering .pt files in {data_dir / 'processed'} …")
    pt_files = discover_pt_files(data_dir, filter_csv)
    if not pt_files:
        print("ERROR: No .pt files found. Check --data-dir.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        pt_files = pt_files[: args.limit]

    print(f"Found {len(pt_files):,} proteins to process.")

    # ── warm-up: time the first 10 serially ──────────────────────────────────
    WARMUP_N = min(10, len(pt_files))
    print(f"Timing first {WARMUP_N} proteins serially …")
    _worker_init()   # initialise models in main process for warmup
    t0 = time.perf_counter()
    for p in pt_files[:WARMUP_N]:
        compute_properties(p)
    warmup_elapsed = time.perf_counter() - t0
    per_protein_s  = warmup_elapsed / WARMUP_N
    total_s_est    = per_protein_s * len(pt_files) / args.workers
    h = int(total_s_est // 3600)
    m = int((total_s_est % 3600) // 60)
    print(
        f"Per-protein (wall): {per_protein_s:.2f} s  →  "
        f"Estimated runtime: {h} hours {m} minutes for {len(pt_files):,} proteins "
        f"({args.workers} workers)"
    )

    # ── open CSV and write header ─────────────────────────────────────────────
    already_done: set = set()
    write_header = not output_csv.exists()
    if output_csv.exists():
        # Resume: collect already-processed pdb_ids
        import pandas as pd  # type: ignore
        try:
            done_df = pd.read_csv(output_csv, usecols=["pdb_id"])
            already_done = set(done_df["pdb_id"].dropna())
            print(f"Resuming: {len(already_done):,} proteins already in output CSV.")
        except Exception:
            write_header = True

    csv_file = open(output_csv, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=COLUMNS)
    if write_header:
        writer.writeheader()

    # Filter out already-done
    if already_done:
        pt_files = [p for p in pt_files if p.stem not in already_done]
        print(f"{len(pt_files):,} proteins remaining.")

    # ── main parallel loop ────────────────────────────────────────────────────
    failures: Dict[str, int] = {col: 0 for col in COLUMNS}
    processed_count = 0
    batch_buffer: List[Dict] = []
    BATCH_SIZE = 100

    run_start = time.perf_counter()

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_worker_init,
    ) as pool:
        futures = {pool.submit(compute_properties, p): p for p in pt_files}

        with tqdm(total=len(pt_files), unit="prot", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as e:
                    pth = futures[future]
                    sys.stderr.write(f"[worker crash] {pth}: {e}\n")
                    row = {col: float("nan") for col in COLUMNS}
                    row["pdb_id"] = pth.stem

                # Tally failures
                for col in COLUMNS:
                    val = row.get(col, float("nan"))
                    if col != "pdb_id" and (
                        val is None
                        or (isinstance(val, float) and math.isnan(val))
                    ):
                        failures[col] += 1

                batch_buffer.append(row)
                processed_count += 1
                pbar.update(1)

                if len(batch_buffer) >= BATCH_SIZE:
                    writer.writerows(batch_buffer)
                    csv_file.flush()
                    batch_buffer.clear()

    # flush remainder
    if batch_buffer:
        writer.writerows(batch_buffer)
        csv_file.flush()
    csv_file.close()

    # ── summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - run_start
    h2 = int(total_elapsed // 3600)
    m2 = int((total_elapsed % 3600) // 60)
    print(f"\n{'='*60}")
    print(f"Done. Processed: {processed_count:,} proteins in {h2}h {m2}m")
    print(f"Output: {output_csv}")
    print("\nFailures per property:")
    for col, n in sorted(failures.items(), key=lambda x: -x[1]):
        if n > 0 and col != "pdb_id":
            pct = 100.0 * n / max(processed_count, 1)
            print(f"  {col:<40s}  {n:>6,}  ({pct:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
