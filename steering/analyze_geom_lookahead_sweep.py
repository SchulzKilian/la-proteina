"""Analysis for the geometric look-ahead sweep.

Computes, per arm, the PAIRED change in the steered property (guided vs the
matched-seed unguided baseline) plus a controller-behaviour summary. Property
deltas are length-binned and z-scored within each length using the unguided
spread at that length (pooling 300+400 under-reports and can hide per-length
sign flips — known trap for size-coupled properties like Rg/CO).

Designability is NOT computed here (separate GPU step). This script is CPU-only
and read-only, runnable as soon as structures exist.

    python -m steering.analyze_geom_lookahead_sweep
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
SWEEP = _ROOT / "results" / "geom_lookahead_sweep"
UNGUIDED = _ROOT / "results" / "sanity_unsteered_seed42_45" / "unguided"

CA_IDX = 1            # OpenFold atom order: CA = index 1
CONTACT_ANGSTROM = 8.0
MIN_SEQ_SEP = 3
DESIG_THRESH = 2.0    # scRMSD_ca_min < 2.0 A = designable


def _ca(pt_path: Path) -> np.ndarray:
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    coords = d["coords_openfold"]            # [L,37,3] Angstroms
    return coords[:, CA_IDX, :].float().numpy()


def rg(ca: np.ndarray) -> float:
    c = ca - ca.mean(0, keepdims=True)
    return float(np.sqrt((c ** 2).sum(1).mean()))


def contact_order(ca: np.ndarray) -> float:
    L = ca.shape[0]
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    iu = np.triu_indices(L, k=MIN_SEQ_SEP)
    sep = (iu[1] - iu[0]).astype(float)
    contact = d[iu] < CONTACT_ANGSTROM
    nc = contact.sum()
    if nc == 0:
        return 0.0
    return float((sep * contact).sum() / (L * nc))


PROP_FN = {"rg": rg, "contact_order": contact_order}
ARM_RE = re.compile(r"^(rg|contact_order)_(baseline|prop)_w(\d+)$")


def load_scrmsd(arm_dir: Path) -> dict[tuple[int, int], float]:
    """{(length, seed): scRMSD_ca_min} from {arm}/scRMSD_guided.csv, if present."""
    csv_path = arm_dir / "scRMSD_guided.csv"
    out: dict[tuple[int, int], float] = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m = re.match(r"s(\d+)_n(\d+)", row["protein_id"])
            if not m:
                continue
            try:
                out[(int(m.group(2)), int(m.group(1)))] = float(row["scRMSD_ca_min"])
            except (ValueError, KeyError):
                out[(int(m.group(2)), int(m.group(1)))] = float("inf")
    return out


def desig_stats(scr: dict, length: int) -> tuple[int, float, float]:
    """(n_folded, designable_fraction, mean_scRMSD) for one length."""
    vals = [v for (L, _), v in scr.items() if L == length]
    if not vals:
        return 0, float("nan"), float("nan")
    n = len(vals)
    rate = sum(v < DESIG_THRESH for v in vals) / n
    finite = [v for v in vals if v != float("inf")]
    return n, rate, (float(np.mean(finite)) if finite else float("inf"))


def main():
    # Cache unguided property values per (length, seed).
    ung_cache: dict[str, dict[tuple[int, int], float]] = {"rg": {}, "contact_order": {}}
    ung_files = sorted(UNGUIDED.glob("s*_n*.pt"))
    for f in ung_files:
        m = re.match(r"s(\d+)_n(\d+)", f.stem)
        if not m:
            continue
        seed, length = int(m.group(1)), int(m.group(2))
        ca = _ca(f)
        ung_cache["rg"][(length, seed)] = rg(ca)
        ung_cache["contact_order"][(length, seed)] = contact_order(ca)

    # Per-length unguided std (for z-scoring), per property.
    ung_std: dict[tuple[str, int], float] = {}
    for prop in ("rg", "contact_order"):
        for length in sorted({L for (L, _) in ung_cache[prop]}):
            vals = [v for (L, _), v in ung_cache[prop].items() if L == length]
            ung_std[(prop, length)] = float(np.std(vals)) if len(vals) > 1 else float("nan")

    # Unguided designability reference (per length).
    ung_scr = load_scrmsd(SWEEP / "_unguided_baseline")
    ung_desig = {L: desig_stats(ung_scr, L)[1] for L in sorted({L for (L, _) in ung_scr})}

    rows = []
    per_protein = []   # the per-protein (achieved Δ, designable) records for the tradeoff curve
    for arm_dir in sorted(SWEEP.glob("*")):
        m = ARM_RE.match(arm_dir.name)
        if not m or not (arm_dir / "guided").is_dir():
            continue
        target, mode, lam = m.group(1), m.group(2), int(m.group(3))
        prop_fn = PROP_FN[target]
        arm_scr = load_scrmsd(arm_dir)

        per_len: dict[int, dict] = {}
        for gpt in sorted((arm_dir / "guided").glob("s*_n*.pt")):
            sm = re.match(r"s(\d+)_n(\d+)", gpt.stem)
            seed, length = int(sm.group(1)), int(sm.group(2))
            if (length, seed) not in ung_cache[target]:
                continue
            g_val = prop_fn(_ca(gpt))
            u_val = ung_cache[target][(length, seed)]
            d = per_len.setdefault(length, {"g": [], "u": [], "delta": []})
            d["g"].append(g_val); d["u"].append(u_val); d["delta"].append(g_val - u_val)
            scr = arm_scr.get((length, seed))   # may be None if not folded yet
            per_protein.append({
                "target": target, "mode": mode, "lambda": lam,
                "length": length, "seed": seed,
                "guided": round(g_val, 4), "unguided": round(u_val, 4),
                "delta": round(g_val - u_val, 4), "abs_delta": round(abs(g_val - u_val), 4),
                "scRMSD_min": round(scr, 4) if scr is not None else "",
                "designable": (int(scr < DESIG_THRESH) if scr is not None else ""),
            })

        # Controller summary from diagnostics.
        s_all, dp_pos = [], []
        for dj in (arm_dir / "diagnostics").glob("*_diagnostics.json"):
            for e in json.load(open(dj)):
                if e.get("skipped"):
                    continue
                if "s" in e:
                    s_all.append(e["s"])
                if "delta_p" in e:
                    dp_pos.append(1.0 if e["delta_p"] > 0 else 0.0)
        s_mean = float(np.mean(s_all)) if s_all else float("nan")
        s_min = float(np.min(s_all)) if s_all else float("nan")
        frac_dp = float(np.mean(dp_pos)) if dp_pos else float("nan")

        for length, d in sorted(per_len.items()):
            mean_delta = float(np.mean(d["delta"]))
            sig = ung_std.get((target, length), float("nan"))
            n_fold, desig_rate, mean_scr = desig_stats(arm_scr, length)
            rows.append({
                "arm": arm_dir.name, "target": target, "mode": mode, "lambda": lam,
                "length": length, "n": len(d["delta"]),
                "unguided_mean": round(float(np.mean(d["u"])), 4),
                "guided_mean": round(float(np.mean(d["g"])), 4),
                "mean_delta": round(mean_delta, 4),
                "delta_sigma": round(mean_delta / sig, 3) if sig and not np.isnan(sig) else float("nan"),
                "s_mean": round(s_mean, 3), "s_min": round(s_min, 3),
                "frac_dP_pos": round(frac_dp, 3),
                "n_folded": n_fold,
                "desig_rate": round(desig_rate, 3) if not np.isnan(desig_rate) else float("nan"),
                "ung_desig_rate": round(ung_desig.get(length, float("nan")), 3) if ung_desig.get(length) is not None else float("nan"),
                "mean_scRMSD": round(mean_scr, 3) if not np.isnan(mean_scr) else float("nan"),
            })

    if not rows:
        print("No arms with structures yet.")
        return
    out_csv = SWEEP / "analysis_property_delta.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Pretty print sorted by target, length, mode, lambda.
    hdr = (f"{'arm':32s} {'L':>4s} {'n':>3s} {'ung':>7s} {'guid':>7s} {'Δ':>8s} {'Δσ':>7s} "
           f"{'s̄':>6s} {'desig':>6s} {'(ung)':>6s} {'nfold':>6s}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["target"], x["length"], x["mode"], x["lambda"])):
        print(f"{r['arm']:32s} {r['length']:>4d} {r['n']:>3d} {r['unguided_mean']:>7.3f} "
              f"{r['guided_mean']:>7.3f} {r['mean_delta']:>+8.3f} {r['delta_sigma']:>7} "
              f"{r['s_mean']:>6} {r['desig_rate']:>6} {r['ung_desig_rate']:>6} {r['n_folded']:>6}")
    print(f"\nwrote {out_csv}")

    # --- Per-protein records + the designability-vs-achieved-Δ tradeoff ---
    pp_csv = SWEEP / "analysis_per_protein.csv"
    with open(pp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_protein[0].keys()))
        w.writeheader(); w.writerows(per_protein)
    print(f"wrote {pp_csv}")

    # Bin by achieved |Δ| (per target/length) and compare designability of
    # baseline vs prop AT MATCHED STEERING. This is the only fair comparison:
    # at the same achieved property change, which method stays designable?
    BIN_EDGES = {
        "rg": [0, 1, 2, 3, 4, 6, 1e9],
        "contact_order": [0, 0.01, 0.02, 0.04, 0.08, 1e9],
    }
    folded = [p for p in per_protein if p["designable"] != ""]
    print("\n=== Designability vs achieved |Δ| (matched-steering tradeoff) ===")
    print("    (only folded proteins; d/n = designable/total in that |Δ| bin)")
    for target in sorted({p["target"] for p in folded}):
        edges = BIN_EDGES[target]
        for length in sorted({p["length"] for p in folded if p["target"] == target}):
            print(f"\n  {target}  L={length}")
            print(f"    {'|Δ| bin':>14s}   {'baseline d/n':>14s}   {'prop d/n':>14s}")
            for lo, hi in zip(edges[:-1], edges[1:]):
                cell = {}
                for mode in ("baseline", "prop"):
                    pts = [p for p in folded if p["target"] == target and p["length"] == length
                           and p["mode"] == mode and lo <= p["abs_delta"] < hi]
                    if pts:
                        d = sum(p["designable"] for p in pts)
                        cell[mode] = f"{d}/{len(pts)}"
                    else:
                        cell[mode] = "·"
                if cell["baseline"] == "·" and cell["prop"] == "·":
                    continue
                hi_s = "inf" if hi >= 1e9 else f"{hi:g}"
                print(f"    {f'[{lo:g},{hi_s})':>14s}   {cell['baseline']:>14s}   {cell['prop']:>14s}")


if __name__ == "__main__":
    main()
