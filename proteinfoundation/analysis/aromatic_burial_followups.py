#!/usr/bin/env python3
"""Follow-up analyses on top of aromatic_burial.py output.

Reads per_residue.parquet (residue, rsa, protein_id, set) from a previous
aromatic_burial run and produces three independent experiments:

  EXP1 — compositional breakdown of the aromatic pool: per-set W/F/Y/H
         fractions among aromatic residues with bootstrap CIs, plus a
         counterfactual gen group-level burial ratio reweighted by PDB
         aromatic-pool proportions.

  EXP2 — curve shape vs amplitude for F and Y: KDE of P(residue is X | RSA)
         on continuous RSA, raw + area-normalized overlays with bootstrap
         bands, plus a logistic slope comparison.

  EXP3 — per-protein F burial-targeting ratio P(F|buried)/P(F|exposed),
         filtered (≥10 F total, ≥2 buried, ≥2 exposed). Falls back to the
         W+F+Y aromatic group when fewer than 30 gen proteins survive.

Bootstrap is over PROTEINS (1000 resamples). Buried <0.20, exposed ≥0.50
(matching aromatic_burial.py).

Example:
    python proteinfoundation/analysis/aromatic_burial_followups.py \\
        --in-file results/aromatic_burial/per_residue.parquet \\
        --out-dir results/aromatic_burial/followups
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

AROMATIC = ["W", "F", "Y", "H"]
GROUP_NONH = ["W", "F", "Y"]  # H excluded from group fallback (amphipathic)
N_BOOT = 1000
KDE_GRID = np.linspace(0.0, 1.0, 200)
KDE_BANDWIDTH = "scott"  # scipy gaussian_kde scott rule; flagged in run.log
KDE_SUBSAMPLE_MAX = 8000  # cap residues per KDE call (per bootstrap iter)
BURIED_T = 0.20
EXPOSED_T = 0.50

PROTEIN_FILTER = dict(min_total=10, min_buried=2, min_exposed=2)
GEN_FALLBACK_THRESHOLD = 30


# ---------- shared utilities ----------

def ensure_sample_id(df: pd.DataFrame, log) -> pd.DataFrame:
    """Provide a `sample_id` column.

    Newer aromatic_burial.py writes `sample_idx` directly (1 unique value
    per sampling instance — length-matched ref draws WITH replacement, so
    the same protein_id can occupy multiple sample slots). Older parquets
    lacking this column fall back to a row-order transition heuristic
    that misses adjacent same-stem duplicates (≤3% under-count in
    practice — flagged below).
    """
    if "sample_idx" in df.columns:
        df = df.copy()
        df["sample_id"] = df["sample_idx"].astype(int)
        log.info("  using explicit sample_idx column from parquet")
        return df
    log.info("  WARN: no sample_idx column; deriving sample_id from row "
             "transitions (may under-count adjacent same-stem duplicates "
             "from length_match by ~3%)")
    blocks = []
    next_id = 0
    for s in ("gen", "ref"):
        sub = df[df["set"] == s].copy()
        if len(sub) == 0:
            continue
        prev = sub["protein_id"].shift()
        is_new_block = (sub["protein_id"] != prev) | prev.isna()
        sub["sample_id"] = is_new_block.cumsum().astype(int) + next_id - 1
        next_id = int(sub["sample_id"].max()) + 1
        blocks.append(sub)
    return pd.concat(blocks, ignore_index=True)


def boot_indices(rng, n, n_boot=N_BOOT):
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


# ============================================================
# EXPERIMENT 1 — composition breakdown + counterfactual
# ============================================================

def exp1_composition(df: pd.DataFrame, rng, log) -> dict:
    log.info("=" * 70)
    log.info("EXP1: compositional breakdown of aromatic pool + counterfactual")
    log.info("=" * 70)
    out = {"per_set": {}, "ratios": {}}

    # bootstrap over proteins (sampling-instances; length-matched draws are
    # preserved as separate units); per-protein counts of W/F/Y/H + total aromatic
    for s in ("gen", "ref"):
        sub = df[df["set"] == s].copy()
        per_p = (
            sub.groupby("sample_id")["residue"]
            .apply(lambda x: pd.Series({a: int((x == a).sum()) for a in AROMATIC}))
            .unstack(fill_value=0)
        )
        per_p["arom"] = per_p[AROMATIC].sum(axis=1)
        # observed (pooled) fractions, just from sums
        totals = per_p[AROMATIC].sum()
        arom_total = int(per_p["arom"].sum())
        if arom_total == 0:
            raise SystemExit(f"[{s}] no aromatic residues — aborting")
        observed = (totals / arom_total).to_dict()

        # bootstrap fractions over proteins
        proteins = per_p.index.values
        cnt_arr = per_p[AROMATIC].values  # (P, 4)
        arom_arr = per_p["arom"].values
        boots = []
        for idx in boot_indices(rng, len(proteins)):
            num = cnt_arr[idx].sum(axis=0)
            den = arom_arr[idx].sum()
            if den == 0:
                continue
            boots.append(num / den)
        boots = np.array(boots)  # (B, 4)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out["per_set"][s] = {
            a: {"frac": observed[a], "lo": float(lo[i]), "hi": float(hi[i])}
            for i, a in enumerate(AROMATIC)
        }
        log.info(f"  [{s}] aromatic-pool composition (mean [95%CI]):")
        for i, a in enumerate(AROMATIC):
            log.info(f"    {a}: {observed[a]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]")
        s_check = sum(observed.values())
        log.info(f"    sum = {s_check:.4f} (sanity: should be 1.0)")

    # per-residue gen burial ratios r_i = P(i|buried)/P(i|exposed) — pool counts
    gen = df[df["set"] == "gen"]
    ratios_gen = {}
    n_buried = (gen["rsa"] < BURIED_T).sum()
    n_exposed = (gen["rsa"] >= EXPOSED_T).sum()
    for a in AROMATIC:
        nb = ((gen["residue"] == a) & (gen["rsa"] < BURIED_T)).sum()
        ne = ((gen["residue"] == a) & (gen["rsa"] >= EXPOSED_T)).sum()
        if ne == 0 or n_buried == 0 or n_exposed == 0:
            ratios_gen[a] = float("nan")
        else:
            ratios_gen[a] = (nb / n_buried) / (ne / n_exposed)
    out["per_residue_gen_ratios"] = ratios_gen
    log.info(f"  per-residue gen burial ratios: " +
             ", ".join(f"{a}={ratios_gen[a]:.3f}" for a in AROMATIC))

    p_gen = {a: out["per_set"]["gen"][a]["frac"] for a in AROMATIC}
    p_ref = {a: out["per_set"]["ref"][a]["frac"] for a in AROMATIC}
    observed_gen_group = sum(p_gen[a] * ratios_gen[a] for a in AROMATIC)
    counterfactual = sum(p_ref[a] * ratios_gen[a] for a in AROMATIC)

    # also compute the strict empirical group ratio for reference
    arom_buried_gen = ((gen["residue"].isin(AROMATIC)) & (gen["rsa"] < BURIED_T)).sum()
    arom_exposed_gen = ((gen["residue"].isin(AROMATIC)) & (gen["rsa"] >= EXPOSED_T)).sum()
    strict_gen = (arom_buried_gen / n_buried) / (arom_exposed_gen / n_exposed) \
        if arom_exposed_gen > 0 else float("nan")
    out["ratios"] = {
        "observed_gen_group_pweighted": float(observed_gen_group),
        "counterfactual_gen_group_pdb_weighted": float(counterfactual),
        "empirical_gen_group_ratio": float(strict_gen),
    }

    rel = abs(counterfactual - observed_gen_group) / max(abs(observed_gen_group), 1e-9)
    if rel > 0.20:
        flag = (f"COMPOSITIONAL: counterfactual {counterfactual:.2f} vs "
                f"observed {observed_gen_group:.2f} differ by {100*rel:.1f}% "
                f">20% → group-level preservation is driven by aromatic "
                f"composition, not behavioral preservation.")
    else:
        flag = (f"NOT COMPOSITIONAL: counterfactual {counterfactual:.2f} vs "
                f"observed {observed_gen_group:.2f} within {100*rel:.1f}% "
                f"(≤20%) → group preservation is not explained by "
                f"composition reweighting.")
    out["flag"] = flag
    log.info(f"  observed (p_gen·r_gen) = {observed_gen_group:.3f}")
    log.info(f"  counterfactual (p_pdb·r_gen) = {counterfactual:.3f}")
    log.info(f"  empirical gen group ratio (raw counts) = {strict_gen:.3f}")
    log.info(f"  → {flag}")
    return out


def write_exp1_md(out: dict, fp) -> None:
    fp.write("## Experiment 1 — Compositional breakdown of the aromatic pool\n\n")
    fp.write("Fraction of aromatic residues that are W/F/Y/H per set "
             "(bootstrap over proteins, 1000 resamples, 95% CI):\n\n")
    fp.write("| set | W | F | Y | H | sum |\n")
    fp.write("|-----|---|---|---|---|-----|\n")
    for s in ("gen", "ref"):
        cells = []
        total = 0.0
        for a in AROMATIC:
            d = out["per_set"][s][a]
            cells.append(f"{d['frac']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")
            total += d["frac"]
        fp.write(f"| {s} | " + " | ".join(cells) + f" | {total:.3f} |\n")
    fp.write("\n**Counterfactual decomposition of the gen aromatic-group "
             "burial ratio** (same per-residue gen ratios, reweighted by "
             "PDB aromatic-pool fractions):\n\n")
    r = out["ratios"]
    rg = out["per_residue_gen_ratios"]
    fp.write("| quantity | value |\n|---|---|\n")
    fp.write(f"| per-residue gen burial ratios | "
             f"W={rg['W']:.2f}, F={rg['F']:.2f}, "
             f"Y={rg['Y']:.2f}, H={rg['H']:.2f} |\n")
    fp.write(f"| observed gen group ratio (p_gen · r_gen) | "
             f"{r['observed_gen_group_pweighted']:.3f} |\n")
    fp.write(f"| empirical gen group ratio (raw counts) | "
             f"{r['empirical_gen_group_ratio']:.3f} |\n")
    fp.write(f"| counterfactual (p_pdb · r_gen) | "
             f"{r['counterfactual_gen_group_pdb_weighted']:.3f} |\n\n")
    fp.write(f"**Flag:** {out['flag']}\n\n")


# ============================================================
# EXPERIMENT 2 — curve shape vs amplitude (F, Y)
# ============================================================

def _subsample(values: np.ndarray, k: int, rng) -> np.ndarray:
    if len(values) <= k:
        return values
    idx = rng.choice(len(values), size=k, replace=False)
    return values[idx]


def kde_density(values: np.ndarray, grid: np.ndarray, bw) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(grid)
    kd = sstats.gaussian_kde(values, bw_method=bw)
    return kd(grid)


def cond_prob_curve(rsa_target: np.ndarray, rsa_all: np.ndarray,
                     grid: np.ndarray, bw, rng,
                     subsample_max: int = KDE_SUBSAMPLE_MAX) -> np.ndarray:
    """P(target|RSA=x) ≈ f_t(x) * P(target) / f_a(x), with f_t, f_a KDEs.

    Both target and all are subsampled (without replacement) to at most
    `subsample_max` to keep bootstrap KDE compute tractable. KDE is
    intensity-normalised (integrates to 1) so subsampling does not affect
    the estimated densities; the marginal probability P(target) =
    |target|/|all| is computed from the ORIGINAL (pre-subsample) counts
    so the absolute amplitude of the curve is preserved.
    """
    if len(rsa_target) < 2 or len(rsa_all) < 2:
        return np.full_like(grid, np.nan)
    p_target = len(rsa_target) / len(rsa_all)  # original-scale marginal
    rt = _subsample(rsa_target, subsample_max, rng)
    ra = _subsample(rsa_all, subsample_max, rng)
    f_t = kde_density(rt, grid, bw)
    f_a = kde_density(ra, grid, bw)
    out = np.full_like(grid, np.nan)
    m = f_a > 1e-9
    out[m] = p_target * f_t[m] / f_a[m]
    return out


def fit_logistic_slope(rsa: np.ndarray, y: np.ndarray, rng,
                        subsample_max: int = 30000) -> float:
    """Fit P(y|RSA) = sigmoid(a + b*RSA), return slope b. Subsampled (uniform
    over residues) to cap compute when ref bootstrap concatenates >500K rows.
    """
    if y.sum() < 2 or y.sum() == len(y):
        return float("nan")
    if len(rsa) > subsample_max:
        idx = rng.choice(len(rsa), size=subsample_max, replace=False)
        rsa, y = rsa[idx], y[idx]
        if y.sum() < 2 or y.sum() == len(y):
            return float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    lr.fit(rsa.reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def exp2_curve_shape(df: pd.DataFrame, rng, log, out_dir: Path) -> dict:
    log.info("=" * 70)
    log.info("EXP2: curve shape vs amplitude for F and Y (KDE + logistic)")
    log.info("=" * 70)
    log.info(f"  KDE bw_method = {KDE_BANDWIDTH!r} (scipy default Scott rule);"
             f" CHOICE-FLAG: per-bootstrap adaptive, common default")
    out = {"per_residue": {}, "verdicts": {}}

    # Pre-stash per-sampling-instance arrays so we can resample at protein level
    # (length-matched ref has duplicate stems — sample_id preserves the 1002 units).
    by_set = {}
    for s in ("gen", "ref"):
        proteins = []
        for pid, g in df[df["set"] == s].groupby("sample_id", sort=True):
            proteins.append((pid, g["residue"].values, g["rsa"].values))
        by_set[s] = proteins
        log.info(f"  [{s}] {len(proteins)} sampling instances")

    for residue in ("F", "Y"):
        log.info(f"  -- residue {residue} --")
        curves = {}
        slopes = {}
        for s in ("gen", "ref"):
            proteins = by_set[s]
            P = len(proteins)
            curves_raw_b = np.zeros((N_BOOT, len(KDE_GRID)))
            curves_norm_b = np.zeros((N_BOOT, len(KDE_GRID)))
            slopes_b = np.zeros(N_BOOT)
            for bi, idx in enumerate(boot_indices(rng, P)):
                rsa_target = np.concatenate(
                    [proteins[i][2][proteins[i][1] == residue] for i in idx]
                )
                rsa_all = np.concatenate([proteins[i][2] for i in idx])
                y = np.concatenate(
                    [(proteins[i][1] == residue).astype(int) for i in idx]
                )
                c = cond_prob_curve(rsa_target, rsa_all, KDE_GRID,
                                     KDE_BANDWIDTH, rng)
                curves_raw_b[bi] = c
                area = np.trapz(np.nan_to_num(c, nan=0.0), KDE_GRID)
                curves_norm_b[bi] = (c / area) if area > 1e-9 else np.nan
                slopes_b[bi] = fit_logistic_slope(rsa_all, y, rng)
            # mean / band
            raw_mean = np.nanmean(curves_raw_b, axis=0)
            raw_lo, raw_hi = np.nanpercentile(curves_raw_b, [2.5, 97.5], axis=0)
            norm_mean = np.nanmean(curves_norm_b, axis=0)
            norm_lo, norm_hi = np.nanpercentile(curves_norm_b, [2.5, 97.5], axis=0)
            slope_mean = float(np.nanmean(slopes_b))
            slope_lo, slope_hi = np.nanpercentile(slopes_b, [2.5, 97.5])
            curves[s] = dict(raw=(raw_mean, raw_lo, raw_hi),
                             norm=(norm_mean, norm_lo, norm_hi))
            slopes[s] = (slope_mean, float(slope_lo), float(slope_hi))
            log.info(f"    [{s}] logistic slope b = {slope_mean:.2f} "
                     f"[{slope_lo:.2f}, {slope_hi:.2f}]")

        # plots
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title in [(axes[0], "raw", f"P(is {residue} | RSA) — raw"),
                                (axes[1], "norm",
                                 f"P(is {residue} | RSA) — area-normalized")]:
            for s, color in [("gen", "C0"), ("ref", "C1")]:
                m, lo, hi = curves[s][key]
                ax.plot(KDE_GRID, m, color=color, lw=1.5, label=s)
                ax.fill_between(KDE_GRID, lo, hi, color=color, alpha=0.25, lw=0)
            ax.axvline(BURIED_T, color="k", ls=":", alpha=0.4)
            ax.axvline(EXPOSED_T, color="k", ls=":", alpha=0.4)
            ax.set_xlabel("RSA")
            ax.set_title(title)
            ax.legend(frameon=False)
        axes[0].set_ylabel(f"P({residue} | RSA)")
        axes[1].set_ylabel("normalized density")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"exp2_{residue}_curves.{ext}", dpi=150)
        plt.close(fig)

        # verdict
        gs, glo, ghi = slopes["gen"]
        rs, rlo, rhi = slopes["ref"]
        overlap = not (ghi < rlo or rhi < glo)
        if overlap:
            verdict = (f"{residue}: gen slope {gs:.2f}[{glo:.2f},{ghi:.2f}] "
                       f"vs PDB slope {rs:.2f}[{rlo:.2f},{rhi:.2f}] — CIs "
                       f"OVERLAP → SAME SHAPE; amplitude difference is the "
                       f"likely driver (consistent with supply-limited "
                       f"placement, not broken machinery).")
        else:
            looser = "FLATTER" if abs(gs) < abs(rs) else "STEEPER"
            verdict = (f"{residue}: gen slope {gs:.2f}[{glo:.2f},{ghi:.2f}] "
                       f"vs PDB slope {rs:.2f}[{rlo:.2f},{rhi:.2f}] — CIs DO "
                       f"NOT OVERLAP, gen is {looser} → placement preference "
                       f"differs in shape from PDB.")
        log.info(f"    verdict: {verdict}")
        out["per_residue"][residue] = {"slopes": slopes,
                                       "ci_overlap": bool(overlap)}
        out["verdicts"][residue] = verdict
    return out


def write_exp2_md(out: dict, fp) -> None:
    fp.write("## Experiment 2 — Curve shape vs amplitude (F, Y)\n\n")
    fp.write("KDE `bw_method='scott'` (scipy default; flagged in run.log). "
             "Bootstrap over proteins (1000 resamples). Logistic regression "
             "fits `P(is X) = sigmoid(a + b·RSA)` separately for gen and "
             "PDB on each bootstrap replicate. Per-iteration KDE / logistic "
             "inputs are uniformly subsampled (≤8000 / ≤30000 residues) for "
             "tractable compute; KDE marginal P(target) uses original counts.\n\n")
    fp.write("| residue | gen slope b [95% CI] | PDB slope b [95% CI] | "
             "CI overlap? |\n")
    fp.write("|---|---|---|---|\n")
    for r in ("F", "Y"):
        gs, glo, ghi = out["per_residue"][r]["slopes"]["gen"]
        rs, rlo, rhi = out["per_residue"][r]["slopes"]["ref"]
        ov = "yes" if out["per_residue"][r]["ci_overlap"] else "**no**"
        fp.write(f"| {r} | {gs:.2f} [{glo:.2f}, {ghi:.2f}] | "
                 f"{rs:.2f} [{rlo:.2f}, {rhi:.2f}] | {ov} |\n")
    fp.write("\nPlots: `exp2_F_curves.{png,pdf}`, `exp2_Y_curves.{png,pdf}` "
             "(raw + area-normalized side-by-side).\n\n")
    fp.write("**Verdicts:**\n\n")
    for r in ("F", "Y"):
        fp.write(f"- {out['verdicts'][r]}\n")
    fp.write("\n")


# ============================================================
# EXPERIMENT 3 — per-protein burial-targeting distribution
# ============================================================

def per_protein_ratios(df: pd.DataFrame, set_label: str,
                       residues: list[str]) -> tuple[list[float], int, int]:
    """Per-PHYSICAL-protein burial ratio: dedupe on protein_id (the same
    physical protein appearing multiple times in length-matched ref would
    just contribute identical ratios and inflate KS/N.).
    """
    sub = df[df["set"] == set_label]
    # take a single sampling instance per unique physical protein
    first_sample = sub.drop_duplicates(subset=["protein_id"])["sample_id"].values
    sub = sub[sub["sample_id"].isin(first_sample)]
    sub = sub.assign(is_target=sub["residue"].isin(residues),
                     bin_b=sub["rsa"] < BURIED_T,
                     bin_e=sub["rsa"] >= EXPOSED_T)
    ratios = []
    n_total = sub["protein_id"].nunique()
    for pid, g in sub.groupby("protein_id"):
        n_t = int(g["is_target"].sum())
        if n_t < PROTEIN_FILTER["min_total"]:
            continue
        n_t_b = int((g["is_target"] & g["bin_b"]).sum())
        n_t_e = int((g["is_target"] & g["bin_e"]).sum())
        if n_t_b < PROTEIN_FILTER["min_buried"] or n_t_e < PROTEIN_FILTER["min_exposed"]:
            continue
        n_b = int(g["bin_b"].sum())
        n_e = int(g["bin_e"].sum())
        if n_b == 0 or n_e == 0:
            continue
        p_t_b = n_t_b / n_b
        p_t_e = n_t_e / n_e
        if p_t_e <= 0:
            continue
        ratios.append(p_t_b / p_t_e)
    return ratios, len(ratios), n_total


def exp3_per_protein(df: pd.DataFrame, log, out_dir: Path) -> dict:
    log.info("=" * 70)
    log.info("EXP3: per-protein burial-targeting distribution")
    log.info("=" * 70)
    log.info(f"  Filter thresholds: ≥{PROTEIN_FILTER['min_total']} target total, "
             f"≥{PROTEIN_FILTER['min_buried']} buried, "
             f"≥{PROTEIN_FILTER['min_exposed']} exposed")
    out = {"used_residues": ["F"], "fallback": False, "per_set": {}, "verdict": ""}

    gen_ratios, gen_kept, gen_total = per_protein_ratios(df, "gen", ["F"])
    ref_ratios, ref_kept, ref_total = per_protein_ratios(df, "ref", ["F"])
    log.info(f"  F-only: gen survives = {gen_kept}/{gen_total}, "
             f"ref survives = {ref_kept}/{ref_total}")

    if gen_kept < GEN_FALLBACK_THRESHOLD:
        log.info(f"  Only {gen_kept} gen proteins survive (<{GEN_FALLBACK_THRESHOLD}); "
                 f"falling back to W+F+Y aromatic group (H excluded — amphipathic).")
        out["fallback"] = True
        out["used_residues"] = GROUP_NONH
        gen_ratios, gen_kept, gen_total = per_protein_ratios(df, "gen", GROUP_NONH)
        ref_ratios, ref_kept, ref_total = per_protein_ratios(df, "ref", GROUP_NONH)
        log.info(f"  Group(W+F+Y): gen survives = {gen_kept}/{gen_total}, "
                 f"ref survives = {ref_kept}/{ref_total}")

    label = "+".join(out["used_residues"])
    if gen_kept < GEN_FALLBACK_THRESHOLD:
        log.info(f"  Even after fallback, only {gen_kept} gen proteins "
                 f"survive — per-protein analysis is UNDERPOWERED. No "
                 f"histogram drawn; reporting summary stats with a strong "
                 f"caveat.")
        g = np.array(gen_ratios) if gen_ratios else np.array([np.nan])
        r = np.array(ref_ratios) if ref_ratios else np.array([np.nan])
        out["per_set"]["gen"] = dict(
            n=gen_kept, total=gen_total, ratios=gen_ratios,
            median=float(np.nanmedian(g)) if gen_ratios else float("nan"),
            iqr=(float(np.nanpercentile(g, 25)) if gen_ratios else float("nan"),
                 float(np.nanpercentile(g, 75)) if gen_ratios else float("nan")))
        out["per_set"]["ref"] = dict(
            n=ref_kept, total=ref_total, ratios=ref_ratios,
            median=float(np.nanmedian(r)) if ref_ratios else float("nan"),
            iqr=(float(np.nanpercentile(r, 25)) if ref_ratios else float("nan"),
                 float(np.nanpercentile(r, 75)) if ref_ratios else float("nan")))
        if gen_ratios and ref_ratios:
            ks_stat, ks_p = sstats.ks_2samp(g, r)
            out["ks"] = {"stat": float(ks_stat), "p": float(ks_p)}
        out["verdict"] = (f"underpowered to distinguish (gen N={gen_kept}, "
                          f"ref N={ref_kept}; histogram suppressed; medians "
                          f"reported with caveat)")
        log.info(f"  verdict: {out['verdict']}")
        if gen_ratios:
            log.info(f"  (caveat) gen median={out['per_set']['gen']['median']:.2f} "
                     f"(N={gen_kept}); ref median={out['per_set']['ref']['median']:.2f} "
                     f"(N={ref_kept})")
        return out

    g = np.array(gen_ratios)
    r = np.array(ref_ratios)
    g_med, r_med = float(np.median(g)), float(np.median(r))
    g_iqr = (float(np.percentile(g, 25)), float(np.percentile(g, 75)))
    r_iqr = (float(np.percentile(r, 25)), float(np.percentile(r, 75)))
    ks_stat, ks_p = sstats.ks_2samp(g, r)
    out["per_set"] = {
        "gen": dict(n=gen_kept, total=gen_total, ratios=gen_ratios,
                    median=g_med, iqr=g_iqr),
        "ref": dict(n=ref_kept, total=ref_total, ratios=ref_ratios,
                    median=r_med, iqr=r_iqr),
    }
    out["ks"] = {"stat": float(ks_stat), "p": float(ks_p)}
    log.info(f"  median: gen={g_med:.2f} (IQR {g_iqr[0]:.2f}-{g_iqr[1]:.2f}), "
             f"ref={r_med:.2f} (IQR {r_iqr[0]:.2f}-{r_iqr[1]:.2f})")
    log.info(f"  KS: stat={ks_stat:.3f}, p={ks_p:.3g}  "
             f"(N_gen={gen_kept}, N_ref={ref_kept}; "
             f"p-value noisy at small N)")

    # plot — log-scale x
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g_log = np.log10(np.clip(g, 1e-3, None))
    r_log = np.log10(np.clip(r, 1e-3, None))
    lo = float(min(g_log.min(), r_log.min()))
    hi = float(max(g_log.max(), r_log.max()))
    bins = np.linspace(lo - 0.05, hi + 0.05, 30)
    ax.hist(g_log, bins=bins, alpha=0.55, color="C0",
            label=f"gen (N={gen_kept})", density=True)
    ax.hist(r_log, bins=bins, alpha=0.55, color="C1",
            label=f"PDB (N={ref_kept})", density=True)
    ax.axvline(np.log10(g_med), color="C0", ls="--", lw=1.2,
               label=f"gen median {g_med:.2f}")
    ax.axvline(np.log10(r_med), color="C1", ls="--", lw=1.2,
               label=f"PDB median {r_med:.2f}")
    ax.set_xlabel(f"log10  P({label}|buried) / P({label}|exposed)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-protein burial-targeting ratio  ({label})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"exp3_per_protein_{label}.{ext}", dpi=150)
    plt.close(fig)

    # heuristic verdict on shape
    g_shifted = g_med / r_med if r_med > 0 else float("nan")
    long_tail = (np.percentile(g, 95) / max(g_med, 1e-9) > 5)
    bimodal = False
    try:
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde as _kde
        kg = _kde(g_log, bw_method=0.3)
        xs = np.linspace(lo, hi, 200)
        ys = kg(xs)
        peaks, _ = find_peaks(ys, prominence=0.05 * ys.max())
        bimodal = len(peaks) >= 2
    except Exception:
        pass

    if bimodal:
        verdict = (f"bimodal (gen median {g_med:.2f} vs PDB {r_med:.2f}, "
                   f"KS p={ks_p:.2g}; ≥2 peaks in gen log-density)")
    elif long_tail:
        verdict = (f"long tail driven by subset (gen median {g_med:.2f}, "
                   f"95th pct/median = "
                   f"{np.percentile(g,95)/max(g_med,1e-9):.1f}, KS p={ks_p:.2g})")
    elif abs(np.log(max(g_shifted, 1e-9))) > 0.2:
        verdict = (f"uniform shift (gen median {g_med:.2f} vs PDB {r_med:.2f}; "
                   f"factor {g_shifted:.2f}; KS p={ks_p:.2g})")
    else:
        verdict = (f"overlapping distributions (gen median {g_med:.2f} vs "
                   f"PDB {r_med:.2f}, KS p={ks_p:.2g})")
    out["verdict"] = verdict
    log.info(f"  verdict: {verdict}")
    return out


def write_exp3_md(out: dict, fp) -> None:
    label = "+".join(out["used_residues"])
    fp.write(f"## Experiment 3 — Per-protein burial-targeting distribution "
             f"({label})\n\n")
    if out["fallback"]:
        fp.write("**FALLBACK:** fewer than 30 gen proteins survived the "
                 "F-only filter; reporting on the W+F+Y aromatic group "
                 "(H excluded — amphipathic). The per-residue F question "
                 "is **unanswered for lack of data**.\n\n")
    g, r = out["per_set"]["gen"], out["per_set"]["ref"]
    fp.write(f"Survived filter "
             f"(≥{PROTEIN_FILTER['min_total']} target total, "
             f"≥{PROTEIN_FILTER['min_buried']} buried, "
             f"≥{PROTEIN_FILTER['min_exposed']} exposed):\n\n")
    fp.write(f"- gen: **{g['n']}** / {g['total']} proteins\n")
    fp.write(f"- PDB: **{r['n']}** / {r['total']} proteins\n\n")
    underpowered = g["n"] < GEN_FALLBACK_THRESHOLD
    if "median" in g and not np.isnan(g["median"]):
        fp.write("| set | median | IQR | N |\n|---|---|---|---|\n")
        fp.write(f"| gen | {g['median']:.2f} | "
                 f"{g['iqr'][0]:.2f}-{g['iqr'][1]:.2f} | {g['n']} |\n")
        fp.write(f"| PDB | {r['median']:.2f} | "
                 f"{r['iqr'][0]:.2f}-{r['iqr'][1]:.2f} | {r['n']} |\n\n")
        if "ks" in out:
            ks = out["ks"]
            fp.write(f"KS two-sample test: D = {ks['stat']:.3f}, "
                     f"p = {ks['p']:.3g} (small-N caveat: p-values are noisy).\n\n")
        if underpowered:
            fp.write("**Histogram suppressed** — fewer than 30 gen "
                     "proteins survive even the group-level filter; the "
                     "shape of the per-protein distribution cannot be "
                     "reliably read off this N.\n\n")
        else:
            fp.write(f"Plot: `exp3_per_protein_{label}.{{png,pdf}}` "
                     f"(log10 x-axis, gen and PDB medians marked).\n\n")
    fp.write(f"**Verdict:** {out['verdict']}\n\n")


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-file", type=Path, required=True,
                    help="per_residue.parquet (or .csv) from aromatic_burial.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        handlers=[logging.FileHandler(log_path, mode="w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("followups")

    in_path = args.in_file
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)
    log.info(f"loaded {len(df):,} rows from {in_path}  (sets: "
             f"{sorted(df['set'].unique().tolist())})")
    df = ensure_sample_id(df, log)
    n_uniq_gen = df.loc[df['set']=='gen','protein_id'].nunique()
    n_uniq_ref = df.loc[df['set']=='ref','protein_id'].nunique()
    n_inst_gen = df.loc[df['set']=='gen','sample_id'].nunique()
    n_inst_ref = df.loc[df['set']=='ref','sample_id'].nunique()
    log.info(f"  gen: {n_uniq_gen} unique proteins, {n_inst_gen} sampling instances")
    log.info(f"  ref: {n_uniq_ref} unique proteins, {n_inst_ref} sampling instances "
             f"(length-matched ref draws WITH replacement)")

    rng = np.random.default_rng(args.seed)

    e1 = exp1_composition(df, rng, log)
    e2 = exp2_curve_shape(df, rng, log, args.out_dir)
    e3 = exp3_per_protein(df, log, args.out_dir)

    md = args.out_dir / "results.md"
    with open(md, "w") as fp:
        fp.write("# Aromatic-burial follow-ups\n\n")
        fp.write(f"Source: `{in_path}`. Buried RSA<{BURIED_T}, "
                 f"exposed RSA≥{EXPOSED_T}. Bootstrap over proteins, "
                 f"{N_BOOT} resamples.\n\n")
        write_exp1_md(e1, fp)
        write_exp2_md(e2, fp)
        write_exp3_md(e3, fp)
    log.info(f"\nWrote {md}")
    log.info(f"Done. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
