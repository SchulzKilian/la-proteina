# Aromatic-burial follow-ups

Source: `/home/ks2218/la-proteina/results/aromatic_burial_afdb/per_residue.parquet`. Buried RSA<0.2, exposed RSA≥0.5. Bootstrap over proteins, 1000 resamples.

## Experiment 1 — Compositional breakdown of the aromatic pool

Fraction of aromatic residues that are W/F/Y/H per set (bootstrap over proteins, 1000 resamples, 95% CI):

| set | W | F | Y | H | sum |
|-----|---|---|---|---|-----|
| gen | 0.080 [0.052, 0.111] | 0.279 [0.229, 0.332] | 0.476 [0.412, 0.542] | 0.165 [0.121, 0.215] | 1.000 |
| ref | 0.131 [0.126, 0.136] | 0.372 [0.365, 0.379] | 0.282 [0.275, 0.288] | 0.215 [0.209, 0.222] | 1.000 |

**Counterfactual decomposition of the gen aromatic-group burial ratio** (same per-residue gen ratios, reweighted by PDB aromatic-pool fractions):

| quantity | value |
|---|---|
| per-residue gen burial ratios | W=5.35, F=2.30, Y=5.14, H=1.16 |
| observed gen group ratio (p_gen · r_gen) | 3.710 |
| empirical gen group ratio (raw counts) | 2.967 |
| counterfactual (p_pdb · r_gen) | 3.256 |

**Flag:** NOT COMPOSITIONAL: counterfactual 3.26 vs observed 3.71 within 12.2% (≤20%) → group preservation is not explained by composition reweighting.

## Experiment 2 — Curve shape vs amplitude (F, Y)

KDE `bw_method='scott'` (scipy default; flagged in run.log). Bootstrap over proteins (1000 resamples). Logistic regression fits `P(is X) = sigmoid(a + b·RSA)` separately for gen and PDB on each bootstrap replicate. Per-iteration KDE / logistic inputs are uniformly subsampled (≤8000 / ≤30000 residues) for tractable compute; KDE marginal P(target) uses original counts.

| residue | gen slope b [95% CI] | PDB slope b [95% CI] | CI overlap? |
|---|---|---|---|
| F | -2.29 [-4.13, -0.74] | -1.71 [-2.00, -1.44] | yes |
| Y | -1.43 [-1.98, -0.88] | -1.52 [-1.83, -1.24] | yes |

Plots: `exp2_F_curves.{png,pdf}`, `exp2_Y_curves.{png,pdf}` (raw + area-normalized side-by-side).

**Verdicts:**

- F: gen slope -2.29[-4.13,-0.74] vs PDB slope -1.71[-2.00,-1.44] — CIs OVERLAP → SAME SHAPE; amplitude difference is the likely driver (consistent with supply-limited placement, not broken machinery).
- Y: gen slope -1.43[-1.98,-0.88] vs PDB slope -1.52[-1.83,-1.24] — CIs OVERLAP → SAME SHAPE; amplitude difference is the likely driver (consistent with supply-limited placement, not broken machinery).

## Experiment 3 — Per-protein burial-targeting distribution (W+F+Y)

**FALLBACK:** fewer than 30 gen proteins survived the F-only filter; reporting on the W+F+Y aromatic group (H excluded — amphipathic). The per-residue F question is **unanswered for lack of data**.

Survived filter (≥10 target total, ≥2 buried, ≥2 exposed):

- gen: **18** / 138 proteins
- PDB: **654** / 835 proteins

| set | median | IQR | N |
|---|---|---|---|
| gen | 1.24 | 0.54-2.06 | 18 |
| PDB | 2.62 | 1.65-3.93 | 654 |

KS two-sample test: D = 0.465, p = 0.000545 (small-N caveat: p-values are noisy).

**Histogram suppressed** — fewer than 30 gen proteins survive even the group-level filter; the shape of the per-protein distribution cannot be reliably read off this N.

**Verdict:** underpowered to distinguish (gen N=18, ref N=654; histogram suppressed; medians reported with caveat)

