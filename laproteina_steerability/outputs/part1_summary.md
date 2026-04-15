# Part 1: Latent Geometry Summary

## Dataset
- Proteins: 56008
- Total residues: 22657969
- Latent dimensions: 8

## Key results

| Metric | Value |
|--------|-------|
| Participation ratio | 7.694 / 8 |
| Effective rank | 90%: 7D, 95%: 8D, 99%: 8D |
| Max off-diagonal Pearson | 0.102 |
| Collapsed dims (var < 1% of max) | 0 |

## Interpretation
- Participation ratio close to 8 means all dimensions contribute.
- No collapsed dimensions = no posterior collapse.
- High off-diagonal correlation (>0.5) suggests redundancy.

## Figures
- `figures/latent_marginals.{png,pdf}` — per-dim histograms + KDE
- `figures/latent_correlations.{png,pdf}` — Pearson + Spearman heatmaps
- `figures/latent_mutual_information.{png,pdf}` — MI heatmap
- `figures/pca_analysis.{png,pdf}` — scree + cumulative variance
- `figures/dim_utilization.{png,pdf}` — within vs between variance
- `figures/length_sensitivity.{png,pdf}` — latent norm vs length

## Tables
- `tables/marginal_stats.csv`
- `tables/pearson_correlation.csv`, `tables/spearman_correlation.csv`
- `tables/mutual_information.csv`
- `tables/pca_results.csv`
- `tables/dim_utilization.csv`
- `tables/length_stats.csv`
