# Design Notes

One-line-per-decision log of choices made during implementation.

1. **SE(3) normalization**: PCA-axis rotation with sign-fix (largest |coord| positive per axis). Known to be discontinuous for near-degenerate CA clouds. No cleaner cheap alternative found; Procrustes needs a reference.
2. **MI estimation**: Using sklearn `mutual_info_regression` pairwise (D^2 calls). Slow but correct. Could speed up with k-NN MI estimators (e.g. NPEET) but adds a dependency.
3. **Probe scaling**: StandardScaler applied inside each CV fold (fit on train, transform test). No scaling on y — probes should handle raw property values.
4. **Noised latent sampling**: For t=1.0 (clean), `n_samples` copies are created but all identical. This means clean probes have 1x data, noised probes have 8x. Intentional: averaging over noise realizations is the point.
5. **UMAP before or after probes**: UMAP runs after probes so best R² can be annotated on panels. UMAP is for illustration only, not diagnostic.
6. **Property count**: Spec says "nine total" but lists 8 properties. Using 8 as configured. Config is the source of truth.
7. **Grouped CV for protein-level probes**: Even for mean-pooled protein-level features, groups ensure a protein never leaks across folds when noised samples expand the dataset.
8. **t-convention**: All code uses t=0 noise, t=1 clean (matching La-Proteina codebase). Prominently logged in all outputs. Steering predictor convention is UNVERIFIED — flagged in code and summaries.
9. **Synthetic smoke test**: Properties are deliberately correlated with latent dims 0 and 1 so probes produce non-trivial R² values in smoke tests.
10. **Decision thresholds**: All thresholds are config-driven under `part2.decisions`. No hardcoded values in the logic function.
