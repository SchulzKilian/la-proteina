# Layer-selective sparse-vs-dense inference diagnostic

**Inference-only stress test on dense-trained weights.** Replaces attention computation in subsets of the 14 transformer layers with the existing sparse path (`PairBiasAttention._attn_sparse`) while loading the dense baseline checkpoint `best_val_00000026_000000002646.ckpt` (E019; 87 % / 87 % / 53 % at L=50 / 100 / 200, N=30).

**Provenance of seed**: `seed=5` is inferred from cousin entries (E054, E049, etc.); not directly attested inline in E019.

## 16-cell table

| L | K | layer_split | designable (N=30) | Δ vs all_dense |
|---|---|---|---|---|
| 50 | 64 | all_dense | 24/30 (80.0%) | -3.3 pp |
| 50 | 64 | lower_half_sparse | 26/30 (86.7%) | +3.3 pp |
| 50 | 64 | upper_half_sparse | 27/30 (90.0%) | +6.7 pp |
| 50 | 64 | all_sparse | 27/30 (90.0%) | +6.7 pp |
| 50 | 128 | all_dense | 25/30 (83.3%) | +0.0 pp |
| 50 | 128 | lower_half_sparse | 28/30 (93.3%) | +10.0 pp |
| 50 | 128 | upper_half_sparse | 23/30 (76.7%) | -6.7 pp |
| 50 | 128 | all_sparse | 23/30 (76.7%) | -6.7 pp |
| 100 | 64 | all_dense | 26/30 (86.7%) | +3.3 pp |
| 100 | 64 | lower_half_sparse | 24/30 (80.0%) | -3.3 pp |
| 100 | 64 | upper_half_sparse | 5/30 (16.7%) | -66.7 pp |
| 100 | 64 | all_sparse | 3/30 (10.0%) | -73.3 pp |
| 100 | 128 | all_dense | 25/30 (83.3%) | +0.0 pp |
| 100 | 128 | lower_half_sparse | 24/30 (80.0%) | -3.3 pp |
| 100 | 128 | upper_half_sparse | 26/30 (86.7%) | +3.3 pp |
| 100 | 128 | all_sparse | 27/30 (90.0%) | +6.7 pp |

Note: `all_dense` is K-independent (no sparse layers consume the neighbor list); the K=64 and K=128 rows reuse the same run, identical numbers.

## Interpretation

**1. Control reproduces dense baseline?** L=50: 25/30 vs baseline 26/30 (87 %). L=100: 25/30 vs baseline 26/30 (87 %). Bit-identity unit test (`script_utils/test_layer_selective_sanity.py`) confirmed `layer_sparse_mask=[False]*14` is exactly equal to the global dense path (max-abs-diff 0.00e+00, fp32).

**2. Lower vs upper at K=128, L=50.** Lower 28/30 (93.3%) vs upper 23/30 (76.7%) — Δ = +16.7 pp. Reading: **lower-half-sparse is closer to all-dense than upper-half**.

**3. K monotonicity flags.** Expected K=128 ≥ K=64. Violations (potential bug or noise):
  - upper_half_sparse L=50: K=64 90.0% > K=128 76.7% (Δ -13.3 pp)
  - all_sparse L=50: K=64 90.0% > K=128 76.7% (Δ -13.3 pp)

**4. Upper bound on layer-hybrid viability.** This test uses dense-trained weights with partially-sparsified inference. Dense-trained upper layers expect complete pair representations from lower layers; they were never exposed to sparsified intermediate representations during training. The degradation observed here is therefore an UPPER BOUND on what a properly trained layer-hybrid would show — a from-scratch hybrid training run could perform better because the upper layers would adapt. A non-catastrophic result here is NECESSARY-BUT-NOT-SUFFICIENT evidence for the hybrid being worth training; a catastrophic result kills the simple form of the hypothesis.

## Pipeline notes

- Wiring change: `LocalLatentsTransformer.layer_sparse_mask` (default `None` → bit-identical to pre-existing forward). When set to a list of 14 bools, layer `i` uses sparse attention iff `mask[i]` is True. Mutex with `use_downsampling`, `n_global_tokens > 0`, `router_sparse_K`, `update_pair_repr=True`.
- Inference hook: `cfg.generation.args.layer_sparse_mask` + `force_sparse_attention_on` in `generate.py:load_ckpt_n_configure_inference`.
- Bit-identity sanity check passed: `[True]*14` ≡ existing global sparse path; `[False]*14` ≡ existing global dense path (max-abs-diff 0.00e+00, fp32, B=2, N=64).