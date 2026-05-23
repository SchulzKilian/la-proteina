# Steering audit — unified matrix

Metric in both tables: per-protein gap = (predictor's last-step claim of TANGO) − (real `tango_total` from TANGO binary).
Gap < 0 ⇒ predictor under-claims real TANGO (classical hacking direction).
Gap ≈ 0 ⇒ predictor is honest about real-property change.

## Table 1 — n=4 smoke (seeds {42-45}, L=300, w=16, tango_min)

| Cell | Predictor | Fold | Smoothing | n | pred mean | real mean | **gap mean** | gap std | source |
|---|---|---|---|---|---|---|---|---|---|
| E028 ref | clean | ens5 | σ=0.1, K=4 | 4 | 378.5 | 581.9 | **-203.5** | 49.3 | `results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w16` |
| E029 ref | NA-v1 | fold2 | off | 4 | 540.0 | 587.5 | **-47.5** | 47.9 | `results/noise_aware_smoke/tango_min_w16_fold2` |
| Tier 1.1 — clean f0 | clean | fold0 | off | 4 | 381.0 | 601.9 | **-220.9** | 64.0 | `results/audit_matrix/clean_fold0` |
| Tier 1.1 — clean f2 | clean | fold2 | off | 4 | 215.2 | 602.1 | **-386.9** | 113.0 | `results/audit_matrix/clean_fold2` |
| Tier 1.1 — clean ens5 (no smooth) | clean | ens5 | off | 4 | 390.8 | 594.7 | **-203.9** | 65.0 | `results/audit_matrix/clean_ensemble_nosmoothing` |
| Tier 1.2 — NA-v1 f0 | NA-v1 | fold0 | off | 4 | 540.5 | 601.8 | **-61.3** | 45.8 | `results/audit_matrix/na_v1_fold0` |
| Tier 1.2 — NA-v1 f1 | NA-v1 | fold1 | off | 4 | 526.7 | 588.9 | **-62.2** | 26.6 | `results/audit_matrix/na_v1_fold1` |
| Tier 1.2 — NA-v1 f3 | NA-v1 | fold3 | off | 4 | 537.5 | 596.9 | **-59.4** | 86.1 | `results/audit_matrix/na_v1_fold3` |
| Tier 1.2 — NA-v1 f4 | NA-v1 | fold4 | off | 4 | 501.3 | 598.3 | **-97.0** | 46.5 | `results/audit_matrix/na_v1_fold4` |

## Table 2 — n=48 full sweep (seeds 42-57 × L ∈ {300,400,500}, tango_min)

| Cell | Predictor | Fold | Smoothing | w | n | pred mean | real mean | gap mean | Δpred (vs w=1) | Δreal (vs w=1) | **Δratio** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 1 | 48 | 959.8 | 877.9 | +81.9 | +0.0 | +0.0 | **n/a** |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 2 | 48 | 938.6 | 874.4 | +64.2 | -21.2 | -3.5 | **6.01×** |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 4 | 48 | 897.8 | 868.5 | +29.3 | -62.0 | -9.4 | **6.62×** |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 8 | 48 | 819.8 | 861.5 | -41.7 | -140.0 | -16.4 | **8.54×** |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 16 | 48 | 671.9 | 843.9 | -172.0 | -288.0 | -34.0 | **8.47×** |
| E032 ref full sweep | NA-v1 | ens5 | off | 1 | 48 | 1011.7 | 893.3 | +118.5 | +0.0 | +0.0 | **n/a** |
| E032 ref full sweep | NA-v1 | ens5 | off | 2 | 48 | 999.3 | 889.5 | +109.8 | -12.4 | -3.8 | **3.31×** |
| E032 ref full sweep | NA-v1 | ens5 | off | 4 | 48 | 975.9 | 884.2 | +91.8 | -35.8 | -9.1 | **3.93×** |
| E032 ref full sweep | NA-v1 | ens5 | off | 8 | 48 | 931.2 | 872.3 | +58.8 | -80.6 | -20.9 | **3.85×** |
| E032 ref full sweep | NA-v1 | ens5 | off | 16 | 48 | 837.2 | 833.4 | +3.8 | -174.5 | -59.9 | **2.91×** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 1 | 48 | 1030.3 | 893.3 | +136.9 | +0.0 | +0.0 | **n/a** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 2 | 48 | 1013.0 | 892.7 | +120.3 | -17.3 | -0.6 | **29.08×** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 4 | 48 | 976.3 | 883.8 | +92.5 | -54.0 | -9.5 | **5.67×** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 8 | 48 | 905.4 | 876.7 | +28.8 | -124.8 | -16.7 | **7.49×** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 16 | 48 | 768.4 | 855.9 | -87.5 | -261.8 | -37.4 | **7.00×** |

## Smoothing key
- σ=0.1, K=4 = randomised gradient smoothing (4 N(0, 0.1²) draws averaged).
- off = single deterministic gradient.
- All cells use unit-norm gradient + w-scaling (matched effective step size).
