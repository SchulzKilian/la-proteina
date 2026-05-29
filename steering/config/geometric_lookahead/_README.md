# Geometric look-ahead steering configs

Closed-form Cα look-ahead guidance (no learned predictor). Run on the **LD+AE
official model** (`--proteina_config inference_ucond_notri_long`) at
**nsteps=400** (HARD RULE — structure-quality eval).

```bash
python -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/geometric_lookahead/rg_lookahead_prop.yaml \
  --lengths 300 --seeds 42 43 44 45 46 47 \
  --output_dir results/geom_lookahead/rg_prop \
  --device cuda:0
```

## Units
The `bb_ca` channel is in **nanometres** during sampling. All proxy / target
constants here are nm (bond 0.38, clash 0.40, contact 0.80). Rg target values
are nm; contact_order is relative (dimensionless).

## Ablation axes
- `mode`: `baseline` (s≡1) | `lookahead_gated` (hard threshold) | `lookahead_proportional` (saturating s)
- `f_map`: `exp` | `reciprocal` | `logistic`; sharpness `beta`
- `schedule.w_max` = λ0 (unthrottled guidance weight)
- `t_floor`: optional, disable throttle below this t

## Reading the diagnostics
`results/.../diagnostics/*.json` logs per step: `p_base`, `p_guided`,
`delta_p`, `s`, `lambda0`, `lambda_eff`. Sanity checks:
- `s` stuck near 1.0 every step → throttle too soft (raise `beta` / lower `gate_threshold`).
- `s` collapsing to ~0 early → too sharp (lower `beta` / raise `gate_threshold`).
- `delta_p <= 0` ⇒ guidance did not worsen geometry ⇒ `s == 1` (expected).
