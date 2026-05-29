"""Geometric look-ahead steering sweep driver.

Loads the LD+AE model ONCE and loops over all arms:
    target ∈ {rg, contact_order}   (direction: minimize)
    mode   ∈ {baseline, lookahead_proportional}
    λ0     ∈ {4, 8, 16, 32}        (schedule.w_max)
    seed   ∈ 42..47                (paired against on-disk unguided)
    length ∈ {300, 400}

Unguided is NOT regenerated — it lives at
`results/sanity_unsteered_seed42_45/unguided/` (same model, nsteps=400,
seeds 42-57). Guided runs share the same seed ⇒ same starting noise ⇒ the only
difference is the deterministic guidance, so pairing on (seed, length) isolates
the steering effect.

Per-structure resumable: skips any (arm, seed, length) whose guided .pt already
exists. Run from repo root:

    CUDA_VISIBLE_DEVICES=<idle-gpu-uuid> nohup \
      /home/ks2218/.conda/envs/laproteina_env/bin/python \
      -m steering.run_geom_lookahead_sweep --device cuda:0 \
      > results/geom_lookahead_sweep/sweep.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from steering.generate import generate_one, save_protein, load_proteina_config
from steering.guide_geometric import GeometricLookaheadGuide
from proteinfoundation.generate import load_ckpt_n_configure_inference

# --- Sweep grid -----------------------------------------------------------
TARGETS = ["rg", "contact_order"]
MODES = {"baseline": "baseline", "prop": "lookahead_proportional"}
LAMBDAS = [4, 8, 16, 32]
SEEDS = [42, 43, 44, 45, 46, 47]
LENGTHS = [300, 400]
PROTEINA_CONFIG = "inference_ucond_notri_long"
OUT_BASE = _ROOT / "results" / "geom_lookahead_sweep"
UNGUIDED_DIR = _ROOT / "results" / "sanity_unsteered_seed42_45" / "unguided"


def build_steering_cfg(target: str, mode_full: str, lam: float) -> dict:
    cfg = {
        "method": "geometric_lookahead",
        "enabled": True,
        "channel": "bb_ca",
        "mode": mode_full,
        "objectives": [{"property": target, "direction": "minimize", "weight": 1.0}],
        "schedule": {"type": "linear_ramp", "w_max": float(lam), "t_start": 0.3, "t_end": 0.8},
        "proxy": {"bond_target_nm": 0.38, "clash_radius_nm": 0.40, "lambda_clash": 1.0},
        "gradient_norm": "unit",
        "gradient_clip": 0.0,
        "log_diagnostics": True,
    }
    if mode_full == "lookahead_proportional":
        cfg["f_map"] = "exp"
        cfg["beta"] = 20.0
    if target == "contact_order":
        cfg["contact_order"] = {"threshold_nm": 0.80, "temperature_nm": 0.05, "min_seq_sep": 3}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS,
                    help="schedule.w_max values to sweep (default: the full grid)")
    ap.add_argument("--targets", type=str, nargs="+", default=TARGETS,
                    choices=["rg", "contact_order"])
    ap.add_argument("--modes", type=str, nargs="+", default=list(MODES.keys()),
                    choices=list(MODES.keys()), help="baseline and/or prop")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={device} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    if not UNGUIDED_DIR.exists():
        print(f"[sweep] WARNING: unguided baseline dir missing: {UNGUIDED_DIR}", flush=True)

    # Load model ONCE.
    print(f"[sweep] loading model {PROTEINA_CONFIG} ...", flush=True)
    cfg = load_proteina_config(PROTEINA_CONFIG)
    model = load_ckpt_n_configure_inference(cfg).to(device)
    model.eval()
    print(f"[sweep] model loaded. nsteps={model.inf_cfg.args.nsteps}", flush=True)

    sel_modes = {ms: MODES[ms] for ms in args.modes}
    arms = [(t, ms, mf, lam) for t in args.targets for ms, mf in sel_modes.items() for lam in args.lambdas]
    total = len(arms) * len(LENGTHS) * len(SEEDS)
    print(f"[sweep] {len(arms)} arms x {len(LENGTHS)} L x {len(SEEDS)} seeds = {total} guided gens", flush=True)

    done = 0
    skipped = 0
    t0 = time.time()
    for target, mode_short, mode_full, lam in arms:
        arm = f"{target}_{mode_short}_w{int(lam)}"
        arm_dir = OUT_BASE / arm
        guided_dir = arm_dir / "guided"
        diag_dir = arm_dir / "diagnostics"
        guided_dir.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(parents=True, exist_ok=True)

        steer_cfg = build_steering_cfg(target, mode_full, lam)
        steer_cfg["device"] = str(device)
        with open(arm_dir / "steering_config.yaml", "w") as f:
            yaml.safe_dump({"steering": steer_cfg}, f, default_flow_style=False)
        guide = GeometricLookaheadGuide(steer_cfg)
        print(f"\n[sweep] === ARM {arm} (mode={mode_full}, w_max={lam}) ===", flush=True)

        for length in LENGTHS:
            for seed in SEEDS:
                done += 1
                pid = f"s{seed}_n{length}"
                pt_path = guided_dir / f"{pid}.pt"
                if pt_path.exists():
                    skipped += 1
                    print(f"[sweep][{done}/{total}] skip {arm}/{pid} (exists)", flush=True)
                    continue

                model.steering_guide = guide
                coors, res_type, mask, extra = generate_one(model, length, seed, device)
                save_protein(coors, res_type, mask, pid, guided_dir)

                diag = extra.get("steering_diagnostics", [])
                with open(diag_dir / f"{pid}_diagnostics.json", "w") as f:
                    json.dump([
                        {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict)
                             else (v if isinstance(v, (int, float, bool, str)) else str(v)))
                         for k, v in d.items()}
                        for d in diag
                    ], f, indent=2)

                # quick controller summary for the log
                act = [d for d in diag if not d.get("skipped")]
                if act:
                    import statistics as st
                    s_vals = [d["s"] for d in act]
                    rg_first = act[0].get("predicted_properties", {}).get(target)
                    rg_last = act[-1].get("predicted_properties", {}).get(target)
                    print(f"[sweep][{done}/{total}] {arm}/{pid} "
                          f"s[min={min(s_vals):.3f} mean={st.mean(s_vals):.3f}] "
                          f"{target}:{rg_first:.3f}->{rg_last:.3f}", flush=True)
                else:
                    print(f"[sweep][{done}/{total}] {arm}/{pid} (no active steps)", flush=True)

                el = time.time() - t0
                rate = (done - skipped) / el if el > 0 else 0
                print(f"[sweep]   {el:.0f}s elapsed, ~{(total-done)/rate if rate>0 else 0:.0f}s remaining", flush=True)

    print(f"\n[sweep] DONE. {done} processed ({skipped} skipped). Output: {OUT_BASE}", flush=True)


if __name__ == "__main__":
    main()
