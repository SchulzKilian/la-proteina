"""Stage-1 calibration for the score-magnitude look-ahead throttle.

Runs a BASELINE (s≡1, full guidance at the target λ) generation per
(target, seed) and, along that single trajectory, logs m_base / m_guided / ΔP / r
for ALL proxies — geometric, score_algebraic (A), and score_faithful (B) at each
probe time t_p ∈ {0.85, 0.9, 0.95}. No throttle is applied and no designability
is computed; this stage only characterises the ΔP / r distributions so we can
propose a STARTING β per proxy (the real β lives in Stage 2 against the frontier).

Baseline structures are identical regardless of proxy (s≡1), so one trajectory
suffices to measure every proxy. B costs +2 model forwards per t_p per step
(≈7 forwards/step total with 3 t_p), all on a SINGLE GPU.

    CUDA_VISIBLE_DEVICES=<idle-gpu-uuid> nohup \
      /home/ks2218/.conda/envs/laproteina_env/bin/python \
      -m steering.run_geom_lookahead_calibration --device cuda:0 \
      > results/geom_lookahead_calib/calib.log 2>&1 &
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

from steering.generate import generate_one, load_proteina_config
from steering.guide_geometric import GeometricLookaheadGuide
from proteinfoundation.generate import load_ckpt_n_configure_inference

# Stage-1 grid: target λ = the Stage-2 first-run λ ("just above free-lunch").
TARGET_LAMBDA = {"rg": 8.0, "contact_order": 12.0}
SEEDS = [42, 43]
LENGTH = 400                       # thesis-relevant regime only
CALIB_T_P = [0.85, 0.9, 0.95]
PROTEINA_CONFIG = "inference_ucond_notri_long"
OUT_BASE = _ROOT / "results" / "geom_lookahead_calib"


def build_calib_cfg(target: str, lam: float, device: str) -> dict:
    cfg = {
        "method": "geometric_lookahead",
        "enabled": True,
        "channel": "bb_ca",
        "mode": "baseline",                # s≡1: measure the un-throttled trajectory
        "proxy_type": "geometric",         # cheapest main path; calibration probes all
        "calibration_probe": True,
        "calibration_t_p": CALIB_T_P,
        "objectives": [{"property": target, "direction": "minimize", "weight": 1.0}],
        "schedule": {"type": "linear_ramp", "w_max": float(lam), "t_start": 0.3, "t_end": 0.8},
        "proxy": {"bond_target_nm": 0.38, "clash_radius_nm": 0.40, "lambda_clash": 1.0},
        "gradient_norm": "unit",
        "gradient_clip": 0.0,
        "log_diagnostics": True,
        "device": device,
    }
    if target == "contact_order":
        cfg["contact_order"] = {"threshold_nm": 0.80, "temperature_nm": 0.05, "min_seq_sep": 3}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--targets", type=str, nargs="+", default=list(TARGET_LAMBDA.keys()),
                    choices=list(TARGET_LAMBDA.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--length", type=int, default=LENGTH)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[calib] device={device} name={name}", flush=True)

    print(f"[calib] loading model {PROTEINA_CONFIG} ...", flush=True)
    cfg = load_proteina_config(PROTEINA_CONFIG)
    model = load_ckpt_n_configure_inference(cfg).to(device)
    model.eval()
    print(f"[calib] model loaded. nsteps={model.inf_cfg.args.nsteps}", flush=True)

    total = len(args.targets) * len(args.seeds)
    done = 0
    t0 = time.time()
    for target in args.targets:
        lam = TARGET_LAMBDA[target]
        diag_dir = OUT_BASE / target
        diag_dir.mkdir(parents=True, exist_ok=True)
        steer_cfg = build_calib_cfg(target, lam, str(device))
        with open(diag_dir / "calib_config.yaml", "w") as f:
            yaml.safe_dump({"steering": steer_cfg}, f, default_flow_style=False)

        for seed in args.seeds:
            done += 1
            pid = f"s{seed}_n{args.length}_w{int(lam)}"
            out_json = diag_dir / f"{pid}_diagnostics.json"
            if out_json.exists():
                print(f"[calib][{done}/{total}] skip {target}/{pid} (exists)", flush=True)
                continue

            guide = GeometricLookaheadGuide(steer_cfg)
            model.steering_guide = guide
            print(f"[calib][{done}/{total}] {target}/{pid} (λ={lam}, baseline) generating ...", flush=True)
            _, _, _, extra = generate_one(model, args.length, seed, device)

            diag = extra.get("steering_diagnostics", [])
            with open(out_json, "w") as f:
                json.dump(diag, f, indent=2)

            act = [d for d in diag if not d.get("skipped") and d.get("cal_A_dP") is not None]
            if act:
                import statistics as st
                rA = [d["cal_A_r"] for d in act]
                rB = [d.get("cal_B_tp0.9_r", 0.0) for d in act]
                rG = [d["cal_geom_r"] for d in act]
                print(f"[calib][{done}/{total}] {target}/{pid} steps_active={len(act)} "
                      f"r_med[geom={st.median(rG):.3f} A={st.median(rA):.3f} B@.9={st.median(rB):.3f}]",
                      flush=True)
            el = time.time() - t0
            print(f"[calib]   {el:.0f}s elapsed", flush=True)

    print(f"\n[calib] DONE. {done} structures. Diagnostics under {OUT_BASE}", flush=True)


if __name__ == "__main__":
    main()
