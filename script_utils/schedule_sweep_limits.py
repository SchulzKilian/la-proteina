"""Limit-testing schedules — push the 'early-heavy / late-can-starve' pattern
discovered by schedule_sweep.py. All sum=280 (matched flat-K=40 budget).

Test concepts:
  1. mirror_decreasing — exact mirror of the failed increasing ramp.
  2. front_plateau_then_taper — sustained early high, gradual late taper.
  3. L0_spike_only — only layer 0 boosted; rest baseline.
  4. late_starve_K8 — push K[6] to extreme low; tests dense layer-7 rescue.
  5. front4_then_crash — first 4 layers plateau high, last 3 crash.

Imports run_one and K_DECOMP from schedule_sweep.py so the YAML / runner
plumbing is shared.
"""
import os
import sys
from pathlib import Path

REPO = Path("/home/ks2218/la-proteina")
sys.path.insert(0, str(REPO / "script_utils"))

from schedule_sweep import run_one  # noqa: E402

LIMITS_SCHEDULES = [
    ("mirror_decreasing",        [64, 56, 48, 40, 32, 24, 16]),  # sum 280
    ("front_plateau_then_taper", [56, 56, 56, 40, 32, 24, 16]),  # sum 280
    ("L0_spike_only",            [64, 32, 32, 40, 40, 40, 32]),  # sum 280
    ("late_starve_K8",           [56, 48, 48, 40, 40, 40,  8]),  # sum 280
    ("front4_then_crash",        [56, 56, 56, 56, 24, 16, 16]),  # sum 280
]


def main():
    requested = set(sys.argv[1:])
    to_run = LIMITS_SCHEDULES if not requested else [
        (n, k) for (n, k) in LIMITS_SCHEDULES if n in requested
    ]
    if requested and not to_run:
        sys.exit(f"No matching schedule for {sorted(requested)}; "
                 f"choices: {[n for n,_ in LIMITS_SCHEDULES]}")
    print(f"=== schedule_sweep_limits: {len(to_run)} schedules, "
          f"GPU env CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} ===",
          flush=True)
    statuses = []
    for name, k_list in to_run:
        s = run_one(name, k_list)
        statuses.append((name, s))
    print("\n=== ALL LIMIT-TESTING SCHEDULES PROCESSED ===")
    for n, s in statuses:
        print(f"  {n}: {s}")


if __name__ == "__main__":
    main()
