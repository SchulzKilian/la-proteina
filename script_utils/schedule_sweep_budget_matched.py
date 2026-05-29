"""Budget-matched 7-sparse + 7-dense rearrangement sweep.

E095 cut dense layers (not budget-matched). This sweep keeps the same 7+7
split and the same sum K ≈ 280 across sparse layers, but varies WHERE the
dense layers sit (concentrated at end vs split vs bookends vs alternating)
and pairs each with an appropriate K-schedule.

Hypothesis to test: dense-at-end (lower_half_sparse) is the only viable
arrangement, OR there exist alternative 7+7 placements that work equally well.
"""
import os
import sys
from pathlib import Path

REPO = Path("/home/ks2218/la-proteina")
sys.path.insert(0, str(REPO / "script_utils"))

from schedule_sweep_whole_model import run_one  # noqa: E402


# (name, mask, K_per_sparse_layer). All 7 sparse + 7 dense, all sum K ≈ 280.
SCHEDULES = [
    # Dense split 3 early + 4 late: dense at 4-6 (3 layers) and 10-13 (4 layers).
    # Sparse positions: 0,1,2,3,7,8,9.
    # K-schedule: front-heavy on the 0-3 block (close to model entry), front-heavy on the 7-9 block.
    ("budget_dense_split_3_4",
     [True, True, True, True, False, False, False, True, True, True, False, False, False, False],
     [56, 48, 32, 16, 56, 40, 32]),

    # Dense middle block (sparse at edges): sparse at 0-3 and 11-13.
    # K-schedule: early sparse heavy at front (matches model entry); late sparse ramps up
    # to prepare for output (no dense after layer 13, so the sparse block must self-integrate).
    ("budget_dense_middle",
     [True, True, True, True, False, False, False, False, False, False, False, True, True, True],
     [48, 40, 32, 16, 32, 56, 56]),

    # Dense bookends (sparse in middle): dense at 0-3 and 11-13, sparse at 4-10.
    # The first sparse layer is layer 4, fed by 4 dense layers; the last sparse layer is
    # layer 10, followed by 3 dense. Sparse stack of 7 in the middle.
    # K-schedule: front-heavy across the sparse-only block (since "early sparse" = layer 4).
    ("budget_dense_bookends",
     [False, False, False, False, True, True, True, True, True, True, True, False, False, False],
     [56, 56, 56, 40, 32, 24, 16]),

    # Strict alternating: T,F,T,F,T,F,T,F,T,F,T,F,T,F.
    # Sparse at 0, 2, 4, 6, 8, 10, 12. 7 sparse / 7 dense.
    # K-schedule: front-heavy (layer 0 most exposed), gentle taper.
    ("budget_alternating_strict",
     [True, False, True, False, True, False, True, False, True, False, True, False, True, False],
     [56, 48, 40, 32, 32, 40, 32]),

    # Dense split 4 early + 3 late: dense at 3-6 (4 layers) and 11-13 (3 layers).
    # Sparse positions: 0,1,2,7,8,9,10.
    # K-schedule: front-heavy on first block, front-heavy + decreasing on second.
    ("budget_dense_split_4_3",
     [True, True, True, False, False, False, False, True, True, True, True, False, False, False],
     [56, 48, 32, 56, 40, 32, 16]),
]


def main():
    requested = set(sys.argv[1:])
    to_run = SCHEDULES if not requested else [
        s for s in SCHEDULES if s[0] in requested
    ]
    if requested and not to_run:
        sys.exit(f"No match for {sorted(requested)}; "
                 f"choices: {[s[0] for s in SCHEDULES]}")
    print(f"=== schedule_sweep_budget_matched: {len(to_run)} schedules, "
          f"GPU env CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} ===",
          flush=True)
    statuses = []
    for name, mask, k_list in to_run:
        # Quick budget sanity log.
        n_sparse = sum(mask); n_dense = len(mask) - n_sparse
        print(f"  scheduled: {name}  ({n_sparse} sparse, {n_dense} dense, "
              f"sum K={sum(k_list)}, mean per sparse {sum(k_list)/len(k_list):.1f})",
              flush=True)
    print()
    for name, mask, k_list in to_run:
        s = run_one(name, mask, k_list)
        statuses.append((name, s))
    print("\n=== ALL BUDGET-MATCHED SCHEDULES PROCESSED ===")
    for n, s in statuses:
        print(f"  {n}: {s}")


if __name__ == "__main__":
    main()
