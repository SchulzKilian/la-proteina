# Prompt for the other Claude Code installation

Paste this in a fresh `claude code` session at the repo root on the machine
that holds the noise-aware predictor checkpoints and the E028 / E032 result
trees (the cluster / device where the original Finding 10 work was done).

---

## Context — why this audit exists

Finding 10 in `content_masterarbeit.md` claims that closing the
predictor-vs-real gradient-hacking gap in latent-flow steering needs **two**
compositional fixes:

1. **Noise-aware predictor training** — fine-tune the property predictor on
   `z_t = (1-t)·ε + t·z_1 + σ_L·√(t(1-t))·ε_2` over the steering window, so
   it sees the same input distribution as the sampling-time gradient call.
2. **5-fold ensemble averaging** — average predictions across folds so
   fold-specific adversarial shortcuts cancel.

Either fix alone leaves the gap at order ~10–20× of real-property change;
combined, the gap closes to within regression noise.

A subsequent audit (the one that produced this script) identified **five
sanity-check questions** that a reviewer would reasonably raise about that
claim. Three were already addressed in `experiments.md`; two were not:

- **Fold confounding**: every "single noise-aware fold" cell uses fold 2,
  picked because it had the highest `val_r²_noisy`. Folds 0, 1, 3, 4 were
  never individually probed → "is fold 2 specially smart?" is open.
- **Smoothing confounding**: the clean+ensemble baseline (E028) used Gaussian
  smoothing σ=0.1, K=4. Every other cell has smoothing off. So E028 vs E032
  conflates "ensemble alone" with "ensemble + smoothing".
- **Reporting inconsistency**: smokes report `per-protein gap = pred − real`;
  full sweeps report `Δratio = Δpred / Δreal`. Two metrics, no unified table
  across all four corners of the predictor × ensemble matrix.

Plus three structural gaps in the 2×2 matrix:

- Clean + single fold has **never been measured in the smoke gap format**
  used by E028→E032 (E025 measured it differently).
- Noise-aware single fold has only been measured at **n=4** (E029 / E031);
  no n=48 head-to-head against E028 / E032's full sweeps.
- A clean + ensemble + **no smoothing** cell has never been run, so the
  smoothing contribution can't be isolated from the ensemble contribution.

This script fills all of that in one resumable pipeline so Finding 10 can be
defended against the audit without hand-waving.

## What to run

The script is at **`script_utils/audit_steering_matrix.py`** (already on
disk). It has four stages, each idempotent:

```bash
python script_utils/audit_steering_matrix.py write-configs
python script_utils/audit_steering_matrix.py generate --device cuda:0
PYTHON_EXEC=$(which python) python script_utils/audit_steering_matrix.py eval
python script_utils/audit_steering_matrix.py report --markdown audit_report.md
```

Or all in one: `python script_utils/audit_steering_matrix.py all --device cuda:0`.

Stage breakdown:

- **`write-configs`** emits 13 YAMLs under `steering/config/audit_matrix/`
  (8 n=4 smoke cells + 5 n=48 sweep cells for the NA-v1 fold 2 sweep).
- **`generate`** runs `python -m steering.generate` with `nsteps=400` for
  each cell. Skips cells whose expected PDBs already exist. Wall-clock
  rough estimate: ~15 min for the 8 smokes + ~2-3 h for the 240-PDB
  fold-2 sweep on a single A100.
- **`eval`** runs `evaluate_directory` (real TANGO via the local TANGO
  binary) on each `guided/` dir. Skips cells with `properties_guided.csv`
  already present. ~5-10 min.
- **`report`** aggregates the new cells alongside the existing E028
  (clean + ensemble + smoothing) and E032 (NA-v1 + ensemble) reference
  cells, builds two markdown tables (n=4 smoke matrix, n=48 sweep
  matrix), and writes them to `audit_report.md` if `--markdown` is set.

### Prerequisites

- **Predictor checkpoints**: the script defaults to
  - clean: `laproteina_steerability/logs/multitask_t1/20260427_161809/checkpoints/fold_{0..4}_best.pt`
  - noise-aware-v1: `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_{0..4}_best.pt`

  If the paths on this machine are different, override via
  `--pred-clean PATH` and `--pred-na PATH` on every stage that takes them.

- **Reference cells** (read-only, for the report's anchor rows):
  - `results/steering_camsol_tango_L500_ensemble_smoothed/tango_min_w*/`
    (E028 clean+ensemble+smoothing, used at both n=4 and n=48 sweep)
  - `results/noise_aware_smoke/tango_min_w16_fold2/` (E029 single-fold
    NA-v1 reference)
  - `results/noise_aware_ensemble_sweep/tango_min_w*/` (E032 NA-v1 +
    ensemble full sweep)

  If they're at different paths, edit the constants `REF_E028`, `REF_E029`,
  `REF_E032_SWEEP` near the top of the script.

- **TANGO binary** on PATH or via `TANGO_EXE=...`.
- **`PYTHON_EXEC`** env var pointing at the laproteina env when running
  `eval` or `all` (so the TANGO subprocess inherits the right Python — see
  `feedback_export_python_exec.md`).
- **nsteps=400** is hardwired — do not lower it under any circumstance.

### What the script will NOT do

- No designability / scRMSD eval (would be ~10 h compute, audit doesn't ask).
- No `camsol_max` direction (no local real-CamSol available; tango_min only).
- No UG K=5 + noise-aware ensemble cell (separate open question from E030
  caveat — flag this if results suggest it would change the picture).

## What to do with the results

**Step 1 — Append a new entry to `experiments.md`** (next free `Eddd`).
Required content per `CLAUDE.md`:

- ID + date (today, absolute date).
- Status: finished.
- **Why ran**: the 5 sanity-check questions and 3 matrix gaps listed in the
  context section above. This entry's job is to defend Finding 10 against
  audit-style review without hand-waving.
- **Configs**: point at the `audit_matrix/` config tree, list the 13 cells,
  cite the script (`script_utils/audit_steering_matrix.py`), the predictor
  checkpoint paths used, the seed grids (smokes: {42-45}, sweep: 42-57),
  and `nsteps=400`.
- **Results**: copy in both tables from `audit_report.md` verbatim. Add a
  short prose block per table that calls out the headline numbers:
  - Did fold 2 turn out to be unusual, or do all 5 NA folds look similar?
    (If they're similar, the "ensemble cancels fold-specific shortcuts"
    mechanism story is weaker than F10 currently claims; if they're spread
    out, the story is stronger.)
  - Does clean+ensemble-no-smoothing close the gap noticeably more than
    clean alone, or is most of E028's gap reduction actually from smoothing?
  - Does the n=48 NA single-fold gap stay close to E029's −47 pilot, or
    does it crossover-flip the way E032 did (n=4 pilot −1.6 → n=48 +3.8)?
- **Possible narrative**: if the new numbers reinforce F10, propose
  promoting key sentences from this entry into F10's writeup. If they
  weaken any sub-claim of F10, write that explicitly and propose the F10
  edits.
- **Methodological caveats**: still tango_min only (no CamSol verification);
  predictor checkpoints unchanged from E029/E032 (no re-training); n=4
  smokes still have wide CIs; sweep is single-fold (fold 2) only — folds
  0, 1, 3, 4 are not at n=48.

**Step 2 — Update Finding 10 in `content_masterarbeit.md`** if the audit
results require it. The likely edits:

- Add a paragraph after the "Two fixes layered" section that explicitly
  addresses the audit. Reference the new `experiments.md` entry by ID.
- Update the "Methodische Einschränkungen" section to reflect what the
  audit confirmed vs. what remains open.
- If the results show that one of the two "fixes" is doing most of the
  work and the other is marginal, soften F10's "compositional, both needed"
  framing accordingly. Honest reporting > defending the original story.

**Step 3 — If results suggest the UG K=5 + NA-v1 ensemble cell would
matter** (e.g. if F10's gap closure looks tighter under audit), flag that
as the obvious follow-up. Do not run it without the user confirming —
it's a separate ~30 min experiment, not part of the Tier 1 scope.

## Critical reminders from `CLAUDE.md`

- `nsteps=400` is non-negotiable for any structure-evaluated run.
- Auto-update `experiments.md` the same turn results land — do not batch
  the write-up.
- Memory persistence: do not write any new memory files for this audit;
  the existing `feedback_steering_must_use_official_ld_ae.md` and
  `feedback_export_python_exec.md` already cover the operational gotchas.
- Do not commit anything until the user explicitly asks.
