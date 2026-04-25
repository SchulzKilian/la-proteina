"""Pull native-resolution training/val traces for the chained CA-only training
runs and look for restart-aligned features.

The training is chained via SLURM requeue; each ~4h slot is a separate wandb
run. We collect all CA-only runs, stitch by trainer/global_step, plot val and
train loss with restart boundaries annotated.

Native resolution via run.scan_history (every logged event, no downsampling).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# Wandb run IDs from slurm_ca_1gpu_*.out, in chronological order.
CHAIN = [
    ("txlwbhao", 28126469),
    ("oz20mwk3", 28149253),
    ("emsldaeq", 28197198),
    ("d1k1587u", 28231457),
    ("ncosp2v7", 28260296),
    ("jeponiu5", 28262493),
]

ENTITY = "kilianschulz-university-of-cambridge"
PROJECT = "laproteina"

OUT = Path("/home/ks2218/la-proteina/analysis_cheap_diagnostics")
OUT.mkdir(exist_ok=True)


def scan_metric(run, metric, step_col="trainer/global_step"):
    """Pull every logged value of `metric` paired with its global_step."""
    rows = []
    for r in run.scan_history(keys=[metric, step_col, "_runtime"]):
        if r.get(metric) is None or r.get(step_col) is None:
            continue
        rows.append({
            "global_step": int(r[step_col]),
            metric: float(r[metric]),
            "_runtime": float(r.get("_runtime", float("nan"))),
        })
    return pd.DataFrame(rows)


def main():
    api = wandb.Api()

    chain_meta = []
    val_frames = []
    train_frames = []
    epoch_dur_frames = []

    for run_id, slurm_id in CHAIN:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        except Exception as e:
            print(f"[skip] {run_id}: {e}")
            continue
        print(f"\n{run_id} (slurm {slurm_id})  state={run.state}  created={run.created_at}")
        print(f"  _runtime={run.summary.get('_runtime')}s  "
              f"max_global_step={run.summary.get('trainer/global_step')}")

        chain_meta.append({
            "wandb_id": run_id, "slurm_id": slurm_id,
            "name": run.name, "state": run.state, "created_at": str(run.created_at),
            "duration_s": run.summary.get("_runtime"),
            "max_global_step": run.summary.get("trainer/global_step"),
            "final_val_loss_epoch": run.summary.get("validation_loss/loss_epoch"),
            "final_train_loss_epoch": run.summary.get("train/loss_epoch"),
        })

        # Pull val loss (epoch-level — should be the headline number user looks at)
        v = scan_metric(run, "validation_loss/loss_epoch")
        if not v.empty:
            v["wandb_id"] = run_id
            val_frames.append(v)
            print(f"  val_loss_epoch: n={len(v)}, "
                  f"step range [{int(v.global_step.min())}, {int(v.global_step.max())}], "
                  f"val range [{v['validation_loss/loss_epoch'].min():.4g}, "
                  f"{v['validation_loss/loss_epoch'].max():.4g}]")

        # Pull train loss (step-level)
        t = scan_metric(run, "train/loss_step")
        if not t.empty:
            t["wandb_id"] = run_id
            train_frames.append(t)
            print(f"  train_loss_step: n={len(t)}")

        # Epoch duration (helps locate restart boundaries)
        ed = scan_metric(run, "train_info/epoch_duration_secs")
        if not ed.empty:
            ed["wandb_id"] = run_id
            epoch_dur_frames.append(ed)

    # Save raw
    pd.DataFrame(chain_meta).to_csv(OUT / "wandb_chain_meta.csv", index=False)

    val_df = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    epoch_dur_df = pd.concat(epoch_dur_frames, ignore_index=True) if epoch_dur_frames else pd.DataFrame()

    val_df.to_csv(OUT / "wandb_val_loss_chained.csv", index=False)
    train_df.to_csv(OUT / "wandb_train_loss_chained.csv", index=False)

    print("\n" + "=" * 70)
    print("VAL LOSS over global_step (epoch-level), full chain:")
    print("=" * 70)
    if val_df.empty:
        print("(no val data)")
    else:
        ordered = val_df.sort_values("global_step").reset_index(drop=True)
        print(ordered[["wandb_id", "global_step", "validation_loss/loss_epoch", "_runtime"]].to_string(index=False))

    # Per-run train loss summary at start, mid, end
    print("\n" + "=" * 70)
    print("TRAIN LOSS (step-level) per run:")
    print("=" * 70)
    if not train_df.empty:
        for rid, _ in CHAIN:
            sub = train_df[train_df["wandb_id"] == rid].sort_values("global_step")
            if sub.empty:
                continue
            steps = sub["global_step"].values
            loss = sub["train/loss_step"].values
            n = len(sub)
            mid = n // 2
            print(f"  {rid}: n={n}  "
                  f"step start={steps[0]}, end={steps[-1]}  "
                  f"loss start={loss[0]:.4g}, mid={loss[mid]:.4g}, end={loss[-1]:.4g}  "
                  f"rolling_std(last_50)={pd.Series(loss[-50:]).std():.4g}")

    # Look for restart-alignment in val loss
    print("\n" + "=" * 70)
    print("RESTART-ALIGNMENT CHECK")
    print("=" * 70)
    print("For each run boundary, list the val_loss values just BEFORE and just AFTER:")
    if not val_df.empty:
        ordered = val_df.sort_values("global_step").reset_index(drop=True)
        # Run boundaries = first global_step of each new run, sorted by appearance order
        boundary_steps = []
        seen = set()
        for _, row in ordered.iterrows():
            if row["wandb_id"] not in seen:
                seen.add(row["wandb_id"])
                boundary_steps.append((row["wandb_id"], row["global_step"]))
        print("Run boundaries (wandb_id, first global_step seen in val_loss):")
        for rid, gs in boundary_steps:
            print(f"  {rid}: first val_step seen at global_step={gs}")

        # Compute pre/post deltas at each restart
        print("\nVal loss across each restart (last 3 of prev run vs first 3 of new run):")
        for i in range(1, len(boundary_steps)):
            prev_rid = boundary_steps[i-1][0]
            cur_rid = boundary_steps[i][0]
            prev = ordered[ordered["wandb_id"] == prev_rid].tail(3)
            cur = ordered[ordered["wandb_id"] == cur_rid].head(3)
            print(f"\n  {prev_rid} -> {cur_rid}")
            print("  PREV last 3:")
            print("    " + prev[["global_step", "validation_loss/loss_epoch"]].to_string(index=False).replace("\n", "\n    "))
            print("  NEW first 3:")
            print("    " + cur[["global_step", "validation_loss/loss_epoch"]].to_string(index=False).replace("\n", "\n    "))

    # Best val checkpoint location: which run had the lowest val loss?
    if not val_df.empty:
        idx = val_df["validation_loss/loss_epoch"].idxmin()
        best = val_df.loc[idx]
        print("\n" + "=" * 70)
        print(f"BEST VAL: {best['validation_loss/loss_epoch']:.4g}  in run {best['wandb_id']}  "
              f"at global_step={int(best['global_step'])}")
        print("=" * 70)


if __name__ == "__main__":
    main()
