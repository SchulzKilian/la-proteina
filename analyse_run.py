import json
import os
from datetime import datetime, timedelta

def analyze_local_run(run_dir):
    # Paths to the local W&B state files
    summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
    metadata_path = os.path.join(run_dir, "files", "wandb-metadata.json")

    # Fallback if structure is different
    if not os.path.exists(summary_path):
        summary_path = os.path.join(run_dir, "wandb-summary.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(run_dir, "wandb-metadata.json")

    if not os.path.exists(summary_path) or not os.path.exists(metadata_path):
        print(f"Error: Could not find JSON files in {run_dir}")
        return

    # Load the data
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 1. Timeline (Start/End/Duration)
    # W&B stores startedAt in ISO format
    start_str = metadata.get("startedAt")
    start_dt = datetime.strptime(start_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    
    runtime_sec = summary.get("_runtime", 0)
    end_dt = start_dt + timedelta(seconds=runtime_sec)

    # 2. Throughput (Samples per second)
    # Pulls nsamples_processed from your Proteina.py logging
    total_samples = summary.get("scaling/nsamples_processed", 0)
    samples_per_sec = total_samples / runtime_sec if runtime_sec > 0 else 0

    # 3. GPU Power
    # Note: Summary only stores the LAST recorded value. 
    # The average is usually computed by the W&B server from binary logs.
    gpu_power_last = summary.get("system.gpu.0.powerUsageWatts")

    # 4. Final Losses
    losses = {k: v for k, v in summary.items() if "loss" in k.lower() and isinstance(v, (int, float))}

    print(f"--- Analysis for: {run_dir} ---")
    print(f"Start Time:     {start_dt}")
    print(f"End Time:       {end_dt}")
    print(f"Active Runtime: {str(timedelta(seconds=int(runtime_sec)))}")
    print(f"\nThroughput:      {samples_per_sec:.2f} samples/sec")
    print(f"Total Samples:   {total_samples:,}")
    
    if gpu_power_last:
        print(f"Last GPU Power:  {gpu_power_last:.2f} W")
    else:
        print("GPU Power:       Not found in local summary (check history).")

    print("\nFinal Loss Metrics:")
    for name, val in losses.items():
        print(f"  - {name}: {val:.6f}")

if __name__ == "__main__":
    # Your provided path
    run_folder = "/home/ks2218/la-proteina/wandb/latest-run"
    analyze_local_run(run_folder)