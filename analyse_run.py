import json
import os
from datetime import datetime, timedelta

def analyze_local_run(run_dir):
    # Paths to the local W&B state files
    summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
    metadata_path = os.path.join(run_dir, "files", "wandb-metadata.json")

    # Fallback if the folder structure is slightly different
    if not os.path.exists(summary_path):
        summary_path = os.path.join(run_dir, "wandb-summary.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(run_dir, "wandb-metadata.json")

    if not os.path.exists(summary_path):
        print(f"Error: Could not find wandb-summary.json in {run_dir}")
        return

    # Load the summary data
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # 1. Timeline (Start/End/Duration)
    # Most reliable method: End Time - Runtime = Start Time
    runtime_sec = summary.get("_runtime", 0)
    end_timestamp = summary.get("_timestamp")

    if end_timestamp and runtime_sec:
        end_dt = datetime.fromtimestamp(end_timestamp)
        start_dt = end_dt - timedelta(seconds=runtime_sec)
    else:
        # Last resort fallback to file modification time
        end_timestamp = os.path.getmtime(summary_path)
        end_dt = datetime.fromtimestamp(end_timestamp)
        start_dt = end_dt - timedelta(seconds=runtime_sec)

    # 2. Throughput (Samples per second)
    # Pulls nsamples_processed from your Proteina.py logging
    total_samples = summary.get("scaling/nsamples_processed", 0)
    samples_per_sec = total_samples / runtime_sec if runtime_sec > 0 else 0

    # 3. GPU Power
    # Summary stores the LAST recorded value from system metrics
    gpu_power_last = summary.get("system.gpu.0.powerUsageWatts")

    # 4. Final Losses
    # Filter for anything with 'loss' in the name that is a number
    losses = {k: v for k, v in summary.items() if "loss" in k.lower() and isinstance(v, (int, float))}

    print(f"--- Analysis for: {run_dir} ---")
    print(f"Start Time:     {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:       {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Active Runtime: {str(timedelta(seconds=int(runtime_sec)))}")
    print(f"\nThroughput:      {samples_per_sec:.2f} samples/sec")
    print(f"Total Samples:   {total_samples:,}")
    
    if gpu_power_last:
        print(f"Last GPU Power:  {gpu_power_last:.2f} W")
    else:
        print("GPU Power:       Not found in local summary.")

    print("\nFinal Loss Metrics:")
    # Sort losses to make them easier to read
    for name in sorted(losses.keys()):
        print(f"  - {name}: {losses[name]:.6f}")

if __name__ == "__main__":
    # Your specific latest-run path
    run_folder = "/home/ks2218/la-proteina/wandb/latest-run"
    analyze_local_run(run_folder)