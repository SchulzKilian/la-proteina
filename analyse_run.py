import wandb
import datetime
import pandas as pd

# Replace with your actual entity (username) and project name from train.py
ENTITY = "kilianschulz" 
PROJECT = "test_release_diffusion" # From cfg_exp.log.wandb_project

def analyze_latest_run():
    api = wandb.Api()
    
    # Get the latest run from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    if not runs:
        print("No runs found.")
        return
    
    run = runs[0] # The most recent run
    print(f"Analyzing Run: {run.name} ({run.id})")
    print("-" * 30)

    # 1. Start and End Times
    start_dt = datetime.datetime.fromtimestamp(run.summary.get("_timestamp") - run.summary.get("_runtime"))
    end_dt = datetime.datetime.fromtimestamp(run.summary.get("_timestamp"))
    print(f"Timeline: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Duration: {str(datetime.timedelta(seconds=int(run.summary.get('_runtime'))))}")

    # 2. Average GPU Power Usage
    # Note: Power usage is in system metrics, which we average over the history
    system_metrics = run.history(stream="system")
    power_keys = [k for k in system_metrics.columns if "gpu.0.powerUsageWatts" in k]
    if power_keys:
        avg_power = system_metrics[power_keys[0]].mean()
        print(f"Avg GPU 0 Power Usage: {avg_power:.2f} Watts")
    else:
        print("GPU Power Usage not found (Check if system metrics were logged).")

    # 3. Final Losses
    # We pull from the 'summary' which holds the last logged values
    losses = {k: v for k, v in run.summary.items() if "loss" in k and "epoch" not in k}
    print("\nFinal Loss Metrics:")
    for k, v in losses.items():
        print(f"  - {k}: {v:.4f}")

    # 4. Throughput (Samples/Sec)
    # Using your logged 'nsamples_processed' and total active runtime
    total_samples = run.summary.get("scaling/nsamples_processed", 0)
    runtime = run.summary.get("_runtime", 0)
    
    if runtime > 0:
        samples_per_sec = total_samples / runtime
        print(f"\nThroughput: {samples_per_sec:.2f} samples/sec")
    
    # Step-based throughput (using your step_duration_secs)
    avg_step_time = run.summary.get("train_info/step_duration_secs")
    if avg_step_time:
        # Effective batch size = samples / global_steps
        eff_batch = total_samples / run.summary.get("trainer/global_step", 1)
        print(f"Instantaneous Throughput: {eff_batch / avg_step_time:.2f} samples/sec (based on last step)")

if __name__ == "__main__":
    analyze_latest_run()