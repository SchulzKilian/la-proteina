import time
import resource
import contextlib
import json
import os
from typing import Dict

import torch
from loguru import logger

@contextlib.contextmanager
def measure_performance(metrics: Dict, task_name: str = "Task"):
    """
    A context manager to measure the execution time and peak memory usage.
    Yields a dictionary that will be populated with results after the block finishes.
    """
    # This dictionary acts as a container that persists outside the context
    metrics = {}
    
    # Synchronize and start timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.perf_counter()

    # Hand control back to the 'with' block
    yield metrics

    # Block has finished, now measure everything
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_bytes = torch.cuda.max_memory_allocated()
        metrics["peak_gpu_memory_mb"] = peak_gpu_bytes / (1024 * 1024)

    end_time = time.perf_counter()
    # ru_maxrss is in KB on Linux
    peak_cpu_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    metrics["elapsed_time_seconds"] = end_time - start_time
    metrics["peak_cpu_memory_mb"] = peak_cpu_kb / 1024.0

    logger.info(f"Performance for '{task_name}':")
    logger.info(f"  Time taken: {metrics['elapsed_time_seconds']:.4f}s")
    logger.info(f"  Peak CPU: {metrics['peak_cpu_memory_mb']:.2f} MB")
    if "peak_gpu_memory_mb" in metrics:
        logger.info(f"  Peak GPU: {metrics['peak_gpu_memory_mb']:.2f} MB")

def save_performance_metrics(root_path: str, task_name: str, metrics: Dict) -> None:
    """
    Saves performance metrics to a JSON file.
    """
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    metrics_file_path = os.path.join(root_path, "performance_metrics.json")
    all_metrics = {}

    if os.path.exists(metrics_file_path):
        try:
            with open(metrics_file_path, "r") as f:
                all_metrics = json.load(f)
        except json.JSONDecodeError:
            all_metrics = {}

    all_metrics[task_name] = metrics

    with open(metrics_file_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Performance metrics for '{task_name}' saved to {metrics_file_path}")

# --- Example Usage ---
if __name__ == "__main__":
    save_dir = "./results"
    task = "inference_test"
    
    # IMPORTANT: Use 'as m' to capture the dictionary yielded
    with measure_performance(task) as m:
        # Simulate work (e.g., model inference)
        time.sleep(1) 
        if torch.cuda.is_available():
            _ = torch.randn(1000, 1000).cuda()
            
    # Once the block exits, 'm' is fully populated
    save_performance_metrics(save_dir, task, m)