# 1. Add this import at the top
from lightning.pytorch.callbacks import DeviceStatsMonitor 

# 2. Update this function
def initialize_callbacks(cfg_exp):
    callbacks = [SeedCallback()]
    
    # This specifically logs GPU utilization and memory to W&B every step
    callbacks.append(DeviceStatsMonitor(cpu_stats=False)) 
    
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    # ... rest of code