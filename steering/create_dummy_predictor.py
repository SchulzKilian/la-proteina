"""Create a dummy predictor checkpoint for testing the steering pipeline.

This creates a PropertyTransformer with random weights and fake z-score stats.
The gradient direction tests still work (random weights have consistent gradients),
but predictions are meaningless.

Usage:
    python -m steering.create_dummy_predictor --output steering/checkpoints/dummy_fold_0_best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "laproteina_steerability"))

from src.multitask_predictor.model import PropertyTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="steering/checkpoints/dummy_fold_0_best.pt")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = PropertyTransformer(
        latent_dim=8, d_model=128, n_heads=4, n_layers=3,
        ffn_expansion=4, dropout=0.1, n_properties=13, max_len=1024,
    )

    # Reasonable z-score stats (won't match real data, but won't cause NaN/overflow)
    stats_mean = np.zeros(13, dtype=np.float32)
    stats_std = np.ones(13, dtype=np.float32)

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "val_r2_mean": 0.0,
        "val_results": {},
        "stats_mean": stats_mean,
        "stats_std": stats_std,
    }, str(out_path))

    print(f"Dummy predictor checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
