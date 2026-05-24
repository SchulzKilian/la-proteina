"""
Hydra config-load smoke for the stochastic-K training config.

Composes `training_ca_only_sparse_K64_stochastic`, instantiates a Proteina
LightningModule from it, and asserts:
  (a) The NN exposes stochastic_k_enabled=True and the configured p_full.
  (b) Switching the module to train/eval flips _last_sampled_full correctly
      across a synthetic forward batch.

Runs in <10s with no data files — only config + NN init + a 2x30 forward.
"""

import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteina import Proteina


def main() -> int:
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="training_ca_only_sparse_K64_stochastic")

    sk = cfg.training.get("stochastic_k", None)
    print(f"  training.stochastic_k = {sk}")
    assert sk is not None and sk.enabled, "stochastic_k missing or disabled in resolved config"
    assert 0.0 <= sk.p_full <= 1.0

    model = Proteina(cfg_exp=cfg, store_dir="/tmp/_stochk_smoke")
    nn = model.nn
    print(f"  nn.stochastic_k_enabled = {nn.stochastic_k_enabled}")
    print(f"  nn.stochastic_k_p_full  = {nn.stochastic_k_p_full}")
    print(f"  nn.curriculum_neighbors = {nn.curriculum_neighbors}")
    assert nn.stochastic_k_enabled is True
    assert abs(nn.stochastic_k_p_full - float(sk.p_full)) < 1e-9
    assert nn.curriculum_neighbors is False

    # Synthetic forward to verify train vs eval gating end-to-end.
    B, N = 2, 30
    batch = {
        "x_t": {"bb_ca": torch.randn(B, N, 3) * 0.5},
        "t": {"bb_ca": torch.full((B,), 0.7)},
        "mask": torch.ones(B, N, dtype=torch.bool),
        "strict_feats": False,
    }
    nn.train()
    full_seen = False
    canon_seen = False
    canonical_k = (
        2 * nn.n_seq_neighbors + nn.n_spatial_neighbors + nn.n_random_neighbors
    )
    for i in range(200):
        torch.manual_seed(3000 + i)
        with torch.no_grad():
            _ = nn(batch)
        if nn._last_sampled_full:
            full_seen = True
        else:
            canon_seen = True
            assert nn._last_sampled_k == canonical_k
        if full_seen and canon_seen:
            break
    assert full_seen and canon_seen, "did not observe both K=full and K=canonical in 200 train forwards"
    print("  train mode: observed both K=full and K=canonical")

    nn.eval()
    for i in range(20):
        torch.manual_seed(4000 + i)
        with torch.no_grad():
            _ = nn(batch)
        assert nn._last_sampled_k == canonical_k
        assert not nn._last_sampled_full
    print(f"  eval  mode: 20 forwards, K stayed at canonical {canonical_k}")

    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
