"""Quick diagnostic: is v2's input-Jacobian norm larger than v1's?

If yes, that confirms the sharp-minimum story for why v2's predictor is more
hackable despite higher val r². For each predictor, sample z_t from the same
distribution the steering hook actually feeds in (forward interpolant +
σ_langevin Brownian-bridge term over t∈[0.3, 0.8]), compute
    ‖∂P_tango / ∂z_t‖_2
per-protein, report mean/median/p95.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laproteina_steerability"))
sys.path.insert(0, str(ROOT))

from src.multitask_predictor.model import PropertyTransformer
from src.multitask_predictor.dataset import PROPERTY_NAMES
from steering.registry import PROPERTY_TO_INDEX

V1_CKPT = ROOT / "laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_2_best.pt"
V2_CKPT = ROOT / "laproteina_steerability/logs/multitask_t1_noise_aware/20260505_123607/checkpoints/fold_2_best.pt"
LATENT_DIR = ROOT / "data/pdb_train/processed_latents_300_800"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PROTEINS = 50
T_MIN, T_MAX = 0.3, 0.8
SIGMA_L = 0.1
TANGO_IDX = PROPERTY_TO_INDEX["tango"]


def load_model(ckpt_path: Path) -> PropertyTransformer:
    src = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PropertyTransformer(
        latent_dim=8, d_model=128, n_heads=4, n_layers=3,
        ffn_expansion=4, dropout=0.1, n_properties=len(PROPERTY_NAMES), max_len=1024,
    ).to(DEVICE)
    model.load_state_dict(src["model_state_dict"])
    model.eval()
    return model


def sample_zt(L: int, t: float, z_1: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    eps_1 = torch.randn(L, 8, generator=gen, device=DEVICE)
    eps_2 = torch.randn(L, 8, generator=gen, device=DEVICE)
    bridge = (t * (1.0 - t)) ** 0.5
    return (1.0 - t) * eps_1 + t * z_1 + SIGMA_L * bridge * eps_2


def jacobian_norm(model: PropertyTransformer, z_t: torch.Tensor, t_val: float) -> float:
    """Per-protein ||∂P_tango / ∂z_t||_2, returned as a Python float."""
    z = z_t.unsqueeze(0).clone().requires_grad_(True)            # [1, L, 8]
    mask = torch.ones(1, z.size(1), dtype=torch.bool, device=DEVICE)
    t = torch.tensor([t_val], device=DEVICE)
    pred = model(z, mask, t)                                       # [1, P]
    tango = pred[0, TANGO_IDX]
    g, = torch.autograd.grad(tango, z, create_graph=False)
    return float(g.detach().norm(2).cpu().item())


def main():
    # Pick N_PROTEINS random .pt files from the latent dir
    files = sorted(LATENT_DIR.rglob("*.pt"))
    if len(files) < N_PROTEINS:
        raise RuntimeError(f"Only {len(files)} .pt files found in {LATENT_DIR}")
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(files), size=N_PROTEINS, replace=False)

    m_v1 = load_model(V1_CKPT)
    m_v2 = load_model(V2_CKPT)
    print(f"v1 ckpt: {V1_CKPT.name}  (epoch={torch.load(V1_CKPT, map_location='cpu', weights_only=False).get('epoch','?')})")
    print(f"v2 ckpt: {V2_CKPT.name}  (epoch={torch.load(V2_CKPT, map_location='cpu', weights_only=False).get('epoch','?')})")
    print(f"Sampling {N_PROTEINS} proteins, t ~ U({T_MIN},{T_MAX}), σ_langevin={SIGMA_L}")

    # Use the same t and same z_t for both models — paired comparison.
    gen = torch.Generator(device=DEVICE).manual_seed(123)
    norms_v1, norms_v2, lens = [], [], []
    for idx in chosen:
        d = torch.load(files[idx], map_location=DEVICE, weights_only=False)
        z_1 = d["mean"].to(DEVICE).float()       # [L, 8]
        L = z_1.size(0)
        t = float(torch.empty(1, device="cpu").uniform_(T_MIN, T_MAX, generator=torch.Generator().manual_seed(int(idx))))
        z_t = sample_zt(L, t, z_1, gen)
        norms_v1.append(jacobian_norm(m_v1, z_t, t))
        norms_v2.append(jacobian_norm(m_v2, z_t, t))
        lens.append(L)

    a1, a2 = np.array(norms_v1), np.array(norms_v2)
    print()
    print(f"||∂P_tango/∂z_t||_2  (n={len(a1)} proteins)")
    print(f"            v1                 v2                 ratio v2/v1")
    print(f"  mean    {a1.mean():.3f}            {a2.mean():.3f}            {a2.mean()/a1.mean():.2f}×")
    print(f"  median  {np.median(a1):.3f}            {np.median(a2):.3f}            {np.median(a2)/np.median(a1):.2f}×")
    print(f"  p95     {np.percentile(a1,95):.3f}            {np.percentile(a2,95):.3f}            {np.percentile(a2,95)/np.percentile(a1,95):.2f}×")
    print(f"  max     {a1.max():.3f}            {a2.max():.3f}            {a2.max()/a1.max():.2f}×")
    print()
    n_v2_larger = int((a2 > a1).sum())
    print(f"v2 has larger ‖J‖ on {n_v2_larger}/{len(a1)} proteins ({n_v2_larger/len(a1)*100:.0f}%)")
    print(f"per-protein ratio v2/v1:  median {np.median(a2/a1):.2f}×, mean {np.mean(a2/a1):.2f}×")


if __name__ == "__main__":
    main()
