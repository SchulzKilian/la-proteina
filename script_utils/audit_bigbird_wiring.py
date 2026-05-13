"""Read-only audit of BigBird + pair-update wiring in the lowtsoft variant.

Loads the trained checkpoint at step 1200 and runs:
  A1. Attention mass to globals per layer × per t-bucket.
  A2. Variance of globals' K across the 4 global tokens (collapse check).
  A3. Cosine similarity between global token embeddings (collapse check).
  B1. One forward+backward on a fresh-init model, logging grad norms for
      global params vs a representative trunk parameter.
  B2. Same as B1 but with `global_cond_emb` overridden to be the time embedding
      (broadcast of cond[:, 0, :]) instead of the zero-init learnable parameter.

Run from repo root:
    /home/ks2218/conda_envs/laproteina_env/bin/python script_utils/audit_bigbird_wiring.py

CPU-only (no GPU on login node). Wall time ~1-2 min.
"""
import json
import os
import sys
import torch

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention

CKPT = "/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft/1778520282/checkpoints/last.ckpt"
EXP_CFG = "/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft/1778520282/checkpoints/exp_config_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft.json"


# ---------------------------------------------------------------------------- #
# 0. Build kwargs / model
# ---------------------------------------------------------------------------- #
def build_kwargs() -> dict:
    with open(EXP_CFG) as f:
        cfg = json.load(f)
    nn_kwargs = dict(cfg["nn"])
    nn_kwargs.pop("name", None)
    # Training-side knobs plumbed by proteina.py at runtime
    tr = cfg.get("training", {})
    nn_kwargs["curriculum_neighbors"] = bool(tr.get("curriculum_neighbors", False))
    cls = tr.get("curriculum_low_t_split", [32, 0, 0])
    nn_kwargs["curriculum_low_t_split"] = tuple(cls) if cls is not None else (32, 0, 0)
    nn_kwargs["sc_neighbors"] = bool(tr.get("sc_neighbors") or False)
    nn_kwargs["latent_dim"] = None  # CA-only
    return nn_kwargs


def load_trained_state(model: LocalLatentsTransformer) -> tuple:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Strip torch.compile wrapper prefix `nn._orig_mod.` and the `nn.` Lightning prefix
    clean = {}
    for k, v in sd.items():
        if k.startswith("nn._orig_mod."):
            clean[k[len("nn._orig_mod."):]] = v
        elif k.startswith("nn."):
            clean[k[len("nn."):]] = v
    info = model.load_state_dict(clean, strict=False)
    return info, ckpt.get("global_step", "?")


def build_batch(B: int = 3, N: int = 100, t_values=(0.1, 0.5, 0.9)) -> dict:
    assert B == len(t_values)
    mask = torch.ones(B, N, dtype=torch.bool)
    x_t_ca = torch.randn(B, N, 3) * 0.3
    x_sc_ca = torch.randn(B, N, 3) * 0.3
    t = torch.tensor(t_values, dtype=torch.float32)
    return {
        "mask": mask,
        "x_t": {"bb_ca": x_t_ca},
        "x_sc": {"bb_ca": x_sc_ca},
        "t": {"bb_ca": t},
        "residue_type": torch.zeros(B, N, dtype=torch.long),
        "ca_coors_nm": x_t_ca.clone(),
        "ca_coors_nm_mask": mask.clone(),
        "res_type_mask": mask.clone(),
    }


# ---------------------------------------------------------------------------- #
# 1. Hook PairBiasAttention._attn_sparse to record per-layer attention mass on globals
# ---------------------------------------------------------------------------- #
def make_hooked_attn_sparse(K_canonical: int, G: int, records: dict):
    """Returns a replacement for PairBiasAttention._attn_sparse that:
      - reproduces the original computation
      - additionally records, for each call, the attention mass that residue
        queries (indices < N_residues) place on the G global slots.
    Records keyed by id(self).
    """
    from einops import rearrange  # noqa
    from torch import einsum  # noqa
    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa
    K_total = K_canonical + G

    def _attn_sparse(self, q, k, v, b, neighbor_idx, seq_mask, slot_valid=None):
        B, H, NplusG, D = q.shape
        K = neighbor_idx.shape[-1]
        BH = B * H

        k_bh = k.reshape(BH, NplusG, D)
        v_bh = v.reshape(BH, NplusG, D)
        idx_bh = neighbor_idx.unsqueeze(1).expand(B, H, NplusG, K).reshape(BH, NplusG, K)
        idx_flat = idx_bh.reshape(BH, NplusG * K)
        idx_flat_d = idx_flat.unsqueeze(-1).expand(BH, NplusG * K, D)
        k_sparse = k_bh.gather(1, idx_flat_d).reshape(BH, NplusG, K, D)
        v_sparse = v_bh.gather(1, idx_flat_d).reshape(BH, NplusG, K, D)

        q_bh = q.reshape(BH, NplusG, D)
        sim = torch.einsum("bnd,bnkd->bnk", q_bh, k_sparse) * self.scale
        sim = sim.reshape(B, H, NplusG, K)

        if seq_mask is not None:
            B_idx = torch.arange(B, device=seq_mask.device).view(B, 1, 1).expand(B, NplusG, K)
            nbr_valid = seq_mask[B_idx, neighbor_idx]
            i_valid = seq_mask[:, :, None].expand(B, NplusG, K)
            attn_mask = nbr_valid & i_valid
            if slot_valid is not None:
                attn_mask = attn_mask & slot_valid
            sim = sim.masked_fill(~attn_mask.unsqueeze(1), max_neg_value(sim))

        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)  # [B, H, NplusG, K]
        attn_bh = attn.reshape(BH, NplusG, K)
        out = torch.einsum("bnk,bnkd->bnd", attn_bh, v_sparse)

        # Record: residue queries only (first N_residues rows). K_total = K_canonical + G.
        # Last G slots of the K axis are the global slots.
        if K == K_total:
            N_residues = NplusG - G
            # Per-batch per-residue, sum attn weight over the G global slots, then average over heads
            attn_to_globals = attn[:, :, :N_residues, K_canonical:].sum(dim=-1)  # [B, H, N_res]
            mean_global_mass_per_batch = attn_to_globals.mean(dim=(1, 2))         # [B]
            # Also: max global mass per layer (per batch) — does any residue heavily attend?
            max_global_mass_per_batch = attn_to_globals.amax(dim=(1, 2))          # [B]
            # k variance across the 4 globals (uses global rows' own k → seq positions [N, N+G))
            global_keys = k[:, :, N_residues:NplusG, :]  # [B, H, G, D]
            # Std across G tokens, averaged over (B, H, D)
            std_across_globals = global_keys.std(dim=2).mean().item()
            mean_norm_globals = global_keys.norm(dim=-1).mean().item()
            layer_idx = getattr(self, "_layer_idx", -1)
            records.setdefault(layer_idx, []).append({
                "mean_global_mass_per_batch": mean_global_mass_per_batch.detach().cpu().tolist(),
                "max_global_mass_per_batch": max_global_mass_per_batch.detach().cpu().tolist(),
                "std_across_globals": std_across_globals,
                "mean_norm_globals": mean_norm_globals,
            })

        return out.reshape(B, H, NplusG, D)

    return _attn_sparse


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    print("=" * 80)
    print("BigBird wiring audit — reads", CKPT)
    print("=" * 80)

    kw = build_kwargs()
    print(f"\n[setup] kwargs of interest:")
    for k in ["nlayers", "token_dim", "nheads", "n_global_tokens", "sparse_attention",
              "n_seq_neighbors", "n_spatial_neighbors", "n_random_neighbors",
              "update_pair_repr", "update_pair_repr_every_n",
              "curriculum_neighbors", "curriculum_low_t_split"]:
        print(f"  {k}: {kw.get(k)}")

    K_canonical = 2 * kw["n_seq_neighbors"] + kw["n_spatial_neighbors"] + kw["n_random_neighbors"]
    G = kw["n_global_tokens"]
    print(f"  K_canonical={K_canonical}, G={G}, K_total={K_canonical+G}")

    # --------- A3 (cheap pre-check, no model run) — cos sim between global tokens
    print("\n" + "-" * 80)
    print("[A3] cosine similarity between trained global_token_emb rows")
    print("-" * 80)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    gtok = sd.get("nn._orig_mod.global_token_emb")
    if gtok is None:
        gtok = sd.get("nn.global_token_emb")
    assert gtok is not None, "global_token_emb missing from ckpt"
    print(f"  global_token_emb shape: {tuple(gtok.shape)}")
    cs = torch.nn.functional.cosine_similarity(
        gtok.unsqueeze(1).float(), gtok.unsqueeze(0).float(), dim=-1
    )
    print(f"  pairwise cosine similarity (G×G):")
    for row in cs.tolist():
        print("    " + "  ".join(f"{x:+.4f}" for x in row))

    # --------- Build trained model
    print("\n" + "-" * 80)
    print("[setup] building model with TRAINED weights")
    print("-" * 80)
    model = LocalLatentsTransformer(**kw)
    info, step = load_trained_state(model)
    print(f"  loaded global_step={step}")
    print(f"  missing keys ({len(info.missing_keys)}): {info.missing_keys[:3]}")
    print(f"  unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:3]}")
    model.eval()

    # --------- A1 + A2 — attention mass and key-collapse
    print("\n" + "-" * 80)
    print("[A1+A2] forward at t=(0.1, 0.5, 0.9); per-layer attention mass on globals")
    print("-" * 80)
    records: dict = {}
    PairBiasAttention._attn_sparse = make_hooked_attn_sparse(K_canonical, G, records)
    # Tag layers
    for i, m in enumerate(model.transformer_layers):
        m.mhba.mha._layer_idx = i  # MultiheadAttnAndTransition.mhba → MultiHeadBiasedAttentionADALN_MM.mha → PairBiasAttention

    batch = build_batch(B=3, N=100, t_values=(0.1, 0.5, 0.9))
    with torch.no_grad():
        out = model(batch)
    print(f"  forward OK; output shape={tuple(out['bb_ca']['v'].shape)}")
    print(f"  layers recorded: {sorted(records.keys())}")

    print(f"\n  per-layer attention mass on globals (mean over residue queries, all heads):")
    print(f"  {'layer':>5s} | {'t=0.1':>10s} | {'t=0.5':>10s} | {'t=0.9':>10s} | "
          f"{'max(t=0.1)':>10s} | {'std_globK':>10s}")
    baseline = G / (K_canonical + G)
    print(f"  baseline (uniform attention over K_total): {baseline:.4f}")
    for li in sorted(records.keys()):
        for entry in records[li]:
            m = entry["mean_global_mass_per_batch"]
            mx = entry["max_global_mass_per_batch"][0]
            sg = entry["std_across_globals"]
            print(f"  {li:>5d} | {m[0]:>10.4f} | {m[1]:>10.4f} | {m[2]:>10.4f} | "
                  f"{mx:>10.4f} | {sg:>10.4f}")

    # --------- B1 — grad-norm comparison on fresh-init model
    print("\n" + "-" * 80)
    print("[B1] fresh-init forward+backward; grad norms")
    print("-" * 80)
    # Restore original _attn_sparse for the fresh-init runs (don't want measurement overhead)
    from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention as _PBA
    # Re-import to get the original method bound to the class — easiest: reload module
    import importlib
    pba_mod = importlib.import_module("proteinfoundation.nn.modules.pair_bias_attn")
    importlib.reload(pba_mod)
    # Re-import LocalLatentsTransformer too so it uses the reloaded class
    llt_mod = importlib.import_module("proteinfoundation.nn.local_latents_transformer")
    importlib.reload(llt_mod)
    LLT_fresh = llt_mod.LocalLatentsTransformer

    torch.manual_seed(123)
    fresh = LLT_fresh(**kw)
    fresh.train()
    batch_b = build_batch(B=2, N=100, t_values=(0.2, 0.8))
    out_fresh = fresh(batch_b)
    target = torch.randn_like(out_fresh["bb_ca"]["v"])
    loss = ((out_fresh["bb_ca"]["v"] - target) * batch_b["mask"][..., None]).pow(2).mean()
    loss.backward()
    print(f"  loss={loss.item():.4f}")

    def gnorm(name: str) -> tuple:
        p = dict(fresh.named_parameters()).get(name)
        if p is None or p.grad is None:
            return ("MISSING", float("nan"), float("nan"))
        return ("OK", p.grad.norm().item(), p.norm().item())

    rows = [
        # globals
        "global_token_emb",
        "global_cond_emb",
        "global_pair_bias_res_to_glob",
        "global_pair_bias_glob_to_res",
        "global_pair_bias_glob_to_glob",
        # representative trunk
        "transformer_layers.0.mhba.mha.to_qkv.weight",
        "transformer_layers.0.mhba.mha.to_bias.weight",
        "transformer_layers.0.mhba.scale_output.to_adaln_zero_gamma.0.weight",
        "pair_update_layers.0.linear_x.weight",
        "ca_linear.1.weight",
    ]
    print(f"  {'param':<60s} | {'status':<7s} | {'|grad|':>12s} | {'|param|':>12s} | {'ratio':>10s}")
    # Compute a reference grad for normalisation
    trunk_g = dict(fresh.named_parameters()).get("transformer_layers.0.mhba.mha.to_qkv.weight").grad.norm().item()
    for name in rows:
        st, g, p = gnorm(name)
        if st == "OK":
            ratio = g / max(trunk_g, 1e-20)
            print(f"  {name:<60s} | {st:<7s} | {g:>12.4e} | {p:>12.4e} | {ratio:>10.4f}")
        else:
            print(f"  {name:<60s} | {st:<7s} | {'-':>12s} | {'-':>12s} | {'-':>10s}")
    print(f"  (ratio = |grad|(this) / |grad|(transformer_layers.0.mhba.mha.to_qkv.weight))")

    # --------- B2 — fresh-init forward+backward with global_cond = broadcast of time-emb
    print("\n" + "-" * 80)
    print("[B2] fresh-init with global_cond_emb overridden = cond[:, 0, :] (time-emb)")
    print("-" * 80)
    torch.manual_seed(123)
    fresh2 = LLT_fresh(**kw)
    fresh2.train()
    # Monkey-patch _attach_globals on this instance to use cond[:, 0:1, :] as the global cond
    original_attach = fresh2._attach_globals.__func__

    def patched_attach(self, seqs, cond, mask, neighbor_idx, slot_valid, pair_rep):
        B, N, K_canonical = neighbor_idx.shape
        G = self.n_global_tokens
        device = seqs.device
        # Use time-emb broadcast (assumes cond is broadcast-constant across residues,
        # which is true for feats_cond_seq=[time_emb_bb_ca]).
        global_seqs = self.global_token_emb.unsqueeze(0).expand(B, G, self.token_dim)
        global_cond = cond[:, :1, :].expand(B, G, cond.shape[-1])  # <-- THE CHANGE
        global_mask = torch.ones(B, G, dtype=mask.dtype, device=device)
        seqs_ext = torch.cat([seqs, global_seqs], dim=1)
        cond_ext = torch.cat([cond, global_cond], dim=1)
        mask_ext = torch.cat([mask, global_mask], dim=1)

        global_idx_row = torch.arange(N, N + G, device=device, dtype=neighbor_idx.dtype)
        global_idx_for_res = global_idx_row.view(1, 1, G).expand(B, N, G)
        res_rows = torch.cat([neighbor_idx, global_idx_for_res], dim=-1)
        lens = mask.sum(dim=-1).to(torch.long)
        lens_clamped = lens.clamp(min=1)
        lin = torch.linspace(0.0, 1.0, steps=K_canonical, device=device)
        glob_res_idx = (lin.view(1, K_canonical) * (lens_clamped - 1).view(B, 1).float())
        glob_res_idx = glob_res_idx.round().to(torch.long)
        glob_res_idx_expanded = glob_res_idx.unsqueeze(1).expand(B, G, K_canonical)
        glob_glob_idx = global_idx_row.view(1, 1, G).expand(B, G, G)
        glob_rows = torch.cat([glob_res_idx_expanded, glob_glob_idx], dim=-1)
        neighbor_idx_ext = torch.cat([res_rows, glob_rows], dim=1)

        glob_slot_for_res = torch.ones(B, N, G, dtype=torch.bool, device=device)
        res_slot_valid = torch.cat([slot_valid, glob_slot_for_res], dim=-1)
        valid_res_pos = glob_res_idx_expanded < lens.view(B, 1, 1)
        first_occ = torch.ones_like(valid_res_pos)
        first_occ[..., 1:] = glob_res_idx_expanded[..., 1:] != glob_res_idx_expanded[..., :-1]
        valid_res_pos = valid_res_pos & first_occ
        glob_glob_valid = torch.ones(B, G, G, dtype=torch.bool, device=device)
        glob_slot_valid = torch.cat([valid_res_pos, glob_glob_valid], dim=-1)
        slot_valid_ext = torch.cat([res_slot_valid, glob_slot_valid], dim=1)

        res_to_glob = self.global_pair_bias_res_to_glob.view(1, 1, G, self.pair_repr_dim).expand(B, N, G, self.pair_repr_dim)
        res_pair_rep = torch.cat([pair_rep, res_to_glob], dim=2)
        glob_to_res = self.global_pair_bias_glob_to_res.view(1, G, 1, self.pair_repr_dim).expand(B, G, K_canonical, self.pair_repr_dim)
        glob_to_glob = self.global_pair_bias_glob_to_glob.view(1, G, G, self.pair_repr_dim).expand(B, G, G, self.pair_repr_dim)
        glob_pair_rep = torch.cat([glob_to_res, glob_to_glob], dim=2)
        pair_rep_ext = torch.cat([res_pair_rep, glob_pair_rep], dim=1)
        return seqs_ext, cond_ext, mask_ext, neighbor_idx_ext, slot_valid_ext, pair_rep_ext

    import types
    fresh2._attach_globals = types.MethodType(patched_attach, fresh2)

    out_fresh2 = fresh2(batch_b)
    loss2 = ((out_fresh2["bb_ca"]["v"] - target) * batch_b["mask"][..., None]).pow(2).mean()
    loss2.backward()
    print(f"  loss={loss2.item():.4f}")

    def gnorm2(name: str) -> tuple:
        p = dict(fresh2.named_parameters()).get(name)
        if p is None or p.grad is None:
            return ("MISSING", float("nan"), float("nan"))
        return ("OK", p.grad.norm().item(), p.norm().item())

    trunk_g2 = dict(fresh2.named_parameters()).get("transformer_layers.0.mhba.mha.to_qkv.weight").grad.norm().item()
    print(f"  {'param':<60s} | {'|grad|(B2)':>12s} | {'ratio(B2)':>10s} | {'ratio(B1)':>10s} | {'speedup':>10s}")
    for name in rows[:5]:  # only globals
        st1, g1, _ = gnorm(name)
        st2, g2, _ = gnorm2(name)
        if st1 == "OK" and st2 == "OK":
            r1 = g1 / max(trunk_g, 1e-20)
            r2 = g2 / max(trunk_g2, 1e-20)
            sp = r2 / max(r1, 1e-20)
            print(f"  {name:<60s} | {g2:>12.4e} | {r2:>10.4f} | {r1:>10.4f} | {sp:>10.2f}x")

    # --------- summary
    print("\n" + "=" * 80)
    print("[summary]")
    print(f"  - A3 cos-sim diagonal=1, off-diagonal indicates global-emb collapse if > ~0.9.")
    print(f"  - A1 baseline (uniform attention) = G/K_total = {G/(K_canonical+G):.4f}; "
          f"if every layer reports ~ this and never grows with depth, globals are inert.")
    print(f"  - A2 std_across_globals near 0 means the 4 globals have collapsed to ~1 token.")
    print(f"  - B1 ratio « 1 means globals get structurally tiny gradient under wd=0.05 LR=2e-4.")
    print(f"  - B2 speedup » 1 means the time-agnostic global_cond_emb is the gradient bottleneck.")
    print("=" * 80)


if __name__ == "__main__":
    main()
