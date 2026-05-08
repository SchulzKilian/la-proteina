"""
Sparse neighbor-list computation for SALAD-style sparse attention.

Builds a [b, n, K] index tensor where K = 2*n_seq + n_spatial + n_random.
The three groups are concatenated in that order so callers can slice them apart
if needed, but normally just use the full K neighbors.
"""
import torch


@torch.no_grad()
def build_neighbor_idx(
    ca_coors: torch.Tensor,   # [b, n, 3]  Cα positions (nm or Å — units don't matter)
    mask: torch.Tensor,       # [b, n]  bool, True = valid residue
    n_seq: int = 8,           # sequential neighbors on each side  → 2*n_seq total
    n_spatial: int = 8,       # nearest Cα neighbors (not already selected)
    n_random: int = 16,       # random neighbors ∝ 1/d³ (not already selected)
) -> tuple:
    """
    Returns (neighbor_idx, slot_valid), both [b, n, K], K = 2*n_seq + n_spatial + n_random.

    neighbor_idx: int64 tensor of neighbor residue indices.
    slot_valid:   bool tensor; True = real neighbor, False = padding slot.
                  Padding only occurs when the protein is shorter than K (~48 residues).
                  Without this guard, padding slots would point to residue 0 (which has
                  seq_mask=True) and be incorrectly treated as real neighbors.

    Selection order:
      1. Sequential: n_seq nearest by |i-j| on each side of i.
      2. Spatial   : n_spatial nearest by Cα distance, excluding (1).
      3. Random    : n_random sampled without replacement ∝ 1/d_Cα³, excluding (1)+(2).
    """
    B, N, _ = ca_coors.shape
    device = ca_coors.device
    INF = 1e6  # sentinel; larger than any realistic inter-residue distance in nm

    # ------------------------------------------------------------------ #
    # Pairwise Cα distances  [b, n, n]
    # ------------------------------------------------------------------ #
    dists = torch.cdist(ca_coors.float(), ca_coors.float())  # [b, n, n]

    # Base invalid mask: only masked residues are never valid neighbors. Self
    # is now allowed in the K-set (lands in slot 0 of the sequential group via
    # seq_dist[i, i] = 0). Pair-bias bins for |i - j| = 0 and d = 0 receive
    # gradients during training.
    invalid_j = ~mask[:, None, :].expand(B, N, N)          # [b, n, n]
    base_invalid = invalid_j                                # [b, n, n]

    # `selected` tracks positions already chosen (or intrinsically invalid).
    # Used to exclude them from subsequent selection rounds.
    selected = base_invalid.clone()                         # [b, n, n]

    # ------------------------------------------------------------------ #
    # 1. Sequential neighbors — top-k by |i - j|
    # ------------------------------------------------------------------ #
    idx = torch.arange(N, device=device, dtype=torch.float32)
    seq_dist = (idx.view(N, 1) - idx.view(1, N)).abs()     # [n, n]
    seq_dist = seq_dist.unsqueeze(0).expand(B, N, N)
    seq_dist_masked = seq_dist.masked_fill(base_invalid, INF)

    k_seq = min(2 * n_seq, N - 1)
    _, seq_nbrs = seq_dist_masked.topk(k_seq, dim=-1, largest=False)   # [b, n, k_seq]
    selected.scatter_(2, seq_nbrs, True)

    # ------------------------------------------------------------------ #
    # 2. Spatial neighbors — nearest Cα not already selected
    # ------------------------------------------------------------------ #
    dists_sp = dists.masked_fill(selected, INF)
    k_sp = min(n_spatial, max(N - 1 - k_seq, 0))
    if k_sp > 0:
        _, sp_nbrs = dists_sp.topk(k_sp, dim=-1, largest=False)        # [b, n, k_sp]
        selected.scatter_(2, sp_nbrs, True)
    else:
        sp_nbrs = seq_nbrs.new_zeros(B, N, 0)

    # ------------------------------------------------------------------ #
    # 3. Random neighbors — Gumbel-max trick for sampling ∝ 1/d³
    # ------------------------------------------------------------------ #
    dists_rnd = dists.masked_fill(selected, INF)
    valid_rnd = dists_rnd < (INF / 2)
    weights = torch.where(
        valid_rnd,
        dists_rnd.clamp(min=1e-4).pow(-3),
        dists_rnd.new_zeros(()),
    )
    log_w = weights.log().masked_fill(~valid_rnd, -float("inf"))
    gumbel = -torch.log(-torch.log(torch.rand_like(log_w).clamp(min=1e-20)))
    k_rnd = min(n_random, max(N - 1 - k_seq - k_sp, 0))
    if k_rnd > 0:
        _, rnd_nbrs = (log_w + gumbel).topk(k_rnd, dim=-1, largest=True)   # [b, n, k_rnd]
    else:
        rnd_nbrs = seq_nbrs.new_zeros(B, N, 0)

    # ------------------------------------------------------------------ #
    # Pad each group to its requested size, then concatenate
    # ------------------------------------------------------------------ #
    def _pad(t: torch.Tensor, target: int) -> torch.Tensor:
        if t.shape[-1] == target:
            return t
        if t.shape[-1] > target:
            return t[..., :target]
        pad = t.new_zeros(*t.shape[:-1], target - t.shape[-1])
        return torch.cat([t, pad], dim=-1)

    # Track actual counts before padding so we can mark padding slots invalid.
    # _pad only adds zeros when N is very small (< 48); all padded slots point
    # to residue 0 which has seq_mask=True, so without the slot_valid guard
    # they would be incorrectly treated as real neighbors.
    k_seq_actual = seq_nbrs.shape[-1]
    k_sp_actual  = sp_nbrs.shape[-1]
    k_rnd_actual = rnd_nbrs.shape[-1]

    seq_nbrs = _pad(seq_nbrs, 2 * n_seq)     # [b, n, 2*n_seq]
    sp_nbrs  = _pad(sp_nbrs,  n_spatial)      # [b, n, n_spatial]
    rnd_nbrs = _pad(rnd_nbrs, n_random)       # [b, n, n_random]

    neighbor_idx = torch.cat([seq_nbrs, sp_nbrs, rnd_nbrs], dim=-1)  # [b, n, K]

    # Build slot validity mask: True = real neighbor, False = padding slot.
    # The counts are scalar (depend only on N, not on individual residues),
    # so the mask is identical for every (b, i) and we broadcast via expand.
    K_total = 2 * n_seq + n_spatial + n_random
    slot_valid = torch.ones(B, N, K_total, dtype=torch.bool, device=device)
    if k_seq_actual < 2 * n_seq:
        slot_valid[:, :, k_seq_actual: 2 * n_seq] = False
    if k_sp_actual < n_spatial:
        slot_valid[:, :, 2 * n_seq + k_sp_actual: 2 * n_seq + n_spatial] = False
    if k_rnd_actual < n_random:
        slot_valid[:, :, 2 * n_seq + n_spatial + k_rnd_actual:] = False

    return neighbor_idx, slot_valid  # both [b, n, K]
