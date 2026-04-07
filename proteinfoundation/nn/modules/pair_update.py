import torch
from torch.utils.checkpoint import checkpoint

from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicationOutgoing(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            self.tri_mult_in = TriangleMultiplicationIncoming(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def forward(self, x, pair_rep, mask, neighbor_idx=None, slot_valid=None):
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, [b, n, n, pair_dim] (dense) or [b, n, K, pair_dim] (sparse)
            mask: binary mask, shape [b, n]
            neighbor_idx: optional [b, n, K] for sparse mode
            slot_valid: optional [b, n, K] bool; True = real neighbor slot (not padding)

        Returns:
            Updated pair representation, same shape as input pair_rep.
        """
        x = x * mask[..., None]  # [b, n, token_dim]
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(2, dim=-1)

        if neighbor_idx is not None:
            # Sparse mode: pair_rep is [b, n, K, pair_dim]
            if self.use_tri_mult:
                raise ValueError(
                    "use_tri_mult=True is incompatible with sparse attention "
                    "(triangular updates require the full n×n pair representation)."
                )
            B, N, K = neighbor_idx.shape
            B_idx = torch.arange(B, device=mask.device).view(B, 1, 1).expand(B, N, K)
            # Match dense convention: proj_1 is j-feature, proj_2 is i-feature.
            # Dense: pair_rep[i,j] += proj_1[j] + proj_2[i]
            # Sparse: pair_rep[i,k] += proj_1[neighbor[i,k]] + proj_2[i]
            x_proj_1_j = x_proj_1[B_idx, neighbor_idx]              # [b, n, K, pair_dim]
            pair_rep = pair_rep + x_proj_2[:, :, None, :] + x_proj_1_j  # [b, n, K, pair_dim]
            # Build sparse pair mask [b, n, K]
            nbr_valid = mask[B_idx, neighbor_idx]
            i_valid   = mask[:, :, None].expand(B, N, K)
            pair_mask = i_valid & nbr_valid
            # Guard against padding slots pointing to valid-but-spurious residue 0
            if slot_valid is not None:
                pair_mask = pair_mask & slot_valid
            pair_mask = pair_mask.float()                               # [b, n, K]
            pair_rep = pair_rep * pair_mask[..., None]
            pair_rep = pair_rep + checkpoint(self.transition_out, *(pair_rep, pair_mask))
            pair_rep = pair_rep * pair_mask[..., None]
        else:
            # Dense mode (original behaviour)
            pair_mask = mask[:, None, :] * mask[:, :, None]             # [b, n, n]
            pair_rep = (
                pair_rep + x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
            )  # [b, n, n, pair_dim]
            pair_rep = self._apply_mask(pair_rep, pair_mask)
            if self.use_tri_mult:
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_out, *(pair_rep, pair_mask * 1.0)
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_in, *(pair_rep, pair_mask * 1.0)
                )
                pair_rep = self._apply_mask(pair_rep, pair_mask)
            pair_rep = pair_rep + checkpoint(
                self.transition_out, *(pair_rep, pair_mask * 1.0)
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)

        return pair_rep
