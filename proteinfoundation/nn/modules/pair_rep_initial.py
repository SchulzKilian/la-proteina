import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

    def forward(self, batch_nn, neighbor_idx=None, slot_valid=None):
        """
        Args:
            batch_nn: batch dict
            neighbor_idx: optional [b, n, K] int tensor for sparse mode.
                          When provided, returns [b, n, K, dim] instead of [b, n, n, dim].
            slot_valid: optional [b, n, K] bool; True = real neighbor slot (not padding)
        """
        mask = batch_nn["mask"]  # [b, n]

        if neighbor_idx is not None:
            # Sparse: build [b, n, K] validity mask for adaln
            B, N, K = neighbor_idx.shape
            B_idx = torch.arange(B, device=mask.device).view(B, 1, 1).expand(B, N, K)
            pair_mask = mask[:, :, None].expand(B, N, K) & mask[B_idx, neighbor_idx]  # [b, n, K]
            # Guard against padding slots pointing to valid-but-spurious residue 0
            if slot_valid is not None:
                pair_mask = pair_mask & slot_valid
        else:
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]  — original

        repr = self.init_repr_factory(batch_nn, neighbor_idx=neighbor_idx, slot_valid=slot_valid)

        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn, neighbor_idx=neighbor_idx, slot_valid=slot_valid)
            repr = self.adaln(repr, cond, pair_mask)

        return repr

