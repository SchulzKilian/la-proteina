from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK
from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.modules.pair_rep_initial import PairReprBuilder
from proteinfoundation.nn.modules.downsampling import DownsampleBlock, UpsampleBlock
from proteinfoundation.nn.modules.sparse_neighbors import build_neighbor_idx


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(
        dtype=torch.bool, device=device
    )  # [21, 37]


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(LocalLatentsTransformer, self).__init__()

        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]
        self.use_downsampling = kwargs.get("use_downsampling", False)
        
        if self.use_downsampling:
            print("Initializing Downsampling/Upsampling modules...")
            self.seq_downsample = DownsampleBlock(self.token_dim)
            self.cond_downsample = DownsampleBlock(kwargs["dim_cond"])
            self.seq_upsample = UpsampleBlock(self.token_dim)
            # pair_rep is built at n/2 directly — no separate pair downsampler needed
        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        # Sparse attention config
        self.sparse_attention = kwargs.get("sparse_attention", False)
        self.n_seq_neighbors     = kwargs.get("n_seq_neighbors",     8)
        self.n_spatial_neighbors = kwargs.get("n_spatial_neighbors", 8)
        self.n_random_neighbors  = kwargs.get("n_random_neighbors",  16)

        self.latent_dim = kwargs.get("latent_dim", None)
        if self.latent_dim is not None:
            self.local_latents_linear = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, self.latent_dim, bias=False),
            )
        else:
            self.local_latents_linear = None

        assert "local_latents" not in self.output_param or self.local_latents_linear is not None, \
            "output_parameterization contains 'local_latents' but latent_dim is None. " \
            "Either provide latent_dim or remove 'local_latents' from output_parameterization."
        assert "local_latents" not in self.output_param or self.latent_dim is not None, \
            "Contradiction: output_parameterization requests local_latents but no latent_dim given."

        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

    def _build_neighbor_idx(
        self, ca_coors: torch.Tensor, mask: torch.Tensor
    ) -> tuple:
        """Compute sparse neighbor indices and slot validity.

        Returns:
            neighbor_idx: [b, n, K] int64 — neighbor residue indices
            slot_valid:   [b, n, K] bool  — True for real neighbors, False for padding slots
                          (padding only occurs for proteins shorter than K=2*n_seq+n_spatial+n_random)
        """
        return build_neighbor_idx(
            ca_coors,
            mask,
            n_seq=self.n_seq_neighbors,
            n_spatial=self.n_spatial_neighbors,
            n_random=self.n_random_neighbors,
        )

    def _subsample_input(self, inp: Dict, n: int, stride: int = 2) -> Dict:
        """
        Shallow-copies the input dict, subsampling any tensor whose dim-1 equals n.
        Recurses into nested dicts (e.g. x_t, x_sc) automatically.
        Scalars, time tensors [b], and non-sequence tensors are passed through unchanged.
        """
        out = {}
        for k, v in inp.items():
            if isinstance(v, dict):
                out[k] = self._subsample_input(v, n, stride)
            elif isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == n:
                out[k] = v[:, ::stride].contiguous()
            else:
                out[k] = v
        return out

    # @torch.compile
    def forward(self, input: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        if self.use_downsampling:
            original_n = seqs.shape[1]

            # 1. Downsample sequence and conditioning via learned blocks
            seqs = self.seq_downsample(seqs)   # [b, n/2, token_dim]
            c = self.cond_downsample(c)         # [b, n/2, dim_cond]

            # 2. Downsample mask (max pool: True if any residue in window is valid)
            mask_float = mask.float().unsqueeze(1)  # [b, 1, n]
            mask = F.max_pool1d(mask_float, kernel_size=2, stride=2).squeeze(1) > 0.5  # [b, n/2]

            # 3. Build pair representation directly at n/2 using stride-2 subsampled coords.
            #    This avoids computing the full [b, n, n, d] tensor and then pooling it down.
            input_ds = self._subsample_input(input, original_n, stride=2)
            # Compute sparse neighbor indices on downsampled coords if needed
            if self.sparse_attention:
                neighbor_idx, slot_valid = self._build_neighbor_idx(input_ds["x_t"]["bb_ca"], mask)
            else:
                neighbor_idx, slot_valid = None, None
            pair_rep = self.pair_repr_builder(input_ds, neighbor_idx=neighbor_idx, slot_valid=slot_valid)

            # Re-apply mask to sequence
            seqs = seqs * mask[..., None]
        else:
            # Compute sparse neighbor indices on current CA coords if needed
            if self.sparse_attention:
                neighbor_idx, slot_valid = self._build_neighbor_idx(input["x_t"]["bb_ca"], mask)
            else:
                neighbor_idx, slot_valid = None, None
            pair_rep = self.pair_repr_builder(input, neighbor_idx=neighbor_idx, slot_valid=slot_valid)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask, neighbor_idx=neighbor_idx, slot_valid=slot_valid
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask, neighbor_idx=neighbor_idx, slot_valid=slot_valid
                        )

        if self.use_downsampling:
            seqs = self.seq_upsample(seqs, target_length=original_n)
            # Restore original mask for final output
            mask = input["mask"] 
            seqs = seqs * mask[..., None]

        # Get outputs
        assert "bb_ca" in self.output_param, \
            "output_parameterization must contain 'bb_ca'."

        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]
        assert ca_nm_out.shape[-1] == 3, \
            f"CA output should have dim 3, got {ca_nm_out.shape[-1]}"

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}

        if self.local_latents_linear is not None:
            assert "local_latents" in self.output_param, \
                "local_latents_linear exists but 'local_latents' missing from output_parameterization."
            local_latents_out = (
                self.local_latents_linear(seqs) * mask[..., None]
            )  # [b, n, latent_dim]
            nn_out["local_latents"] = {
                self.output_param["local_latents"]: local_latents_out
            }

        return nn_out
