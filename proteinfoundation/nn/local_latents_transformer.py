from typing import Dict

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
            # Use AvgPool2d for the pair representation
            self.pair_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
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

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

    @torch.compile
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

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        if self.use_downsampling:
            original_n = seqs.shape[1]
            
            # 1. Downsample Sequence and Conditioning
            seqs = self.seq_downsample(seqs) 
            c = self.cond_downsample(c)
            
            # 2. Downsample Mask (Max pool keeps 'True' if any residue in window is valid)
            mask_float = mask.float().unsqueeze(1) # [b, 1, n]
            mask_down = F.max_pool1d(mask_float, kernel_size=2, stride=2).squeeze(1)
            mask = mask_down > 0.5 
            
            # 3. Downsample Pair Representation (2D)
            # [b, n, n, d] -> [b, d, n, n]
            pair_rep = pair_rep.permute(0, 3, 1, 2)
            pair_rep = self.pair_downsample(pair_rep)
            # [b, d, n', n'] -> [b, n', n', d]
            pair_rep = pair_rep.permute(0, 2, 3, 1)
            
            # Re-apply mask to sequence
            seqs = seqs * mask[..., None]

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]

        if self.use_downsampling:
            seqs = self.seq_upsample(seqs, target_length=original_n)
            # Restore original mask for final output
            mask = input["mask"] 
            seqs = seqs * mask[..., None]

        # Get outputs
        local_latents_out = (
            self.local_latents_linear(seqs) * mask[..., None]
        )  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {
            self.output_param["local_latents"]: local_latents_out
        }
        return nn_out
