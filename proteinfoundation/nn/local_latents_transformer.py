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
            self.seq_upsample = UpsampleBlock(self.token_dim)
            # cond is broadcast-constant across positions for the canonical config
            # (feats_cond_seq=[time_emb_bb_ca]), so a learned downsample wastes
            # ~dim_cond^2 params. Use a parameter-free mean pool at the call site.
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

        # Fix C2: optionally build the sparse neighbor list from the self-conditioning
        # coordinates (x_sc) instead of the noisy x_t when t is below a threshold. Plumbed
        # in from cfg_exp.training in proteina.py — source of truth lives next to self_cond.
        self.sc_neighbors = kwargs.get("sc_neighbors", False)
        self.sc_neighbors_t_threshold = kwargs.get("sc_neighbors_t_threshold", 0.4)
        if self.sc_neighbors:
            print(
                f"[Fix C2] sc_neighbors=True (t_threshold={self.sc_neighbors_t_threshold}): "
                f"sparse neighbors will be built from x_sc when t < threshold and x_sc is "
                f"present; otherwise falls back to x_t."
            )

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

    def _subsample_input(self, inp: Dict, n: int, mask: Optional[torch.Tensor] = None, stride: int = 2) -> Dict:
        """
        Shallow-copies the input dict, subsampling any tensor whose dim-1 equals n.
        Recurses into nested dicts (e.g. x_t, x_sc) automatically.

        Pooling rule:
          - With mask provided AND stride==2 AND tensor is float with ndim>=3
            (i.e. coords / latents): mask-weighted CENTROID pool. Output token i
            is the midpoint of input residues 2i and 2i+1 (correctly normalised
            at boundaries so distances aren't halved next to padding).
          - Otherwise: stride-pool (every-other index) — original behavior. Used
            for integer labels (residue_type), boolean masks, and any case where
            the caller didn't supply a mask.

        The mask, if provided, must be the FULL-N mask (pre-downsample).
        """
        out = {}
        for k, v in inp.items():
            if isinstance(v, dict):
                out[k] = self._subsample_input(v, n, mask=mask, stride=stride)
            elif isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == n:
                if mask is not None and stride == 2 and v.is_floating_point() and v.ndim >= 3:
                    assert v.shape[1] % 2 == 0, \
                        f"centroid pool requires even n, got {v.shape[1]}"
                    mask_f = mask.to(dtype=v.dtype)
                    while mask_f.ndim < v.ndim:
                        mask_f = mask_f.unsqueeze(-1)  # broadcast over trailing dims
                    v_masked = v * mask_f
                    v_sum = v_masked[:, 0::2] + v_masked[:, 1::2]
                    w_sum = mask_f[:, 0::2] + mask_f[:, 1::2]
                    out[k] = (v_sum / w_sum.clamp(min=1.0)).contiguous()
                else:
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
            # Skip source: pre-trunk featurization at full N. Carries fine-grained
            # per-residue info that the downsample compresses away.
            pre_trunk_seqs = seqs
            # Full-N mask used for boundary-aware blur and centroid pooling below.
            full_mask = mask

            # 1. Downsample sequence via learned block (mask-aware blur)
            seqs = self.seq_downsample(seqs, mask=full_mask)   # [b, n/2, token_dim]

            # 2. Mean-pool conditioning (no params; equivalent to learned downsample
            #    when cond is broadcast-constant, but tolerates non-constant cond too)
            c = 0.5 * (c[:, 0::2] + c[:, 1::2])  # [b, n/2, dim_cond]

            # 3. Downsample mask (max pool: True if any residue in window is valid)
            mask_float = mask.float().unsqueeze(1)  # [b, 1, n]
            mask = F.max_pool1d(mask_float, kernel_size=2, stride=2).squeeze(1) > 0.5  # [b, n/2]

            # 4. Build pair_rep at full N then 2D mask-aware pool to N/2 — trunk
            #    attends with biases from real residue distances (and full-stride
            #    seq separation), not midpoint-CA distances. Padded (i,j) cells in
            #    pair_full are zero (re-masked at feature_factory.py:2038 and
            #    adaptive_ln_scale.py:40), so avg_pool2d on values + avg_pool2d on
            #    the pair-mask gives the correct mask-renormalized average — the
            #    /4 factors cancel. Shapes unchanged so old checkpoints load, but
            #    warm-starting from a pooled-coord-path checkpoint will diverge.
            assert not self.sparse_attention, \
                "use_downsampling + sparse_attention not supported on this path."

            pair_full = self.pair_repr_builder(input, neighbor_idx=None, slot_valid=None)
            fm = full_mask.to(pair_full.dtype)
            pm_full = (fm[:, :, None] * fm[:, None, :]).unsqueeze(1)                  # [b, 1, N, N]
            num = F.avg_pool2d(pair_full.permute(0, 3, 1, 2).contiguous(), 2, 2)      # [b, d, N/2, N/2]
            den = F.avg_pool2d(pm_full, 2, 2)                                          # [b, 1, N/2, N/2]
            pair_rep = (num / den.clamp(min=1e-6)).permute(0, 2, 3, 1).contiguous()
            hm = mask.to(pair_rep.dtype)
            pair_rep = pair_rep * (hm[:, :, None] * hm[:, None, :])[..., None]
            neighbor_idx, slot_valid = None, None

            # Re-apply mask to sequence
            seqs = seqs * mask[..., None]
        else:
            # Compute sparse neighbor indices on current CA coords if needed
            if self.sparse_attention:
                # Default: build neighbors from x_t (noisy current-step coords).
                coords_for_neighbors = input["x_t"]["bb_ca"]
                # Fix C2: at low t (high noise), x_t is essentially noise and the
                # spatial+random neighbors carry no information. When sc_neighbors is on
                # AND x_sc is present in the batch (training: 50% of batches per the
                # self-cond coin flip; inference step>=1, or step==0 if bootstrap), swap
                # in x_sc per-protein for samples whose t is below the threshold. The
                # 50% no-x_sc training case intentionally falls back to x_t — we want
                # the model to remain robust to x_sc-absent at all t (matches the
                # inference step==0 no-bootstrap path).
                if self.sc_neighbors and "x_sc" in input \
                        and isinstance(input.get("x_sc"), dict) \
                        and "bb_ca" in input["x_sc"]:
                    t_bb_ca = input["t"]["bb_ca"]                     # [B]
                    use_sc = t_bb_ca < self.sc_neighbors_t_threshold  # [B] bool
                    x_sc_ca = input["x_sc"]["bb_ca"]                  # [B, N, 3]
                    coords_for_neighbors = torch.where(
                        use_sc[:, None, None],
                        x_sc_ca,
                        coords_for_neighbors,
                    )
                neighbor_idx, slot_valid = self._build_neighbor_idx(coords_for_neighbors, mask)
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
            # UpsampleBlock has zero-init out_proj + zero-init parity_emb, so at
            # init seqs ≈ pre_trunk_seqs. The trunk learns deltas on top of the
            # original featurization rather than having to reconstruct fine-grained
            # per-residue info from coarse tokens.
            mask = input["mask"]  # restore full-N mask for the upsampled output
            seqs = self.seq_upsample(seqs, target_length=original_n, mask=mask) + pre_trunk_seqs
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
