import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

root = os.path.abspath(".")
sys.path.insert(0, root)  # Adds project's root directory
# isort: split

import argparse
import json

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.performance_utils import measure_performance, save_performance_metrics

torch.set_float32_matmul_precision('high')

def parse_args_and_cfg() -> Tuple[Dict, Dict, str]:
    """
    Parses command line arguments and loads the corresponding config file.

    Returns:
        Command line arguments (dict)
        Config file (dict)
        config_name (string)
    """
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        "--config-name",
        type=str,
        default="inference_base",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job id for this config to determine which split to use.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="inference",
        help="Name of the task to be performed.",
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Name of the data path",
    )
    args = parser.parse_args()
    if args.data_path is not None:
        os.environ["DATA_PATH"] = args.data_path
    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../configs"
    else:
        config_path = f"../configs/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")

    return args, cfg, config_name


def setup(
    cfg: Dict, create_root: bool = True, config_name: str = ".", job_id: int = 0
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    assert (
        not (
            cfg.generation.metric.compute_designability
            or cfg.generation.metric.compute_novelty_pdb
            or cfg.generation.metric.compute_novelty_afdb
        )
        or not cfg.generation.metric.compute_fid
    ), "Designability/Novelty cannot be computed together with FID"

    # Set root path for this inference run
    if "motif_task_name" in cfg.generation.dataset:
        root_path = (
            f"./inference/{config_name}_{cfg.generation.dataset.motif_task_name}"
        )
    else:
        root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    # Set seed
    cfg.seed = cfg.seed + job_id  # Different seeds for different splits ids
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def check_cfg_validity(cfg_data: Dict, cfg_sample_args: Dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} and autoguidance ratio {cfg_sample_args.ag_ratio}."
        )
        assert (
            cfg_sample_args.ag_ratio >= 0.0 and cfg_sample_args.ag_ratio <= 1.0
        ), f"Autoguidance ratio should be between 0 and 1, but now is {cfg_sample_args.ag_ratio}."
        assert (cfg_sample_args.ag_ratio == 0.0) or (
            cfg_sample_args.ag_ckpt_path is not None
        ), f"Autoguidance checkpoint path should be provided"
    else:
        logger.info(f"Guidance is turned off.")

    # Logging conditional generation
    if cfg_sample_args.fold_cond:
        logger.info("Conditional generation is turned on.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is not None
        ), "Empirical (len, cath_code) distribution file should be provided when using conditional generation."
    else:
        logger.info("Conditional generation is turned off.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is None
        ), "Empirical (len, cath_code) distribution file shouldn't be provided when using unconditional generation."


def load_ag_ckpt(cfg: Dict) -> Union[None, torch.nn.Module]:
    """
    Loads the neural network for the "bad" checkpoint in autoguidance, if requested.

    Returns:
        A nn module, if autogudance enabled.
    """
    nn_ag = None
    if cfg.ag_ratio > 0 and cfg.guidance_w != 1.0:
        logger.info(
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file, strict=False)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info(
                "Removing autoencoder encoder from autoguidance model during generation to save memory"
            )
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def load_ckpt_n_configure_inference(cfg: Dict) -> Proteina:
    """
    Loads the model, potentially the autoguidance checkpoint as well, if requested.

    Returns:
        Model (Proteina)
    """
    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    model = Proteina.load_from_checkpoint(ckpt_file, strict=False, autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path", None))

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    model.configure_inference(cfg.generation, nn_ag=nn_ag)

    # Neighbor-list curriculum (sparse-attention models only). Defaults to
    # off; existing configs unchanged. The schedule itself (3-bucket
    # reallocation of K=64 across sequential/spatial/random by t) lives
    # inside `LocalLatentsTransformer._build_neighbor_idx`.
    cur_on = cfg.generation.args.get("curriculum_neighbors", None)
    if cur_on:
        if not getattr(model.nn, "sparse_attention", False):
            logger.warning(
                "curriculum_neighbors=True but model is not sparse-attention — flag is a no-op."
            )
        model.nn.curriculum_neighbors = True
        logger.info(
            "[Curriculum neighbors] ON — 3-bucket K=64 reallocation: "
            "t<0.33 → (n_seq=32, n_sp=0, n_rd=0); "
            "0.33≤t<0.66 → (16, 8, 24); "
            "t≥0.66 → (8, 16, 32)."
        )
    elif cur_on is False:
        # Force off against a curriculum-trained ckpt → use the static
        # (n_seq_neighbors, n_spatial_neighbors, n_random_neighbors) at all t.
        # Distribution-shift test (the model was trained against the schedule),
        # not a clean architectural comparison — see the K-bump caveat below.
        model.nn.curriculum_neighbors = False
        logger.info(
            "[Curriculum neighbors] OFF — static neighbor split at all t "
            f"(n_seq={getattr(model.nn, 'n_seq_neighbors', '?')} per side, "
            f"n_spatial={getattr(model.nn, 'n_spatial_neighbors', '?')}, "
            f"n_random={getattr(model.nn, 'n_random_neighbors', '?')})."
        )

    # Optional inference-time override for the low-t bucket split (only meaningful
    # when curriculum_neighbors is left ON). Distribution-shift test against a
    # ckpt trained with (32, 0, 0) at low t — same caveat as the K-bump override.
    low_t_override = cfg.generation.args.get("curriculum_low_t_split", None)
    if low_t_override is not None and getattr(model.nn, "curriculum_neighbors", False):
        split = tuple(low_t_override)
        assert 2 * split[0] + split[1] + split[2] == 64, (
            f"curriculum_low_t_split={split} must sum to K=64."
        )
        model.nn.curriculum_low_t_split = split
        logger.info(
            f"[Curriculum neighbors] low-t bucket overridden to "
            f"(n_seq={split[0]}, n_sp={split[1]}, n_rd={split[2]}) for t<0.33."
        )

    # Inference-only K-bump (sparse-attention models only). Override the
    # trained 8/8/16 (K=40) neighbor counts post-load. Works because K is just
    # an int attribute consumed at the call site — `build_neighbor_idx` and
    # the sparse attention/pair paths derive K from `neighbor_idx.shape[-1]`,
    # not from any pre-allocated buffer. q/k/v projections and softmax
    # dynamics remain calibrated for K=40, so this is a distribution-shift
    # test, not a clean architectural test.
    if (
        "n_seq_neighbors_override" in cfg.generation.args
        or "n_spatial_neighbors_override" in cfg.generation.args
        or "n_random_neighbors_override" in cfg.generation.args
    ):
        if not getattr(model.nn, "sparse_attention", False):
            logger.warning(
                "K-bump override set but model is not sparse-attention — overrides are no-ops."
            )
        n_seq = cfg.generation.args.get("n_seq_neighbors_override",
                                         model.nn.n_seq_neighbors)
        n_sp = cfg.generation.args.get("n_spatial_neighbors_override",
                                        model.nn.n_spatial_neighbors)
        n_rd = cfg.generation.args.get("n_random_neighbors_override",
                                        model.nn.n_random_neighbors)
        model.nn.n_seq_neighbors = n_seq
        model.nn.n_spatial_neighbors = n_sp
        model.nn.n_random_neighbors = n_rd
        K_total = 2 * n_seq + n_sp + n_rd
        logger.info(
            f"[K-bump] override → n_seq={n_seq} (×2 = {2*n_seq} sequential), "
            f"n_spatial={n_sp}, n_random={n_rd} → K_total={K_total} "
            f"(trained at K=40)."
        )

    # Per-layer sparse/dense routing (inference-only diagnostic; see
    # script_utils/test_layer_selective_sparse_inference.py). When
    # cfg.generation.args.layer_sparse_mask is a list of nlayers bools,
    # each layer i uses sparse attention iff mask[i] is True. Required
    # companion flag `force_sparse_attention_on=True` turns sparse_attention
    # on post-load when the ckpt was dense-trained — no parameter changes,
    # just enables the neighbor list at forward time.
    layer_mask = cfg.generation.args.get("layer_sparse_mask", None)
    force_sparse_on = cfg.generation.args.get("force_sparse_attention_on", False)
    if force_sparse_on:
        if not getattr(model.nn, "sparse_attention", False):
            logger.info("[layer_sparse_mask] forcing model.nn.sparse_attention=True "
                        "(was False from ckpt) — diagnostic on dense-trained weights.")
            model.nn.sparse_attention = True
    if layer_mask is not None:
        layer_mask_list = [bool(x) for x in list(layer_mask)]
        expected_len = getattr(model.nn, "nlayers", None)
        assert expected_len is None or len(layer_mask_list) == expected_len, (
            f"layer_sparse_mask len {len(layer_mask_list)} != nlayers {expected_len}"
        )
        assert getattr(model.nn, "sparse_attention", False), (
            "layer_sparse_mask requires sparse_attention=True. "
            "Set force_sparse_attention_on=True for a dense-trained ckpt."
        )
        model.nn.layer_sparse_mask = layer_mask_list
        n_sparse = sum(layer_mask_list)
        logger.info(
            f"[layer_sparse_mask] per-layer routing ON: "
            f"{n_sparse}/{len(layer_mask_list)} sparse, "
            f"{len(layer_mask_list)-n_sparse} dense. Mask: {layer_mask_list}"
        )

    # Inference-only sc_neighbors override (Fix C2 at inference on a ckpt that
    # was NOT trained with sc_neighbors). Mutates both:
    #   (a) model.cfg_exp.training.sc_neighbors — read by full_simulation
    #       at proteina.py:869 as sc_neighbors_active, gates the step-0
    #       bootstrap forward.
    #   (b) model.nn.sc_neighbors — read by LocalLatentsTransformer's forward
    #       at every step to swap x_t for x_sc in the neighbor-list builder
    #       when t < sc_neighbors_t_threshold.
    # Distribution-shift test against a ckpt trained without sc_neighbors —
    # the trunk never saw clean-coord-built neighbors at low t. Self-cond must
    # be enabled (inf_cfg.args.self_cond=True, default) for x_sc to be
    # populated; otherwise the override is a no-op past step 0.
    # Inference-only ROUTER-DERIVED PER-(LAYER, HEAD) K-set for sparse layers.
    # The TopKRouter was trained per-(layer, head, query) (KL loss vs dense's
    # per-(l, h) top-K attention). To honor that training, we swap each lower-
    # half layer's MHA module from MultiHeadBiasedAttentionADALN_MM (dense /
    # per-layer sparse) → MultiHeadBiasedAttentionADALN_MM_RouterSparse (per-
    # (head) sparse). The two classes have *bit-identical parameter layouts*
    # per pair_bias_attn_sparse.py:42-46, so the dense baseline's trained
    # weights load directly into the new module — no projection retraining.
    #
    # Two-pass forward wrapper on model.nn:
    #   PASS 1: ORIGINAL blocks in transformer_layers (dense MHA), with
    #           layer_sparse_mask=[False]*nlayers → genuine all-dense forward.
    #           Forward-pre-hooks on each transformer_layers[i] capture the
    #           layer-i input seqs (same input distribution the router was
    #           trained on).
    #   ROUTER: for each i ∈ router_lower_layers, call
    #             router.forward_one_layer(layer_inputs[i], layer_idx=i)
    #             → [B, H, N, N]. Per-(batch, head, query) top-K=router_K
    #             → router_neighbor_idx_i: [B, H, N, K] (the shape Move-2's
    #             PairBiasAttentionSparse expects).
    #   PASS 2: SWAP router-sparse blocks (weights copied from originals once
    #           at install time) into transformer_layers[i] for i ∈ lower,
    #           then call _orig_forward with layer_sparse_mask=[False]*nlayers
    #           (all positions consume pair_rep_dense; the swapped block reads
    #           its per-head K-set from the per-layer cache; upper layers run
    #           dense). Restore original blocks after.
    # Cost: 2× trunk forward + per-layer router scoring per ODE step. Requires
    # sparse_attention=True (force_sparse_attention_on=True for a dense-trained
    # ckpt) AND a layer_sparse_mask. Mutex with n_global_tokens > 0 and
    # use_downsampling=True.
    router_ckpt_path = cfg.generation.args.get("router_ckpt_path", None)
    if router_ckpt_path is not None:
        import sys as _sys
        if "/home/ks2218/la-proteina" not in _sys.path:
            _sys.path.insert(0, "/home/ks2218/la-proteina")
        from script_utils.load_frozen_router import load_frozen_router as _load_router
        from proteinfoundation.nn.modules.pair_bias_attn_sparse import (
            MultiheadAttnAndTransitionRouterSparse,
        )
        router_K = int(cfg.generation.args.get("router_K", 40))
        router_lower_layers = list(cfg.generation.args.get(
            "router_lower_layers",
            cfg.generation.args.get("router_aggregate_layers", list(range(7)))
        ))
        assert getattr(model.nn, "sparse_attention", False), (
            "router_ckpt_path requires sparse_attention=True (set "
            "force_sparse_attention_on=True for a dense-trained ckpt)."
        )
        assert getattr(model.nn, "layer_sparse_mask", None) is not None, (
            "router_ckpt_path requires layer_sparse_mask to be set."
        )
        assert getattr(model.nn, "n_global_tokens", 0) == 0, (
            "router-derived K-set is not wired through _attach_globals."
        )
        assert not getattr(model.nn, "use_downsampling", False), (
            "router-derived K-set is not wired through use_downsampling path."
        )
        _mask_list = list(model.nn.layer_sparse_mask)
        for _li in router_lower_layers:
            assert _mask_list[_li], (
                f"router_lower_layers includes layer {_li} but "
                f"layer_sparse_mask[{_li}]=False — that layer would not be swapped."
            )

        _router = _load_router(router_ckpt_path, map_location="cpu")
        _device = next(model.parameters()).device
        _dtype = next(model.parameters()).dtype
        _router = _router.to(device=_device, dtype=_dtype)
        _router.eval()
        for _p in _router.parameters():
            _p.requires_grad_(False)

        _nn = model.nn
        _saved_mask = list(_nn.layer_sparse_mask)
        _layer_inputs_list: list = [None] * _nn.nlayers
        _capture_flag = {"on": False}
        _per_layer_K: dict = {}            # {layer_idx: (router_neighbor_idx, slot_valid)}

        def _make_hook(layer_idx: int):
            def _hook(_module, _input):
                _layer_inputs_list[layer_idx] = _input[0].detach()
            return _hook

        for _i, _layer in enumerate(_nn.transformer_layers):
            _layer.register_forward_pre_hook(_make_hook(_i))

        # Read the dims off the existing block (architecture cfg lives there).
        _nheads = getattr(_router, "n_heads")
        _token_dim = _nn.token_dim
        _pair_dim = _nn.pair_repr_dim
        _dim_cond = getattr(_nn, "dim_cond", None)
        if _dim_cond is None:
            # Fallback: AdaptiveLayerNorm.to_beta is Linear(dim_cond → dim), so
            # to_beta.in_features == dim_cond. Stable inspection point.
            _dim_cond = _nn.transformer_layers[0].mhba.adaln.to_beta.in_features
        _use_qkln = getattr(_nn, "use_qkln", True)

        # Build router-sparse blocks once; copy weights from originals so trained
        # projections (q/k/v/out/pair_bias, adaln, scale_output, transition)
        # transfer verbatim. PairBiasAttentionSparse vs PairBiasAttention have
        # bit-identical param layouts per the file's own docstring.
        _original_lower_blocks: dict = {}
        _router_sparse_blocks: dict = {}
        import types as _types
        for _li in router_lower_layers:
            _orig_block = _nn.transformer_layers[_li]
            _original_lower_blocks[_li] = _orig_block
            _rs_block = MultiheadAttnAndTransitionRouterSparse(
                dim_token=_token_dim, dim_pair=_pair_dim, nheads=_nheads,
                dim_cond=_dim_cond, use_qkln=_use_qkln,
            ).to(device=_device, dtype=_dtype)
            # mhba: adaln + mha + scale_output (mha class differs but state_dict layout matches)
            _rs_block.mhba.load_state_dict(_orig_block.mhba.state_dict(), strict=True)
            _rs_block.transition.load_state_dict(_orig_block.transition.state_dict(), strict=True)
            _rs_block.eval()
            for _p in _rs_block.parameters():
                _p.requires_grad_(False)

            # Wrap rs_block.forward so the trunk's layer-loop call signature
            # (x, pair_rep, cond, mask, neighbor_idx=..., slot_valid=...)
            # routes through to router_neighbor_idx= using the per-layer cache.
            def _make_rs_wrapped(layer_idx, rs_orig_forward):
                def _wrapped(self_block, x, pair_rep, cond, mask,
                              neighbor_idx=None, slot_valid=None):
                    rn_idx, rn_valid = _per_layer_K[layer_idx]
                    return rs_orig_forward(
                        self_block, x, pair_rep, cond, mask,
                        router_neighbor_idx=rn_idx, slot_valid=rn_valid,
                    )
                return _wrapped
            _rs_block.forward = _types.MethodType(
                _make_rs_wrapped(_li, type(_rs_block).forward), _rs_block
            )
            _router_sparse_blocks[_li] = _rs_block

        _orig_build = _nn._build_neighbor_idx
        _orig_forward = _nn.forward

        def _patched_build(self_nn, ca_coors, mask, t=None):
            # Capture pass: zeros. Real pass: zeros too — during PASS 2 we set
            # layer_sparse_mask=[False]*nlayers so no layer consumes the upfront
            # neighbor_idx (the swapped router-sparse layers read their per-head
            # K-set from _per_layer_K instead).
            B, N, _ = ca_coors.shape
            return (
                torch.zeros(B, N, router_K, dtype=torch.long, device=ca_coors.device),
                torch.zeros(B, N, router_K, dtype=torch.bool, device=ca_coors.device),
            )

        def _swap_in_router_sparse():
            for _li in router_lower_layers:
                _nn.transformer_layers[_li] = _router_sparse_blocks[_li]

        def _swap_out_router_sparse():
            for _li in router_lower_layers:
                _nn.transformer_layers[_li] = _original_lower_blocks[_li]

        def _patched_forward(input_dict):
            mask_b = input_dict["mask"]
            with torch.no_grad():
                # ---- PASS 1: capture layer inputs (original dense blocks, all-dense) ----
                for _ii in range(_nn.nlayers):
                    _layer_inputs_list[_ii] = None
                _capture_flag["on"] = True
                _nn.layer_sparse_mask = [False] * _nn.nlayers
                try:
                    _ = _orig_forward(input_dict)
                finally:
                    _capture_flag["on"] = False
                    _nn.layer_sparse_mask = _saved_mask

                assert all(t is not None for t in _layer_inputs_list), (
                    "Pre-hooks did not fire on every layer; router capture failed."
                )

                # ---- ROUTER: per-(layer, head) K-sets ----
                _per_layer_K.clear()
                for _li in router_lower_layers:
                    scores_i = _router.forward_one_layer(
                        layer_input=_layer_inputs_list[_li], layer_idx=_li,
                    )  # [B, H, N, N]
                    B_, H_, N_, _ = scores_i.shape
                    neg_inf = torch.finfo(scores_i.dtype).min
                    # Mask invalid keys; broadcast over (H, query).
                    scores_masked = scores_i.masked_fill(
                        ~mask_b[:, None, None, :], neg_inf
                    )
                    # Clamp K to padded length N so topk(k=K) is valid when
                    # K > N (e.g., router_K=64 but generate.py pads L=50 to N=50).
                    # Within a single batch all proteins share L, so K_eff is
                    # consistent across the batch; PairBiasAttentionSparse reads
                    # K from neighbor_idx.shape[-1] and handles K<router_K fine.
                    K_eff = min(router_K, scores_masked.shape[-1])
                    top_idx = scores_masked.topk(k=K_eff, dim=-1).indices       # [B, H, N, K_eff]
                    slot_valid = (
                        mask_b[:, None, None, :].expand(B_, H_, N_, -1)
                        .gather(-1, top_idx)
                    )                                                            # [B, H, N, K_eff]
                    _per_layer_K[_li] = (top_idx, slot_valid)

                # ---- PASS 2: swap router-sparse blocks in, run trunk all-dense-paired ----
                _swap_in_router_sparse()
                _nn.layer_sparse_mask = [False] * _nn.nlayers
                try:
                    out = _orig_forward(input_dict)
                finally:
                    _swap_out_router_sparse()
                    _nn.layer_sparse_mask = _saved_mask
                    _per_layer_K.clear()
            return out

        _nn._build_neighbor_idx = _types.MethodType(_patched_build, _nn)
        _nn.forward = _types.MethodType(lambda self_nn, inp: _patched_forward(inp), _nn)
        logger.info(
            f"[router] frozen router loaded from {router_ckpt_path} "
            f"(hidden={_router.hidden_dim}, score={_router.score_dim}, "
            f"nlayers={_router.n_layers}, nheads={_router.n_heads}). "
            f"PER-(LAYER, HEAD) K-set: MHA on lower layers {router_lower_layers} "
            f"swapped to PairBiasAttentionSparse (weights transferred from dense). "
            f"Each lower layer's per-(head, query) K=router_K={router_K} comes from "
            f"router.forward_one_layer at that layer. Upper layers run dense unchanged. "
            f"Cost: 2× trunk forward per ODE step."
        )

    # Inference-only PER-LAYER content-free K-ramp for the sparse layers.
    # `layer_K_splits` is a list of `(n_seq, n_spatial, n_random)` triples — one
    # per sparse layer (in the order they appear in layer_sparse_mask, i.e.
    # the indices where mask[i]=True). Each sparse layer rebuilds its own
    # content-free neighbor_idx with that layer's K-budget AND its own
    # pair_rep_sparse, leaving the upper dense layers untouched.
    #
    # Mutex with router_ckpt_path (the router patch above already overrides
    # _build_neighbor_idx and patches the lower transformer_layers; combining
    # the two would let the router's per-(head) K-set fight with a content-free
    # per-(layer) K-set and silently produce undefined behaviour).
    layer_K_splits = cfg.generation.args.get("layer_K_splits", None)
    if layer_K_splits is not None:
        assert router_ckpt_path is None, (
            "layer_K_splits and router_ckpt_path are mutually exclusive."
        )
        assert getattr(model.nn, "sparse_attention", False), (
            "layer_K_splits requires sparse_attention=True (set "
            "force_sparse_attention_on=True for a dense-trained ckpt)."
        )
        assert getattr(model.nn, "layer_sparse_mask", None) is not None, (
            "layer_K_splits requires layer_sparse_mask to be set."
        )
        assert getattr(model.nn, "n_global_tokens", 0) == 0, (
            "layer_K_splits is not wired through _attach_globals."
        )
        assert not getattr(model.nn, "use_downsampling", False), (
            "layer_K_splits is not wired through use_downsampling path."
        )
        _mask_list = list(model.nn.layer_sparse_mask)
        _sparse_idxs = [i for i, s in enumerate(_mask_list) if s]
        _splits = [tuple(int(x) for x in s) for s in layer_K_splits]
        assert len(_splits) == len(_sparse_idxs), (
            f"layer_K_splits has {len(_splits)} entries but "
            f"layer_sparse_mask has {len(_sparse_idxs)} sparse layers."
        )
        for s in _splits:
            assert len(s) == 3 and all(v >= 0 for v in s), (
                f"each layer_K_splits entry must be (n_seq, n_spatial, n_random) "
                f"non-negative ints; got {s}"
            )
        _splits_by_layer = dict(zip(_sparse_idxs, _splits))

        _nn = model.nn
        _per_layer_K_ramp: dict = {}        # {layer_idx: (neighbor_idx, slot_valid)}
        _ramp_input_holder = {"input": None}
        _orig_build_ramp = _nn._build_neighbor_idx
        _orig_forward_ramp = _nn.forward
        import types as _types_r

        def _patched_forward_ramp(input_dict):
            with torch.no_grad():
                mask_b = input_dict["mask"]
                ca = input_dict["x_t"]["bb_ca"]
                t_b = input_dict["t"]["bb_ca"]

                _per_layer_K_ramp.clear()
                _saved_seq = _nn.n_seq_neighbors
                _saved_sp  = _nn.n_spatial_neighbors
                _saved_rd  = _nn.n_random_neighbors
                try:
                    for _li, (_ns, _nsp, _nr) in _splits_by_layer.items():
                        _nn.n_seq_neighbors = _ns
                        _nn.n_spatial_neighbors = _nsp
                        _nn.n_random_neighbors = _nr
                        _nbr, _sv = _orig_build_ramp(ca, mask_b, t_b)
                        _per_layer_K_ramp[_li] = (_nbr, _sv)
                finally:
                    _nn.n_seq_neighbors = _saved_seq
                    _nn.n_spatial_neighbors = _saved_sp
                    _nn.n_random_neighbors = _saved_rd

                _ramp_input_holder["input"] = input_dict
                try:
                    out = _orig_forward_ramp(input_dict)
                finally:
                    _ramp_input_holder["input"] = None
                    _per_layer_K_ramp.clear()
            return out

        # Patch lower transformer_layers to rebuild their own pair_rep + use
        # their own neighbor_idx. The trunk's layer loop still passes a global
        # pair_rep_sparse/neighbor_idx via the layer_sparse_mask branch; we
        # ignore those and use the per-layer cache.
        _orig_layer_forwards_ramp: dict = {}
        for _li in _sparse_idxs:
            _layer = _nn.transformer_layers[_li]
            _orig_layer_forwards_ramp[_li] = type(_layer).forward

            def _make_ramp_layer_forward(layer_idx, orig_forward_fn):
                def _patched_layer_forward(self_layer, seqs, _pair_rep_ignored,
                                            c, mask, neighbor_idx=None, slot_valid=None):
                    nbr_i, sv_i = _per_layer_K_ramp[layer_idx]
                    pair_rep_i = _nn.pair_repr_builder(
                        _ramp_input_holder["input"],
                        neighbor_idx=nbr_i, slot_valid=sv_i,
                    )
                    return orig_forward_fn(
                        self_layer, seqs, pair_rep_i, c, mask,
                        neighbor_idx=nbr_i, slot_valid=sv_i,
                    )
                return _patched_layer_forward

            _layer.forward = _types_r.MethodType(
                _make_ramp_layer_forward(_li, type(_layer).forward), _layer
            )

        _nn.forward = _types_r.MethodType(
            lambda self_nn, inp: _patched_forward_ramp(inp), _nn
        )
        _Ks_total = [2 * s[0] + s[1] + s[2] for s in _splits]
        logger.info(
            f"[layer_K_ramp] content-free per-layer K on sparse layers {_sparse_idxs}: "
            f"K_total = {_Ks_total} (mean {sum(_Ks_total)/len(_Ks_total):.1f}). "
            f"Splits (n_seq, n_spatial, n_random): {_splits}."
        )

    sc_nbr_override = cfg.generation.args.get("sc_neighbors_override", None)
    sc_nbr_thr_override = cfg.generation.args.get(
        "sc_neighbors_t_threshold_override", None
    )
    if sc_nbr_override is not None or sc_nbr_thr_override is not None:
        if not getattr(model.nn, "sparse_attention", False):
            logger.warning(
                "sc_neighbors override set but model is not sparse-attention — "
                "overrides are no-ops (dense path ignores neighbor_idx)."
            )
        # cfg.training may be in struct mode and may not have the sc_neighbors
        # keys at all (older ckpts predate Fix C2). Open struct briefly to add.
        OmegaConf.set_struct(model.cfg_exp.training, False)
        if sc_nbr_override is not None:
            model.cfg_exp.training.sc_neighbors = bool(sc_nbr_override)
            model.nn.sc_neighbors = bool(sc_nbr_override)
        if sc_nbr_thr_override is not None:
            thr = float(sc_nbr_thr_override)
            model.cfg_exp.training.sc_neighbors_t_threshold = thr
            model.nn.sc_neighbors_t_threshold = thr
        OmegaConf.set_struct(model.cfg_exp.training, True)
        logger.info(
            f"[sc_neighbors override] sc_neighbors="
            f"{getattr(model.nn, 'sc_neighbors', '?')}, "
            f"t_threshold="
            f"{getattr(model.nn, 'sc_neighbors_t_threshold', '?')} "
            f"(ckpt-trained value overridden post-load; requires self_cond=True)."
        )

    return model


def split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nsamples
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nsamples = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def binder_split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nlens_cfg.random_lens[2]
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nlens_cfg.random_lens[2] = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def save_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    chain_indexes: np.ndarray = None,
    cath_codes: List[List[List[str]]] = None,
) -> None:
    """
    Saves generated samples.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)/
        predictions: List of lists of tuples. Each tuple represents a sample, has to components
            (coors [n, 37, 3], aatype [n])
        job_id: job number, used to store files.
        chain_indexes: chain indexes for each sample, used to store files.
        cath_codes: conditional sampling...
    """
    predictions = [sample for sublist in predictions for sample in sublist]
    # List[tuple] where each tuple is (coors [n, 37, 3], aatype [n])

    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        coors_atom37, residue_type = pred  # [n, 37, 3] and [n]
        n = coors_atom37.shape[-3]
        if chain_indexes:
            chain_index = chain_indexes[j].numpy()
        else:
            chain_index = None

        # Create directory where everything related to this sample will be stored
        suffix = ""
        dir_name = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{suffix}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(
            root_path, dir_name
        )
        os.makedirs(sample_root_path, exist_ok=False)

        # Save generated structure as pdb
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            chain_index=chain_index,
            overwrite=True,
            no_indexing=True,
        )



def save_motif_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    motif_pdb_name: str = None,
) -> None:
    predictions = [sample for sublist in predictions for sample in sublist]
    print([(p[0].shape, p[1].shape) for p in predictions])
    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        coors_atom37, residue_type = pred  # [n, 37, 3] and [n]
        n = coors_atom37.shape[-3]
        dir_name = f"job_{job_id}_id_{j}_motif_{motif_pdb_name}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(root_path, dir_name)
        os.makedirs(sample_root_path, exist_ok=False)
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            overwrite=True,
            no_indexing=True,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="inference_base")
def main(cfg: Dict) -> None:
    load_dotenv()

    # 1. Use the cfg object directly (Hydra has already parsed CLI overrides)
    # These can be overridden via command line (e.g., job_id=1)
    config_name = cfg.get("config_name", "inference_base") 
    job_id = cfg.get("job_id", 0)
    njobs = cfg.get("gen_njobs", 1)

    # 2. Setup paths and seed
    root_path = setup(cfg, create_root=True, config_name=config_name, job_id=job_id)

    # 2b. Save the fully-resolved config (including CLI overrides) so that
    #     evaluate.py can pick up the actual generation parameters.
    resolved_cfg_path = os.path.join(root_path, "resolved_config.yaml")
    with open(resolved_cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    # 3. Check for existing results to avoid redundant work
    csv_filename = f"results_{config_name}_{job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting.")
        sys.exit(0)

    # 4. Check configuration validity and load the model
    # This now correctly uses any overrides passed to ckpt_path or ckpt_name
    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.dataset, cfg_gen.args)
    model = load_ckpt_n_configure_inference(cfg)
    model._generation_base_seed = cfg.seed  # Used by predict_step for per-batch seeding

    # 5. Handle dataset splitting and creation
    motif_cond = cfg_gen.args.get("motif_cond", False)
    cfg_gen = split_by_job(cfg_gen, job_id, njobs)

    if motif_cond or ("motif_task_name" in cfg_gen.dataset):
        motif_csv_path = os.path.join(
            root_path,
            f"{cfg_gen.dataset.get('motif_task_name', 'motif')}_{job_id}_motif_info.csv",
        )
        dataset = GenDataset(motif_csv_path=motif_csv_path, **cfg_gen.dataset)
    else:
        dataset = GenDataset(**cfg_gen.dataset)
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 6. Run prediction with Lightning Trainer
    trainer = L.Trainer(accelerator="gpu", devices=1)
    results = {}
    
    # Measure and save performance metrics
    with measure_performance(results, task_name=config_name) as metrics:
        predictions = trainer.predict(model, dataloader)

    save_performance_metrics(root_path, config_name, metrics)

    # 7. Save the generated PDB files
    if motif_cond or ("motif_task_name" in cfg_gen.dataset):
        save_motif_predictions(
            root_path,
            predictions,
            job_id=job_id,
            motif_pdb_name=cfg_gen.dataset.get("motif_task_name", None),
        )
        import shutil
        motif_csv = f"./{cfg_gen.dataset.get('motif_task_name', '')}_motif_info.csv"
        if os.path.exists(motif_csv):
            shutil.copy(motif_csv, root_path)
    else:
        save_predictions(
            root_path,
            predictions,
            job_id=job_id,
            chain_indexes=None,
            cath_codes=dataset.cath_codes,
        )


if __name__ == "__main__":
    main()
