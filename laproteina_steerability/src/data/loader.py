"""Dataset loader for cached protein latent representations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ProteinRecord:
    """A single protein's cached encoder output.

    Attributes
    ----------
    protein_id : str
        Unique identifier for the protein.
    latents : np.ndarray
        Latent vectors, shape ``[L, latent_dim]`` float32.
    ca_coords : np.ndarray or None
        Alpha-carbon coordinates in nm, shape ``[L, 3]`` float32.
        None if not loaded.
    length : int
        Number of residues.
    """

    protein_id: str
    latents: np.ndarray
    ca_coords: np.ndarray | None
    length: int


def _load_pt(path: Path, field_names: dict[str, str], load_coords: bool) -> ProteinRecord:
    """Load a single .pt file into a ProteinRecord.

    Parameters
    ----------
    path : Path
        Path to the .pt file.
    field_names : dict
        Mapping of semantic names to keys in the file.
    load_coords : bool
        Whether to load CA coordinates.

    Returns
    -------
    ProteinRecord
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both dict and object-with-attributes (e.g. PyG Data)
    def _get(key: str) -> Any:
        if isinstance(data, dict):
            return data[key]
        return getattr(data, key)

    latents = np.asarray(_get(field_names["latents"]), dtype=np.float32)
    length_key = field_names.get("length")
    length = int(_get(length_key)) if length_key else latents.shape[0]

    protein_id_raw = _get(field_names["protein_id"])
    protein_id = str(protein_id_raw)

    ca_coords = None
    if load_coords:
        raw_coords = np.asarray(_get(field_names["ca_coords"]), dtype=np.float32)
        # If coords are full-atom [L, N_atoms, 3], extract CA via ca_atom_index
        ca_idx = field_names.get("ca_atom_index")
        if ca_idx is not None and raw_coords.ndim == 3:
            ca_coords = raw_coords[:, int(ca_idx), :]  # [L, 3]
        elif raw_coords.ndim == 2:
            ca_coords = raw_coords  # already [L, 3]
        else:
            raise ValueError(
                f"ca_coords has shape {raw_coords.shape} but no ca_atom_index "
                f"specified to extract CA from full-atom coordinates"
            )

    return ProteinRecord(
        protein_id=protein_id,
        latents=latents,
        ca_coords=ca_coords,
        length=length,
    )


def _load_npz(path: Path, field_names: dict[str, str], load_coords: bool) -> ProteinRecord:
    """Load a single .npz file into a ProteinRecord.

    Parameters
    ----------
    path : Path
        Path to the .npz file.
    field_names : dict
        Mapping of semantic names to keys in the file.
    load_coords : bool
        Whether to load CA coordinates.

    Returns
    -------
    ProteinRecord
    """
    data = np.load(path, allow_pickle=True)

    latents = data[field_names["latents"]].astype(np.float32)
    length = int(data[field_names["length"]]) if field_names.get("length") else latents.shape[0]
    protein_id = str(data[field_names["protein_id"]])

    ca_coords = None
    if load_coords:
        ca_coords = data[field_names["ca_coords"]].astype(np.float32)

    return ProteinRecord(
        protein_id=protein_id,
        latents=latents,
        ca_coords=ca_coords,
        length=length,
    )


_LOADERS = {
    "pt": _load_pt,
    "npz": _load_npz,
}


def load_dataset(
    latent_dir: str | Path,
    file_format: str,
    field_names: dict[str, str],
    load_coords: bool = False,
    subsample: int | None = None,
    rng: np.random.Generator | None = None,
    length_range: tuple[int, int] | None = None,
) -> list[ProteinRecord]:
    """Load all protein records from a directory of cached encoder outputs.

    Parameters
    ----------
    latent_dir : str or Path
        Directory containing per-protein files.
    file_format : str
        File extension: "pt" or "npz".
    field_names : dict
        Mapping from semantic field names (latents, ca_coords, protein_id,
        length) to the actual keys used in the files. Also supports
        ``ca_atom_index`` (int) to extract CA from full-atom coords.
    load_coords : bool
        Whether to load CA coordinates (needed for Part 2's
        latent_plus_backbone variant).
    subsample : int or None
        If set, randomly subsample this many proteins (after length filter).
    rng : np.random.Generator or None
        Random generator for subsampling. Required if subsample is set.
    length_range : tuple[int, int] or None
        If set, only keep proteins with length in ``[min, max]`` inclusive.

    Returns
    -------
    list[ProteinRecord]
        Loaded protein records, sorted by protein_id for reproducibility.
    """
    latent_dir = Path(latent_dir)
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

    loader_fn = _LOADERS.get(file_format)
    if loader_fn is None:
        raise ValueError(f"Unsupported file format: {file_format}. Supported: {list(_LOADERS.keys())}")

    # Collect all files (may be in subdirectories for sharded layouts)
    logger.info("Scanning for .%s files in %s ...", file_format, latent_dir)
    files = sorted(latent_dir.rglob(f"*.{file_format}"))
    if not files:
        raise FileNotFoundError(f"No .{file_format} files found in {latent_dir}")

    logger.info("Found %d .%s files", len(files), file_format)

    # Pre-subsample files when dataset is large and subsample is requested,
    # to avoid loading hundreds of thousands of files we'll discard.
    # Over-sample by 2x to leave room for length filtering and load failures.
    if subsample is not None and subsample < len(files):
        if rng is None:
            raise ValueError("rng is required when subsample is set")
        pre_n = min(len(files), subsample * 2)
        indices = rng.choice(len(files), size=pre_n, replace=False)
        files = [files[i] for i in sorted(indices)]
        logger.info("Pre-subsampled to %d files for loading", len(files))

    from tqdm import tqdm
    records: list[ProteinRecord] = []
    n_failed = 0
    for path in tqdm(files, desc="Loading proteins", disable=len(files) < 100):
        try:
            rec = loader_fn(path, field_names, load_coords)
            records.append(rec)
        except Exception as e:
            n_failed += 1
            logger.warning("Failed to load %s: %s", path, e)

    if n_failed > 0:
        logger.warning("Failed to load %d / %d files", n_failed, len(files))

    # Length filter
    if length_range is not None:
        lo, hi = length_range
        before = len(records)
        records = [r for r in records if lo <= r.length <= hi]
        logger.info("Length filter [%d, %d]: kept %d / %d proteins",
                     lo, hi, len(records), before)

    # Subsample (after length filter so the filter doesn't shrink a pre-subsampled set)
    if subsample is not None and subsample < len(records):
        if rng is None:
            raise ValueError("rng is required when subsample is set")
        indices = rng.choice(len(records), size=subsample, replace=False)
        records = [records[i] for i in sorted(indices)]
        logger.info("Subsampled to %d proteins", len(records))

    records.sort(key=lambda r: r.protein_id)
    logger.info("Loaded %d proteins (%d total residues)",
                len(records), sum(r.length for r in records))
    return records


def pool_latents(records: list[ProteinRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Pool all per-residue latents into a single array.

    Parameters
    ----------
    records : list[ProteinRecord]
        Loaded protein records.

    Returns
    -------
    all_latents : np.ndarray
        Shape ``[N_total_residues, latent_dim]``.
    protein_ids : np.ndarray
        Shape ``[N_total_residues]``, dtype object. The protein_id for each
        residue row.
    """
    latent_chunks = []
    id_chunks = []
    for rec in records:
        latent_chunks.append(rec.latents)
        id_chunks.append(np.full(rec.length, rec.protein_id, dtype=object))
    return np.concatenate(latent_chunks, axis=0), np.concatenate(id_chunks, axis=0)


def make_synthetic_dataset(
    n_proteins: int = 10,
    length: int = 50,
    latent_dim: int = 8,
    rng: np.random.Generator | None = None,
    include_coords: bool = True,
) -> list[ProteinRecord]:
    """Create a synthetic dataset for smoke testing.

    Latents are drawn from a multivariate normal with a random covariance
    (not identity — tests should catch code that assumes independence).
    CA coordinates are random points in a 3D box.

    Parameters
    ----------
    n_proteins : int
        Number of proteins to generate.
    length : int
        Number of residues per protein (fixed for simplicity).
    latent_dim : int
        Dimensionality of latent vectors.
    rng : np.random.Generator or None
        Random generator. Defaults to Generator(PCG64(0)).
    include_coords : bool
        Whether to generate CA coordinates.

    Returns
    -------
    list[ProteinRecord]
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Random covariance for latents
    A = rng.standard_normal((latent_dim, latent_dim))
    cov = A @ A.T / latent_dim
    mean = rng.standard_normal(latent_dim) * 0.5

    # 1000 real PDB IDs sampled from processed_latents for realistic smoke tests
    _REAL_IDS = [
        '10px_1H', '180l_B', '1a7r_H', '1aar_A', '1ag6_A', '1aqd_G', '1ay9_A', '1b24_A',
        '1ca9_E', '1chp_F', '1cwl_A', '1dck_A', '1dsb_B', '1ejx_B', '1f51_G', '1f9e_K',
        '1fgz_A', '1fm0_E', '1fzb_D', '1gk4_E', '1id3_D', '1j89_D', '1jre_J', '1jts_X',
        '1jzk_B', '1kqs_N', '1ksh_B', '1kvw_A', '1kyy_B', '1l38_A', '1muq_E', '1n61_A',
        '1na1_D', '1nji_1', '1o6d_A', '1om9_B', '1otf_A', '1ows_A', '1pi2_A', '1pk6_C',
        '1pqo_A', '1pyb_A', '1q0m_F', '1q7y_O', '1q86_V', '1qhn_A', '1qmy_B', '1qsi_B',
        '1rhf_B', '1rhj_C', '1rlc_S', '1rnj_A', '1rsc_P', '1rxq_B', '1s1s_B', '1s56_B',
        '1ssy_A', '1t0n_E', '1t13_E', '1t3r_A', '1td7_A', '1twc_I', '1uli_B', '1vpz_A',
        '1vy5_BH', '1y13_C', '1y23_B', '1y45_B', '1y7c_C', '1yff_H', '1yj9_S', '1ytf_D',
        '1zgl_E', '1zlf_A', '2a4c_B', '2ajf_F', '2avv_E', '2bl0_A', '2bs0_B', '2c45_E',
        '2de7_D', '2doq_B', '2een_B', '2eiu_A', '2fmm_B', '2gia_A', '2gkv_E', '2h2y_C',
        '2hbf_B', '2hbz_B', '2hth_B', '2huz_A', '2hyi_H', '2iqy_A', '2jjz_D', '2nnd_A',
        '2ovi_A', '2p49_B', '2pii_A', '2qb0_C', '2qgb_B', '2qie_G', '2qqd_A', '2qzj_F',
        '2r50_C', '2r92_F', '2rtg_D', '2uwr_A', '2vn2_D', '2vwa_D', '2vxg_B', '2vxi_H',
        '2wg9_B', '2wgm_K', '2wqz_C', '2wvj_C', '2z3q_A', '2zl0_E', '2zt4_A', '3a4e_A',
        '3ab2_N', '3b3a_A', '3ba3_B', '3bat_D', '3bow_B', '3brv_B', '3cjt_N', '3clx_A',
        '3cq3_D', '3cxc_W', '3da7_D', '3dl3_F', '3dna_A', '3ds3_B', '3dxd_A', '3e6q_E',
        '3eql_O', '3eyt_C', '3f6z_A', '3f7g_A', '3fjf_B', '3frw_C', '3gkt_A', '3gsh_A',
        '3hty_J', '3i55_M', '3ise_A', '3j6j_C', '3j92_j', '3j9y_j', '3jai_RR', '3jan_SM',
        '3jvz_Y', '3ks8_D', '3l34_B', '3l5v_A', '3lrh_A', '3lrx_A', '3luq_B', '3mnn_G',
        '3n7a_N', '3omy_A', '3pe9_B', '3qd8_O', '3qf2_B', '3rbc_O', '3rej_D', '3rgd_I',
        '3rgu_A', '3s1i_C', '3s85_J', '3t72_N', '3v3l_A', '3vh7_C', '3w63_A', '3w98_F',
        '3wh2_A', '3whb_A', '3wjt_A', '3wuu_D', '3zgx_Z', '3zoe_B', '3zsr_B', '3zte_L',
        '3zw1_A', '4a3j_H', '4a6a_E', '4al2_B', '4avr_A', '4bmc_A', '4bts_A5', '4c3i_K',
        '4cpb_D', '4f3z_D', '4f7h_A', '4f88_H', '4f8c_D', '4fav_E', '4fbw_C', '4fq0_C',
        '4g88_C', '4gfa_A', '4gk7_N', '4ha2_B', '4hbm_C', '4hfv_A', '4hgo_A', '4hi9_B',
        '4hqb_E', '4hr9_A', '4iit_B', '4itw_F', '4iut_A', '4j3y_C', '4j8w_H', '4ji7_M',
        '4jiw_N', '4jj2_B', '4k6b_B', '4k7s_B', '4kf5_B', '4l71_QM', '4li2_B', '4lpz_A',
        '4lr3_D', '4lvi_A', '4m77_H', '4n8x_B', '4n9f_u', '4och_A', '4oy7_C', '4pa7_C',
        '4q7o_A', '4qh8_F', '4qie_F', '4rft_m', '4rp5_B', '4rsu_L', '4u24_BV', '4u3u_O0',
        '4u4n_C7', '4u4o_M3', '4u4q_n4', '4u4y_o7', '4u50_N6', '4u50_o2', '4u51_N8', '4u53_c9',
        '4u56_O5', '4u8u_g', '4ui1_D', '4ui4_A', '4uy8_X', '4v4q_BX', '4v51_DQ', '4v54_AU',
        '4v54_BM', '4v57_BR', '4v5e_BQ', '4v5p_DT', '4v5y_BM', '4v63_CI', '4v64_BN', '4v6c_BP',
        '4v7k_B6', '4v7r_CS', '4v7z_D5', '4v7z_DH', '4v8g_D2', '4v8u_CG', '4v9d_DH', '4v9n_AF',
        '4v9p_FL', '4v9q_BJ', '4v9q_DG', '4w9f_I', '4wf1_B3', '4woi_DT', '4wqf_AH', '4wu1_BA',
        '4wzd_62', '4wzo_L8', '4xks_I', '4xvn_C', '4y6t_B', '4ybb_BF', '4ym2_C', '4z8c_1i',
        '5aj4_BO', '5apo_o', '5b18_D', '5bpu_F', '5bpx_A', '5bsi_B', '5chz_A', '5clo_K',
        '5d6y_A', '5d8s_G', '5dfe_RR', '5dvw_C', '5dz5_B', '5e6g_B', '5e6i_N', '5e7d_C',
        '5eew_C', '5el5_J8', '5exv_F', '5f8k_1q', '5fdu_1T', '5fq6_J', '5g4v_C', '5ggr_Y',
        '5h3i_D', '5hz4_E', '5i4l_m9', '5ib7_7A', '5it8_CU', '5j30_XT', '5j4c_1V', '5j4d_GC',
        '5j4d_WA', '5j88_AU', '5jds_B', '5jns_A', '5jus_TB', '5kcr_10', '5ken_O', '5kk1_A',
        '5kpw_Q', '5kv7_A', '5ler_4A', '5lp0_B', '5lpk_B', '5lxq_A', '5lzs_P', '5lzu_LL',
        '5lzv_t', '5lzy_TT', '5m3l_J', '5m80_B', '5mbv_A', '5mhr_A', '5mmi_0', '5mq4_D',
        '5mq9_A', '5n06_A', '5ndk_15', '5o31_K', '5o6a_F', '5o7s_B', '5oai_A', '5oba_E',
        '5ocl_F', '5oct_F', '5pp8_A', '5ppi_A', '5pru_B', '5ptl_B', '5qj2_H', '5qm0_A',
        '5qmo_A', '5qmy_A', '5qnb_A', '5qot_A', '5tbw_DQ', '5tgm_o4', '5tvb_B', '5u0i_A',
        '5u0k_D', '5u7n_B', '5utz_E', '5v50_B', '5v6d_B', '5vge_B', '5vmo_A', '5vpf_C',
        '5vpo_YX', '5w5u_B', '5w8t_B', '5wit_1Y', '5xcv_B', '5xtb_G', '5y6p_z7', '5yrq_B',
        '5zgb_5', '5zz8_l', '6a52_A', '6a6j_C', '6bc0_F', '6bk8_n', '6bok_PB', '6bok_ZC',
        '6bra_A', '6buw_QH', '6bz8_QQ', '6c5f_C', '6cao_I', '6cfk_15', '6cfl_2j', '6cit_B',
        '6cpr_B', '6dhe_V', '6dzi_a', '6e5s_J', '6eci_L', '6elz_d', '6em3_M', '6et5_l',
        '6ex3_A', '6ff7_l', '6foi_W', '6ftx_O', '6ghb_A', '6gk4_B', '6gsk_AA', '6gwt_L',
        '6gzz_R3', '6h72_C', '6hax_C', '6hde_C', '6hif_h', '6i5a_A', '6ic3_G', '6id5_B',
        '6j0n_G', '6j2p_B', '6jfw_B', '6jlu_17', '6jy3_H', '6ke9_G', '6kgx_O4', '6kgx_U2',
        '6kgx_U8', '6kmv_T', '6kvo_B', '6lp5_D', '6lui_A', '6mcv_E', '6mx0_A', '6myi_D',
        '6n8n_o', '6n8n_r', '6n9e_1H', '6ncv_H', '6nep_B', '6nfc_B', '6nta_RU', '6o5x_B',
        '6ogi_g', '6oj0_M', '6olz_AT', '6ope_QQ', '6oqt_Y', '6ore_Z', '6osi_QF', '6oxi_RU',
        '6oxi_XN', '6oxt_A', '6p5n_AS', '6p8k_H', '6pxi_A', '6pyt_a', '6q8o_9', '6q95_P',
        '6qc4_f2', '6qc7_S7', '6qzl_B', '6qzp_Lb', '6r0n_A', '6r0z_W', '6r6p_U', '6r7m_A',
        '6rbd_Y', '6rd6_7', '6rdh_A', '6rdl_8', '6rdt_C', '6ro6_B', '6rvv_MA', '6rvv_PE',
        '6rvv_TC', '6rvv_WD', '6ryu_E', '6shb_A', '6sic_m', '6sic_q', '6sjj_F', '6st2_A',
        '6sv4_XM', '6t4q_LQ', '6t76_C', '6tb9_A2', '6tgc_C', '6tmf_X', '6tz5_KB', '6u54_B',
        '6ucv_M', '6uxp_E', '6uzz_B', '6v21_F', '6v8o_C', '6vid_D', '6vjs_J', '6w1t_U',
        '6wd2_n', '6wd4_i', '6wd5_f', '6wdl_n', '6wgh_B', '6wnw_f', '6wqd_A', '6wrs_J',
        '6wz9_C', '6x81_D', '6xhw_1T', '6xhw_2R', '6xo1_A', '6xqe_1H', '6xqe_1n', '6xqe_1Q',
        '6xu7_CM', '6xve_A', '6xyo_J', '6xyw_AC', '6yef_R', '6yfc_FU', '6yfg_ND', '6yfh_AJ',
        '6yfp_AZ', '6yfs_BB', '6yft_DA', '6yn1_c', '6yoa_A', '6ywv_0', '6z5y_B', '6z6j_LT',
        '6z6n_Lk', '6z6p_D', '6zb0_AAA', '6zfb_e', '6zkg_X', '6zkk_f', '6zn6_A', '6zqc_DQ',
        '6zsd_t2', '6zsg_XO', '6zt4_B', '6zu5_SF0', '6zvj_x', '7a01_k2', '7a4g_AF', '7a4h_EI',
        '7a5f_a3', '7ah9_3L', '7ah9_4Q', '7ajt_CF', '7b18_D', '7b5k_r', '7bgd_r', '7bl4_U',
        '7br8_Y', '7c00_A', '7ced_F', '7ckw_N', '7cpi_C', '7cub_D', '7cwu_F', '7d1l_C',
        '7da4_B', '7dkj_D', '7dlv_J', '7dvn_A', '7e4u_B', '7eq9_CA', '7eq9_P', '7ew7_C',
        '7ewa_B', '7ext_S4', '7ezx_S9', '7f0l_A', '7fq5_A', '7goc_B', '7h1b_A', '7hj1_A',
        '7hq7_A', '7hq7_B', '7hxx_C', '7it2_A', '7jfo_B', '7jqb_Z', '7jqm_1G', '7jvq_G',
        '7k4m_C', '7k51_q', '7k5j_W', '7kef_L', '7khw_w', '7lh5_AM', '7lx9_A', '7m0n_C',
        '7m2t_DD', '7mdz_WW', '7mri_C', '7n1h_K', '7n1p_SG', '7n2c_SE', '7n2u_LE', '7nar_T',
        '7nas_F', '7nav_V', '7nbu_R', '7nhq_H', '7nld_A', '7nqh_Be', '7nsh_HL', '7nyv_I',
        '7nzf_BBB', '7o0v_AD', '7o4j_L', '7o81_BV', '7obp_A', '7of0_3', '7ohv_Q', '7ohw_V',
        '7oii_a', '7oj0_E', '7p48_p', '7p5w_I', '7p81_H', '7p81_N', '7pe1_JF', '7pff_L',
        '7pka_A', '7ppb_A', '7psa_p', '7pzl_D', '7q0r_f', '7q47_B', '7q7v_A', '7qaj_C',
        '7qca_SF0', '7qhm_J', '7qi4_K', '7qi4_O', '7qi5_J', '7qi6_U', '7r72_c', '7r72_Z',
        '7rem_A', '7rqc_2R', '7s97_A', '7sc7_AW', '7st6_T', '7std_C', '7stk_D', '7syu_S',
        '7tdq_B', '7tjy_9', '7top_AL40', '7toq_AS05', '7tor_AL17', '7tut_H', '7tvk_A', '7u50_B',
        '7u5l_X', '7uck_PP', '7uio_AJ', '7unu_V', '7uph_i', '7upi_A', '7upk_L', '7uq2_E',
        '7uvx_X', '7uxz_BBB', '7v2k_U', '7v30_W', '7vbl_p', '7vmr_I', '7w1p_T', '7w2k_a',
        '7w30_B', '7w4j_k', '7w4l_d', '7wf3_J', '7wqu_B', '7wt8_H', '7wtt_P', '7x2t_D',
        '7xxf_K', '7y4l_R4', '7y5e_DE', '7y5e_NP', '7y5e_VM', '7y7a_a6', '7y7a_Bh', '7y7a_cA',
        '7y7a_EM', '7y7a_HS', '7y7a_jd', '7y7a_KS', '7y7a_Wi', '7zj3_J', '7zm7_S', '7zp8_k',
        '8a22_Aq', '8a57_N', '8abh_F', '8aex_I', '8alu_A', '8ap6_L', '8ap6_V2', '8apj_X1',
        '8b9z_K', '8ba0_j', '8bf7_J', '8bge_m', '8bgh_E', '8bhp_w', '8bpx_e', '8bpx_O',
        '8bqd_AT', '8bqs_Es', '8br8_Lb', '8btk_BR', '8btr_LX', '8bwm_B', '8c3a_BV', '8c3b_AAA',
        '8c7h_B', '8ccs_3', '8ce1_d', '8cmy_D', '8cro_N', '8cty_D', '8cvl_12', '8cvm_0',
        '8d11_A', '8dbp_I', '8dil_A', '8eg5_H', '8ej5_B', '8ekb_1t', '8eng_F', '8ert_C',
        '8esw_B8', '8evj_F', '8evq_BR', '8evs_Bb', '8evt_Bb', '8exh_Q', '8f4y_B', '8fl9_LK',
        '8fmw_AU', '8fop_I', '8g0c_2', '8g2z_3X', '8g5v_C', '8g6x_x', '8g7n_1', '8gbk_A',
        '8gcb_A', '8gn4_A', '8gzu_85', '8gzu_a2', '8h65_G', '8hag_A', '8hkx_S17E', '8hky_AS2P',
        '8hkz_L14P', '8ib6_f', '8ifc_z', '8ip8_na', '8iuj_4F', '8iyj_h1', '8j07_2Y', '8j07_5X',
        '8jas_T', '8jax_X', '8jdl_b', '8jjr_D', '8jla_G', '8k2d_SU', '8k82_Lp', '8k82_SW',
        '8kd1_C', '8khc_H', '8ofj_A', '8oi5_AP', '8oit_BS', '8p0y_W', '8p2h_Q', '8p4m_P',
        '8p4w_A', '8p5x_G', '8p8m_RO', '8p8u_RA', '8p8u_RT', '8pj2_D', '8pjo_A', '8pm5_G',
        '8pp5_B', '8ppk_Z', '8pw6_q1', '8q37_B', '8q47_c', '8q49_T', '8q4a_X', '8qj0_U',
        '8r59_KA', '8r5a_kA', '8r6a_B', '8r79_A', '8rgi_B', '8roy_D', '8rp7_CCC', '8rvq_V',
        '8rwg_3', '8s2t_C', '8s4h_A', '8seh_A', '8sn3_M', '8t1l_H', '8t3b_Ac', '8t4d_D',
        '8t8c_2G', '8te6_A', '8to2_G', '8toc_BY', '8tvh_D', '8ubi_D', '8uca_A2', '8ud6_1O',
        '8ud7_1Y', '8uej_DH', '8uep_1V', '8uer_1f', '8uql_Z', '8ut0_Lo', '8uuq_A', '8uyz_A',
        '8v3z_l', '8v50_D', '8vb4_N', '8vft_KK', '8vkv_x', '8vmu_A', '8vna_A', '8voo_n',
        '8vos_W', '8vrm_A', '8vv4_P', '8w31_C', '8wkk_K', '8wln_M', '8wqy_G', '8wv3_E',
        '8wwm_C', '8xch_K', '8xgc_R', '8xkw_B', '8xnp_S', '8xsz_LX', '8xx5_G', '8xz3_F',
        '8y5n_Y', '8y5t_Z', '8yfq_F', '8yfr_D', '8yj2_B', '8ylr_Se', '8yur_R', '8z5d_D',
        '8z5g_UA', '8z71_f', '8zfg_i', '8zfi_p', '8zgs_C', '8zlx_C', '8zn2_D', '8znl_C',
        '8zra_O', '8zv9_H', '8zxj_B', '8zyg_A', '9axt_Af', '9bct_F', '9c4d_D', '9chl_D',
        '9ci1_2', '9cn3_i', '9cqj_D', '9cqv_A', '9d0g_1n', '9d0g_1s', '9dfd_10', '9dfd_24',
        '9djw_x', '9dth_A', '9e1o_C', '9e3h_F', '9e5f_A', '9e6q_AF', '9eea_C', '9ejf_L',
        '9ev2_TY', '9eyp_B', '9f8i_B', '9f9s_CN', '9f9s_Mo', '9fe8_C', '9fov_A', '9fq7_4I',
        '9fq8_4P', '9fxf_A', '9fy3_k', '9g30_L', '9g3m_CB', '9g3m_NC', '9g5c_Q', '9g7k_H',
        '9gae_o', '9gaz_E', '9gha_K', '9gq6_B', '9gu6_D', '9h6d_B', '9ha6_W', '9hca_O',
        '9hei_H', '9hes_AP', '9hes_AY', '9hiy_H', '9hqv_Bl', '9htl_B', '9i1w_Lo', '9i3l_k',
        '9ihh_M', '9iju_H', '9ikf_Q', '9itp_H', '9j9h_b', '9j9i_w', '9jg7_C', '9jm5_D',
        '9jpm_0', '9jte_C', '9jvz_E', '9jy1_a', '9jzv_A', '9kbq_B', '9kmv_J', '9lle_N',
        '9ly8_Pe', '9mkb_On', '9mog_hq', '9mq4_x', '9mtt_2U', '9mvu_D', '9n5t_BH', '9n96_A',
        '9ndp_MM', '9ng8_B', '9njv_13', '9njv_su', '9nkl_20', '9nlq_sg', '9nn1_J', '9nrz_Y',
        '9nvk_B', '9o15_I', '9o3l_2r', '9o9v_J', '9olb_C', '9onz_E', '9oog_e', '9p9h_Lm',
        '9p9i_SH', '9q87_H', '9qb3_f', '9qex_F', '9qlo_LZ', '9qw2_A', '9r9p_SR', '9rb3_F',
        '9rpv_SV', '9rpz_B', '9spf_SN', '9sqt_H', '9u6l_Y', '9v1i_lW', '9v2k_Z', '9vc7_B',
        '9w74_E', '9wzd_D', '9y1y_A', '9y46_E', '9ypt_a', '9ypv_p', '9ypw_o', '9zeo_C',
    ]

    records = []
    for i in range(n_proteins):
        L = length + rng.integers(-10, 10)  # slight length variation
        L = max(L, 10)
        latents = rng.multivariate_normal(mean, cov, size=L).astype(np.float32)
        ca_coords = rng.standard_normal((L, 3)).astype(np.float32) * 0.5 if include_coords else None
        pid = _REAL_IDS[i % len(_REAL_IDS)] if i < len(_REAL_IDS) else f"synth_{i:04d}"
        records.append(ProteinRecord(
            protein_id=pid,
            latents=latents,
            ca_coords=ca_coords,
            length=L,
        ))
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = make_synthetic_dataset(n_proteins=10, length=50)
    print(f"Created {len(records)} synthetic proteins")
    all_lat, all_ids = pool_latents(records)
    print(f"Pooled latents shape: {all_lat.shape}")
    print(f"Unique protein IDs: {len(np.unique(all_ids))}")
    print(f"Latent mean: {all_lat.mean(0)}")
    print(f"Latent std:  {all_lat.std(0)}")
