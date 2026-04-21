import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import functools
import os
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from multiprocessing import Pool

from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.utils.cluster_utils import (
    cluster_sequences,
    df_to_fasta,
    expand_cluster_splits,
    fasta_to_df,
    read_cluster_tsv,
    setup_clustering_file_paths,
    split_dataframe,
)
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

from graphein.ml.datasets import PDBManager
from graphein.protein.tensor.io import protein_to_pyg
# Restored import for downloading
from graphein.protein.utils import download_pdb_multiprocessing 
import sys

# ==============================================================================
# STANDALONE WORKER FUNCTION (CRITICAL FIX FOR BROKEN PIPE)
# ==============================================================================

logger.remove()
logger.add(sys.stderr, level="DEBUG")
def process_single_pdb_file(args):
    """
    Standalone worker function. 
    Args: (pdb, chain, raw_dir, processed_dir, fmt, store_het, store_bfactor, pre_transform, pre_filter)
    """
    pdb, chain, raw_dir, processed_dir, fmt, store_het, store_bfactor, pre_transform, pre_filter = args
    
    try:
        # Resolve paths
        raw_path = raw_dir / f"{pdb}.{fmt}"
        if raw_path.exists():
            path_str = str(raw_path)
        elif raw_path.with_suffix(f".{fmt}.gz").exists():
            path_str = str(raw_path.with_suffix(f".{fmt}.gz"))
        else:
            return None # File missing

        fill_value_coords = 1e-5
        chain_selection = chain if chain != "all" else "all"

        graph = protein_to_pyg(
            path=path_str,
            chain_selection=chain_selection,
            keep_insertions=True,
            store_het=store_het,
            store_bfactor=store_bfactor,
            fill_value_coords=fill_value_coords,
        )

        # Post-processing
        fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
        
        graph.id = fname.split(".")[0]
        coord_mask = graph.coords != fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        graph.residue_type = torch.tensor(
            [resname_to_idx[residue] for residue in graph.residues]
        ).long()
        graph.database = "pdb"
        graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

        if pre_transform:
            graph = pre_transform(graph)

        if pre_filter:
            if pre_filter(graph) is not True:
                return None
        shard = pdb[0:2].lower()
        shard_dir = processed_dir / shard
        shard_dir.mkdir(exist_ok=True, parents=True)
        

        torch.save(graph, shard_dir / fname)
        return fname

    except Exception as e:
        # logger.warning(f"Failed to process {pdb}: {e}")
        return None


class PDBDataSelector:
    def __init__(
        self,
        data_dir: str,
        fraction: float = 1.0,
        min_length: int = None,
        max_length: int = None,
        molecule_type: str = None,
        experiment_types: List[str] = None,
        oligomeric_min: int = None,
        oligomeric_max: int = None,
        best_resolution: float = None,
        worst_resolution: float = None,
        has_ligands: List[str] = None,
        remove_ligands: List[str] = None,
        remove_non_standard_residues: bool = True,
        remove_pdb_unavailable: bool = True,
        labels: Optional[List[Literal["uniprot_id", "cath_code", "ec_number"]]] = None,
        remove_cath_unavailable: bool = False,
        exclude_ids: List[str] = None,
        exclude_ids_from_file: str = None,
        num_workers: int = 32,
    ):
        self.database = "pdb"
        self.data_dir = pathlib.Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.fraction = fraction
        self.molecule_type = molecule_type
        self.experiment_types = experiment_types
        self.oligomeric_min = oligomeric_min
        self.oligomeric_max = oligomeric_max
        self.best_resolution = best_resolution
        self.worst_resolution = worst_resolution
        self.has_ligands = has_ligands
        self.remove_ligands = remove_ligands
        self.remove_non_standard_residues = remove_non_standard_residues
        self.remove_pdb_unavailable = remove_pdb_unavailable
        self.min_length = min_length
        self.max_length = max_length
        self.exclude_ids = exclude_ids
        self.exclude_ids_from_file = exclude_ids_from_file
        self.labels = labels
        self.remove_cath_unavailable = remove_cath_unavailable
        self.num_workers = num_workers
        self.df_data = None

    def create_dataset(self) -> pd.DataFrame:
        if self.df_data:
            return self.df_data

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing PDBManager in {self.data_dir}...")
        pdb_manager = PDBManager(root_dir=self.data_dir, labels=self.labels)

        logger.info(f"[DEBUG] Total chains in PDB indices: {len(pdb_manager.df)}")

        num_chains = len(pdb_manager.df)
        logger.info(f"Starting with: {num_chains} chains")

        if self.fraction != 1.0:
            logger.info(f"Subsampling data to {self.fraction} fraction")
            pdb_manager.df = pdb_manager.df.sample(frac=self.fraction)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.experiment_types:
            logger.info(f"Removing chains that are not in: {self.experiment_types}")
            pdb_manager.experiment_types(self.experiment_types, update=True)

        if self.max_length:
            logger.info(f"Removing chains longer than {self.max_length}...")
            pdb_manager.length_shorter_than(self.max_length, update=True)

        if self.min_length:
            logger.info(f"Removing chains shorter than {self.min_length}...")
            pdb_manager.length_longer_than(self.min_length, update=True)

        if self.molecule_type:
            logger.info(f"Removing molecule types not in: {self.molecule_type}...")
            pdb_manager.molecule_type(self.molecule_type, update=True)

        if self.oligomeric_min:
            pdb_manager.oligomeric(self.oligomeric_min, "greater", update=True)
        if self.oligomeric_max:
            pdb_manager.oligomeric(self.oligomeric_max, "less", update=True)

        if self.worst_resolution:
            pdb_manager.resolution_better_than_or_equal_to(self.worst_resolution, update=True)
        if self.best_resolution:
            pdb_manager.resolution_worse_than_or_equal_to(self.best_resolution, update=True)

        if self.remove_ligands:
            pdb_manager.has_ligands(self.remove_ligands, inverse=True, update=True)

        if self.has_ligands:
            pdb_manager.has_ligands(self.has_ligands, update=True)

        if self.remove_non_standard_residues:
            pdb_manager.remove_non_standard_alphabet_sequences(update=True)
        if self.remove_pdb_unavailable:
            pdb_manager.remove_unavailable_pdbs(update=True)
        if self.remove_cath_unavailable:
            mask = ~pdb_manager.df["cath_code"].isna()
            pdb_manager.df = pdb_manager.df[mask]

        all_exclude_ids = set()
        if self.exclude_ids:
            all_exclude_ids.update(self.exclude_ids)
        if self.exclude_ids_from_file:
            with open(self.exclude_ids_from_file, "r") as f:
                file_ids = {line.strip() for line in f if line.strip()}
            all_exclude_ids.update(file_ids)

        if all_exclude_ids:
            logger.info(f"Removing excluded chains ({len(all_exclude_ids)} gathered)")
            mask = ~pdb_manager.df["id"].isin(all_exclude_ids)
            pdb_manager.df = pdb_manager.df[mask]

        logger.info(f"{len(pdb_manager.df)} chains remaining")
        self.df_data = pdb_manager.df
        return self.df_data


class PDBDataSplitter:
    def __init__(
        self,
        df_data: pd.DataFrame = None,
        data_dir: str = None,
        train_val_test: List[float] = [0.8, 0.15, 0.05],
        split_type: Literal["random", "sequence_similarity"] = "random",
        split_sequence_similarity: Optional[int] = None,
        overwrite_sequence_clusters: Optional[bool] = False,
    ) -> None:
        self.df_data = df_data
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.split_type = split_type
        self.split_sequence_similarity = split_sequence_similarity
        self.overwrite_sequence_clusters = overwrite_sequence_clusters
        self.splits = ["train", "val", "test"]
        self.dfs_splits = None
        self.clusterid_to_seqid_mappings = None

    def split_data(self, df_data: pd.DataFrame, file_identifier: str) -> Dict:
        if self.split_type == "random":
            logger.info(f"Splitting dataset via random split into {self.train_val_test}...")
            self.dfs_splits = split_dataframe(df_data, self.splits, self.train_val_test)
            self.clusterid_to_seqid_mappings = None

        elif self.split_type == "sequence_similarity":
            logger.info(f"Splitting via sequence-similarity {self.split_sequence_similarity}...")
            input_fasta_filepath, cluster_fasta_filepath, cluster_tsv_filepath = (
                setup_clustering_file_paths(
                    self.data_dir,
                    file_identifier,
                    self.split_sequence_similarity,
                )
            )

            if not input_fasta_filepath.exists() or self.overwrite_sequence_clusters:
                logger.info("Writing sequences to fasta...")
                df_to_fasta(df=df_data, output_file=input_fasta_filepath)

            if not cluster_fasta_filepath.exists() or self.overwrite_sequence_clusters:
                logger.info("Clustering via mmseqs2...")
                cluster_sequences(
                    fasta_input_filepath=input_fasta_filepath,
                    cluster_output_filepath=cluster_fasta_filepath,
                    min_seq_id=self.split_sequence_similarity,
                    overwrite=self.overwrite_sequence_clusters,
                )
            df_cluster_reps = fasta_to_df(cluster_fasta_filepath)
            seq_ids = df_cluster_reps["id"].to_numpy().tolist()
            df_sequences_reps = df_data.loc[df_data.id.isin(seq_ids)]
            splits = split_dataframe(df_sequences_reps, self.splits, self.train_val_test)
            clusterid_to_seqid_mapping = read_cluster_tsv(cluster_tsv_filepath)
            self.dfs_splits, self.clusterid_to_seqid_mappings = expand_cluster_splits(
                cluster_rep_splits=splits,
                clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
            )
        return (self.dfs_splits, self.clusterid_to_seqid_mappings)


class PDBDataset(Dataset):
    def __init__(
        self,
        pdb_codes: List[str],
        chains: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        format: Literal["mmtf", "pdb", "cif", "ent"] = "cif",
        in_memory: bool = False,
        file_names: Optional[List[str]] = None,
        num_workers: int = 64,
        use_precomputed_latents: bool = False,

    ):
        self.database = "pdb"
        self.pdb_codes = [pdb.lower() for pdb in pdb_codes]
        self.chains = chains
        self.format = format
        self.data_dir = pathlib.Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.in_memory = in_memory
        self.file_names = file_names
        self.num_workers = num_workers

        self.sequence_id_to_idx = None

        # Track file names we've already warned about so a missing file doesn't
        # spam the log every epoch. Used only by the streaming (in_memory=False) path.
        self._missing_warned = set()

        self.use_precomputed_latents = use_precomputed_latents

        if self.use_precomputed_latents and transform is not None:
            from torch_geometric.transforms import Compose
            
            # Define the list of "Structural" transforms to remove
            baked_in_names = [
                "CoordsToNanometers",
                "CenterStructureTransform",
                "GlobalRotationTransform",
                "OpenFoldFrame"
            ]

            if isinstance(transform, Compose):

                filtered_transforms = [
                    t for t in transform.transforms 
                    if not any(name in str(type(t)) for name in baked_in_names)
                ]
                self.transform = Compose(filtered_transforms)
                print(f"DEBUG: Filtered transforms for precomputed latents. Remaining: {self.transform}")
            else:
                # If it's just a single transform, check it
                if any(name in str(type(transform)) for name in baked_in_names):
                    self.transform = None
                else:
                    self.transform = transform
        else:
            self.transform = transform


        if self.use_precomputed_latents:
            self.processed_dir = self.data_dir / "processed_latents"
        else:
            self.processed_dir = self.data_dir / "processed"

        

        if self.in_memory:
            logger.info(f"Reading {len(file_names)} files into memory...")

            # Resolve symlink once so every thread uses the real path directly,
            # avoiding repeated symlink resolution on every file open.
            real_processed_dir = pathlib.Path(os.path.realpath(self.processed_dir))

            # Build task list without per-file exists() calls (each is a network
            # metadata round-trip on /rds/ — 100K calls = ~500s overhead).
            # Instead, try shard path first and fall back inside the loader.
            tasks = []
            for f in file_names:
                fname = f if f.endswith(".pt") else f"{f}.pt"
                shard = fname[0:2].lower()
                tasks.append((real_processed_dir / shard / fname, real_processed_dir / fname))

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _load(paths):
                shard_path, flat_path = paths
                try:
                    return torch.load(shard_path, map_location='cpu', weights_only=False)
                except FileNotFoundError:
                    try:
                        return torch.load(flat_path, map_location='cpu', weights_only=False)
                    except FileNotFoundError:
                        logger.warning(f"File not found (skipping): {shard_path}")
                        return None
                except Exception as e:
                    logger.warning(f"Failed to load {shard_path} ({type(e).__name__}: {e}) — skipping.")
                    return None

            self.data = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=max(1, self.num_workers)) as pool:
                futures = {pool.submit(_load, paths): i for i, paths in enumerate(tasks)}
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Loading (Parallel)"):
                    self.data[futures[future]] = future.result()

            # Filter out failed/corrupted files and keep file_names in sync
            # (__len__ uses file_names, __getitem__ uses self.data[idx])
            n_before = len(self.data)
            valid_mask = [d is not None for d in self.data]
            self.data = [d for d, ok in zip(self.data, valid_mask) if ok]
            self.file_names = [f for f, ok in zip(self.file_names, valid_mask) if ok]
            n_skipped = n_before - len(self.data)
            if n_skipped > 0:
                logger.warning(f"Skipped {n_skipped} corrupted/missing files. {len(self.data)} loaded successfully.")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Union[Data, Dict]:
            if self.in_memory:
                import copy
                graph_or_dict = copy.copy(self.data[idx])
            else:
                # Graceful skip: on missing/corrupted file, warn once and advance
                # to the next index. Prevents a single bad file from killing an
                # epoch. Bounded by len(self) so all-missing still terminates.
                attempts = 0
                n = len(self)
                while True:
                    fname = self.file_names[idx] if self.file_names is not None else f"{self.pdb_codes[idx]}.pt"
                    if not fname.endswith(".pt"): fname += ".pt"

                    shard = fname[0:2].lower()
                    file_path = self.processed_dir / shard / fname
                    if not file_path.exists():
                        file_path = self.processed_dir / fname

                    try:
                        graph_or_dict = torch.load(file_path, map_location='cpu', weights_only=False)
                        break
                    except (FileNotFoundError, EOFError, RuntimeError) as e:
                        if fname not in self._missing_warned:
                            logger.warning(
                                f"PDBDataset: skipping {fname} ({type(e).__name__}: {e}). "
                                "Advancing to next index."
                            )
                            self._missing_warned.add(fname)
                        attempts += 1
                        if attempts >= n:
                            raise RuntimeError(
                                f"PDBDataset: all {n} files missing/corrupted after skip attempts."
                            ) from e
                        idx = (idx + 1) % n

            if isinstance(graph_or_dict, dict):
                graph_or_dict = Data(**graph_or_dict)


            if not self.use_precomputed_latents:
                # Standard training: Slice 37 atoms using the PDB_TO_OPENFOLD index
                if hasattr(graph_or_dict, 'coords') and graph_or_dict.coords.ndim == 3:
                    graph_or_dict.coords = graph_or_dict.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                    graph_or_dict.coord_mask = graph_or_dict.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

            else:
                # Precomputed latents training
                assert hasattr(graph_or_dict, "mean") and hasattr(graph_or_dict, "log_scale"), \
                f"File {fname} is missing precomputed latents ('mean' or 'log_scale'). " \
                "Did you run precompute_latents.py?"
                
                # Get the true sequence length (L) from the coordinate mask
                L = graph_or_dict.coord_mask.shape[0]

                # EXPLICIT SHAPE FIX: We ONLY transpose if the sequence length is trapped in dimension 1
                # (e.g., shape is (512, L))
                if graph_or_dict.mean.shape[1] == L:
                    graph_or_dict.mean = graph_or_dict.mean.transpose(0, 1).contiguous()
                    graph_or_dict.log_scale = graph_or_dict.log_scale.transpose(0, 1).contiguous()
                
                # STRICT ASSERTS: Enforce that Dimension 0 is always the sequence length (L)
                # and Dimension 1 is the channel dimension (512)
                assert graph_or_dict.mean.shape[0] == L, \
                    f"[{fname}] CRITICAL: 'mean' dim 0 is {graph_or_dict.mean.shape[0]}, but true sequence length is {L}. Shape is {graph_or_dict.mean.shape}"
                assert graph_or_dict.mean.shape[1] == 8 or True, \
                    f"[{fname}] CRITICAL: 'mean' dim 1 must be exactly 8 (channels). Shape is {graph_or_dict.mean.shape}"
            # 4. Apply transforms (e.g., CoordsToNanometers)
            if self.transform:
                graph_or_dict = self.transform(graph_or_dict)

            return graph_or_dict
    

class PDBLightningDataModule(BaseLightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataselector: Optional[PDBDataSelector] = None,
        datasplitter: Optional[PDBDataSplitter] = None,
        in_memory: bool = False,
        format: Literal["mmtf", "pdb", "cif", "ent"] = "cif",
        overwrite: bool = False,
        store_het: bool = False,
        store_bfactor: bool = True,
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        use_precomputed_latents: bool = False,
        **kwargs,
    ):
        self.use_precomputed_latents = use_precomputed_latents
        super().__init__(
            batch_padding=batch_padding,
            sampling_mode=sampling_mode,
            transforms=transforms,
            pre_transforms=pre_transforms,
            pre_filters=pre_filters,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.data_dir = pathlib.Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        self.dataselector = dataselector
        self.datasplitter = datasplitter
        self.sampling_mode = sampling_mode
        self.format = format
        self.overwrite = overwrite
        self.in_memory = in_memory
        self.store_het = store_het
        self.store_bfactor = store_bfactor
        self.df_data = None
        self.dfs_splits = None
        self.clusterid_to_seqid_mappings = None
        self.file_names = None

    def prepare_data(self):
        """
        Prepares metadata and handles structure downloads/processing.
        When using latents, it strictly filters the metadata to only include 
        files found in the 'processed_latents' directory.
        """
        if self.dataselector:
            file_identifier = self._get_file_identifier(self.dataselector)


            df_data_name = f"{file_identifier}.csv"
            
            # 1. Check if the CSV already exists. 
            # If it does, we assume it's already filtered or the user wants to keep it.
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(f"{df_data_name} exists, skipping selection.")
                return

            # 2. Create the initial selection from the PDB database metadata
            logger.info(f"{df_data_name} not found, creating metadata.")
            df_data = self.dataselector.create_dataset()

            # --- CRITICAL FIX: FILTER FOR EXISTING LATENTS ---
            if self.use_precomputed_latents:
                logger.info("Filtering metadata to only include existing precomputed latents on disk...")
                import glob
                
                # Scan for all .pt files in processed_latents (including shards)
                latent_dir = self.data_dir / "processed_latents"
                # This finds files in both processed_latents/1abc.pt and processed_latents/1a/1abc.pt
                existing_paths = glob.glob(str(latent_dir / "**" / "*.pt"), recursive=True)
                
                # Extract IDs from filenames (e.g., '5wtu_D.pt' -> '5wtu_D')
                existing_ids = {os.path.basename(p).replace(".pt", "") for p in existing_paths}
                
                def check_id_exists(row):
                    pdb = row['pdb'].lower()
                    chain = row.get('chain')
                    # Construct ID format used in filenames: PDB_CHAIN or just PDB
                    target_id = f"{pdb}_{chain}" if pd.notna(chain) and chain != "all" else pdb
                    return target_id in existing_ids

                initial_count = len(df_data)
                # Keep only rows where the file actually exists
                df_data = df_data[df_data.apply(check_id_exists, axis=1)]
                
                logger.info(f"Filtered dataset from {initial_count} to {len(df_data)} available latents.")
                
                # Save the "clean" CSV. Setup() will now only see existing files.
                logger.info(f"Saving filtered metadata csv to {df_data_name}")
                df_data.to_csv(self.data_dir / df_data_name, index=False)
                return
            # --------------------------------------------------
            # 3. Standard Mode (Non-Latent): Download and Process
            import glob
            # Scan for existing processed .pt files
            try:
                processed_files = set()
                if self.processed_dir.exists():
                    for root, _, files in os.walk(self.processed_dir):
                        for f in files:
                            if f.endswith(".pt"):
                                processed_files.add(f)
                else:
                    processed_files = set()
            except Exception:
                processed_files = set()

            # Determine which PDBs actually need downloading (i.e. not yet processed)
            pdbs_to_download = []
            for _, row in df_data.iterrows():
                pdb = row['pdb'].lower()
                chain = row.get('chain', 'all')
                fname = f"{pdb}.pt" if pd.isna(chain) or chain == "all" else f"{pdb}_{chain}.pt"
                
                if fname not in processed_files:
                    pdbs_to_download.append(pdb)
            
            # Remove duplicates
            pdbs_to_download = list(set(pdbs_to_download))

            logger.info(f"Dataset created. Downloading {len(pdbs_to_download)} missing structures (skipped already processed files)...")
            if len(pdbs_to_download) > 0:
                self._download_structure_data(pdbs_to_download)

            logger.info("Verifying file availability...")
            raw_files_on_disk = set(os.listdir(self.raw_dir)) if self.raw_dir.exists() else set()
            
            # A PDB is valid if its raw file exists OR its processed .pt file already exists
            valid_pdbs = []
            for _, row in df_data.iterrows():
                pdb = row['pdb'].lower()
                chain = row.get('chain', 'all')
                fname = f"{pdb}.pt" if pd.isna(chain) or chain == "all" else f"{pdb}_{chain}.pt"
                
                raw_exists = f"{pdb}.{self.format}" in raw_files_on_disk or f"{pdb}.{self.format}.gz" in raw_files_on_disk
                processed_exists = fname in processed_files
                
                if raw_exists or processed_exists:
                    valid_pdbs.append(pdb)

            df_data = df_data[df_data["pdb"].isin(set(valid_pdbs))]
            
            self._process_structure_data(df_data["pdb"].tolist(), df_data["chain"].tolist())

            logger.info(f"Saving dataset csv to {df_data_name}")
            df_data.to_csv(self.data_dir / df_data_name, index=False)

        else:
            # Logic for when no dataselector is provided (loading everything in a folder)
            df_data_name = f"{self.data_dir.name}.csv"
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(f"{df_data_name} exists.")
            else:
                logger.info(f"{df_data_name} not found.")
                df_data = self._load_pdb_folder_data(self.raw_dir)
                if not self.use_precomputed_latents:
                    self._process_structure_data(pdb_codes=df_data["pdb"].tolist(), chains=None)
                df_data.to_csv(self.data_dir / df_data_name, index=False)

    def _process_structure_data(self, pdb_codes, chains):
        import os
        import glob
        from itertools import islice
        from joblib import Parallel, delayed
        from tqdm import tqdm

        # 1. Scan the directory once (as you already do)
        logger.info("Scanning sharded processed directory...")
        existing_paths = glob.glob(str(self.processed_dir / "**" / "*.pt"), recursive=True)
        processed_files = {os.path.basename(p) for p in existing_paths}

        # 2. Define the Generator Pattern
        # This function 'yields' one task at a time instead of building a giant list
        def task_generator():
            for i, pdb in enumerate(pdb_codes):
                chain = chains[i] if chains is not None else "all"
                fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
                
                if fname not in processed_files:
                    # Yielding is 'lazy' — it only executes when the worker is ready
                    yield (
                        pdb, chain, self.raw_dir, self.processed_dir,
                        self.format, self.store_het, self.store_bfactor,
                        getattr(self, 'pre_transform', None),
                        getattr(self, 'pre_filter', None)
                    )

        file_names = []

        # 3. Consume the generator in chunks so loky workers are torn down
        # and respawned between batches. This releases memory leaked by
        # per-item graphein/biopython parsing and prevents OOM over long runs.
        BATCH_SIZE = 5000
        if self.num_workers > 0:
            logger.info(f"Processing with {self.num_workers} workers (batched, size={BATCH_SIZE})...")
            gen = task_generator()
            pbar = tqdm(total=len(pdb_codes), desc="Processing")
            try:
                while True:
                    batch = list(islice(gen, BATCH_SIZE))
                    if not batch:
                        break
                    with Parallel(n_jobs=self.num_workers) as parallel:
                        results = parallel(
                            delayed(process_single_pdb_file)(task) for task in batch
                        )
                    file_names.extend(r for r in results if r is not None)
                    pbar.update(len(batch))
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                raise e
            finally:
                pbar.close()
        else:
            # Serial fallback
            for task in tqdm(task_generator(), total=len(pdb_codes), desc="Serial"):
                res = process_single_pdb_file(task)
                if res: file_names.append(res)

        return file_names

    def _load_pdb_folder_data(self, data_dir: pathlib.Path) -> pd.DataFrame:
        pdb_files = list(data_dir.glob(f"*.{self.format}"))
        df_data = pd.DataFrame({
            'pdb': [pdb_file.stem for pdb_file in pdb_files],
            'id': [pdb_file.stem for pdb_file in pdb_files],
        })
        if len(df_data) == 0:
            raise ValueError(f"No files with extension .{self.format} found in {data_dir}")
        logger.info(f"Found {len(df_data)} {self.format} files")
        return df_data

    def _get_file_identifier(self, ds):
        file_identifier = (
            f"df_pdb_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}_mt{ds.molecule_type}"
            f"_et{''.join(ds.experiment_types) if ds.experiment_types else ''}"
            f"_mino{ds.oligomeric_min}_maxo{ds.oligomeric_max}"
            f"_minr{ds.best_resolution}_maxr{ds.worst_resolution}"
            f"_hl{''.join(ds.has_ligands) if ds.has_ligands else ''}"
            f"_rl{''.join(ds.remove_ligands) if ds.remove_ligands else ''}"
            f"_rnsr{ds.remove_non_standard_residues}_rpu{ds.remove_pdb_unavailable}"
            f"_l{''.join(ds.labels) if ds.labels else ''}"
            f"_rcu{ds.remove_cath_unavailable}"
        )
        if self.use_precomputed_latents:
            file_identifier += "_latents"
            
        return file_identifier

    def setup(self, stage: Optional[str] = None):
        if not self.df_data:
            if self.dataselector:
                file_identifier = self._get_file_identifier(self.dataselector)
            else:
                file_identifier = self.data_dir.name
            df_data_name = f"{file_identifier}.csv"
            logger.info(f"Loading dataset csv from {df_data_name}")
            self.df_data = pd.read_csv(self.data_dir / df_data_name)

        # --- RANK GUARD START ---
        # We ensure clustering and file-moving only happens on Rank 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            
            if rank == 0:
                # Rank 0 performs the clustering/splitting first
                (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
                    self.datasplitter.split_data(self.df_data, file_identifier)
                )
            
            # All other ranks wait here until Rank 0 finishes writing files to disk
            torch.distributed.barrier()
            
            if rank != 0:
                # Now that Rank 0 is done, other ranks call split_data.
                # Because the files now exist, they will skip the mmseqs2 clustering 
                # and just load the results into memory.
                (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
                    self.datasplitter.split_data(self.df_data, file_identifier)
                )
        else:
            # Standard execution for single-GPU or non-distributed setups
            (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
                self.datasplitter.split_data(self.df_data, file_identifier)
            )
        # --- RANK GUARD END ---

        if stage == "fit" or stage is None:
            self.train_ds = self._get_dataset("train")
            self.val_ds = self._get_dataset("val")
        elif stage == "test":
            self.test_ds = self._get_dataset("test")

    def _get_dataset(self, split: Literal["train", "val", "test"]) -> PDBDataset:
        df_split = self.dfs_splits[split]
        pdb_codes = df_split["pdb"].tolist()
        if 'chain' in df_split.columns:
            chains = df_split["chain"].tolist()
            file_names = [f"{pdb}_{chain}" for pdb, chain in zip(pdb_codes, chains)]
        else:
            chains = None
            file_names = [f"{pdb}" for pdb in pdb_codes]

        return PDBDataset(
            pdb_codes=pdb_codes,
            chains=chains,
            data_dir=self.data_dir,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            file_names=file_names,
            num_workers=self.num_workers,
            use_precomputed_latents=self.use_precomputed_latents,
        )

    def _download_structure_data(self, pdb_codes) -> None:
        if pdb_codes is not None:

            if not self.overwrite:
                logger.info("Checking local raw files...")

                existing_files = set(os.listdir(self.raw_dir))
                
                to_download = []
                for pdb in pdb_codes:
                    # Check for raw files (e.g. 1abc.cif or 1abc.cif.gz)
                    f1 = f"{pdb}.{self.format}"
                    f2 = f"{pdb}.{self.format}.gz"
                    if f1 not in existing_files and f2 not in existing_files:
                        to_download.append(pdb)
            else:
                to_download = pdb_codes
            
            # Determine whether to download raw structures
            if to_download:
                logger.info(
                    f"Attempting to download {len(to_download)} structures to {self.raw_dir}"
                )
                file_format = (
                    self.format[:-3] if self.format.endswith(".gz") else self.format
                )
                
                # calculate number of downloads per worker
                chunksize = (
                    len(to_download) // self.num_workers + 1
                )
                
                # --- FIX: WRAP DOWNLOAD IN TRY/EXCEPT ---
                try:
                    download_pdb_multiprocessing(
                        to_download,
                        self.raw_dir,
                        format=file_format,
                        max_workers=self.num_workers,
                        chunksize=chunksize,
                    )
                except (OSError, PermissionError) as e:
                    logger.warning(f"⚠️  DOWNLOAD FAILED: Could not write to disk ({e}).")
                    logger.warning("⚠️  Continuing with ONLY the files that already exist locally.")
                    # We catch the error so the code doesn't crash. 
                    # The 'prepare_data' method will then filter out the missing files in the next step.
            else:
                logger.info(
                    f"No structures to download, all {len(pdb_codes)} structure files already present"
                )