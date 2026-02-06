import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import functools

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


# ==============================================================================
# STANDALONE WORKER FUNCTION (CRITICAL FIX FOR BROKEN PIPE)
# ==============================================================================
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

        torch.save(graph, processed_dir / fname)
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
        self.transform = transform
        self.sequence_id_to_idx = None

        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [torch.load(self.processed_dir / f) for f in tqdm(file_names)]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Data:
        if self.in_memory:
            graph = self.data[idx]
        else:
            if self.file_names is not None:
                fname = f"{self.file_names[idx]}.pt"
            elif self.chains is not None:
                fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
            else:
                fname = f"{self.pdb_codes[idx]}.pt"

            graph = torch.load(self.data_dir / "processed" / fname, weights_only=False)

        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

        if self.transform:
            graph = self.transform(graph)

        return graph


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
        **kwargs,
    ):
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
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
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
        if self.dataselector:
            file_identifier = self._get_file_identifier(self.dataselector)
            df_data_name = f"{file_identifier}.csv"
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(f"{df_data_name} exists, skipping selection.")
            else:
                logger.info(f"{df_data_name} not found, creating dataset.")
                df_data = self.dataselector.create_dataset()

                # --- STEP 1: DOWNLOAD MISSING FILES ---
                logger.info(f"Dataset created with {len(df_data)} entries. Downloading structures...")
                self._download_structure_data(df_data["pdb"].tolist())

                # --- STEP 2: CHECK FILE AVAILABILITY (Robustness) ---
                logger.info("Verifying file availability...")
                existing_pdbs = []
                unique_pdbs = df_data["pdb"].unique()
                
                # Check which files actually exist (in case download failed for some)
                for pdb in tqdm(unique_pdbs, desc="Verifying files"):
                    if (self.raw_dir / f"{pdb}.{self.format}").exists() or \
                       (self.raw_dir / f"{pdb}.{self.format}.gz").exists():
                        existing_pdbs.append(pdb)
                
                existing_pdbs_set = set(existing_pdbs)
                missing_count = len(unique_pdbs) - len(existing_pdbs)
                
                if missing_count > 0:
                    logger.warning(f"⚠️  Skipping {missing_count} PDBs that failed to download or are missing.")
                    df_data = df_data[df_data["pdb"].isin(existing_pdbs_set)]

                logger.info(f"Dataset filtered to {len(df_data)} available entries.")
                
                self._process_structure_data(
                    df_data["pdb"].tolist(), df_data["chain"].tolist()
                )

                logger.info(f"Saving dataset csv to {df_data_name}")
                df_data.to_csv(self.data_dir / df_data_name, index=False)

        else:
            df_data_name = f"{self.data_dir.name}.csv"
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(f"{df_data_name} exists.")
            else:
                logger.info(f"{df_data_name} not found.")
                df_data = self._load_pdb_folder_data(self.raw_dir)
                self._process_structure_data(
                    pdb_codes=df_data["pdb"].tolist(),
                    chains=None,
                )
                logger.info(f"Saving dataset csv to {df_data_name}")
                df_data.to_csv(self.data_dir / df_data_name, index=False)

    def _process_structure_data(self, pdb_codes, chains):
        """Process raw data. Supports serial execution if num_workers=0."""
        tasks = []
        for i, pdb in enumerate(pdb_codes):
            chain = chains[i] if chains is not None else "all"
            
            fname = f"{pdb}.pt" if chain == "all" else f"{pdb}_{chain}.pt"
            if (self.processed_dir / fname).exists():
                continue
                
            tasks.append((
                pdb,
                chain,
                self.raw_dir,
                self.processed_dir,
                self.format,
                self.store_het,
                self.store_bfactor,
                getattr(self, 'pre_transform', None),
                getattr(self, 'pre_filter', None)
            ))

        file_names = []
        if len(tasks) > 0:
            if self.num_workers > 0:
                # --- PARALLEL MODE (Use Pool) ---
                n_workers = self.num_workers
                logger.info(f"Processing {len(tasks)} files with {n_workers} workers...")
                
                try:
                    with Pool(processes=n_workers) as pool:
                        results = list(tqdm(
                            pool.imap(process_single_pdb_file, tasks), 
                            total=len(tasks),
                            desc="Processing (Parallel)",
                            unit="file"
                        ))
                    file_names = [r for r in results if r is not None]
                except Exception as e:
                    logger.error("Parallel processing crashed!")
                    raise e
            else:
                # --- SERIAL MODE (Main Process) ---
                logger.info(f"Processing {len(tasks)} files in SERIAL mode...")
                for task in tqdm(tasks, desc="Processing (Serial)", unit="file"):
                    res = process_single_pdb_file(task)
                    if res is not None:
                        file_names.append(res)
        
        logger.info("Completed processing.")
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

        (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
            self.datasplitter.split_data(self.df_data, file_identifier)
        )

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
        )

    def _download_structure_data(self, pdb_codes) -> None:
        if pdb_codes is not None:
            to_download = (
                pdb_codes
                if self.overwrite
                else [
                    pdb
                    for pdb in pdb_codes
                    if not (
                        (self.raw_dir / f"{pdb}.{self.format}").exists()
                        or (self.raw_dir / f"{pdb}.{self.format}.gz").exists()
                    )
                ]
            )
            to_download = list(set(to_download))
            
            # Determine whether to download raw structures
            if to_download:
                logger.info(
                    f"Attempting to download {len(to_download)} structures to {self.processed_dir}"
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