import os
import sys
import ssl
import wget
import time
ssl._create_default_https_context = ssl._create_unverified_context
root = os.path.abspath(".")
sys.path.insert(0, root)  # Adds project's root directory
# --- MONKEYPATCH FIX FOR BROKEN CATH URL ---
import graphein.ml.datasets.pdb_data
NEW_CATH_URL = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"

# 2. Update the global variable in the library (so other parts see it)
graphein.ml.datasets.pdb_data.CATH_ID_CATH_CODE_URL = NEW_CATH_URL

# 3. Define the download function that renames the file
def fixed_download_cath(self):
    # Ensure we use the correct URL (overwriting any stale instance variable)
    self.cath_id_cath_code_url = NEW_CATH_URL
    
    # We force the filename to be 'cath-b-newest-all.txt'
    # This is the "Rename" step. Even though the URL ends in 'cath-domain-list.txt',
    # we save it as 'cath-b-newest-all.txt' so the rest of the code works.
    target_path = self.root_dir / "cath-b-newest-all.txt"
    
    print(f"[Patch] Checking for file: {target_path}")
    print(f"[Patch] URL source: {self.cath_id_cath_code_url}")

    if not target_path.exists():
        print(f"[Patch] File not found. Downloading...")
        try:
            # This is the line you want to contribute to the library
            wget.download(self.cath_id_cath_code_url, out=str(target_path))
            print("\n[Patch] Download complete!")
        except Exception as e:
            print(f"\n[Patch] Python download failed: {e}")
            print("[Patch] Attempting fallback to system curl...")
            import subprocess
            # Fallback for your cluster (keep this local, don't put in PR)
            subprocess.run(["curl", "-o", str(target_path), self.cath_id_cath_code_url], check=True)
    else:
        print("[Patch] File already exists. Skipping download.")


# --- ADD THIS TO THE MONKEYPATCH SECTION IN proteinfoundation/train.py ---

def robust_parse_ligand_map(self):
    """
    Robust version of Graphein's ligand map parser that skips empty or malformed lines.
    """
    path = self.root_dir / "ligand_map.txt"
    if not path.exists():
        # Optional: You could try to download it here, but PDBManager usually handles it.
        # If it's missing, just return empty to avoid the crash.
        return {}
    
    ligand_map = {}
    with open(path, "r") as f:
        for line in f:
            params = line.strip().split("\t")
            # The original code fails here if params is empty or has only 1 element
            if len(params) > 1:
                ligand_map[params[0]] = params[1:]
    return ligand_map

# Apply the patch

graphein.ml.datasets.pdb_data.PDBManager._parse_ligand_map = robust_parse_ligand_map
# 4. Apply the monkeypatch
graphein.ml.datasets.pdb_data.PDBManager._download_cath_id_cath_code_map = fixed_download_cath
import json
import pickle
from pathlib import Path

import hydra
import lightning as L
import loralib as lora
import torch
import torch.multiprocessing
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.fold_utils import (
    transform_global_percentage_to_mask_dropout,
)
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
)


@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def create_dir(ckpt_path_store, parents=True, exist_ok=True):
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)



def load_cfg_exp(config_name, single_gpu, is_cluster_run):
    """
    Loads experiment config.
    """
    config_path = "../configs/experiment_config"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=config_name)
        if not is_cluster_run or single_gpu:
            # Rewrite number of GPUs and nodes for local runs or if single flag is used
            cfg_exp.hardware.ngpus_per_node_ = 1
            cfg_exp.hardware.nnodes_ = 1
            cfg_exp.run_name_ = cfg_exp.run_name_ + "_local"
        log_info(f"Exp config {cfg_exp}")

    cfg_exp = handle_cath_conditioning(cfg_exp)
    return cfg_exp


def handle_cath_conditioning(cfg_exp):
    """
    Setups dropping ration for cath conditioning, based on global percentage.
    """
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(
            cfg_exp.training.fold_label_sample_ratio
        )
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )
    return cfg_exp


def get_run_dirs(cfg_exp):
    """
    Get root directory for run and directory to store checkpoints.
    Resume is opt-in: set environment variable RESUME=1 (or =true) to scan for
    an existing last.ckpt and resume from there. Default is a fresh run.
    """
    run_name = cfg_exp.run_name_
    log_info(f"Job name: {run_name}")
    store_base = os.path.join(".", "store", run_name)

    resume_enabled = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")

    root_run = None
    if resume_enabled and os.path.isdir(store_base):
        log_info("RESUME=1 set — scanning for existing checkpoint to resume from")
        subdirs = sorted(
            (d for d in os.listdir(store_base) if os.path.isdir(os.path.join(store_base, d))),
            reverse=True,
        )
        for subdir in subdirs:
            candidate_root = os.path.join(store_base, subdir)
            candidate_ckpts = os.path.join(candidate_root, "checkpoints")
            if fetch_last_ckpt(candidate_ckpts) is not None:
                root_run = candidate_root
                log_info(f"Resuming from existing run directory: {root_run}")
                break
    elif not resume_enabled:
        log_info("RESUME not set — starting a fresh run (set RESUME=1 to auto-resume)")

    if root_run is None:
        root_run = os.path.join(store_base, f"{int(time.time())}")

    log_info(f"Root run: {root_run}")
    ckpt_path_store = os.path.join(root_run, "checkpoints")
    log_info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def initialize_callbacks(cfg_exp):
    """
    Initializes general training callbacks.
    """
    callbacks = [SeedCallback()]

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    callbacks.append(LogEpochTimeCallback())
    callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))
    return callbacks


def get_training_precision(cfg_exp, is_cluster_run):
    """
    Gets and sets correct training precision.
    """
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        if is_cluster_run:
            precision = "bf16-mixed"
        else:
            precision = "16"
    else:
        torch.set_float32_matmul_precision("high")
    return precision


def load_data_module(cfg_exp, is_cluster_run):
    """
    Loads data config file and creates corresponding datamodule.
    """
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(
        f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}"
    )
    cfg_data = cfg_exp.dataset

    cfg_data.datamodule.num_workers = num_cpus  # Overwrite number of cpus
    if cfg_data.get("exclude_id_pkl_path") is not None:
        with open(cfg_data.exclude_id_pkl_path, "rb") as fin:
            exclude_ids = pickle.load(fin)
        if cfg_data.datamodule.dataselector.exclude_ids is not None:
            cfg_data.datamodule.dataselector.exclude_ids += exclude_ids
        else:
            cfg_data.datamodule.dataselector.exclude_ids = exclude_ids
    if not is_cluster_run:
        cfg_data["datamodule"]["batch_size"] = 2
        log_info("Local run, setting batch size to 2")
    log_info(f"Data config {cfg_data}")

    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    return cfg_data, datamodule


def _log_resume_banner(last_ckpt_path, ckpt_path_store):
    """Human-readable banner for easy grep/scan in slurm logs when resuming.

    Note: the 'wall clock since run dir created' number includes queue/startup
    time from every prior SLURM job in this run dir and idle time between jobs;
    it is NOT the actual training wall time. For accurate training time, read
    global_step from the checkpoint after Lightning loads it.
    """
    from datetime import datetime
    run_dir_name = os.path.basename(os.path.dirname(ckpt_path_store))
    try:
        start_ts = int(run_dir_name)
        start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        start_ts, start_str = None, None
    try:
        save_ts = os.path.getmtime(last_ckpt_path)
        save_str = datetime.fromtimestamp(save_ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        save_ts, save_str = None, None
    dur_h = (save_ts - start_ts) / 3600.0 if (start_ts and save_ts) else None
    job_id = os.environ.get("SLURM_JOB_ID", "")
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    log_info("=" * 70)
    log_info("RESUMING TRAINING FROM CHECKPOINT")
    log_info(f"  checkpoint:            {last_ckpt_path}")
    if start_str:
        log_info(f"  run dir created:       {start_str} (run dir: {run_dir_name})")
    else:
        log_info(f"  previous run dir:      {run_dir_name}")
    if save_str:
        log_info(f"  last checkpoint saved: {save_str}")
    if dur_h is not None:
        log_info(f"  wall clock since dir:  {dur_h:.2f} h (INCLUDES startup+queue+idle, not just training)")
    if job_id or job_name:
        log_info(f"  resuming under SLURM:  job={job_id} name={job_name}")
    log_info("=" * 70)


def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store):
    """
    Loads the model and the checkpoint to start training from. This could be just a set
    of parameters (`pretrain_ckpt_path`) or resuming training (`last`). It also handles
    LoRA layers if requested.
    """
    model = Proteina(cfg_exp)

    # get last ckpt if needs to resume training from there
    last_ckpt_name = fetch_last_ckpt(ckpt_path_store)
    if last_ckpt_name is not None:
        last_ckpt_path = os.path.join(ckpt_path_store, last_ckpt_name)
    else:
        last_ckpt_path = None
    log_info(f"Last checkpoint: {last_ckpt_path}")
    if last_ckpt_path is not None:
        _log_resume_banner(last_ckpt_path, ckpt_path_store)

    # If LoRA is turned on, replace Linear with LoRA layers
    # Note: We do not use LoRA in the La-Proteina paper.
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model, cfg_exp.lora.r, cfg_exp.lora.lora_alpha, cfg_exp.lora.lora_dropout
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # If not resuming from `last` ckpt training set seed
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    return model, last_ckpt_path


def setup_ckpt(cfg_exp, ckpt_path_store):
    """
    Created checkpointing callbacks and creates directory to store checkpoints.
    """
    args_ckpt_last = {
        "dirpath": ckpt_path_store,
        "save_weights_only": False,
        "filename": "ignore",
        "every_n_train_steps": cfg_exp.log.last_ckpt_every_n_steps,
        "save_last": True,
    }
    args_ckpt = {
        "dirpath": ckpt_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "chk_{epoch:08d}_{step:012d}",
        "every_n_train_steps": cfg_exp.log.checkpoint_every_n_steps,
        "monitor": "train_loss",
        "save_top_k": 10000,
        "mode": "min",
    }
    args_ckpt_best = {
        "dirpath": ckpt_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "best_val_{epoch:08d}_{step:012d}",
        "monitor": "validation_loss/loss_epoch",
        "mode": "min",
        "save_top_k": 3,
        "auto_insert_metric_name": False,
    }
    checkpoint_callback = EmaModelCheckpoint(**args_ckpt)
    checkpoint_callback_last = EmaModelCheckpoint(**args_ckpt_last)
    checkpoint_callback_best = EmaModelCheckpoint(**args_ckpt_best)

    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    return [checkpoint_callback, checkpoint_callback_last, checkpoint_callback_best]


@rank_zero_only
def store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger):
    """
    Stores config files locally and logs them to wandb run.
    """

    def store_n_log_config(cfg, cfg_path, wandb_logger):
        # Config dump is metadata — never fatal. On Cambridge HPC the RDS mount
        # can transiently go read-only on individual compute nodes; we don't
        # want to kill a training job because we couldn't write a JSON snapshot.
        try:
            with open(cfg_path, "w") as f:
                cfg_aux = OmegaConf.to_container(cfg, resolve=True)
                json.dump(cfg_aux, f, indent=4, sort_keys=True)
        except OSError as e:
            logger.warning(f"Could not write config snapshot to {cfg_path}: {e}. Continuing.")
            return

        if wandb_logger is not None:
            try:
                artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
                artifact.add_file(cfg_path)
                wandb_logger.experiment.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Could not log config artifact to wandb: {e}. Continuing.")

    cfg_exp_file = os.path.join(ckpt_path_store, f"exp_config_{run_name}.json")
    cfg_data_file = os.path.join(ckpt_path_store, f"data_config_{run_name}.json")

    store_n_log_config(cfg_exp, cfg_exp_file, wandb_logger)
    store_n_log_config(cfg_data, cfg_data_file, wandb_logger)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="training_local_latents",  # overridden by --config-name from submit script
)
def main(cfg_exp) -> None:
    load_dotenv()

    is_cluster_run = "SLURM_JOB_ID" in os.environ
    nolog = cfg_exp.get(
        "nolog", False
    )  # To use do `python proteinfoundation/train.py +nolog=true`
    single = cfg_exp.get("single", False)
    show_prog_bar = True
    
    if not is_cluster_run or single:
        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
        is_cluster_run = False # Treat as local for plugin purposes
        log_info("Mode: LOCAL/SINGLE-GPU")

    log_info(f"Exp config {cfg_exp}")

    run_name, root_run, ckpt_path_store = get_run_dirs(cfg_exp)
    callbacks = initialize_callbacks(cfg_exp)
    cfg_data, datamodule = load_data_module(cfg_exp, is_cluster_run)

    # Create model, warm-up or last ckpt
    model, resume_ckpt_path = get_model_n_ckpt_resume(cfg_exp, ckpt_path_store)

    # logger
    resume_enabled = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")

    wandb_logger = None
    if cfg_exp.log.log_wandb and not nolog:
        # resume="never" forces wandb to create a new run rather than resuming a
        # prior one with matching name/id. Opt in by setting RESUME=1.
        wandb_logger = WandbLogger(
            project=cfg_exp.log.wandb_project,
            name=run_name,
            resume="allow" if resume_enabled else "never",
        )

    # checkpoints
    if cfg_exp.log.checkpoint and not nolog:
        ckpt_callbacks = setup_ckpt(cfg_exp, ckpt_path_store)
        callbacks += ckpt_callbacks
        store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger)

    # Train. SLURM auto-requeue also gated behind RESUME=1 so that an interrupted
    # job doesn't silently resume from checkpoint.
    plugins = [SLURMEnvironment(auto_requeue=resume_enabled)] if is_cluster_run else []
    show_prog_bar = show_prog_bar or not is_cluster_run
    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg_exp.log.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        limit_val_batches=100,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=1,
        precision=get_training_precision(cfg_exp, is_cluster_run),
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)
    # If resume_ckpt_path is None then it creates a new optimizer


if __name__ == "__main__":
    main()
