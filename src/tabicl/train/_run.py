from __future__ import annotations

import os
import random
import json
import timeit
import warnings
import functools
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import math
import numpy as np
import psutil

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

from tabicl._model.tabicl import TabICL
from tabicl._model.attention import set_flash_attn3_enabled
from tabicl.prior._dataset import PriorDataset
from tabicl.prior._genload import LoadPriorDataset, make_prior_dataloader, seed_worker
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._optim import get_scheduler
from tabicl.train._muon import Muon
from tabicl.train._train_config import build_parser
from tabicl.train._checkpoint_io import atomic_torch_save
from tabicl.train._identity_rng import SAMPLER_VERSION, TrainerIdentityRNG
from tabicl.train._provenance import (
    FORMAL_PROTOCOL_CONFIG_FIELDS,
    ParentTrust,
    build_checkpoint_provenance,
    build_optimizer_protocol,
    load_source_manifest,
    make_manifest,
    runtime_environment_manifest,
    validate_parent_trust,
)
from tabicl.train._rng_state import (
    gather_all_rank_rng_state,
    restore_all_rank_rng_state,
    select_rank_rng_state,
    validate_full_resume_checkpoint,
)

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_identity_rng()
        self.configure_wandb()
        # W&B initialization is outside the model-initialization RNG boundary.
        # Re-seed here so all treatment arms start from matched parameters.
        self.seed()
        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.configure_formal_provenance()
        self._loaded_full_resume = False
        self.load_checkpoint()
        if not self._loaded_full_resume:
            self.seed()

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(
                original_batch_size / self.ddp_world_size
            )

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_identity_rng(self):
        """Create the rank-local Temporary Identity stream."""
        identity_seed = getattr(self.config, "identity_rng_seed", None)
        if identity_seed is None:
            identity_seed = self.config.torch_seed
        self.config.identity_rng_seed = identity_seed
        self.config.identity_sampler_version = (
            SAMPLER_VERSION if self.config.row_identity_mode == "temporary" else None
        )
        self.identity_rng = TrainerIdentityRNG(
            identity_mode=self.config.row_identity_mode,
            base_seed=identity_seed,
            rank=self.ddp_rank,
            world_size=self.ddp_world_size,
        )

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""

        # Determine the task type. regression_method=None trains for classification;
        # "quantile" trains for quantile regression (max_classes=0) with a pinball loss.
        self.regression = self.config.regression_method is not None
        if self.regression and self.config.regression_method != "quantile":
            raise NotImplementedError(
                f"regression_method='{self.config.regression_method}' is not supported. "
                "Only None (classification) and 'quantile' (pinball regression) are available."
            )
        if self.regression and self.config.num_quantiles <= 0:
            raise ValueError(
                "For quantile regression, num_quantiles must be greater than 0."
            )

        # Map the private-style --norm_type to the public model's bias_free_ln flag.
        if self.config.norm_type == "default":
            bias_free_ln = False
        elif self.config.norm_type == "layernorm_nobias":
            bias_free_ln = True
        else:
            raise NotImplementedError(
                f"norm_type='{self.config.norm_type}' is not supported. "
                "Use 'default' or 'layernorm_nobias'."
            )

        # FlashAttention-3 runs attention in fp16; the v2 recipe enables it only for stages 2 & 3.
        set_flash_attn3_enabled(self.config.use_flash_attn3)

        self.model_config = {
            "max_classes": 0 if self.regression else self.config.max_classes,
            "num_quantiles": self.config.num_quantiles,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "col_affine": self.config.col_affine,
            "col_feature_group": self.config.col_feature_group,
            "col_feature_group_size": self.config.col_feature_group_size,
            "col_target_aware": self.config.col_target_aware,
            "col_ssmax": self.config.ssmax_type if self.config.col_ssmax else False,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "row_rope_interleaved": self.config.row_rope_interleaved,
            "row_identity_mode": self.config.row_identity_mode,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "icl_ssmax": self.config.ssmax_type if self.config.icl_ssmax else False,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "bias_free_ln": bias_free_ln,
            "zero_init": self.config.zero_init,
            "recompute": self.config.recompute,
        }

        model = TabICL(**self.model_config)
        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(
                model, device_ids=[self.ddp_local_rank], broadcast_buffers=False
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        """Set up a tabular dataset generator for synthetic data during training."""

        self.prior_cursor = 0
        if self.config.prior_dir is None:
            # Generate prior data on the fly
            prior_config = PriorConfig.from_args(self.config)
            dataset = PriorDataset(
                regression=self.regression,
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                log_n_features=self.config.log_n_features,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                config=prior_config,  # graph_scm prior options
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism; the DataLoader parallelizes across batches
            )
            prior_schema = {
                "schema_version": 1,
                "regression": self.regression,
                "batch_size": self.config.batch_size,
                "batch_size_per_gp": self.config.batch_size_per_gp,
                "min_features": self.config.min_features,
                "max_features": self.config.max_features,
                "max_classes": self.config.max_classes,
                "min_seq_len": self.config.min_seq_len,
                "max_seq_len": self.config.max_seq_len,
                "log_seq_len": self.config.log_seq_len,
                "log_n_features": self.config.log_n_features,
                "seq_len_per_gp": self.config.seq_len_per_gp,
                "min_train_size": self.config.min_train_size,
                "max_train_size": self.config.max_train_size,
                "replay_small": self.config.replay_small,
                "prior_type": self.config.prior_type,
                "prior_device": self.config.prior_device,
                "graph_config": (
                    asdict(prior_config)
                    if self.config.prior_type == "graph_scm"
                    else None
                ),
            }
            dataset.configure_logical_stream(
                schema=json.dumps(prior_schema, sort_keys=True, separators=(",", ":")),
                experiment_seed=self.config.np_seed,
                ddp_rank=self.ddp_rank,
                world_size=self.ddp_world_size,
                cursor=self.prior_cursor,
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        # For on-the-fly generation, parallelize dataset creation across dataloader workers.
        # For pre-generated data loaded from disk, a single worker is enough.
        if self.config.prior_dir is None:
            # psutil.cpu_count(logical=False) can return None on some platforms; fall back safely.
            num_workers = self.config.n_jobs
            if num_workers <= 0:
                num_workers = psutil.cpu_count(logical=False) or os.cpu_count() or 1
            prefetch_factor = 2
        else:
            num_workers = 1
            prefetch_factor = 4

        # Create dataloader for efficient loading and prefetching
        self.prior_dataset = dataset
        pin_memory = self.config.prior_device == "cpu"
        pin_memory_device = self.config.device if pin_memory else ""
        if self.config.prior_dir is None:
            self.dataloader = make_prior_dataloader(
                dataset,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                pin_memory_device=pin_memory_device,
                persistent_workers=num_workers > 0,
            )
        else:
            loader_generator = torch.Generator(device="cpu")
            loader_generator.manual_seed(self.config.torch_seed + self.ddp_rank)
            self.dataloader = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                pin_memory_device=pin_memory_device,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                generator=loader_generator,
            )

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

        if self.config.muon:
            if self.master_process:
                print("Using Muon optimizer.")
            self.optimizer = Muon(
                param_groups=[
                    dict(params=list(self.raw_model.parameters()), use_muon=True)
                ],
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                matched_adamw_rms=0.2,
                momentum=self.config.beta1,
                nesterov=True,
                ns_steps=5,
                adamw_betas=(self.config.beta1, self.config.beta2),
                adamw_eps=1e-8,
                use_cautious_wd=self.config.use_cautious_wd,
            )
        else:
            self.optimizer = optim.AdamW(
                params=self.raw_model.parameters(),
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print("Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.float16 if self.config.dtype == "float16" else torch.float32
                ),
            )
        else:
            self.amp_ctx = nullcontext()

    def _formal_optimizer_config(self):
        warmup_steps = (
            self.config.max_steps * self.config.warmup_proportion
            if self.config.warmup_proportion >= 0
            else self.config.warmup_steps
        )
        scheduler_config = {"max_steps": self.config.max_steps}
        if self.config.scheduler != "constant":
            scheduler_config["warmup_steps"] = warmup_steps
        if self.config.scheduler == "cosine_with_restarts":
            scheduler_config.update(
                {
                    "num_cycles": self.config.cosine_num_cycles,
                    "amplitude_decay": self.config.cosine_amplitude_decay,
                    "lr_end": self.config.cosine_lr_end,
                }
            )
        elif self.config.scheduler == "polynomial_decay_warmup":
            scheduler_config.update(
                {
                    "lr_end": self.config.poly_decay_lr_end,
                    "power": self.config.poly_decay_power,
                }
            )
        return build_optimizer_protocol(
            self.raw_model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            scheduler_algorithm=self.config.scheduler,
            scheduler_config=scheduler_config,
        )

    def _formal_parent_manifest(self):
        stage = self.config.formal_stage
        parent_names = (
            "formal_transaction_ledger",
            "formal_transaction_ledger_sha256",
            "formal_parent_finalized_manifest",
            "formal_parent_stage",
            "formal_parent_upstream_identity",
            "formal_parent_artifact_identity",
            "formal_artifact_root",
        )
        supplied = [getattr(self.config, name) is not None for name in parent_names]
        if stage == "stage1":
            if (
                any(supplied)
                or self.config.checkpoint_path is not None
                or self.config.only_load_model
            ):
                raise ValueError(
                    "formal Stage 1 is fresh-only and requires an explicit null parent"
                )
            return make_manifest("parent", {"parent": None})
        if (
            not all(supplied)
            or self.config.checkpoint_path is None
            or not self.config.only_load_model
        ):
            raise ValueError(
                "formal Stage 2/3 requires an explicit model-only parent and every immutable trust input"
            )
        expected_parent_stage = "stage1" if stage == "stage2" else "stage2"
        if self.config.formal_parent_stage != expected_parent_stage:
            raise ValueError(
                f"formal {stage} parent must be {expected_parent_stage}, got {self.config.formal_parent_stage}"
            )
        trust = ParentTrust(
            checkpoint_path=Path(self.config.checkpoint_path),
            finalized_manifest_path=Path(self.config.formal_parent_finalized_manifest),
            transaction_ledger_path=Path(self.config.formal_transaction_ledger),
            transaction_ledger_sha256=self.config.formal_transaction_ledger_sha256,
            study_id=self.config.formal_study_id,
            arm=self.config.row_identity_mode,
            parent_stage=self.config.formal_parent_stage,
            upstream_identity=self.config.formal_parent_upstream_identity,
            artifact_identity=self.config.formal_parent_artifact_identity,
            artifact_root=Path(self.config.formal_artifact_root),
        )
        validated_parent = validate_parent_trust(trust)
        self._validated_parent_checkpoint = validated_parent.checkpoint
        parent = validated_parent.manifest
        payload = parent["payload"]["parent"]
        expected_current = {
            "np_seed": self.config.np_seed,
            "torch_seed": self.config.torch_seed,
            "identity_rng_seed": self.config.identity_rng_seed,
            "world_size": self.ddp_world_size,
            "max_checkpoint_bytes": self.config.max_checkpoint_bytes,
        }
        for field, expected in expected_current.items():
            if payload[field] != expected:
                raise ValueError(f"formal parent {field} differs from the child run")
        return parent

    def configure_formal_provenance(self):
        """Build immutable checkpoint provenance for opt-in formal runs."""
        if not getattr(self.config, "formal_training", False):
            self.formal_provenance = None
            return
        max_checkpoint_bytes = getattr(self.config, "max_checkpoint_bytes", None)
        if (
            isinstance(max_checkpoint_bytes, bool)
            or not isinstance(max_checkpoint_bytes, int)
            or max_checkpoint_bytes < 1
        ):
            raise ValueError(
                "formal training requires a positive max_checkpoint_bytes overlay ceiling"
            )
        required = (
            "formal_stage",
            "formal_source_manifest",
            "formal_source_sha256",
            "formal_source_commit_sha",
            "formal_source_tree_sha",
            "formal_environment_sha256",
            "formal_study_id",
            "formal_output_id",
        )
        missing = [
            name for name in required if getattr(self.config, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"formal training is missing required inputs: {', '.join(missing)}"
            )
        if self.config.prior_dir is not None:
            raise ValueError("formal training requires the logical on-the-fly prior")
        source = load_source_manifest(
            self.config.formal_source_manifest,
            expected_sha256=self.config.formal_source_sha256,
            expected_commit_sha=self.config.formal_source_commit_sha,
            expected_tree_sha=self.config.formal_source_tree_sha,
        )
        environment = runtime_environment_manifest(require_formal_runtime=True)
        if environment["sha256"] != self.config.formal_environment_sha256:
            raise ValueError("environment does not match external expected sha256")
        parent = self._formal_parent_manifest()
        run_config = {
            key: value
            for key, value in vars(self.config).items()
            if key not in FORMAL_PROTOCOL_CONFIG_FIELDS
            and key not in {"identity_sampler_version", "max_checkpoint_bytes"}
        }
        prior_stream = self.prior_dataset.logical_stream_state_dict(
            cursor=self.prior_cursor
        )
        self.formal_provenance = build_checkpoint_provenance(
            source_manifest=source,
            environment=environment["payload"],
            model_config=self.model_config,
            state_dict=self.raw_model.state_dict(),
            prior_stream=prior_stream,
            optimizer_config=self._formal_optimizer_config(),
            stage=self.config.formal_stage,
            terminal_step=self.config.max_steps,
            np_seed=self.config.np_seed,
            torch_seed=self.config.torch_seed,
            identity_seed=self.config.identity_rng_seed,
            world_size=self.ddp_world_size,
            identity_treatment=self.identity_rng.treatment_manifest(),
            run_config=run_config,
            operational_context={
                "study_id": self.config.formal_study_id,
                "arm": self.config.row_identity_mode,
                "output_id": self.config.formal_output_id,
            },
            parent_manifest=parent,
        )
        if self.config.formal_stage != "stage1":
            parent_payload = parent["payload"]["parent"]
            current = self.formal_provenance["manifests"]
            for field, manifest_name in (
                ("source_sha256", "source"),
                ("environment_sha256", "environment"),
                ("architecture_sha256", "architecture"),
            ):
                if parent_payload[field] != current[manifest_name]["sha256"]:
                    raise ValueError(
                        f"formal parent {field} differs from the child run"
                    )

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
            )[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if (
            getattr(self.config, "formal_training", False)
            and self.config.formal_stage == "stage1"
            and checkpoint_path is not None
        ):
            raise ValueError(
                "formal Stage 1 is fresh-only and refuses any discovered checkpoint"
            )
        formal_parent = getattr(
            self.config, "formal_training", False
        ) and self.config.formal_stage in {"stage2", "stage3"}
        if formal_parent:
            checkpoint = getattr(self, "_validated_parent_checkpoint", None)
            if checkpoint is None:
                raise ValueError(
                    "formal Stage 2/3 requires the same-FD validated parent object"
                )
        elif checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        if not formal_parent:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.device,
                weights_only=True,
            )

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # Validate the treatment and complete stochastic bundle before model
        # state is touched. Intentional model-only stage transitions still
        # require a matching treatment manifest but start fresh RNG streams.
        if self.config.only_load_model:
            self.identity_rng.restore_checkpoint(checkpoint, only_load_model=True)
        else:
            validate_full_resume_checkpoint(checkpoint)
            self.identity_rng.restore_checkpoint(checkpoint, only_load_model=False)
            select_rank_rng_state(
                checkpoint["rng_state"],
                rank=self.ddp_rank,
                world_size=self.ddp_world_size,
            )
            if not hasattr(self.prior_dataset, "load_logical_stream_state_dict"):
                raise ValueError(
                    "full resume requires a logical on-the-fly prior stream"
                )
            self.prior_dataset.load_logical_stream_state_dict(
                checkpoint["prior_stream"]
            )
            if checkpoint["prior_stream"]["cursor"] != checkpoint["curr_step"]:
                raise ValueError("prior stream cursor must equal curr_step")

        self.raw_model.load_state_dict(checkpoint["state_dict"])
        if formal_parent:
            del self._validated_parent_checkpoint

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.scaler.load_state_dict(checkpoint["scaler_state"])
            self.curr_step = checkpoint["curr_step"]
            self.prior_cursor = checkpoint["prior_stream"]["cursor"]
            restore_all_rank_rng_state(
                checkpoint["rng_state"],
                rank=self.ddp_rank,
                world_size=self.ddp_world_size,
            )
            self._loaded_full_resume = True
            print(f"Resuming training at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        identity_fields = self.identity_rng.checkpoint_fields()
        rng_state = gather_all_rank_rng_state(
            rank=self.ddp_rank, world_size=self.ddp_world_size
        )
        if not self.master_process:
            return

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        if hasattr(self.prior_dataset, "logical_stream_state_dict"):
            prior_stream = self.prior_dataset.logical_stream_state_dict(
                cursor=self.prior_cursor
            )
        else:
            # Pre-generated priors remain saveable, but formal full resume is
            # deliberately rejected by load_checkpoint because their worker-
            # local file buffers are not a logical on-the-fly stream.
            prior_stream = {
                "schema_version": 1,
                "kind": "pre_generated",
                "cursor": self.prior_cursor,
            }

        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "curr_step": self.curr_step,
            "rng_state": rng_state,
            "prior_stream": prior_stream,
        }
        checkpoint.update(identity_fields)
        if getattr(self, "formal_provenance", None) is not None:
            checkpoint["provenance"] = self.formal_provenance
        atomic_torch_save(
            checkpoint,
            checkpoint_path,
            max_bytes=getattr(self.config, "max_checkpoint_bytes", None),
        )

    def coordinate_checkpoint_phase(
        self, phase: str, local_error: Exception | None
    ) -> None:
        """Make every rank observe the same checkpoint-boundary outcome."""
        local_failure = None
        if local_error is not None:
            local_failure = {
                "rank": self.ddp_rank,
                "type": type(local_error).__name__,
                "message": str(local_error),
            }

        if self.ddp:
            failures = [None] * self.ddp_world_size
            torch.distributed.all_gather_object(failures, local_failure)
        else:
            failures = [local_failure]

        failures = [failure for failure in failures if failure is not None]
        if failures:
            details = "; ".join(
                f"rank {failure['rank']} {failure['type']}: {failure['message']}"
                for failure in failures
            )
            raise RuntimeError(
                f"checkpoint {phase} phase failed on {details}"
            ) from local_error

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except (IndexError, ValueError):
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            formal_training = getattr(self.config, "formal_training", False)
            removed_any = False
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                    removed_any = True
                except Exception as error:
                    if formal_training:
                        # Retention is part of the formal disk-safety protocol.
                        # The coordinated caller makes every rank stop here.
                        raise
                    print(f"Error removing checkpoint {ckpt_path}: {error}")

            if formal_training and removed_any:
                flags = os.O_RDONLY
                for name in ("O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"):
                    flags |= getattr(os, name, 0)
                directory_fd = os.open(ckpt_dir, flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

    def seed(self):
        """Reset global seeds with the current step. This avoids regenerating
        identical datasets when resuming pretraining with on-the-fly data
        generation.
        """
        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset + self.curr_step)
        random.seed(self.config.np_seed + seed_offset + self.curr_step)
        torch.manual_seed(self.config.torch_seed + seed_offset + self.curr_step)

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        progress_refresh_seconds = getattr(
            self.config, "progress_refresh_seconds", 0.1
        )
        if (
            isinstance(progress_refresh_seconds, bool)
            or not isinstance(progress_refresh_seconds, (int, float))
            or not math.isfinite(progress_refresh_seconds)
            or progress_refresh_seconds <= 0
        ):
            raise ValueError("progress_refresh_seconds must be finite and positive")

        if self.master_process:
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps),
                desc="Step",
                leave=True,
                mininterval=float(progress_refresh_seconds),
            )
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            # Get the next batch
            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            self.curr_step = step + 1
            self.prior_cursor = self.curr_step
            if (
                self.config.empty_cache_every > 0
                and self.curr_step % self.config.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()
            is_temp_save = self.curr_step % self.config.save_temp_every == 0
            is_perm_save = self.curr_step % self.config.save_perm_every == 0
            should_save = is_temp_save or is_perm_save

            hook_error = None
            try:
                if self.master_process:
                    # Add timing information to results
                    results.update({"prior_time": prior_time, "train_time": train_time})

                    # Update progress bar with rounded values for cleaner display
                    step_progress.set_postfix(
                        {
                            k: round(v, 3) if isinstance(v, float) else v
                            for k, v in results.items()
                        },
                        refresh=False,
                    )

                # Logging to Weights & Biases
                if self.wandb_run is not None:
                    # Add learning rate to results
                    results["lr"] = self.scheduler.get_last_lr()[0]
                    wandb.log(results, step=self.curr_step)
            except Exception as error:
                if not should_save:
                    raise
                hook_error = error

            # Capture stochastic state at the true end-of-step boundary, after
            # progress/logging hooks that may themselves consume a global RNG.
            if should_save:
                self.coordinate_checkpoint_phase("hook", hook_error)

                write_error = None
                try:
                    self.save_checkpoint(name=f"step-{self.curr_step}.ckpt")
                except Exception as error:
                    write_error = error
                self.coordinate_checkpoint_phase("write", write_error)

                should_prune = (
                    is_temp_save
                    and not is_perm_save
                    and self.config.max_checkpoints > 0
                )
                if should_prune:
                    prune_error = None
                    if self.master_process:
                        try:
                            self.manage_checkpoint()
                        except Exception as error:
                            prune_error = error
                    self.coordinate_checkpoint_phase("prune", prune_error)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor
            Sequence lengths for each dataset, shape ``(micro_batch_size,)``.

        micro_train_size : Tensor
            Training sizes (split positions) for each dataset, shape
            ``(micro_batch_size,)``.

        Returns
        -------
        seq_len : int
            The common sequence length for the micro batch.

        train_size : int
            The common train size for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same sequence length."
            )

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same training size."
            )

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (``micro_d.max()``) respectively. This
        optimizes memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor
            Input features per dataset of shape ``(B, T, H)``.

        micro_y : Tensor
            Target labels per dataset of shape ``(B, T)``.

        micro_d : Tensor
            Number of active features per dataset of shape ``(B,)``.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        micro_X : Tensor
            Truncated features of shape ``(B, seq_len, micro_d.max())``.

        micro_y : Tensor
            Truncated labels of shape ``(B, seq_len)``.
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors
            for the micro batch.

        micro_batch_idx : int
            Index of the current micro batch.

        num_micro_batches : int
            Total number of micro batches.

        Returns
        -------
        dict
            Result dictionary with 'ce' and 'accuracy' keys.
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        # Set DDP gradient sync for last micro batch only
        if self.ddp:
            self.model.require_backward_grad_sync = (
                micro_batch_idx == num_micro_batches - 1
            )

        # By default (v2), ignore the per-dataset feature count so the model treats all (padded)
        # columns uniformly. This is required for the model's feature grouping and supports
        # variable-feature priors (e.g. graph_scm).
        model_d = None if self.config.ignore_d else micro_d

        row_identity_permutation = self.identity_rng.sample_for_micro_batch(
            batch_size=micro_X.shape[0],
            num_identity_tokens=self.raw_model._num_row_identity_tokens(
                micro_X.shape[-1]
            ),
            device=self.config.device,
        )

        with self.amp_ctx:
            if self.regression:
                # (B, test_size, num_quantiles) predicted quantiles at levels
                # linspace(0, 1, num_quantiles + 2)[1:-1] (matches inference / QuantileDistribution)
                pred = self.model(
                    micro_X,
                    y_train,
                    model_d,
                    row_identity_permutation=row_identity_permutation,
                )
                alphas = torch.linspace(
                    0.0,
                    1.0,
                    self.config.num_quantiles + 2,
                    device=pred.device,
                    dtype=pred.dtype,
                )[1:-1].view(1, 1, -1)
                errors = y_test.unsqueeze(-1) - pred
                loss = torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
            else:
                pred = self.model(
                    micro_X,
                    y_train,
                    model_d,
                    row_identity_permutation=row_identity_permutation,
                )  # (B, test_size, max_classes)
                pred = pred.flatten(end_dim=-2)
                true = y_test.long().flatten()
                loss = F.cross_entropy(pred, true)

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {}
            if self.regression:
                micro_results["pinball"] = scaled_loss.item()
            else:
                micro_results["ce"] = scaled_loss.item()
                accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
                micro_results["accuracy"] = accuracy.item() / num_micro_batches

        return micro_results

    def run_batch(self, batch):
        """Train the model on a batch of datasets.

        Handles gradient accumulation by splitting the batch into micro-batches.
        Supports variable-sized datasets by padding. Skips micro-batches on CUDA
        OOM errors. Updates model parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch : tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence
            lengths).

        Returns
        -------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Pad nested tensors to the same size
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Split the batch into micro-batches along the first dimension
        num_micro_batches = math.ceil(
            self.config.batch_size / self.config.micro_batch_size
        )
        micro_batches = [
            torch.split(t, self.config.micro_batch_size, dim=0) for t in batch
        ]
        micro_batches = list(zip(*micro_batches))

        results = {"pinball": 0.0} if self.regression else {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, idx, num_micro_batches
                )
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Clip the gradient
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
