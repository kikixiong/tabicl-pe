import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tabicl.train._rng_state import (
    capture_rank_rng_state,
    make_all_rank_rng_bundle,
    restore_rank_rng_state,
    select_rank_rng_state,
)


class _Stateful:
    def __init__(self, state):
        self.state = state
        self.load_calls = 0

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.load_calls += 1
        self.state = state


class _CheckpointProgress:
    def __init__(self, values, *, fail_hook=False):
        self.values = values
        self.fail_hook = fail_hook

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, values=None, *, refresh=True, **kwargs):
        if self.fail_hook:
            raise RuntimeError("master hook boom")


def _checkpoint_coordination_worker(
    rank, world_size, init_file, checkpoint_dir, scenario
):
    import tabicl.train._run as run_module
    from tabicl.prior._dataset import PriorDataset
    from tabicl.train._identity_rng import TrainerIdentityRNG
    from tabicl.train._run import Trainer

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            max_steps=1,
            empty_cache_every=0,
            save_temp_every=1,
            save_perm_every=100,
            max_checkpoints=1 if scenario == "prune_failure" else 0,
            checkpoint_dir=checkpoint_dir,
            max_checkpoint_bytes=1 if scenario == "ceiling_failure" else None,
        )
        trainer.ddp = True
        trainer.ddp_rank = rank
        trainer.ddp_world_size = world_size
        trainer.master_process = rank == 0
        trainer.curr_step = 0
        trainer.prior_cursor = 0
        trainer.dataloader = [None]
        trainer.run_batch = lambda batch: {"loss": 0.0}
        trainer.wandb_run = None
        trainer.model_config = {}
        trainer.raw_model = _Stateful({"weight": torch.tensor([rank])})
        trainer.optimizer = _Stateful({"optimizer": rank})
        trainer.scheduler = _Stateful({"scheduler": rank})
        trainer.scaler = _Stateful({"scale": 1.0})
        trainer.identity_rng = TrainerIdentityRNG(
            identity_mode="temporary",
            base_seed=31,
            rank=rank,
            world_size=world_size,
        )
        trainer.prior_dataset = PriorDataset(
            prior_type="dummy",
            batch_size=1,
            min_features=2,
            max_features=2,
            max_classes=2,
            min_seq_len=4,
            max_seq_len=5,
            min_train_size=1,
            max_train_size=3,
        )
        trainer.prior_dataset.configure_logical_stream(
            schema="checkpoint-coordination-v1",
            experiment_seed=19,
            ddp_rank=rank,
            world_size=world_size,
            cursor=0,
        )

        run_module.tqdm = lambda values, **kwargs: _CheckpointProgress(
            values,
            fail_hook=scenario == "hook_failure",
        )
        if scenario == "write_failure":

            def fail_write(*args, **kwargs):
                raise OSError("master write boom")

            run_module.atomic_torch_save = fail_write
        if scenario == "prune_failure":

            def fail_prune():
                raise OSError("master prune boom")

            trainer.manage_checkpoint = fail_prune

        error = None
        try:
            Trainer.train.__wrapped__(trainer)
        except Exception as caught:
            error = caught

        outcomes = [None] * world_size
        dist.all_gather_object(
            outcomes,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        assert outcomes[1:] == outcomes[:-1]
        if scenario == "success":
            assert error is None
        else:
            assert isinstance(error, RuntimeError)
            assert "rank 0" in str(error)
            if scenario == "ceiling_failure":
                assert "write phase" in str(error)
                assert "max_checkpoint_bytes" in str(error)
                if rank == 0:
                    assert not list(Path(checkpoint_dir).iterdir())
            else:
                assert scenario.split("_")[0] in str(error)
                assert "boom" in str(error)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "hook_failure",
        "write_failure",
        "ceiling_failure",
        "prune_failure",
    ],
)
def test_two_rank_checkpoint_boundary_coordinates_outcomes(tmp_path, scenario):
    mp.spawn(
        _checkpoint_coordination_worker,
        args=(
            2,
            str(tmp_path / f"{scenario}-gloo-init"),
            str(tmp_path / scenario),
            scenario,
        ),
        nprocs=2,
        join=True,
    )


def _make_resume_trainer(tmp_path, *, mode="temporary", seed=71, world_size=1):
    from tabicl.prior._dataset import PriorDataset
    from tabicl.train._identity_rng import TrainerIdentityRNG
    from tabicl.train._run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        checkpoint_path=str(tmp_path / "resume.ckpt"),
        checkpoint_dir=None,
        device="cpu",
        only_load_model=False,
    )
    trainer.ddp_rank = 0
    trainer.ddp_world_size = world_size
    trainer.raw_model = _Stateful({"weight": torch.tensor([0.0])})
    trainer.optimizer = _Stateful({"optimizer": 0})
    trainer.scheduler = _Stateful({"scheduler": 0})
    trainer.scaler = _Stateful({"scale": 1.0})
    trainer.identity_rng = TrainerIdentityRNG(
        identity_mode=mode,
        base_seed=seed,
        rank=0,
        world_size=world_size,
    )
    trainer.prior_dataset = PriorDataset(
        prior_type="dummy",
        batch_size=1,
        min_features=2,
        max_features=2,
        max_classes=2,
        min_seq_len=4,
        max_seq_len=5,
        min_train_size=1,
        max_train_size=3,
    )
    trainer.prior_dataset.configure_logical_stream(
        schema="resume-test-v1",
        experiment_seed=19,
        ddp_rank=0,
        world_size=world_size,
        cursor=0,
    )
    trainer.prior_cursor = 0
    trainer.curr_step = 0
    return trainer


def _full_checkpoint(trainer, *, cursor=9):
    checkpoint = {
        "state_dict": {"weight": torch.tensor([3.0])},
        "optimizer_state": {"optimizer": 4},
        "scheduler_state": {"scheduler": 5},
        "scaler_state": {"scale": 128.0, "growth_tracker": 6},
        "curr_step": cursor,
        "prior_stream": trainer.prior_dataset.logical_stream_state_dict(cursor=cursor),
        "rng_state": make_all_rank_rng_bundle(
            [capture_rank_rng_state(rank=0)], world_size=1
        ),
    }
    checkpoint.update(trainer.identity_rng.checkpoint_fields())
    return checkpoint


def _draw_all():
    values = {
        "python": random.random(),
        "numpy": np.random.random(),
        "torch_cpu": torch.rand(4),
    }
    if torch.cuda.is_available():
        values["torch_cuda"] = torch.rand(4, device="cuda").cpu()
    return values


def test_python_numpy_torch_cpu_and_cuda_rng_resume_exactly():
    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(7)
    state = capture_rank_rng_state(rank=0)
    expected = _draw_all()
    _draw_all()

    restore_rank_rng_state(state)
    actual = _draw_all()
    assert actual["python"] == expected["python"]
    assert actual["numpy"] == expected["numpy"]
    torch.testing.assert_close(actual["torch_cpu"], expected["torch_cpu"], rtol=0, atol=0)
    if torch.cuda.is_available():
        torch.testing.assert_close(actual["torch_cuda"], expected["torch_cuda"], rtol=0, atol=0)


def test_all_rank_bundle_selects_exact_rank_and_rejects_wrong_world_size():
    states = [capture_rank_rng_state(rank=rank) for rank in range(2)]
    bundle = make_all_rank_rng_bundle(states, world_size=2)

    assert select_rank_rng_state(bundle, rank=1, world_size=2)["rank"] == 1
    with pytest.raises(ValueError, match="world_size"):
        select_rank_rng_state(bundle, rank=0, world_size=1)


def test_all_rank_bundle_rejects_swapped_outer_rank_keys():
    states = [capture_rank_rng_state(rank=rank) for rank in range(2)]
    bundle = make_all_rank_rng_bundle(states, world_size=2)
    bundle["rank_states"] = {
        "0": bundle["rank_states"]["1"],
        "1": bundle["rank_states"]["0"],
    }

    with pytest.raises(ValueError, match="outer rank key"):
        select_rank_rng_state(bundle, rank=0, world_size=2)


def test_incomplete_legacy_checkpoint_is_not_a_full_resume_checkpoint():
    from tabicl.train._rng_state import validate_full_resume_checkpoint

    legacy = {
        "state_dict": {},
        "optimizer_state": {},
        "scheduler_state": {},
        "curr_step": 10,
    }
    with pytest.raises(ValueError, match="identity_treatment.*rng_state.*prior_stream.*scaler_state"):
        validate_full_resume_checkpoint(legacy)


def test_trainer_full_resume_restores_scaler_prior_cursor_and_rng(tmp_path):
    trainer = _make_resume_trainer(tmp_path)
    checkpoint = _full_checkpoint(trainer, cursor=9)
    restore_rank_rng_state(select_rank_rng_state(checkpoint["rng_state"], rank=0, world_size=1))
    expected = _draw_all()
    _draw_all()
    torch.save(checkpoint, trainer.config.checkpoint_path)

    trainer.load_checkpoint()

    assert trainer.curr_step == 9
    assert trainer.prior_cursor == 9
    assert trainer.prior_dataset.logical_stream_state_dict()["cursor"] == 9
    assert trainer.scaler.state == {"scale": 128.0, "growth_tracker": 6}
    actual = _draw_all()
    assert actual["python"] == expected["python"]
    assert actual["numpy"] == expected["numpy"]
    torch.testing.assert_close(actual["torch_cpu"], expected["torch_cpu"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,seed,world_size,field",
    [
        ("rope", 71, 1, "row_identity_mode"),
        ("temporary", 72, 1, "identity_rng_seed"),
        ("temporary", 71, 2, "world_size"),
    ],
)
def test_resume_identity_drift_fails_before_model_mutation(
    tmp_path, mode, seed, world_size, field
):
    source = _make_resume_trainer(tmp_path)
    torch.save(_full_checkpoint(source), source.config.checkpoint_path)
    target = _make_resume_trainer(
        tmp_path, mode=mode, seed=seed, world_size=world_size
    )

    with pytest.raises(ValueError, match=field):
        target.load_checkpoint()

    assert target.raw_model.load_calls == 0


def test_fresh_model_is_reseeded_after_wandb_initialization():
    from tabicl.train._run import Trainer

    class OrderTrainer(Trainer):
        def configure_ddp(self):
            self.ddp = False
            self.master_process = True
            self.ddp_rank = self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.curr_step = 0
            torch.manual_seed(self.config.torch_seed)

        def configure_identity_rng(self):
            pass

        def configure_wandb(self):
            torch.rand(1)  # A logger is allowed to use a global RNG during init.

        def build_model(self):
            self.initial_weight = torch.rand(1)

        def configure_prior(self):
            pass

        def configure_optimizer(self):
            pass

        def configure_amp(self):
            pass

        def load_checkpoint(self):
            pass

        def seed(self):
            torch.manual_seed(self.config.torch_seed)

    config = SimpleNamespace(torch_seed=314)
    expected_generator = torch.Generator().manual_seed(314)
    expected_weight = torch.rand(1, generator=expected_generator)

    trainer = OrderTrainer(config)

    torch.testing.assert_close(trainer.initial_weight, expected_weight, rtol=0, atol=0)


def test_checkpoint_rng_boundary_is_after_logging(monkeypatch):
    import tabicl.train._run as run_module
    from tabicl.train._run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        max_steps=1,
        empty_cache_every=0,
        save_temp_every=1,
        save_perm_every=1,
        max_checkpoints=0,
    )
    trainer.curr_step = 0
    trainer.prior_cursor = 0
    trainer.master_process = False
    trainer.ddp = False
    trainer.dataloader = [None]
    trainer.run_batch = lambda batch: {"loss": 0.0}
    trainer.scheduler = SimpleNamespace(get_last_lr=lambda: [1.0])
    trainer.wandb_run = object()
    captured = []
    trainer.save_checkpoint = lambda name: captured.append(torch.rand(1))
    monkeypatch.setattr(run_module.wandb, "log", lambda *args, **kwargs: torch.rand(1))

    expected_generator = torch.Generator().manual_seed(2718)
    torch.rand(1, generator=expected_generator)  # consumed by logger
    expected_at_boundary = torch.rand(1, generator=expected_generator)
    torch.manual_seed(2718)

    Trainer.train.__wrapped__(trainer)

    torch.testing.assert_close(captured[0], expected_at_boundary, rtol=0, atol=0)


def test_progress_output_is_rate_limited_without_skipping_training_steps(monkeypatch):
    import tabicl.train._run as run_module
    from tabicl.train._run import Trainer

    calls = {}

    class FakeProgress:
        def __iter__(self):
            yield from range(2)

        def set_postfix(self, values, *, refresh):
            calls.setdefault("postfix", []).append((values, refresh))

    def fake_tqdm(iterable, *, desc, leave, mininterval):
        calls["tqdm"] = (list(iterable), desc, leave, mininterval)
        return FakeProgress()

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        max_steps=2,
        progress_refresh_seconds=30.0,
        empty_cache_every=0,
        save_temp_every=10,
        save_perm_every=10,
        max_checkpoints=0,
    )
    trainer.curr_step = 0
    trainer.prior_cursor = 0
    trainer.master_process = True
    trainer.ddp = False
    trainer.dataloader = [None, None]
    observed = []
    trainer.run_batch = lambda batch: observed.append(batch) or {"loss": 0.5}
    trainer.scheduler = SimpleNamespace(get_last_lr=lambda: [1.0])
    trainer.wandb_run = None
    monkeypatch.setattr(run_module, "tqdm", fake_tqdm)

    Trainer.train.__wrapped__(trainer)

    assert calls["tqdm"] == ([0, 1], "Step", True, 30.0)
    assert len(calls["postfix"]) == 2
    assert all(refresh is False for _, refresh in calls["postfix"])
    assert observed == [None, None]
    assert trainer.curr_step == trainer.prior_cursor == 2


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_progress_refresh_interval_must_be_finite_and_positive(value):
    from tabicl.train._run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        max_steps=0,
        progress_refresh_seconds=value,
    )
    trainer.curr_step = 0
    trainer.master_process = False
    trainer.ddp = False

    with pytest.raises(ValueError, match="finite and positive"):
        Trainer.train.__wrapped__(trainer)


def _cpu_trainer_config(checkpoint_dir, *, checkpoint_path=None):
    from tabicl.train._train_config import build_parser

    args = [
        "--device",
        "cpu",
        "--amp",
        "false",
        "--max_steps",
        "2",
        "--batch_size",
        "2",
        "--micro_batch_size",
        "2",
        "--scheduler",
        "constant",
        "--prior_type",
        "dummy",
        "--prior_device",
        "cpu",
        "--n_jobs",
        "1",
        "--batch_size_per_gp",
        "1",
        "--min_features",
        "2",
        "--max_features",
        "2",
        "--max_classes",
        "2",
        "--min_seq_len",
        "6",
        "--max_seq_len",
        "6",
        "--min_train_size",
        "2",
        "--max_train_size",
        "4",
        "--embed_dim",
        "8",
        "--col_num_blocks",
        "1",
        "--col_nhead",
        "2",
        "--col_num_inds",
        "2",
        "--row_num_blocks",
        "1",
        "--row_nhead",
        "2",
        "--row_num_cls",
        "1",
        "--row_identity_mode",
        "temporary",
        "--identity_rng_seed",
        "29",
        "--icl_num_blocks",
        "1",
        "--icl_nhead",
        "2",
        "--ff_factor",
        "1",
        "--dropout",
        "0.1",
        "--zero_init",
        "false",
        "--np_seed",
        "17",
        "--torch_seed",
        "23",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--save_temp_every",
        "1",
        "--save_perm_every",
        "100",
        "--max_checkpoints",
        "0",
    ]
    if checkpoint_path is not None:
        args.extend(["--checkpoint_path", str(checkpoint_path)])
    return build_parser().parse_args(args)


def _assert_checkpoint_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_checkpoint_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_checkpoint_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_actual_cpu_trainer_step_one_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import tabicl.train._run as run_module

    real_make_prior_dataloader = run_module.make_prior_dataloader

    def make_in_process_loader(dataset, **kwargs):
        return real_make_prior_dataloader(
            dataset,
            num_workers=0,
            pin_memory=False,
        )

    monkeypatch.setattr(run_module, "make_prior_dataloader", make_in_process_loader)
    uninterrupted_dir = tmp_path / "uninterrupted"
    split_dir = tmp_path / "split"
    previous_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        uninterrupted = run_module.Trainer(_cpu_trainer_config(uninterrupted_dir))
        uninterrupted.train()

        first_leg = run_module.Trainer(_cpu_trainer_config(split_dir))
        # Keep the scheduler's two-step horizon while deliberately stopping at
        # the first durable end-of-step checkpoint.
        first_leg.config.max_steps = 1
        first_leg.train()

        resumed = run_module.Trainer(
            _cpu_trainer_config(
                split_dir,
                checkpoint_path=split_dir / "step-1.ckpt",
            )
        )
        resumed.train()
    finally:
        torch.set_num_threads(previous_num_threads)

    uninterrupted_checkpoint = torch.load(
        uninterrupted_dir / "step-2.ckpt", map_location="cpu", weights_only=True
    )
    resumed_checkpoint = torch.load(
        split_dir / "step-2.ckpt", map_location="cpu", weights_only=True
    )
    _assert_checkpoint_tree_equal(resumed_checkpoint, uninterrupted_checkpoint)
