import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
