from __future__ import annotations

import hashlib
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tabicl.train._optim import get_cosine_with_restarts

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_fingerprint_fullsize_continuation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fingerprint_fullsize_continuation", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _scheduler(horizon: int):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=8e-4)
    scheduler = get_cosine_with_restarts(
        optimizer,
        num_warmup_steps=5_000,
        num_training_steps=horizon,
        num_cycles=1,
        amplitude_decay=1.0,
        lr_end=1e-7,
    )
    return optimizer, scheduler


def _advance(optimizer, scheduler, count: int) -> None:
    for _ in range(count):
        optimizer.step()
        scheduler.step()


def test_5k_state_rebinds_exactly_to_uninterrupted_500k_scheduler():
    module = _load_script()
    pilot_optimizer, pilot_scheduler = _scheduler(5_000)
    full_optimizer, full_scheduler = _scheduler(500_000)
    _advance(pilot_optimizer, pilot_scheduler, 5_000)
    _advance(full_optimizer, full_scheduler, 5_000)

    assert pilot_scheduler.state_dict() == full_scheduler.state_dict()
    assert pilot_optimizer.param_groups[0]["lr"] == pytest.approx(8e-4, abs=1e-15)

    resumed_optimizer, resumed_scheduler = _scheduler(500_000)
    resumed_optimizer.load_state_dict(pilot_optimizer.state_dict())
    resumed_scheduler.load_state_dict(pilot_scheduler.state_dict())
    for step in range(5_001, 5_005):
        _advance(full_optimizer, full_scheduler, 1)
        _advance(resumed_optimizer, resumed_scheduler, 1)
        assert resumed_scheduler.last_epoch == step
        assert resumed_scheduler.get_last_lr() == full_scheduler.get_last_lr()
        assert resumed_scheduler.get_last_lr()[0] == pytest.approx(
            module.expected_lr(step), abs=1e-15
        )

    _advance(pilot_optimizer, pilot_scheduler, 1)
    assert pilot_scheduler.get_last_lr()[0] == pytest.approx(1e-7, abs=1e-15)
    assert pilot_scheduler.get_last_lr() != resumed_scheduler.get_last_lr()


def test_expected_500k_lr_has_not_decayed_at_50k():
    module = _load_script()
    assert module.expected_lr(5_000) == pytest.approx(8e-4, abs=1e-15)
    assert module.expected_lr(50_000) == pytest.approx(0.0007837992147971183, abs=1e-15)
    assert module.expected_lr(500_000) == pytest.approx(1e-7, abs=1e-15)


def test_runtime_validator_rejects_a_50k_scheduler_horizon():
    module = _load_script()
    optimizer, scheduler = _scheduler(50_000)
    _advance(optimizer, scheduler, 5_000)
    trainer = SimpleNamespace(
        _loaded_full_resume=True,
        curr_step=5_000,
        prior_cursor=5_000,
        config=SimpleNamespace(only_load_model=False),
        scheduler=scheduler,
        model_config=module._expected_model_config("rope"),
    )
    with pytest.raises(ValueError, match="scheduler horizon/protocol"):
        module._validate_loaded_trainer(trainer, arm="rope", from_step=5_000)


def test_prior_stream_manifest_advances_with_the_logical_cursor(monkeypatch):
    module = _load_script()
    schema = "{}"
    schema_sha256 = hashlib.sha256(schema.encode()).hexdigest()

    def state(step):
        value = {
            "schema_version": 1,
            "algorithm": "sha256-schema-seed-rank-logical-step-v1",
            "schema": schema,
            "schema_sha256": schema_sha256,
            "experiment_seed": module.SEED,
            "ddp_rank": 0,
            "world_size": 1,
            "cursor": step,
        }
        payload = repr({key: value[key] for key in sorted(value)}).encode()
        return {**value, "manifest_sha256": hashlib.sha256(payload).hexdigest()}

    origin = state(5_000)
    continuation = state(5_001)
    assert origin["manifest_sha256"] != continuation["manifest_sha256"]
    monkeypatch.setattr(module, "PRIOR_SCHEMA_SHA256", schema_sha256)
    monkeypatch.setattr(module, "PRIOR_MANIFEST_SHA256", origin["manifest_sha256"])
    module._validate_prior_stream(origin, expected_step=5_000)
    module._validate_prior_stream(continuation, expected_step=5_001)


def test_one_step_smoke_uses_the_worker_invariant_in_process_stream(monkeypatch):
    module = _load_script()
    import tabicl.prior._genload as genload

    calls = {}

    def fake_loader(dataset, **kwargs):
        calls["dataset"] = dataset
        calls.update(kwargs)
        return "in-process-loader"

    monkeypatch.setattr(genload, "make_prior_dataloader", fake_loader)
    dataset = object()
    trainer = SimpleNamespace(prior_dataset=dataset, dataloader="old-loader")
    module._configure_one_step_smoke_dataloader(trainer)
    assert trainer.dataloader == "in-process-loader"
    assert calls == {"dataset": dataset, "num_workers": 0, "pin_memory": False}


def test_manifest_publication_is_canonical_and_write_once(tmp_path):
    module = _load_script()
    record = module._manifest(
        "fingerprint_fullsize_continuation_parent",
        {"study": module.STUDY, "formal_eligible": False},
    )
    path = tmp_path / "parent.json"
    module._publish_json(path, record)
    assert path.read_bytes() == module._canonical(record) + b"\n"
    assert module._load_manifest(path) == record
    with pytest.raises(FileExistsError):
        module._publish_json(path, record)


def test_environment_manifest_round_trips_across_a_json_boundary(tmp_path):
    module = _load_script()
    path = tmp_path / "environment.json"
    module.prepare_environment(Namespace(output=str(path)))
    record = module.validate_environment_manifest(path)
    assert record["kind"] == module.ENVIRONMENT_KIND


def test_training_arguments_bind_full_horizon_and_full_resume(tmp_path):
    module = _load_script()
    values = module._training_config_args(
        arm="fingerprint",
        checkpoint_path=tmp_path / "parent.ckpt",
        checkpoint_dir=tmp_path / "checkpoints",
        wandb_dir=tmp_path / "wandb",
        save_temp_every=1_000,
    )
    arguments = dict(zip(values[::2], values[1::2]))
    assert arguments["--max_steps"] == "500000"
    assert arguments["--warmup_steps"] == "5000"
    assert arguments["--checkpoint_path"].endswith("parent.ckpt")
    assert arguments["--only_load_model"] == "False"
    assert arguments["--row_identity_mode"] == "none"
    assert arguments["--row_fingerprint"] == "True"
    assert arguments["--max_checkpoint_bytes"] == "300000000"


def test_segment_manifest_recursively_binds_exact_parent_lineage(tmp_path, monkeypatch):
    module = _load_script()
    continuation_source = "f" * 40
    environment_sha256 = "e" * 64
    contracts = {
        5_000: {
            "path": str((tmp_path / "origin.ckpt").resolve()),
            "sha256": module.ARM_CONTRACTS["rope"]["origin_checkpoint_sha256"],
            "size_bytes": 220_711_775,
            "curr_step": 5_000,
            "state_tensors": 391,
            "state_elements": 27_552_258,
            "optimizer_state_entries": 390,
            "scheduler_last_epoch": 5_000,
            "scheduler_last_lr": 8e-4,
            "prior_manifest_sha256": module.PRIOR_MANIFEST_SHA256,
            "prior_schema_sha256": module.PRIOR_SCHEMA_SHA256,
        },
        20_000: {
            "path": str((tmp_path / "step-20000.ckpt").resolve()),
            "sha256": "b" * 64,
            "size_bytes": 220_711_775,
            "curr_step": 20_000,
            "state_tensors": 391,
            "state_elements": 27_552_258,
            "optimizer_state_entries": 390,
            "scheduler_last_epoch": 20_000,
            "scheduler_last_lr": module.expected_lr(20_000),
            "prior_manifest_sha256": module.PRIOR_MANIFEST_SHA256,
            "prior_schema_sha256": module.PRIOR_SCHEMA_SHA256,
        },
    }

    def fake_validate(_path, *, arm, expected_step, expected_sha256):
        assert arm == "rope"
        contract = contracts[expected_step]
        assert expected_sha256 == contract["sha256"]
        return dict(contract)

    monkeypatch.setattr(module, "validate_checkpoint", fake_validate)
    origin = module._manifest(
        "fingerprint_fullsize_continuation_parent",
        {
            "study": module.STUDY,
            "formal_eligible": False,
            "arm": "rope",
            "seed": module.SEED,
            "source_commit": module.ORIGIN_SOURCE_COMMIT,
            "origin_completion_sha256": module.ARM_CONTRACTS["rope"][
                "origin_completion_sha256"
            ],
            "origin_scheduler_horizon_steps": 5_000,
            "continuation_scheduler_horizon_steps": module.SCHEDULER_HORIZON_STEPS,
            "continuation_environment_sha256": environment_sha256,
            "checkpoint": contracts[5_000],
        },
    )
    origin_path = tmp_path / "origin.json"
    module._publish_json(origin_path, origin)

    payload = {
        "study": module.STUDY,
        "formal_eligible": False,
        "arm": "rope",
        "seed": module.SEED,
        "source_commit": continuation_source,
        "environment_sha256": environment_sha256,
        "from_step": 5_000,
        "to_step": 20_000,
        "scheduler_horizon_steps": module.SCHEDULER_HORIZON_STEPS,
        "warmup_steps": module.WARMUP_STEPS,
        "parent_manifest_path": str(origin_path.resolve()),
        "parent_manifest_sha256": origin["sha256"],
        "parent_checkpoint_sha256": contracts[5_000]["sha256"],
        "checkpoint": contracts[20_000],
        "runtime": {"python": "3.10", "torch": "2", "cuda": "12", "gpu": "H100"},
    }
    segment_path = tmp_path / "segment.json"
    module._publish_json(
        segment_path,
        module._manifest("fingerprint_fullsize_segment_completion", payload),
    )
    record, checkpoint = module.validate_parent_manifest(
        segment_path,
        arm="rope",
        expected_step=20_000,
        expected_source_commit=continuation_source,
        expected_environment_sha256=environment_sha256,
    )
    assert record["payload"]["parent_manifest_sha256"] == origin["sha256"]
    assert checkpoint == contracts[20_000]

    payload["parent_manifest_sha256"] = "0" * 64
    bad_path = tmp_path / "bad-segment.json"
    module._publish_json(
        bad_path,
        module._manifest("fingerprint_fullsize_segment_completion", payload),
    )
    with pytest.raises(ValueError, match="upstream manifest SHA-256"):
        module.validate_parent_manifest(
            bad_path,
            arm="rope",
            expected_step=20_000,
            expected_source_commit=continuation_source,
            expected_environment_sha256=environment_sha256,
        )
