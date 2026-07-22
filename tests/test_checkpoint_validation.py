from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tabicl.train._identity_rng import TrainerIdentityRNG, build_checkpoint_bundle
from tabicl.train._provenance import (
    CheckpointExpectations,
    FinalizationTrust,
    ParentTrust,
    build_checkpoint_provenance,
    build_optimizer_protocol,
    canonical_json_bytes,
    canonical_sha256,
    checkpoint_sha256,
    finalize_identity_checkpoint,
    make_manifest,
    recover_finalized_checkpoint_manifest,
    validate_identity_checkpoint,
    validate_parent_trust,
)
from tabicl.train._rng_state import capture_rank_rng_state, make_all_rank_rng_bundle


VALIDATOR_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "validate_identity_checkpoint.py"
)


def _prior_stream(*, cursor: int, seed: int = 11, world_size: int = 1):
    schema = '{"max_seq_len":1024,"prior":"dummy"}'
    state = {
        "schema_version": 1,
        "algorithm": "sha256-schema-seed-rank-logical-step-v1",
        "schema": schema,
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
        "experiment_seed": seed,
        "ddp_rank": 0,
        "world_size": world_size,
        "cursor": cursor,
    }
    state["manifest_sha256"] = hashlib.sha256(
        repr({key: state[key] for key in sorted(state)}).encode()
    ).hexdigest()
    return state


def _source_manifest(tag="a"):
    return make_manifest(
        "source",
        {
            "commit_sha": tag * 40,
            "tree_sha": "b" * 40,
            "code_roots": ["scripts", "src/tabicl"],
            "entries": [],
        },
    )


def _identity_fields(mode: str, *, seed: int, world_size: int):
    controllers = [
        TrainerIdentityRNG(
            identity_mode=mode,
            base_seed=seed,
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    ]
    fields = {"identity_treatment": controllers[0].treatment_manifest()}
    if mode == "temporary":
        fields["identity_sampler"] = build_checkpoint_bundle(
            [controller.state_dict() for controller in controllers]
        )
    return fields


def _optimization_checkpoint_fields(
    *,
    terminal_step: int,
    muon: bool = False,
    amp: bool = False,
    freeze_bias: bool = False,
):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

    model = TinyModel()
    if freeze_bias:
        model.linear.bias.requires_grad_(False)
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        model.linear.bias.zero_()
    if muon:
        from tabicl.train._muon import Muon

        optimizer = Muon(
            [dict(params=list(model.parameters()), use_muon=True)],
            lr=1e-4,
            weight_decay=0.0,
            momentum=0.9,
            adamw_betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.GradScaler("cpu" if amp else "cuda", enabled=amp)
    for _ in range(terminal_step):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    protocol = build_optimizer_protocol(
        model,
        optimizer,
        scheduler,
        scaler,
        scheduler_algorithm="constant",
        scheduler_config={"max_steps": terminal_step},
    )
    return {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "optimizer_protocol": protocol,
    }


def _checkpoint(
    *,
    mode="temporary",
    stage="stage1",
    terminal_step=10,
    np_seed=11,
    torch_seed=13,
    identity_seed=17,
    world_size=1,
    parent_manifest=None,
    environment=None,
    muon=False,
    amp=False,
    freeze_bias=False,
):
    optimization = _optimization_checkpoint_fields(
        terminal_step=terminal_step,
        muon=muon,
        amp=amp,
        freeze_bias=freeze_bias,
    )
    state_dict = optimization["state_dict"]
    model_config = {
        "embed_dim": 16,
        "row_nhead": 4,
        "row_identity_mode": mode,
    }
    identity = _identity_fields(mode, seed=identity_seed, world_size=world_size)
    prior_stream = _prior_stream(
        cursor=terminal_step, seed=np_seed, world_size=world_size
    )
    checkpoint = {
        "config": model_config,
        "state_dict": state_dict,
        "optimizer_state": optimization["optimizer_state"],
        "scheduler_state": optimization["scheduler_state"],
        "scaler_state": optimization["scaler_state"],
        "curr_step": terminal_step,
        "rng_state": make_all_rank_rng_bundle(
            [capture_rank_rng_state(rank=rank) for rank in range(world_size)],
            world_size=world_size,
        ),
        "prior_stream": prior_stream,
        **identity,
    }
    checkpoint["provenance"] = build_checkpoint_provenance(
        source_manifest=_source_manifest(),
        environment=environment
        or {
            "python": "test",
            "torch_version": str(torch.__version__),
            "cuda": None,
            "visible_cuda_device_count": 0,
        },
        model_config=model_config,
        state_dict=state_dict,
        prior_stream=prior_stream,
        optimizer_config=optimization["optimizer_protocol"],
        stage=stage,
        terminal_step=terminal_step,
        np_seed=np_seed,
        torch_seed=torch_seed,
        identity_seed=identity_seed,
        world_size=world_size,
        identity_treatment=checkpoint["identity_treatment"],
        run_config={
            "row_identity_mode": mode,
            "np_seed": np_seed,
            "torch_seed": torch_seed,
            "identity_rng_seed": identity_seed,
            "lr": 1e-4,
            "amp": amp,
            "muon": muon,
            "checkpoint_dir": f"outputs/study-a-{mode}",
            "wandb_name": f"study-a-{mode}",
        },
        operational_context={
            "study_id": "study-a",
            "arm": mode,
            "output_id": f"study-a-{mode}",
        },
        parent_manifest=parent_manifest or make_manifest("parent", {"parent": None}),
    )
    return checkpoint


def _expectations(checkpoint, *, parent_trust=None):
    manifests = checkpoint["provenance"]["manifests"]
    seed = manifests["seed"]["payload"]
    stage = manifests["stage"]["payload"]
    return CheckpointExpectations(
        mode=checkpoint["identity_treatment"]["row_identity_mode"],
        np_seed=seed["np_seed"],
        torch_seed=seed["torch_seed"],
        identity_seed=seed["identity_rng_seed"],
        stage=stage["stage"],
        terminal_step=stage["terminal_step"],
        source_sha256=manifests["source"]["sha256"],
        environment_sha256=manifests["environment"]["sha256"],
        prior_sha256=manifests["prior"]["sha256"],
        architecture_sha256=manifests["architecture"]["sha256"],
        optimizer_sha256=manifests["optimizer"]["sha256"],
        scientific_sha256=manifests["scientific_config"]["sha256"],
        cohort_protocol_sha256=manifests["cohort_protocol"]["sha256"],
        arm_protocol_sha256=manifests["arm_protocol"]["sha256"],
        world_size=seed["world_size"],
        cuda_device_count=checkpoint["rng_state"]["rank_states"]["0"][
            "cuda_device_count"
        ],
        max_checkpoint_bytes=64 << 20,
        parent_trust=parent_trust,
    )


def _save(tmp_path: Path, checkpoint, *, step=None, name=None):
    step = checkpoint["curr_step"] if step is None else step
    path = tmp_path / (name or f"step-{step}.ckpt")
    torch.save(checkpoint, path)
    return path


def _validator_args(path: Path, expected: CheckpointExpectations) -> list[str]:
    return [
        sys.executable,
        str(VALIDATOR_SCRIPT),
        "--checkpoint",
        str(path),
        "--mode",
        expected.mode,
        "--np-seed",
        str(expected.np_seed),
        "--torch-seed",
        str(expected.torch_seed),
        "--identity-seed",
        str(expected.identity_seed),
        "--stage",
        expected.stage,
        "--terminal-step",
        str(expected.terminal_step),
        "--source-sha256",
        expected.source_sha256,
        "--environment-sha256",
        expected.environment_sha256,
        "--prior-sha256",
        expected.prior_sha256,
        "--architecture-sha256",
        expected.architecture_sha256,
        "--optimizer-sha256",
        expected.optimizer_sha256,
        "--scientific-sha256",
        expected.scientific_sha256,
        "--cohort-protocol-sha256",
        expected.cohort_protocol_sha256,
        "--arm-protocol-sha256",
        expected.arm_protocol_sha256,
        "--world-size",
        str(expected.world_size),
        "--cuda-device-count",
        str(expected.cuda_device_count),
        "--max-checkpoint-bytes",
        str(expected.max_checkpoint_bytes),
    ]


def _resign_bundle(checkpoint, name, payload):
    provenance = checkpoint["provenance"]
    provenance["manifests"][name] = make_manifest(name, payload)
    body = {
        "schema_version": provenance["schema_version"],
        "manifest_sha256": {
            key: value["sha256"]
            for key, value in sorted(provenance["manifests"].items())
        },
    }
    provenance["manifest_sha256"] = body["manifest_sha256"]
    provenance["bundle_sha256"] = canonical_sha256(body)


def _sync_protocol_manifests(checkpoint):
    manifests = checkpoint["provenance"]["manifests"]
    stage = manifests["stage"]["payload"]
    _resign_bundle(
        checkpoint,
        "cohort_protocol",
        {
            "stage": stage["stage"],
            "terminal_step": stage["terminal_step"],
            "source_sha256": manifests["source"]["sha256"],
            "environment_sha256": manifests["environment"]["sha256"],
            "architecture_sha256": manifests["architecture"]["sha256"],
            "prior_sha256": manifests["prior"]["sha256"],
            "optimizer_sha256": manifests["optimizer"]["sha256"],
            "seed_sha256": manifests["seed"]["sha256"],
            "scientific_config_sha256": manifests["scientific_config"]["sha256"],
        },
    )
    _resign_bundle(
        checkpoint,
        "arm_protocol",
        {
            "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
            "mode": manifests["treatment"]["payload"]["row_identity_mode"],
            "treatment_sha256": manifests["treatment"]["sha256"],
        },
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _resign_rng_bundle(checkpoint):
    bundle = checkpoint["rng_state"]
    hashes = {}
    for outer_rank, state in bundle["rank_states"].items():
        numpy_state = state["numpy"]
        rank_manifest = {
            "schema_version": state["schema_version"],
            "rank": state["rank"],
            "python": state["python"],
            "numpy": {
                "bit_generator": numpy_state["bit_generator"],
                "keys_sha256": _tensor_sha256(numpy_state["keys"]),
                "position": numpy_state["position"],
                "has_gauss": numpy_state["has_gauss"],
                "cached_gaussian": numpy_state["cached_gaussian"],
            },
            "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
            "torch_cuda_sha256": [
                _tensor_sha256(value) for value in state["torch_cuda"]
            ],
            "cuda_device_count": state["cuda_device_count"],
        }
        state["manifest_sha256"] = hashlib.sha256(
            json.dumps(rank_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        hashes[outer_rank] = state["manifest_sha256"]
    bundle["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": bundle["schema_version"],
                "world_size": bundle["world_size"],
                "rank_state_hashes": hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _resign_identity_sampler(checkpoint):
    bundle = checkpoint["identity_sampler"]
    records = []
    for outer_rank in sorted(bundle["rank_states"], key=int):
        state = bundle["rank_states"][outer_rank]
        state["generator_state_sha256"] = _tensor_sha256(state["generator_state"])
        records.append(
            {key: value for key, value in state.items() if key != "generator_state"}
        )
    bundle["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": bundle["schema_version"],
                "sampler_version": bundle["sampler_version"],
                "base_seed": bundle["base_seed"],
                "world_size": bundle["world_size"],
                "rank_states": records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize("mode", ["rope", "temporary", "none"])
def test_valid_formal_checkpoint_passes_weights_only_validation(tmp_path, mode):
    checkpoint = _checkpoint(mode=mode)
    path = _save(tmp_path, checkpoint)
    report = validate_identity_checkpoint(path, _expectations(checkpoint))
    assert report["mode"] == mode
    assert report["terminal_step"] == 10


def test_checkpoint_byte_ceiling_is_external_and_checked_before_deserialization(
    tmp_path, monkeypatch
):
    import tabicl.train._provenance as provenance_module

    checkpoint = _checkpoint(mode="rope")
    path = _save(tmp_path, checkpoint)
    exact = replace(_expectations(checkpoint), max_checkpoint_bytes=path.stat().st_size)
    assert (
        validate_identity_checkpoint(path, exact)["checkpoint_size"]
        == path.stat().st_size
    )

    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run for oversized checkpoints")

    monkeypatch.setattr(provenance_module.torch, "load", forbidden_load)
    too_small = replace(exact, max_checkpoint_bytes=path.stat().st_size - 1)
    with pytest.raises(ValueError, match="byte ceiling"):
        validate_identity_checkpoint(path, too_small)
    assert called is False


@pytest.mark.parametrize("ceiling", [0, True])
def test_checkpoint_byte_ceiling_must_be_a_positive_nonbool_integer(tmp_path, ceiling):
    checkpoint = _checkpoint(mode="rope")
    path = _save(tmp_path, checkpoint)
    expected = replace(_expectations(checkpoint), max_checkpoint_bytes=ceiling)
    with pytest.raises(ValueError, match="positive integer"):
        validate_identity_checkpoint(path, expected)


def test_empty_checkpoint_is_rejected_before_deserialization(tmp_path, monkeypatch):
    import tabicl.train._provenance as provenance_module

    checkpoint = _checkpoint(mode="rope")
    path = tmp_path / "step-10.ckpt"
    path.write_bytes(b"")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run for an empty checkpoint")

    monkeypatch.setattr(provenance_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="at least one byte"):
        validate_identity_checkpoint(path, _expectations(checkpoint))
    assert called is False


def test_checkpoint_validator_cli_requires_all_external_expectations(tmp_path):
    checkpoint = _checkpoint(mode="temporary")
    expected = _expectations(checkpoint)
    path = _save(tmp_path, checkpoint)
    result = subprocess.run(
        _validator_args(path, expected),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checkpoint_sha256"] == checkpoint_sha256(path)

    missing_ceiling = _validator_args(path, expected)
    index = missing_ceiling.index("--max-checkpoint-bytes")
    del missing_ceiling[index : index + 2]
    rejected = subprocess.run(missing_ceiling, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "--max-checkpoint-bytes" in rejected.stderr


@pytest.mark.parametrize(
    "manifest_name,field,value,match",
    [
        ("seed", "np_seed", 999, "np_seed"),
        ("seed", "torch_seed", 999, "torch_seed"),
        ("seed", "identity_rng_seed", 999, "identity_rng_seed"),
        ("stage", "stage", "stage2", "stage"),
        ("stage", "terminal_step", 999, "terminal_step"),
    ],
)
def test_payload_tamper_and_internal_rehash_still_fails_explicit_expectations(
    tmp_path, manifest_name, field, value, match
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    payload = dict(checkpoint["provenance"]["manifests"][manifest_name]["payload"])
    payload[field] = value
    _resign_bundle(checkpoint, manifest_name, payload)
    _sync_protocol_manifests(checkpoint)
    expected = replace(
        expected,
        cohort_protocol_sha256=checkpoint["provenance"]["manifests"]["cohort_protocol"][
            "sha256"
        ],
        arm_protocol_sha256=checkpoint["provenance"]["manifests"]["arm_protocol"][
            "sha256"
        ],
    )
    path = _save(tmp_path, checkpoint)
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(path, expected)


@pytest.mark.parametrize(
    "manifest_name,match",
    [
        ("source", "source"),
        ("prior", "prior"),
        ("architecture", "architecture"),
        ("scientific_config", "scientific_config"),
    ],
)
def test_replacing_and_resigning_locked_manifest_fails_external_hash(
    tmp_path, manifest_name, match
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    payload = dict(checkpoint["provenance"]["manifests"][manifest_name]["payload"])
    payload["attacker"] = "self-consistent"
    _resign_bundle(checkpoint, manifest_name, payload)
    _sync_protocol_manifests(checkpoint)
    path = _save(tmp_path, checkpoint)
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(path, expected)


def test_protocol_budget_mutation_and_full_rehash_fails_external_protocol(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    stage = dict(checkpoint["provenance"]["manifests"]["stage"]["payload"])
    stage["terminal_step"] = 11
    _resign_bundle(checkpoint, "stage", stage)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="cohort_protocol"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_rehashed_stage_payload_rejects_unknown_field_by_exact_schema(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    expected = _expectations(checkpoint)
    stage = copy.deepcopy(checkpoint["provenance"]["manifests"]["stage"]["payload"])
    stage["unexpected"] = "self-consistent"
    _resign_bundle(checkpoint, "stage", stage)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="stage payload keys mismatch"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_protocol_optimizer_mutation_and_full_rehash_fails_external_protocol(
    tmp_path,
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    optimizer = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    optimizer["optimizer"]["groups"][0]["base_lr"] = 2e-4
    _resign_bundle(checkpoint, "optimizer", optimizer)
    _sync_protocol_manifests(checkpoint)
    expected = replace(
        expected,
        optimizer_sha256=checkpoint["provenance"]["manifests"]["optimizer"]["sha256"],
    )

    with pytest.raises(ValueError, match="cohort_protocol"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


@pytest.mark.parametrize("component", ["optimizer", "scheduler", "scaler"])
def test_rehashed_optimizer_protocol_rejects_unsupported_component_type(
    tmp_path, component
):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol[component]["type"] = "attacker.Unsupported"
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)
    manifests = checkpoint["provenance"]["manifests"]
    expected = replace(
        _expectations(checkpoint),
        optimizer_sha256=manifests["optimizer"]["sha256"],
        cohort_protocol_sha256=manifests["cohort_protocol"]["sha256"],
        arm_protocol_sha256=manifests["arm_protocol"]["sha256"],
    )

    with pytest.raises(ValueError, match=f"{component} type.*unsupported"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_prior_mutation_and_full_rehash_still_fails_external_prior(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    prior = dict(checkpoint["provenance"]["manifests"]["prior"]["payload"])
    prior["schema"] = '{"max_seq_len":2048,"prior":"dummy"}'
    prior["schema_sha256"] = hashlib.sha256(prior["schema"].encode()).hexdigest()
    _resign_bundle(checkpoint, "prior", prior)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="prior"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_missing_complete_state_and_nested_nonfinite_are_rejected(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    del checkpoint["scaler_state"]
    with pytest.raises(ValueError, match="scaler_state"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint = _checkpoint()
    checkpoint["optimizer_state"]["state"][0]["exp_avg"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite.*optimizer_state"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_temporary_requires_all_rank_sampler_and_other_modes_reject_it(tmp_path):
    temporary = _checkpoint(mode="temporary")
    expected = _expectations(temporary)
    del temporary["identity_sampler"]
    with pytest.raises(ValueError, match="identity_sampler"):
        validate_identity_checkpoint(_save(tmp_path, temporary), expected)

    rope = _checkpoint(mode="rope")
    rope["identity_sampler"] = _identity_fields("temporary", seed=17, world_size=1)[
        "identity_sampler"
    ]
    with pytest.raises(ValueError, match="must not contain identity_sampler"):
        validate_identity_checkpoint(_save(tmp_path, rope), _expectations(rope))


def test_sampler_rank_coverage_rng_rank_coverage_and_world_size_are_strict(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    sampler = checkpoint["identity_sampler"]
    sampler["rank_states"]["1"] = copy.deepcopy(sampler["rank_states"]["0"])
    with pytest.raises(ValueError, match="rank"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint = _checkpoint()
    checkpoint["rng_state"]["rank_states"] = {}
    with pytest.raises(ValueError, match="RNG.*rank"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_world_size_two_formal_checkpoint_with_all_rank_state_passes(tmp_path):
    checkpoint = _checkpoint(mode="temporary", world_size=2)
    expected = _expectations(checkpoint)

    report = validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    assert report["mode"] == "temporary"
    assert set(checkpoint["rng_state"]["rank_states"]) == {"0", "1"}
    assert set(checkpoint["identity_sampler"]["rank_states"]) == {"0", "1"}


def test_offline_rng_validation_uses_external_cuda_device_count_not_local_runtime(
    tmp_path,
):
    checkpoint = _checkpoint()
    expected = replace(_expectations(checkpoint), cuda_device_count=1)
    with pytest.raises(ValueError, match="CUDA device count"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


@pytest.mark.parametrize("rng_field", ["python", "numpy", "torch_cpu"])
def test_offline_rng_validation_rejects_self_rehashed_unrestorable_state(
    tmp_path, rng_field
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    rank_state = checkpoint["rng_state"]["rank_states"]["0"]
    if rng_field == "python":
        rank_state["python"] = "not-a-random-state"
    elif rng_field == "numpy":
        rank_state["numpy"]["keys"] = torch.tensor([1], dtype=torch.int64)
    else:
        rank_state["torch_cpu"] = torch.tensor([1], dtype=torch.uint8)
    _resign_rng_bundle(checkpoint)

    with pytest.raises(ValueError, match="RNG.*restorable"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_temporary_sampler_rejects_self_rehashed_unrestorable_generator_state(
    tmp_path,
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    checkpoint["identity_sampler"]["rank_states"]["0"]["generator_state"] = (
        torch.tensor([1], dtype=torch.uint8)
    )
    _resign_identity_sampler(checkpoint)

    with pytest.raises(ValueError, match="identity sampler.*restorable"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("foreign_state", "foreign"),
        ("duplicate_param", "duplicate"),
        ("bool_param", "non-bool integer"),
        ("missing_params", "params"),
    ],
)
def test_optimizer_param_id_integrity_is_strict(tmp_path, mutation, match):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    optimizer = checkpoint["optimizer_state"]
    parameter_ids = optimizer["param_groups"][0]["params"]
    if mutation == "foreign_state":
        optimizer["state"][99] = {"step": torch.tensor(1.0)}
    elif mutation == "duplicate_param":
        optimizer["param_groups"][0]["params"] = [
            parameter_ids[0],
            parameter_ids[0],
        ]
    elif mutation == "bool_param":
        optimizer["param_groups"][0]["params"] = [True, parameter_ids[1]]
    else:
        del optimizer["param_groups"][0]["params"]
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_optimizer_static_beta_drift_is_rejected_after_coherent_id_renumber(
    tmp_path,
):
    checkpoint = _checkpoint(mode="rope")
    expected = _expectations(checkpoint)
    optimizer = checkpoint["optimizer_state"]
    old_ids = optimizer["param_groups"][0]["params"]
    new_ids = [41 + index for index in range(len(old_ids))]
    optimizer["state"] = {
        new: optimizer["state"][old] for old, new in zip(old_ids, new_ids)
    }
    optimizer["param_groups"][0]["params"] = new_ids
    path = _save(tmp_path, checkpoint)
    validate_identity_checkpoint(path, expected)

    optimizer["param_groups"][0]["betas"] = (0.8, 0.999)

    with pytest.raises(ValueError, match="betas"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_real_muon_optimizer_state_roundtrip_and_static_drift(tmp_path):
    checkpoint = _checkpoint(mode="rope", muon=True)
    expected = _expectations(checkpoint)
    validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint["optimizer_state"]["param_groups"][0]["adamw_betas"] = (
        0.8,
        0.999,
    )
    with pytest.raises(ValueError, match="adamw_betas"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_frozen_parameter_without_optimizer_slot_is_accepted_end_to_end(tmp_path):
    checkpoint = _checkpoint(mode="rope", freeze_bias=True)
    protocol_group = checkpoint["provenance"]["manifests"]["optimizer"]["payload"][
        "optimizer"
    ]["groups"][0]
    frozen = [
        parameter
        for parameter in protocol_group["parameters"]
        if not parameter["requires_grad"]
    ]

    assert [parameter["name"] for parameter in frozen] == ["linear.bias"]
    assert len(checkpoint["optimizer_state"]["state"]) == 1
    validate_identity_checkpoint(_save(tmp_path, checkpoint), _expectations(checkpoint))


def test_optimizer_protocol_excludes_dynamic_step_state():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.GradScaler("cpu", enabled=True)
    arguments = {
        "scheduler_algorithm": "constant",
        "scheduler_config": {"max_steps": 10},
    }
    before = build_optimizer_protocol(model, optimizer, scheduler, scaler, **arguments)

    loss = model(torch.ones((2, 3))).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    after = build_optimizer_protocol(model, optimizer, scheduler, scaler, **arguments)

    assert after == before
    assert after["scaler"]["static"]["device"] == "cpu"


def test_adamw_pre_29_static_schema_without_decoupled_flag_is_accepted(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol_static = protocol["optimizer"]["groups"][0]["static"]
    state_group = checkpoint["optimizer_state"]["param_groups"][0]
    protocol_static.pop("decoupled_weight_decay", None)
    state_group.pop("decoupled_weight_decay", None)
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    validate_identity_checkpoint(_save(tmp_path, checkpoint), _expectations(checkpoint))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("use_muon", False, "use_muon"),
        ("ns_steps", 0, "ns_steps"),
        ("momentum", 1.0, "momentum"),
        ("adamw_betas", [0.9, 1.0], "adamw_betas"),
        ("adamw_eps", 0.0, "adamw_eps"),
    ],
)
def test_muon_protocol_rejects_invalid_static_semantics(tmp_path, field, value, match):
    checkpoint = _checkpoint(mode="rope", muon=True)
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol["optimizer"]["groups"][0]["static"][field] = value
    checkpoint["optimizer_state"]["param_groups"][0][field] = value
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "keys"),
        ("extra", "keys"),
        ("shape", "shape"),
        ("dtype", "dtype"),
    ],
)
def test_optimizer_moment_schema_is_strict(tmp_path, mutation, match):
    checkpoint = _checkpoint(mode="rope")
    expected = _expectations(checkpoint)
    slot = next(iter(checkpoint["optimizer_state"]["state"].values()))
    if mutation == "missing":
        del slot["exp_avg_sq"]
    elif mutation == "extra":
        slot["attacker"] = torch.zeros(())
    elif mutation == "shape":
        slot["exp_avg"] = torch.zeros(1)
    else:
        slot["exp_avg"] = slot["exp_avg"].double()

    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_adamw_step_rejects_bool_tensor_even_when_numeric_value_matches(tmp_path):
    checkpoint = _checkpoint(mode="rope", terminal_step=1)
    slot = next(iter(checkpoint["optimizer_state"]["state"].values()))
    slot["step"] = torch.tensor(True)

    with pytest.raises(ValueError, match="AdamW optimizer step tensor dtype"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


@pytest.mark.parametrize("muon", [False, True])
def test_configure_optimizer_keeps_frozen_parameters_in_ordinary_param_groups(muon):
    from tabicl.train._run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.raw_model = torch.nn.Linear(3, 2)
    trainer.raw_model.bias.requires_grad = False
    trainer.master_process = False
    trainer.config = SimpleNamespace(
        muon=muon,
        lr=1e-4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        use_cautious_wd=False,
        scheduler="constant",
        warmup_proportion=0.0,
        warmup_steps=0,
        max_steps=10,
    )

    trainer.configure_optimizer()

    grouped = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert grouped == list(trainer.raw_model.parameters())


def test_scheduler_and_enabled_scaler_structures_are_validated(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    checkpoint["scheduler_state"] = {}
    with pytest.raises(ValueError, match="scheduler_state.*last_epoch"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint = _checkpoint()
    optimizer_protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    optimizer_protocol["scaler"]["enabled"] = True
    _resign_bundle(
        checkpoint,
        "optimizer",
        optimizer_protocol,
    )
    _sync_protocol_manifests(checkpoint)
    expected = replace(
        _expectations(checkpoint),
        optimizer_sha256=checkpoint["provenance"]["manifests"]["optimizer"]["sha256"],
        cohort_protocol_sha256=checkpoint["provenance"]["manifests"]["cohort_protocol"][
            "sha256"
        ],
        arm_protocol_sha256=checkpoint["provenance"]["manifests"]["arm_protocol"][
            "sha256"
        ],
    )
    with pytest.raises(ValueError, match="scaler_state"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_enabled_scaler_roundtrip_and_static_corruption(tmp_path):
    checkpoint = _checkpoint(mode="rope", amp=True)
    expected = _expectations(checkpoint)
    validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint["scaler_state"]["growth_factor"] = 3.0
    with pytest.raises(ValueError, match="growth_factor"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_enabled_scaler_rejects_unusable_growth_tracker(tmp_path):
    checkpoint = _checkpoint(mode="rope", amp=True)
    checkpoint["scaler_state"]["_growth_tracker"] = 2**100

    with pytest.raises(ValueError, match="scaler_state.*growth tracker"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


def test_disabled_scaler_protocol_rejects_malformed_static_value(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol["scaler"]["static"]["init_scale"] = "not-a-number"
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="scaler static init_scale"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


def test_scaler_protocol_rejects_unknown_device(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol["scaler"]["static"]["device"] = "attacker"
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="scaler static device"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


def test_adamw_protocol_requires_betas_when_checkpoint_matches_omission(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    del protocol["optimizer"]["groups"][0]["static"]["betas"]
    del checkpoint["optimizer_state"]["param_groups"][0]["betas"]
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match="AdamW static.*betas"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


@pytest.mark.parametrize("mutation", ["base", "current", "coherent_terminal"])
def test_scheduler_lr_binding_rejects_drift(tmp_path, mutation):
    checkpoint = _checkpoint(mode="rope")
    expected = _expectations(checkpoint)
    if mutation == "base":
        checkpoint["scheduler_state"]["base_lrs"][0] = 2e-4
    elif mutation == "current":
        checkpoint["optimizer_state"]["param_groups"][0]["lr"] = 2e-4
    else:
        checkpoint["optimizer_state"]["param_groups"][0]["lr"] = 2e-4
        checkpoint["scheduler_state"]["_last_lr"][0] = 2e-4

    with pytest.raises(ValueError, match="scheduler_state"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


@pytest.mark.parametrize(
    "algorithm",
    [
        "constant",
        "linear_warmup",
        "cosine_warmup",
        "cosine_with_restarts",
        "polynomial_decay_warmup",
    ],
)
def test_real_supported_scheduler_state_roundtrips(algorithm):
    import tabicl.train._provenance as provenance_module
    from tabicl.train._optim import get_scheduler

    max_steps = 4
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = SimpleNamespace(
        scheduler=algorithm,
        max_steps=max_steps,
        warmup_proportion=-1.0,
        warmup_steps=1,
        cosine_num_cycles=1,
        cosine_amplitude_decay=1.0,
        cosine_lr_end=0.0,
        poly_decay_lr_end=0.0,
        poly_decay_power=1.0,
    )
    scheduler = get_scheduler(config, optimizer)
    static = {"max_steps": max_steps}
    if algorithm != "constant":
        static["warmup_steps"] = 1
    if algorithm == "cosine_with_restarts":
        static.update({"num_cycles": 1, "amplitude_decay": 1.0, "lr_end": 0.0})
    elif algorithm == "polynomial_decay_warmup":
        static.update({"lr_end": 0.0, "power": 1.0})
    scaler = torch.GradScaler("cuda", enabled=False)
    protocol = build_optimizer_protocol(
        model,
        optimizer,
        scheduler,
        scaler,
        scheduler_algorithm=algorithm,
        scheduler_config=static,
    )
    for _ in range(max_steps):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    provenance_module._validate_optimizer_protocol(protocol)
    provenance_module._validate_scheduler_state(
        scheduler.state_dict(),
        terminal_step=max_steps,
        protocol=protocol,
        current_lrs=[group["lr"] for group in optimizer.param_groups],
        base_lrs=[group["base_lr"] for group in protocol["optimizer"]["groups"]],
    )


def test_legacy_scheduler_private_state_schema_is_accepted(tmp_path):
    checkpoint = _checkpoint(mode="rope")
    checkpoint["scheduler_state"].pop("_is_initial", None)
    checkpoint["scheduler_state"]["verbose"] = False

    validate_identity_checkpoint(_save(tmp_path, checkpoint), _expectations(checkpoint))


@pytest.mark.parametrize(
    "algorithm,field,value,match",
    [
        ("linear_warmup", "warmup_steps", 11, "warmup_steps"),
        ("cosine_with_restarts", "num_cycles", 0, "num_cycles"),
        ("cosine_with_restarts", "amplitude_decay", 1.1, "amplitude_decay"),
        ("cosine_with_restarts", "lr_end", -1e-5, "lr_end"),
        ("polynomial_decay_warmup", "power", 0.0, "power"),
    ],
)
def test_scheduler_protocol_rejects_invalid_static_semantics(
    tmp_path, algorithm, field, value, match
):
    checkpoint = _checkpoint(mode="rope")
    protocol = copy.deepcopy(
        checkpoint["provenance"]["manifests"]["optimizer"]["payload"]
    )
    protocol["scheduler"]["algorithm"] = algorithm
    static = {"max_steps": 10, "warmup_steps": 1}
    if algorithm == "cosine_with_restarts":
        static.update({"num_cycles": 1, "amplitude_decay": 1.0, "lr_end": 0.0})
    elif algorithm == "polynomial_decay_warmup":
        static.update({"lr_end": 0.0, "power": 1.0})
    static[field] = value
    protocol["scheduler"]["static"] = static
    _resign_bundle(checkpoint, "optimizer", protocol)
    _sync_protocol_manifests(checkpoint)

    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint), _expectations(checkpoint)
        )


@pytest.mark.parametrize(
    "field,value,match",
    [("last_epoch", 9, "terminal step"), ("_step_count", 10, "step count")],
)
def test_scheduler_progress_is_bound_to_checkpoint_terminal_step(
    tmp_path, field, value, match
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    checkpoint["scheduler_state"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


@pytest.mark.parametrize(
    "mutation,match",
    [("shape", "architecture"), ("dtype", "architecture"), ("key", "architecture")],
)
def test_model_key_shape_dtype_schema_is_bound_externally(tmp_path, mutation, match):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    if mutation == "shape":
        checkpoint["state_dict"]["linear.weight"] = torch.zeros((3, 2))
    elif mutation == "dtype":
        checkpoint["state_dict"]["linear.weight"] = checkpoint["state_dict"][
            "linear.weight"
        ].double()
    else:
        checkpoint["state_dict"]["renamed.weight"] = checkpoint["state_dict"].pop(
            "linear.weight"
        )
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_filename_curr_step_prior_cursor_and_terminal_step_must_agree(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    with pytest.raises(ValueError, match="filename"):
        validate_identity_checkpoint(
            _save(tmp_path, checkpoint, name="final.ckpt"), expected
        )

    checkpoint["curr_step"] = 9
    with pytest.raises(ValueError, match="curr_step"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint, step=10), expected)

    checkpoint = _checkpoint()
    checkpoint["prior_stream"]["cursor"] = 9
    prior_without_hash = {
        key: value
        for key, value in checkpoint["prior_stream"].items()
        if key != "manifest_sha256"
    }
    checkpoint["prior_stream"]["manifest_sha256"] = hashlib.sha256(
        repr(
            {key: prior_without_hash[key] for key in sorted(prior_without_hash)}
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="prior.*cursor"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def _write_json(path: Path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _finalization_setup(tmp_path: Path):
    checkpoint = _checkpoint(stage="stage1", terminal_step=10)
    checkpoint_path = _save(tmp_path, checkpoint)
    expected = _expectations(checkpoint)
    manifests = checkpoint["provenance"]["manifests"]
    final_path = tmp_path / "finalized-stage1.json"
    ledger = make_manifest(
        "transaction_ledger",
        {
            "study_id": "study-a",
            "entries": [
                {
                    "arm": "temporary",
                    "stage": "stage1",
                    "terminal_step": 10,
                    "upstream_identity": "study-a:temporary:stage1",
                    "artifact_identity": "temporary-stage1-final",
                    "checkpoint_relpath": checkpoint_path.name,
                    "finalized_manifest_relpath": final_path.name,
                    "np_seed": 11,
                    "torch_seed": 13,
                    "identity_rng_seed": 17,
                    "world_size": 1,
                    "cuda_device_count": 0,
                    "max_checkpoint_bytes": expected.max_checkpoint_bytes,
                    "source_sha256": manifests["source"]["sha256"],
                    "environment_sha256": manifests["environment"]["sha256"],
                    "prior_sha256": manifests["prior"]["sha256"],
                    "architecture_sha256": manifests["architecture"]["sha256"],
                    "optimizer_sha256": manifests["optimizer"]["sha256"],
                    "scientific_sha256": manifests["scientific_config"]["sha256"],
                    "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
                    "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
                }
            ],
        },
    )
    ledger_path = tmp_path / "transaction-ledger.json"
    _write_json(ledger_path, ledger)
    trust = FinalizationTrust(
        transaction_ledger_path=ledger_path,
        transaction_ledger_sha256=ledger["sha256"],
        artifact_root=tmp_path,
        study_id="study-a",
        upstream_identity="study-a:temporary:stage1",
        artifact_identity="temporary-stage1-final",
    )
    return checkpoint_path, final_path, checkpoint, expected, trust


def _rewrite_finalization_ledger_entry(trust: FinalizationTrust, **updates):
    ledger = json.loads(trust.transaction_ledger_path.read_text())
    ledger["payload"]["entries"][0].update(updates)
    ledger = make_manifest("transaction_ledger", ledger["payload"])
    _write_json(trust.transaction_ledger_path, ledger)
    return replace(trust, transaction_ledger_sha256=ledger["sha256"])


def _finalization_cli_args(
    checkpoint_path: Path,
    final_path: Path,
    expected: CheckpointExpectations,
    trust: FinalizationTrust,
) -> list[str]:
    return [
        *_validator_args(checkpoint_path, expected),
        "--finalize-output",
        str(final_path),
        "--finalization-transaction-ledger",
        str(trust.transaction_ledger_path),
        "--finalization-transaction-ledger-sha256",
        trust.transaction_ledger_sha256,
        "--finalization-study-id",
        trust.study_id,
        "--finalization-upstream-identity",
        trust.upstream_identity,
        "--finalization-artifact-identity",
        trust.artifact_identity,
        "--finalization-artifact-root",
        str(trust.artifact_root),
    ]


def test_validator_cli_can_finalize_and_explicitly_recover(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    command = _finalization_cli_args(checkpoint_path, final_path, expected, trust)

    finalized = subprocess.run(command, text=True, capture_output=True)
    assert finalized.returncode == 0, finalized.stderr
    assert json.loads(finalized.stdout) == json.loads(final_path.read_text())

    recovered = subprocess.run(
        [*command, "--recover-finalization"], text=True, capture_output=True
    )
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout) == json.loads(final_path.read_text())


def test_finalized_manifest_is_write_once_durable_and_records_exact_checkpoint(
    tmp_path,
):
    checkpoint_path, final_path, checkpoint, expected, trust = _finalization_setup(
        tmp_path
    )
    manifest = finalize_identity_checkpoint(
        checkpoint_path,
        expected,
        finalized_manifest_path=final_path,
        trust=trust,
    )

    assert json.loads(final_path.read_text()) == manifest
    assert manifest["payload"]["checkpoint_sha256"] == checkpoint_sha256(
        checkpoint_path
    )
    assert manifest["payload"]["checkpoint_size"] == checkpoint_path.stat().st_size
    assert manifest["payload"]["max_checkpoint_bytes"] == expected.max_checkpoint_bytes
    assert (
        manifest["payload"]["provenance_sha256"]
        == checkpoint["provenance"]["bundle_sha256"]
    )

    original = final_path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert final_path.read_bytes() == original


def test_fd_published_final_is_consumable_as_exact_parent(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    finalized = finalize_identity_checkpoint(
        checkpoint_path,
        expected,
        finalized_manifest_path=final_path,
        trust=trust,
    )
    parent_trust = ParentTrust(
        checkpoint_path=checkpoint_path,
        finalized_manifest_path=final_path,
        transaction_ledger_path=trust.transaction_ledger_path,
        transaction_ledger_sha256=trust.transaction_ledger_sha256,
        study_id=trust.study_id,
        arm=expected.mode,
        parent_stage=expected.stage,
        upstream_identity=trust.upstream_identity,
        artifact_identity=trust.artifact_identity,
        artifact_root=trust.artifact_root,
    )

    validated = validate_parent_trust(parent_trust)

    assert validated.checkpoint_sha256 == finalized["payload"]["checkpoint_sha256"]
    assert validated.manifest["payload"]["parent"]["finalized_manifest_sha256"] == (
        finalized["sha256"]
    )


def test_finalization_study_must_match_checkpoint_operational_provenance(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    ledger = json.loads(trust.transaction_ledger_path.read_text())
    payload = dict(ledger["payload"])
    payload["study_id"] = "study-b"
    attacker_ledger = make_manifest("transaction_ledger", payload)
    _write_json(trust.transaction_ledger_path, attacker_ledger)
    attacker_trust = replace(
        trust,
        transaction_ledger_sha256=attacker_ledger["sha256"],
        study_id="study-b",
    )

    with pytest.raises(ValueError, match="study_id.*checkpoint provenance"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=attacker_trust,
        )
    assert not final_path.exists()


def test_finalization_rejects_symlinked_artifact_root(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    alias = tmp_path / "artifact-root-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    trust = replace(trust, artifact_root=alias)

    with pytest.raises(ValueError, match="symlink"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=alias / final_path.name,
            trust=trust,
        )
    assert not final_path.exists()


def test_finalization_rejects_symlinked_ledger_parent(tmp_path):
    checkpoint_path, _final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    real_parent = tmp_path / "real-final-parent"
    real_parent.mkdir()
    alias = tmp_path / "ledger-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    final_path = alias / "final.json"
    trust = _rewrite_finalization_ledger_entry(
        trust, finalized_manifest_relpath=f"{alias.name}/{final_path.name}"
    )

    with pytest.raises(ValueError, match="symlink"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert not (real_parent / final_path.name).exists()


def test_finalization_rejects_symlinked_actual_parent_alias(tmp_path):
    checkpoint_path, _final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    real_parent = tmp_path / "real-final-parent"
    real_parent.mkdir()
    alias = tmp_path / "actual-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    final_path = alias / "final.json"
    trust = _rewrite_finalization_ledger_entry(
        trust, finalized_manifest_relpath=f"{real_parent.name}/{final_path.name}"
    )

    with pytest.raises(ValueError, match="symlink|path"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert not (real_parent / final_path.name).exists()


def test_finalization_rejects_existing_final_symlink_without_touching_target(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    target = tmp_path / "must-not-change"
    target.write_text("sentinel")
    final_path.symlink_to(target)

    with pytest.raises(ValueError, match="already exists|symlink"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert target.read_text() == "sentinel"


def test_finalization_race_has_exactly_one_winner(tmp_path):
    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )

    def attempt():
        try:
            finalize_identity_checkpoint(
                checkpoint_path,
                expected,
                finalized_manifest_path=final_path,
                trust=trust,
            )
            return "created"
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert outcomes.count("created") == 1
    assert sum("already exists" in value for value in outcomes) == 1


def test_finalization_fsync_failure_is_fail_closed_and_cleans_partial_file(
    tmp_path, monkeypatch
):
    import tabicl.train._provenance as provenance_module

    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    real_fsync = provenance_module.os.fsync
    calls = 0

    def fail_first_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(provenance_module.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert not final_path.exists()


def test_finalization_write_interruption_and_publish_failure_never_expose_partial_final(
    tmp_path, monkeypatch
):
    import tabicl.train._provenance as provenance_module

    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    real_write = provenance_module.os.write
    write_calls = 0

    def interrupt_write(fd, payload):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(fd, payload[: max(1, len(payload) // 2)])
        raise OSError("write interrupted")

    monkeypatch.setattr(provenance_module.os, "write", interrupt_write)
    with pytest.raises(OSError, match="write interrupted"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert not final_path.exists()
    assert not list(tmp_path.glob(f".{final_path.name}.tmp-*"))

    monkeypatch.setattr(provenance_module.os, "write", real_write)
    monkeypatch.setattr(
        provenance_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    assert not final_path.exists()
    assert not list(tmp_path.glob(f".{final_path.name}.tmp-*"))


def test_directory_fsync_failure_keeps_complete_final_and_explicit_recovery(
    tmp_path, monkeypatch
):
    import tabicl.train._provenance as provenance_module

    checkpoint_path, final_path, _checkpoint_value, expected, trust = (
        _finalization_setup(tmp_path)
    )
    real_fsync = provenance_module.os.fsync
    calls = 0

    def fail_publish_dir_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:  # file fsync succeeds; first directory fsync fails after link.
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(provenance_module.os, "fsync", fail_publish_dir_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        finalize_identity_checkpoint(
            checkpoint_path,
            expected,
            finalized_manifest_path=final_path,
            trust=trust,
        )
    complete_bytes = final_path.read_bytes()
    assert json.loads(complete_bytes)["kind"] == "finalized_checkpoint"
    assert not list(tmp_path.glob(f".{final_path.name}.tmp-*"))

    monkeypatch.setattr(provenance_module.os, "fsync", real_fsync)
    recovered = recover_finalized_checkpoint_manifest(
        checkpoint_path,
        expected,
        finalized_manifest_path=final_path,
        trust=trust,
    )
    assert canonical_json_bytes(recovered) + b"\n" == complete_bytes


def _parent_trust(
    tmp_path: Path,
    *,
    mode="temporary",
    np_seed=11,
    ledger_np_seed=None,
    parent_stage="stage1",
    terminal_step=10,
    environment=None,
):
    parent = _checkpoint(
        mode=mode,
        stage=parent_stage,
        terminal_step=terminal_step,
        np_seed=np_seed,
        environment=environment,
    )
    if parent_stage != "stage1":
        manifests = parent["provenance"]["manifests"]
        predecessor = "stage1" if parent_stage == "stage2" else "stage2"
        _resign_bundle(
            parent,
            "parent",
            {
                "parent": {
                    "checkpoint_sha256": "0" * 64,
                    "finalized_manifest_sha256": "1" * 64,
                    "transaction_ledger_sha256": "2" * 64,
                    "study_id": "study-a",
                    "arm": mode,
                    "stage": predecessor,
                    "terminal_step": max(1, terminal_step - 10),
                    "upstream_identity": f"study-a:{mode}:{predecessor}",
                    "artifact_identity": f"{mode}-{predecessor}-final",
                    "np_seed": np_seed,
                    "torch_seed": 13,
                    "identity_rng_seed": 17,
                    "world_size": 1,
                    "cuda_device_count": 0,
                    "max_checkpoint_bytes": 64 << 20,
                    "source_sha256": manifests["source"]["sha256"],
                    "environment_sha256": manifests["environment"]["sha256"],
                    "prior_sha256": manifests["prior"]["sha256"],
                    "architecture_sha256": manifests["architecture"]["sha256"],
                    "optimizer_sha256": manifests["optimizer"]["sha256"],
                    "scientific_sha256": manifests["scientific_config"]["sha256"],
                    "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
                    "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
                }
            },
        )
    parent_path = _save(tmp_path, parent, name=f"step-{terminal_step}.ckpt")
    parent_digest = checkpoint_sha256(parent_path)
    manifests = parent["provenance"]["manifests"]
    final = make_manifest(
        "finalized_checkpoint",
        {
            "study_id": "study-a",
            "arm": mode,
            "stage": parent_stage,
            "terminal_step": terminal_step,
            "upstream_identity": f"study-a:{mode}:{parent_stage}",
            "artifact_identity": f"{mode}-{parent_stage}-final",
            "checkpoint_sha256": parent_digest,
            "checkpoint_size": parent_path.stat().st_size,
            "provenance_sha256": parent["provenance"]["bundle_sha256"],
            "source_sha256": manifests["source"]["sha256"],
            "environment_sha256": manifests["environment"]["sha256"],
            "prior_sha256": manifests["prior"]["sha256"],
            "architecture_sha256": manifests["architecture"]["sha256"],
            "optimizer_sha256": manifests["optimizer"]["sha256"],
            "seed_sha256": manifests["seed"]["sha256"],
            "treatment_sha256": manifests["treatment"]["sha256"],
            "scientific_sha256": manifests["scientific_config"]["sha256"],
            "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
            "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
            "cuda_device_count": 0,
            "max_checkpoint_bytes": 64 << 20,
        },
    )
    final_path = tmp_path / f"final-{mode}.json"
    _write_json(final_path, final)
    ledger = make_manifest(
        "transaction_ledger",
        {
            "study_id": "study-a",
            "entries": [
                {
                    "arm": mode,
                    "stage": parent_stage,
                    "upstream_identity": f"study-a:{mode}:{parent_stage}",
                    "artifact_identity": f"{mode}-{parent_stage}-final",
                    "checkpoint_relpath": parent_path.name,
                    "finalized_manifest_relpath": final_path.name,
                    "terminal_step": terminal_step,
                    "np_seed": np_seed if ledger_np_seed is None else ledger_np_seed,
                    "torch_seed": 13,
                    "identity_rng_seed": 17,
                    "world_size": 1,
                    "cuda_device_count": 0,
                    "max_checkpoint_bytes": 64 << 20,
                    "source_sha256": manifests["source"]["sha256"],
                    "environment_sha256": manifests["environment"]["sha256"],
                    "prior_sha256": manifests["prior"]["sha256"],
                    "architecture_sha256": manifests["architecture"]["sha256"],
                    "optimizer_sha256": manifests["optimizer"]["sha256"],
                    "scientific_sha256": manifests["scientific_config"]["sha256"],
                    "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
                    "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
                }
            ],
        },
    )
    ledger_path = tmp_path / f"ledger-{mode}.json"
    _write_json(ledger_path, ledger)
    trust = ParentTrust(
        checkpoint_path=parent_path,
        finalized_manifest_path=final_path,
        transaction_ledger_path=ledger_path,
        transaction_ledger_sha256=ledger["sha256"],
        study_id="study-a",
        arm=mode,
        parent_stage=parent_stage,
        upstream_identity=f"study-a:{mode}:{parent_stage}",
        artifact_identity=f"{mode}-{parent_stage}-final",
        artifact_root=tmp_path,
    )
    parent_manifest = make_manifest(
        "parent",
        {
            "parent": {
                "checkpoint_sha256": parent_digest,
                "finalized_manifest_sha256": final["sha256"],
                "transaction_ledger_sha256": ledger["sha256"],
                "study_id": "study-a",
                "arm": mode,
                "stage": parent_stage,
                "terminal_step": terminal_step,
                "upstream_identity": f"study-a:{mode}:{parent_stage}",
                "artifact_identity": f"{mode}-{parent_stage}-final",
                "np_seed": np_seed,
                "torch_seed": 13,
                "identity_rng_seed": 17,
                "world_size": 1,
                "cuda_device_count": 0,
                "max_checkpoint_bytes": 64 << 20,
                "source_sha256": manifests["source"]["sha256"],
                "environment_sha256": manifests["environment"]["sha256"],
                "prior_sha256": manifests["prior"]["sha256"],
                "architecture_sha256": manifests["architecture"]["sha256"],
                "optimizer_sha256": manifests["optimizer"]["sha256"],
                "scientific_sha256": manifests["scientific_config"]["sha256"],
                "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
                "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
            }
        },
    )
    return trust, parent_manifest


def test_stage1_requires_explicit_null_parent(tmp_path):
    checkpoint = _checkpoint(stage="stage1")
    expected = _expectations(checkpoint)
    parent = checkpoint["provenance"]["manifests"]["parent"]
    _resign_bundle(checkpoint, "parent", {"parent": {"checkpoint_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="Stage 1.*null parent"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)
    assert parent["payload"] == {"parent": None}


def test_parent_ledger_ceiling_is_enforced_before_deserialization(
    tmp_path, monkeypatch
):
    import tabicl.train._provenance as provenance_module

    trust, _parent_manifest = _parent_trust(tmp_path)
    ledger = json.loads(trust.transaction_ledger_path.read_text())
    ledger["payload"]["entries"][0]["max_checkpoint_bytes"] = (
        trust.checkpoint_path.stat().st_size - 1
    )
    ledger = make_manifest("transaction_ledger", ledger["payload"])
    _write_json(trust.transaction_ledger_path, ledger)
    final_payload = json.loads(trust.finalized_manifest_path.read_text())["payload"]
    final_payload["max_checkpoint_bytes"] = ledger["payload"]["entries"][0][
        "max_checkpoint_bytes"
    ]
    _write_json(
        trust.finalized_manifest_path,
        make_manifest("finalized_checkpoint", final_payload),
    )
    trust = replace(trust, transaction_ledger_sha256=ledger["sha256"])
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("parent torch.load must not run above the ceiling")

    monkeypatch.setattr(provenance_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="byte ceiling"):
        validate_parent_trust(trust)
    assert called is False


def test_parent_final_ceiling_must_equal_immutable_ledger(tmp_path):
    trust, _parent_manifest = _parent_trust(tmp_path)
    finalized = json.loads(trust.finalized_manifest_path.read_text())
    finalized["payload"]["max_checkpoint_bytes"] -= 1
    _write_json(
        trust.finalized_manifest_path,
        make_manifest("finalized_checkpoint", finalized["payload"]),
    )

    with pytest.raises(ValueError, match="max_checkpoint_bytes"):
        validate_parent_trust(trust)


def test_parent_ledger_relative_path_rejects_symlink_component(tmp_path):
    trust, _parent_manifest = _parent_trust(tmp_path)
    alias = tmp_path / "artifact-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    ledger = json.loads(trust.transaction_ledger_path.read_text())
    ledger["payload"]["entries"][0][
        "checkpoint_relpath"
    ] = f"{alias.name}/{trust.checkpoint_path.name}"
    ledger = make_manifest("transaction_ledger", ledger["payload"])
    _write_json(trust.transaction_ledger_path, ledger)
    trust = replace(trust, transaction_ledger_sha256=ledger["sha256"])

    with pytest.raises(ValueError, match="symlink"):
        validate_parent_trust(trust)


def test_formal_stage1_rejects_implicit_checkpoint_discovery(tmp_path):
    from tabicl.train._run import Trainer

    (tmp_path / "step-10.ckpt").write_bytes(b"must-not-be-loaded")
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        checkpoint_path=None,
        checkpoint_dir=str(tmp_path),
        formal_training=True,
        formal_stage="stage1",
    )

    with pytest.raises(ValueError, match="fresh-only.*checkpoint"):
        trainer.load_checkpoint()


def test_stage2_parent_is_bound_to_exact_bytes_ledger_and_final_manifest(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    validate_identity_checkpoint(_save(tmp_path, child), expected)


def test_parent_strict_core_rejects_rehashed_prior_cursor_attack(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )

    parent = torch.load(trust.checkpoint_path, map_location="cpu", weights_only=True)
    parent["prior_stream"]["cursor"] = 9
    prior_without_hash = {
        key: value
        for key, value in parent["prior_stream"].items()
        if key != "manifest_sha256"
    }
    parent["prior_stream"]["manifest_sha256"] = hashlib.sha256(
        repr(
            {key: prior_without_hash[key] for key in sorted(prior_without_hash)}
        ).encode()
    ).hexdigest()
    torch.save(parent, trust.checkpoint_path)

    final_payload = json.loads(trust.finalized_manifest_path.read_text())["payload"]
    final_payload["checkpoint_sha256"] = checkpoint_sha256(trust.checkpoint_path)
    final_payload["checkpoint_size"] = trust.checkpoint_path.stat().st_size
    finalized = make_manifest("finalized_checkpoint", final_payload)
    _write_json(trust.finalized_manifest_path, finalized)

    parent_record = copy.deepcopy(parent_manifest["payload"]["parent"])
    parent_record["checkpoint_sha256"] = final_payload["checkpoint_sha256"]
    parent_record["finalized_manifest_sha256"] = finalized["sha256"]
    _resign_bundle(child, "parent", {"parent": parent_record})
    expected = _expectations(child, parent_trust=trust)

    with pytest.raises(ValueError, match="prior cursor"):
        validate_identity_checkpoint(_save(tmp_path, child), expected)


@pytest.mark.parametrize(
    "parent_kwargs,match",
    [
        ({"parent_stage": "stage2", "terminal_step": 20}, "parent stage"),
        ({"mode": "rope"}, "parent arm"),
        ({"np_seed": 99}, "parent np_seed"),
    ],
)
def test_child_validator_independently_locks_parent_stage_arm_and_seed(
    tmp_path, parent_kwargs, match
):
    trust, parent_manifest = _parent_trust(tmp_path, **parent_kwargs)
    child = _checkpoint(
        stage="stage2", terminal_step=30, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, child), expected)


def test_stage2_validator_cli_constructs_complete_parent_trust(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    child_path = _save(tmp_path, child)
    result = subprocess.run(
        [
            *_validator_args(child_path, expected),
            "--parent-checkpoint",
            str(trust.checkpoint_path),
            "--finalized-parent-manifest",
            str(trust.finalized_manifest_path),
            "--transaction-ledger",
            str(trust.transaction_ledger_path),
            "--transaction-ledger-sha256",
            trust.transaction_ledger_sha256,
            "--study-id",
            trust.study_id,
            "--parent-stage",
            trust.parent_stage,
            "--upstream-identity",
            trust.upstream_identity,
            "--artifact-identity",
            trust.artifact_identity,
            "--artifact-root",
            str(trust.artifact_root),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["stage"] == "stage2"


def test_parent_seed_and_world_are_validated_against_immutable_ledger(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path, np_seed=99, ledger_np_seed=11)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    with pytest.raises(ValueError, match="parent.*np_seed"):
        validate_identity_checkpoint(_save(tmp_path, child), expected)


@pytest.mark.parametrize(
    "child_stage,parent_stage,terminal_step",
    [("stage2", "stage1", 10), ("stage3", "stage2", 20)],
)
def test_real_trainer_stage_transition_consumes_ledger_derived_parent_fields(
    tmp_path, child_stage, parent_stage, terminal_step
):
    from tabicl.train._run import Trainer

    trust, expected_parent = _parent_trust(
        tmp_path,
        parent_stage=parent_stage,
        terminal_step=terminal_step,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.ddp_world_size = 1
    trainer.config = SimpleNamespace(
        formal_stage=child_stage,
        formal_transaction_ledger=str(trust.transaction_ledger_path),
        formal_transaction_ledger_sha256=trust.transaction_ledger_sha256,
        formal_parent_finalized_manifest=str(trust.finalized_manifest_path),
        formal_parent_stage=parent_stage,
        formal_parent_upstream_identity=trust.upstream_identity,
        formal_parent_artifact_identity=trust.artifact_identity,
        formal_artifact_root=str(trust.artifact_root),
        formal_study_id="study-a",
        checkpoint_path=str(trust.checkpoint_path),
        only_load_model=True,
        row_identity_mode="temporary",
        np_seed=11,
        torch_seed=13,
        identity_rng_seed=17,
    )

    parent = trainer._formal_parent_manifest()

    assert parent == expected_parent
    assert parent["payload"]["parent"]["world_size"] == 1
    assert (
        parent["payload"]["parent"]["source_sha256"]
        == expected_parent["payload"]["parent"]["source_sha256"]
    )


@pytest.mark.parametrize(
    "field,trust_attribute",
    [
        ("checkpoint_path", "checkpoint_path"),
        ("formal_transaction_ledger", "transaction_ledger_path"),
    ],
)
def test_formal_trainer_rejects_symlinked_parent_trust_input(
    tmp_path, field, trust_attribute
):
    from tabicl.train._run import Trainer

    trust, _parent_manifest = _parent_trust(tmp_path)
    alias = tmp_path / "trust-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    values = {
        "formal_stage": "stage2",
        "formal_transaction_ledger": str(trust.transaction_ledger_path),
        "formal_transaction_ledger_sha256": trust.transaction_ledger_sha256,
        "formal_parent_finalized_manifest": str(trust.finalized_manifest_path),
        "formal_parent_stage": "stage1",
        "formal_parent_upstream_identity": trust.upstream_identity,
        "formal_parent_artifact_identity": trust.artifact_identity,
        "formal_artifact_root": str(trust.artifact_root),
        "formal_study_id": "study-a",
        "checkpoint_path": str(trust.checkpoint_path),
        "only_load_model": True,
        "row_identity_mode": "temporary",
        "np_seed": 11,
        "torch_seed": 13,
        "identity_rng_seed": 17,
    }
    values[field] = str(alias / getattr(trust, trust_attribute).name)
    trainer = Trainer.__new__(Trainer)
    trainer.ddp_world_size = 1
    trainer.config = SimpleNamespace(**values)

    with pytest.raises(ValueError, match="symlink"):
        trainer._formal_parent_manifest()


def test_formal_trainer_consumes_same_fd_validated_parent_after_path_swap(
    tmp_path, monkeypatch
):
    import tabicl.train._run as run_module

    trust, expected_parent = _parent_trust(tmp_path)
    original = torch.load(trust.checkpoint_path, map_location="cpu", weights_only=True)
    trainer = run_module.Trainer.__new__(run_module.Trainer)
    trainer.ddp_world_size = 1
    trainer.ddp_rank = 0
    trainer.config = SimpleNamespace(
        formal_stage="stage2",
        formal_training=True,
        formal_transaction_ledger=str(trust.transaction_ledger_path),
        formal_transaction_ledger_sha256=trust.transaction_ledger_sha256,
        formal_parent_finalized_manifest=str(trust.finalized_manifest_path),
        formal_parent_stage="stage1",
        formal_parent_upstream_identity=trust.upstream_identity,
        formal_parent_artifact_identity=trust.artifact_identity,
        formal_artifact_root=str(trust.artifact_root),
        formal_study_id="study-a",
        checkpoint_path=str(trust.checkpoint_path),
        checkpoint_dir=None,
        only_load_model=True,
        row_identity_mode="temporary",
        np_seed=11,
        torch_seed=13,
        identity_rng_seed=17,
        device="cpu",
    )
    trainer.identity_rng = SimpleNamespace(
        restore_checkpoint=lambda *args, **kwargs: None
    )
    loaded = {}
    trainer.raw_model = SimpleNamespace(
        load_state_dict=lambda state: loaded.update(copy.deepcopy(state))
    )

    assert trainer._formal_parent_manifest() == expected_parent
    attacker = copy.deepcopy(original)
    attacker["state_dict"]["linear.weight"] = torch.full((2, 3), 999.0)
    replacement = tmp_path / "replacement.ckpt"
    torch.save(attacker, replacement)
    replacement.replace(trust.checkpoint_path)

    monkeypatch.setattr(
        run_module.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal parent pathname was reopened")
        ),
    )
    trainer.load_checkpoint()

    assert torch.equal(loaded["linear.weight"], original["state_dict"]["linear.weight"])


@pytest.mark.parametrize(
    "child_stage,parent_stage,parent_step,child_step",
    [("stage2", "stage1", 10, 20), ("stage3", "stage2", 20, 30)],
)
def test_real_trainer_builds_formal_stage2_and_stage3_provenance(
    tmp_path, child_stage, parent_stage, parent_step, child_step
):
    from tabicl.train._provenance import runtime_environment_manifest
    from tabicl.train._run import Trainer
    from tabicl.train._train_config import build_parser

    environment = runtime_environment_manifest()
    trust, expected_parent = _parent_trust(
        tmp_path,
        parent_stage=parent_stage,
        terminal_step=parent_step,
        environment=environment["payload"],
    )
    parent_checkpoint = torch.load(
        trust.checkpoint_path, map_location="cpu", weights_only=True
    )
    source = parent_checkpoint["provenance"]["manifests"]["source"]
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)
    config = build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--amp",
            "false",
            "--prior_type",
            "dummy",
            "--row_identity_mode",
            "temporary",
            "--np_seed",
            "11",
            "--torch_seed",
            "13",
            "--identity_rng_seed",
            "17",
            "--max_steps",
            str(child_step),
            "--checkpoint_path",
            str(trust.checkpoint_path),
            "--only_load_model",
            "true",
            "--formal_training",
            "true",
            "--formal_stage",
            child_stage,
            "--formal_source_manifest",
            str(source_path),
            "--formal_source_sha256",
            source["sha256"],
            "--formal_source_commit_sha",
            "a" * 40,
            "--formal_source_tree_sha",
            "b" * 40,
            "--formal_environment_sha256",
            environment["sha256"],
            "--formal_study_id",
            "study-a",
            "--formal_output_id",
            "study-a-temporary",
            "--formal_transaction_ledger",
            str(trust.transaction_ledger_path),
            "--formal_transaction_ledger_sha256",
            trust.transaction_ledger_sha256,
            "--formal_parent_finalized_manifest",
            str(trust.finalized_manifest_path),
            "--formal_parent_stage",
            parent_stage,
            "--formal_parent_upstream_identity",
            trust.upstream_identity,
            "--formal_parent_artifact_identity",
            trust.artifact_identity,
            "--formal_artifact_root",
            str(trust.artifact_root),
        ]
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.ddp_world_size = 1
    trainer.model_config = parent_checkpoint["config"]
    trainer.raw_model = torch.nn.Module()
    trainer.raw_model.linear = torch.nn.Linear(3, 2)
    trainer.raw_model.load_state_dict(parent_checkpoint["state_dict"])
    trainer.master_process = False
    trainer.configure_optimizer()
    trainer.configure_amp()
    trainer.identity_rng = TrainerIdentityRNG(
        identity_mode="temporary", base_seed=17, rank=0, world_size=1
    )
    child_prior = _prior_stream(cursor=0)
    trainer.prior_dataset = SimpleNamespace(
        logical_stream_state_dict=lambda cursor=0: {
            **child_prior,
            "cursor": cursor,
        }
    )
    trainer.prior_cursor = 0

    trainer.configure_formal_provenance()

    assert trainer.formal_provenance["manifests"]["parent"] == expected_parent
    assert trainer.formal_provenance["manifests"]["stage"]["payload"] == {
        "stage": child_stage,
        "terminal_step": child_step,
    }


def test_parent_rejects_swapped_valid_checkpoint_redirected_sidecar_and_self_reported_hash(
    tmp_path,
):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    child_path = _save(tmp_path, child)
    trusted_final_bytes = trust.finalized_manifest_path.read_bytes()

    other = _checkpoint(mode="rope", stage="stage1", terminal_step=10)
    other_path = _save(tmp_path, other, name="other-valid.ckpt")
    swapped = copy.copy(trust)
    object.__setattr__(swapped, "checkpoint_path", other_path)
    with pytest.raises(ValueError, match="checkpoint path"):
        validate_identity_checkpoint(
            child_path,
            copy.copy(expected).__class__(
                **{**expected.__dict__, "parent_trust": swapped}
            ),
        )

    redirected_final = json.loads(trust.finalized_manifest_path.read_text())
    redirected_final["payload"]["artifact_identity"] = "redirected"
    redirected_final = make_manifest(
        "finalized_checkpoint", redirected_final["payload"]
    )
    trust.finalized_manifest_path.write_bytes(canonical_json_bytes(redirected_final))
    with pytest.raises(
        ValueError, match="finalized.*(sha256|artifact_identity)|parent manifest"
    ):
        validate_identity_checkpoint(child_path, expected)

    trust.finalized_manifest_path.write_bytes(trusted_final_bytes)
    with open(trust.checkpoint_path, "ab") as handle:
        handle.write(b"attacker")
    with pytest.raises(ValueError, match="parent checkpoint sha256"):
        validate_identity_checkpoint(child_path, expected)


def test_parent_finite_weight_mutation_with_unchanged_final_fails(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    expected = _expectations(child, parent_trust=trust)
    parent = torch.load(trust.checkpoint_path, map_location="cpu", weights_only=True)
    parent["state_dict"]["linear.weight"][0, 0] += 1.0
    torch.save(parent, trust.checkpoint_path)

    with pytest.raises(ValueError, match="parent checkpoint sha256"):
        validate_identity_checkpoint(_save(tmp_path, child), expected)


def test_child_parent_checkpoint_digest_one_nibble_drift_fails(tmp_path):
    trust, parent_manifest = _parent_trust(tmp_path)
    child = _checkpoint(
        stage="stage2", terminal_step=20, parent_manifest=parent_manifest
    )
    parent_record = copy.deepcopy(parent_manifest["payload"]["parent"])
    digest = parent_record["checkpoint_sha256"]
    parent_record["checkpoint_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    _resign_bundle(child, "parent", {"parent": parent_record})

    with pytest.raises(ValueError, match="parent manifest"):
        validate_identity_checkpoint(
            _save(tmp_path, child), _expectations(child, parent_trust=trust)
        )
