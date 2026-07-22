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

from tabicl.train._identity_rng import TrainerIdentityRNG
from tabicl.train._provenance import (
    CheckpointExpectations,
    FinalizationTrust,
    ParentTrust,
    build_checkpoint_provenance,
    canonical_json_bytes,
    canonical_sha256,
    checkpoint_sha256,
    finalize_identity_checkpoint,
    make_manifest,
    recover_finalized_checkpoint_manifest,
    validate_identity_checkpoint,
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
    controller = TrainerIdentityRNG(
        identity_mode=mode, base_seed=seed, rank=0, world_size=world_size
    )
    fields = {"identity_treatment": controller.treatment_manifest()}
    if mode == "temporary":
        fields.update(controller.checkpoint_fields())
    return fields


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
):
    state_dict = {
        "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "linear.bias": torch.zeros(2),
    }
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
        "optimizer_state": {
            "state": {0: {"exp_avg": torch.zeros(2), "step": torch.tensor(1.0)}},
            "param_groups": [{"lr": 1e-4, "params": [0]}],
        },
        "scheduler_state": {
            "last_epoch": terminal_step,
            "_step_count": terminal_step + 1,
        },
        "scaler_state": {},
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
        optimizer_config={"name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.999]},
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
            "amp": False,
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


def test_protocol_optimizer_mutation_and_full_rehash_fails_external_protocol(
    tmp_path,
):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    optimizer = dict(checkpoint["provenance"]["manifests"]["optimizer"]["payload"])
    optimizer["lr"] = 2e-4
    _resign_bundle(checkpoint, "optimizer", optimizer)
    _sync_protocol_manifests(checkpoint)
    expected = replace(
        expected,
        optimizer_sha256=checkpoint["provenance"]["manifests"]["optimizer"]["sha256"],
    )

    with pytest.raises(ValueError, match="cohort_protocol"):
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
    if mutation == "foreign_state":
        optimizer["state"][99] = {"step": torch.tensor(1.0)}
    elif mutation == "duplicate_param":
        optimizer["param_groups"].append({"lr": 1e-4, "params": [0]})
    elif mutation == "bool_param":
        optimizer["param_groups"][0]["params"] = [True]
    else:
        del optimizer["param_groups"][0]["params"]
    with pytest.raises(ValueError, match=match):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


def test_scheduler_and_enabled_scaler_structures_are_validated(tmp_path):
    checkpoint = _checkpoint()
    expected = _expectations(checkpoint)
    checkpoint["scheduler_state"] = {}
    with pytest.raises(ValueError, match="scheduler_state.*last_epoch"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)

    checkpoint = _checkpoint()
    scientific = checkpoint["provenance"]["manifests"]["scientific_config"]["payload"]
    _resign_bundle(
        checkpoint,
        "scientific_config",
        {**scientific, "amp": True},
    )
    _sync_protocol_manifests(checkpoint)
    expected = replace(
        _expectations(checkpoint),
        scientific_sha256=checkpoint["provenance"]["manifests"]["scientific_config"][
            "sha256"
        ],
        cohort_protocol_sha256=checkpoint["provenance"]["manifests"]["cohort_protocol"][
            "sha256"
        ],
        arm_protocol_sha256=checkpoint["provenance"]["manifests"]["arm_protocol"][
            "sha256"
        ],
    )
    with pytest.raises(ValueError, match="scaler_state"):
        validate_identity_checkpoint(_save(tmp_path, checkpoint), expected)


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
    parent_path = _save(tmp_path, parent, name=f"parent-{mode}.ckpt")
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
    trainer.raw_model = SimpleNamespace(
        state_dict=lambda: parent_checkpoint["state_dict"]
    )
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
