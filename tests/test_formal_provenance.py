from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from tabicl.train._provenance import (
    OPERATIONAL_CONFIG_FIELDS,
    architecture_manifest,
    build_checkpoint_provenance,
    canonical_sha256,
    make_manifest,
    partition_run_config,
    runtime_environment_manifest,
    validate_cohort_provenance,
    validate_manifest,
    validate_provenance_bundle,
)


def _model_config(mode: str = "rope") -> dict[str, object]:
    return {
        "embed_dim": 16,
        "row_nhead": 4,
        "row_rope_base": 100000.0,
        "row_identity_mode": mode,
    }


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "layer.bias": torch.zeros(2),
        "layer.weight": torch.ones((2, 3), dtype=torch.float32),
    }


def _source_manifest():
    return make_manifest(
        "source",
        {
            "commit_sha": "1" * 40,
            "tree_sha": "2" * 40,
            "code_roots": ["scripts", "src/tabicl"],
            "entries": [
                {
                    "path": "src/tabicl/__init__.py",
                    "mode": "100644",
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            ],
        },
    )


def _prior_stream() -> dict[str, object]:
    schema = '{"prior":"dummy","version":1}'
    import hashlib

    state = {
        "schema_version": 1,
        "algorithm": "sha256-schema-seed-rank-logical-step-v1",
        "schema": schema,
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
        "experiment_seed": 11,
        "ddp_rank": 0,
        "world_size": 1,
        "cursor": 0,
    }
    state["manifest_sha256"] = hashlib.sha256(
        repr({key: state[key] for key in sorted(state)}).encode()
    ).hexdigest()
    return state


def _identity_treatment(mode: str) -> dict[str, object]:
    from tabicl.train._identity_rng import make_identity_treatment

    return make_identity_treatment(mode=mode, seed=17, world_size=1)


def _run_config(mode: str, output_id: str) -> dict[str, object]:
    return {
        "row_identity_mode": mode,
        "np_seed": 11,
        "torch_seed": 13,
        "identity_rng_seed": 17,
        "lr": 1e-4,
        "batch_size": 8,
        "dtype": "float32",
        "checkpoint_dir": f"outputs/{output_id}",
        "checkpoint_path": None,
        "wandb_name": output_id,
        "wandb_id": None,
        "wandb_dir": "logs",
    }


def _bundle(mode: str, *, output_id: str | None = None, run_config=None):
    output_id = output_id or f"study-a-{mode}"
    run_config = run_config or _run_config(mode, output_id)
    return build_checkpoint_provenance(
        source_manifest=_source_manifest(),
        environment={"python": "3.13.5", "torch": "2.7.1", "cuda": None},
        model_config=_model_config(mode),
        state_dict=_state_dict(),
        prior_stream=_prior_stream(),
        optimizer_config={"name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.999]},
        stage="stage1",
        terminal_step=500_000,
        np_seed=11,
        torch_seed=13,
        identity_seed=17,
        world_size=1,
        identity_treatment=_identity_treatment(mode),
        run_config=run_config,
        operational_context={
            "study_id": "study-a",
            "arm": mode,
            "output_id": output_id,
        },
        parent_manifest=make_manifest("parent", {"parent": None}),
    )


def test_canonical_hash_is_order_independent_and_rejects_nonfinite_numbers():
    assert canonical_sha256({"b": [2, 3], "a": 1}) == canonical_sha256(
        {"a": 1, "b": [2, 3]}
    )
    with pytest.raises(ValueError, match="finite"):
        canonical_sha256({"bad": float("nan")})


def test_manifest_validation_is_exact_and_detects_extra_fields():
    manifest = make_manifest("seed", {"np_seed": 1})
    assert validate_manifest(manifest, expected_kind="seed") == manifest["sha256"]

    tampered = {**manifest, "uncommitted_note": "ignored by weak validators"}
    with pytest.raises(ValueError, match="keys"):
        validate_manifest(tampered, expected_kind="seed")


def test_architecture_hash_excludes_exactly_row_identity_mode():
    rope = architecture_manifest(_model_config("rope"), _state_dict())
    temporary = architecture_manifest(_model_config("temporary"), _state_dict())
    assert rope["sha256"] == temporary["sha256"]

    changed = _model_config("rope")
    changed["row_rope_base"] = 10_000.0
    assert architecture_manifest(changed, _state_dict())["sha256"] != rope["sha256"]

    extra = _model_config("rope")
    extra["experimental_identity_knob"] = True
    assert architecture_manifest(extra, _state_dict())["sha256"] != rope["sha256"]


def test_state_dict_schema_is_part_of_architecture_hash():
    baseline = architecture_manifest(_model_config(), _state_dict())
    wrong_shape = _state_dict()
    wrong_shape["layer.weight"] = torch.ones((3, 2))
    wrong_dtype = _state_dict()
    wrong_dtype["layer.weight"] = wrong_dtype["layer.weight"].double()
    missing = _state_dict()
    del missing["layer.bias"]

    assert (
        architecture_manifest(_model_config(), wrong_shape)["sha256"]
        != baseline["sha256"]
    )
    assert (
        architecture_manifest(_model_config(), wrong_dtype)["sha256"]
        != baseline["sha256"]
    )
    assert (
        architecture_manifest(_model_config(), missing)["sha256"] != baseline["sha256"]
    )


def test_run_config_partition_enumerates_operational_fields_and_keeps_unknowns_scientific():
    config = _run_config("rope", "study-a-rope")
    scientific, operational, treatment = partition_run_config(config)

    assert treatment == {"row_identity_mode": "rope"}
    assert set(operational) == set(config) & OPERATIONAL_CONFIG_FIELDS
    assert "lr" in scientific

    drifted = dict(config, undeclared_knob="different")
    drifted_scientific, _, _ = partition_run_config(drifted)
    assert canonical_sha256(drifted_scientific) != canonical_sha256(scientific)


def test_three_arm_cohort_has_one_declared_treatment_difference():
    bundles = {mode: _bundle(mode) for mode in ("rope", "temporary", "none")}
    report = validate_cohort_provenance(bundles)

    assert report["study_id"] == "study-a"
    assert report["arms"] == ["none", "rope", "temporary"]
    assert len(set(report["treatment_sha256"].values())) == 3
    assert (
        len(
            {
                bundle["manifests"]["cohort_protocol"]["sha256"]
                for bundle in bundles.values()
            }
        )
        == 1
    )
    assert (
        len(
            {
                bundle["manifests"]["arm_protocol"]["sha256"]
                for bundle in bundles.values()
            }
        )
        == 3
    )
    assert (
        report["cohort_protocol_sha256"]
        == bundles["rope"]["manifests"]["cohort_protocol"]["sha256"]
    )


def test_protocol_manifests_bind_shared_budget_and_only_declared_treatment():
    bundle = _bundle("temporary")
    manifests = bundle["manifests"]
    shared = manifests["cohort_protocol"]["payload"]
    assert shared == {
        "stage": "stage1",
        "terminal_step": 500_000,
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "seed_sha256": manifests["seed"]["sha256"],
        "scientific_config_sha256": manifests["scientific_config"]["sha256"],
    }
    assert manifests["arm_protocol"]["payload"] == {
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "mode": "temporary",
        "treatment_sha256": manifests["treatment"]["sha256"],
    }


def test_cohort_rejects_undeclared_scientific_drift_and_bad_output_identity():
    bundles = {mode: _bundle(mode) for mode in ("rope", "temporary", "none")}
    drift = _run_config("temporary", "study-a-temporary")
    drift["gradient_clipping"] = 0.5
    bundles["temporary"] = _bundle("temporary", run_config=drift)
    with pytest.raises(ValueError, match="scientific_config"):
        validate_cohort_provenance(bundles)

    bundles = {mode: _bundle(mode) for mode in ("rope", "temporary", "none")}
    bad = copy.deepcopy(bundles["none"])
    bad["manifests"]["operational_config"] = make_manifest(
        "operational_config",
        {
            **bad["manifests"]["operational_config"]["payload"],
            "context": {
                "study_id": "study-a",
                "arm": "none",
                "output_id": "../escape",
            },
        },
    )
    body = {
        "schema_version": bad["schema_version"],
        "manifest_sha256": {
            name: value["sha256"] for name, value in sorted(bad["manifests"].items())
        },
    }
    bad["manifest_sha256"] = body["manifest_sha256"]
    bad["bundle_sha256"] = canonical_sha256(body)
    bundles["none"] = bad
    with pytest.raises(ValueError, match="output_id"):
        validate_cohort_provenance(bundles)


def test_provenance_bundle_rejects_rehashed_shared_manifest_without_protocol_rehash():
    bundle = _bundle("rope")
    validate_provenance_bundle(bundle)

    tampered = copy.deepcopy(bundle)
    tampered["manifests"]["seed"] = make_manifest(
        "seed",
        {**tampered["manifests"]["seed"]["payload"], "torch_seed": 999},
    )
    body = {
        "schema_version": tampered["schema_version"],
        "manifest_sha256": {
            name: value["sha256"]
            for name, value in sorted(tampered["manifests"].items())
        },
    }
    tampered["manifest_sha256"] = body["manifest_sha256"]
    tampered["bundle_sha256"] = canonical_sha256(body)
    with pytest.raises(ValueError, match="cohort protocol"):
        validate_provenance_bundle(tampered)


def _formal_trainer(tmp_path, *, source_sha=None):
    from tabicl.train._identity_rng import TrainerIdentityRNG
    from tabicl.train._run import Trainer
    from tabicl.train._train_config import build_parser

    source = _source_manifest()
    source_path = tmp_path / "source.json"
    source_path.write_bytes(
        __import__(
            "tabicl.train._provenance", fromlist=["canonical_json_bytes"]
        ).canonical_json_bytes(source)
    )
    environment = runtime_environment_manifest()
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
            "500000",
            "--checkpoint_dir",
            str(tmp_path / "checkpoints"),
            "--formal_training",
            "true",
            "--formal_stage",
            "stage1",
            "--formal_source_manifest",
            str(source_path),
            "--formal_source_sha256",
            source_sha or source["sha256"],
            "--formal_source_commit_sha",
            "1" * 40,
            "--formal_source_tree_sha",
            "2" * 40,
            "--formal_environment_sha256",
            environment["sha256"],
            "--formal_study_id",
            "study-a",
            "--formal_output_id",
            "study-a-temporary",
        ]
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model_config = _model_config("temporary")
    trainer.raw_model = torch.nn.Linear(3, 2)
    trainer.identity_rng = TrainerIdentityRNG(
        identity_mode="temporary", base_seed=17, rank=0, world_size=1
    )
    trainer.prior_dataset = SimpleNamespace(
        logical_stream_state_dict=lambda cursor=0: {
            **_prior_stream(),
            "cursor": cursor,
            "manifest_sha256": hashlib.sha256(
                repr(
                    {
                        key: value
                        for key, value in {
                            **_prior_stream(),
                            "cursor": cursor,
                        }.items()
                        if key != "manifest_sha256"
                    }
                ).encode()
            ).hexdigest(),
        }
    )

    # Keep the test stream canonical in the same sorted-repr form as PriorDataset.
    def prior_state(cursor=0):
        state = {**_prior_stream(), "cursor": cursor}
        state.pop("manifest_sha256")
        state["manifest_sha256"] = hashlib.sha256(
            repr({key: state[key] for key in sorted(state)}).encode()
        ).hexdigest()
        return state

    trainer.prior_dataset.logical_stream_state_dict = prior_state
    trainer.prior_cursor = 0
    trainer.curr_step = 1
    trainer.ddp = False
    trainer.ddp_rank = 0
    trainer.ddp_world_size = 1
    trainer.master_process = True
    trainer.optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {}, "param_groups": []}
    )
    trainer.scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": 1})
    trainer.scaler = SimpleNamespace(state_dict=lambda: {})
    return trainer


def test_formal_parser_defaults_off_and_exposes_explicit_trust_inputs():
    from tabicl.train._train_config import build_parser

    config = build_parser().parse_args([])
    assert config.formal_training is False
    for name in (
        "formal_stage",
        "formal_source_manifest",
        "formal_source_sha256",
        "formal_source_commit_sha",
        "formal_source_tree_sha",
        "formal_environment_sha256",
        "formal_study_id",
        "formal_output_id",
        "formal_transaction_ledger",
    ):
        assert hasattr(config, name)


def test_trainer_formal_checkpoint_persists_validated_provenance(tmp_path):
    trainer = _formal_trainer(tmp_path)
    trainer.configure_formal_provenance()
    trainer.save_checkpoint("step-1.ckpt")

    checkpoint = torch.load(
        tmp_path / "checkpoints/step-1.ckpt", map_location="cpu", weights_only=True
    )
    validate_provenance_bundle(checkpoint["provenance"])
    assert checkpoint["provenance"] == trainer.formal_provenance
    assert checkpoint["provenance"]["manifests"]["stage"]["payload"] == {
        "stage": "stage1",
        "terminal_step": 500_000,
    }


def test_formal_trainer_fails_closed_on_external_source_digest_drift(tmp_path):
    trainer = _formal_trainer(tmp_path, source_sha="0" * 64)
    with pytest.raises(ValueError, match="source.*external expected"):
        trainer.configure_formal_provenance()
