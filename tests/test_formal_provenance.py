from __future__ import annotations

import base64
import copy
import hashlib
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tabicl.train._provenance import (
    OPERATIONAL_CONFIG_FIELDS,
    _FORMAL_RUNTIME_DISTRIBUTIONS,
    _FORMAL_RUNTIME_MODULES,
    _distribution_fingerprint,
    _distribution_record_integrity,
    _module_origin_fingerprint,
    architecture_manifest,
    build_git_source_manifest,
    build_checkpoint_provenance,
    canonical_json_bytes,
    canonical_sha256,
    load_source_manifest,
    make_manifest,
    partition_run_config,
    validate_cohort_provenance,
    validate_manifest,
    validate_provenance_bundle,
)


def _git(*args: str, cwd) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _fake_distribution(metadata, version):
    content = {"METADATA": "metadata", "RECORD": "record", "WHEEL": "wheel"}
    return SimpleNamespace(
        metadata=metadata,
        version=version,
        read_text=lambda name: content.get(name),
    )


def test_distribution_fingerprint_is_canonical_and_path_free():
    fingerprint = _distribution_fingerprint(
        _fake_distribution({"Name": "Example_Pkg"}, "1.0+cuda")
    )

    assert fingerprint["name"] == "example-pkg"
    assert fingerprint["version"] == "1.0+cuda"
    assert set(fingerprint) == {
        "name",
        "version",
        "metadata_sha256",
        "record_sha256",
        "wheel_sha256",
    }
    assert "/" not in str(fingerprint) and "\\" not in str(fingerprint)


@pytest.mark.parametrize(
    "metadata,version",
    [({}, "1.0"), ({"Name": "bad/name"}, "1.0"), ({"Name": "valid"}, "../1")],
)
def test_distribution_fingerprint_rejects_malformed_metadata(metadata, version):
    with pytest.raises(ValueError, match="installed distribution"):
        _distribution_fingerprint(_fake_distribution(metadata, version))


def test_module_origin_is_inside_selected_distribution_and_matches_record(tmp_path):
    origin = tmp_path / "example" / "__init__.py"
    origin.parent.mkdir()
    origin.write_bytes(b"value = 1\n")
    digest = hashlib.sha256(origin.read_bytes()).digest()

    class RecordPath(str):
        hash = SimpleNamespace(
            mode="sha256",
            value=base64.urlsafe_b64encode(digest).decode().rstrip("="),
        )

    distribution = SimpleNamespace(
        locate_file=lambda _path: tmp_path,
        files=[RecordPath("example/__init__.py")],
    )
    module = SimpleNamespace(__file__=str(origin))

    assert _module_origin_fingerprint(
        distribution, module, distribution_name="example"
    ) == {
        "module_origin_relative_path": "example/__init__.py",
        "module_origin_sha256": digest.hex(),
    }

    origin.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="differs from RECORD"):
        _module_origin_fingerprint(
            distribution, module, distribution_name="example"
        )


def _record_path(path: str, raw: bytes, *, size: int | None = None):
    digest = hashlib.sha256(raw).digest()

    class RecordPath(str):
        hash = SimpleNamespace(
            mode="sha256",
            value=base64.urlsafe_b64encode(digest).decode().rstrip("="),
        )

    value = RecordPath(path)
    value.size = len(raw) if size is None else size
    return value


def test_distribution_record_integrity_hashes_all_files_inside_installation(tmp_path):
    installation = tmp_path / "venv"
    site_packages = installation / "lib/python/site-packages"
    package = site_packages / "example/__init__.py"
    command = installation / "bin/example"
    package.parent.mkdir(parents=True)
    command.parent.mkdir(parents=True)
    package.write_bytes(b"PACKAGE\n")
    command.write_bytes(b"COMMAND\n")
    files = [
        _record_path("example/__init__.py", b"PACKAGE\n"),
        _record_path("../../../bin/example", b"COMMAND\n"),
    ]
    distribution = SimpleNamespace(
        locate_file=lambda _path: site_packages,
        files=files,
    )

    proof, verified = _distribution_record_integrity(
        distribution,
        distribution_name="example",
        installation_roots=(installation,),
    )

    assert proof == {
        "record_verified_file_count": 2,
        "record_verified_total_bytes": 16,
        "record_verified_files_sha256": canonical_sha256(verified),
        "record_pyc_mismatch_count": 0,
    }
    assert [entry["path"] for entry in verified] == [
        "bin/example",
        "lib/python/site-packages/example/__init__.py",
    ]


@pytest.mark.parametrize(
    "failure", ["symlink", "parent_symlink", "missing", "mismatch", "escape"]
)
def test_distribution_record_integrity_fails_closed(tmp_path, failure):
    installation = tmp_path / "venv"
    site_packages = installation / "lib/python/site-packages"
    site_packages.mkdir(parents=True)
    expected = b"EXPECTED\n"
    record_name = "example/value.bin"
    if failure == "escape":
        record_name = "../../../../outside.bin"
    else:
        target = site_packages / record_name
        if failure == "parent_symlink":
            real_parent = tmp_path / "real-parent"
            real_parent.mkdir()
            (site_packages / "example").symlink_to(
                real_parent, target_is_directory=True
            )
        else:
            target.parent.mkdir(parents=True)
        if failure == "symlink":
            real = tmp_path / "real.bin"
            real.write_bytes(expected)
            target.symlink_to(real)
        elif failure == "mismatch":
            target.write_bytes(b"MISMATCH\n")
        elif failure == "parent_symlink":
            target.write_bytes(expected)
        elif failure != "missing":
            target.write_bytes(expected)
    distribution = SimpleNamespace(
        locate_file=lambda _path: site_packages,
        files=[_record_path(record_name, expected)],
    )

    with pytest.raises(ValueError, match="RECORD"):
        _distribution_record_integrity(
            distribution,
            distribution_name="example",
            installation_roots=(installation,),
        )


def test_distribution_record_integrity_binds_pyc_runtime_mismatch(tmp_path):
    installation = tmp_path / "venv"
    site_packages = installation / "lib/python/site-packages"
    target = site_packages / "example/__pycache__/value.pyc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"RUNTIME-PYC\n")
    distribution = SimpleNamespace(
        locate_file=lambda _path: site_packages,
        files=[_record_path("example/__pycache__/value.pyc", b"WHEEL-PYC\n")],
    )

    proof, verified = _distribution_record_integrity(
        distribution,
        distribution_name="example",
        installation_roots=(installation,),
    )

    assert proof["record_pyc_mismatch_count"] == 1
    assert verified[0]["matches_record"] is False
    assert verified[0]["declared_sha256"] != verified[0]["actual_sha256"]
    assert proof["record_verified_files_sha256"] == canonical_sha256(verified)


def test_git_source_manifest_rejects_tracked_gitlinks(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    _git("config", "user.name", "Test", cwd=nested)
    _git("config", "user.email", "test@example.invalid", cwd=nested)
    (nested / "payload.py").write_text("SENTINEL = True\n")
    _git("add", "payload.py", cwd=nested)
    _git("commit", "-qm", "nested", cwd=nested)

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(nested),
        "vendor/nested",
        cwd=repo,
    )
    _git("commit", "-qm", "gitlink", cwd=repo)

    with pytest.raises(ValueError, match="unsupported Git object"):
        build_git_source_manifest(repo, commit_sha="HEAD")


def test_git_source_manifest_rejects_dirty_extracted_bytes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src/tabicl").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    tracked = repo / "src/tabicl/__init__.py"
    tracked.write_text("SOURCE = 'committed'\n")
    _git("add", "src/tabicl/__init__.py", cwd=repo)
    _git("commit", "-qm", "source", cwd=repo)
    tracked.write_text("SOURCE = 'dirty'\n")

    with pytest.raises(ValueError, match="clean exact checkout"):
        build_git_source_manifest(repo, commit_sha="HEAD")


def test_git_source_manifest_rejects_symlink_in_repository_path(tmp_path):
    actual = tmp_path / "actual"
    repo = actual / "repo"
    (repo / "src/tabicl").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    (repo / "src/tabicl/__init__.py").write_text("SOURCE = 'trusted'\n")
    _git("add", "src/tabicl/__init__.py", cwd=repo)
    _git("commit", "-qm", "source", cwd=repo)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_git_source_manifest(alias / "repo", commit_sha="HEAD")


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


def test_library_source_manifest_loader_rejects_parent_symlink(tmp_path):
    manifest = _source_manifest()
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    path = trusted / "source.json"
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    alias = tmp_path / "alias"
    alias.symlink_to(trusted, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        load_source_manifest(
            alias / path.name,
            expected_sha256=manifest["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
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


def _environment_payload(*, visible_cuda_device_count=1):
    return {
        "python_version": "3.13.5",
        "python_implementation": "CPython",
        "python_executable_sha256": "e" * 64,
        "python_cache_tag": "cpython-313",
        "python_soabi": "cpython-313-x86_64-linux-gnu",
        "platform_system": "Linux",
        "platform_release": "test",
        "platform_machine": "x86_64",
        "torch_version": "2.7.1",
        "numpy_version": "2.0.0",
        "cuda_runtime_version": "12.8",
        "cudnn_version": 9000,
        "environment_fingerprint_schema_version": 2,
        "installed_distributions_sha256": "d" * 64,
        "formal_runtime_distributions": [
            {
                "name": name,
                "version": "1.0",
                "metadata_sha256": "a" * 64,
                "record_sha256": "b" * 64,
                "wheel_sha256": "c" * 64,
                "module": _FORMAL_RUNTIME_MODULES[name],
                "module_version": "1.0",
                "module_origin_relative_path": f"{name}/__init__.py",
                "module_origin_sha256": "f" * 64,
                "record_verified_file_count": 1,
                "record_verified_total_bytes": 1,
                "record_verified_files_sha256": "9" * 64,
                "record_pyc_mismatch_count": 0,
            }
            for name in _FORMAL_RUNTIME_DISTRIBUTIONS
        ],
        "unavailable_formal_runtime_distributions": [],
        "flash_attn3_available": True,
        "nccl_version": [2, 27, 5],
        "visible_cuda_device_count": visible_cuda_device_count,
    }


def _optimizer_protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "optimizer": {
            "type": "torch.optim.adamw.AdamW",
            "groups": [
                {
                    "parameters": [
                        {
                            "name": "layer.bias",
                            "shape": [2],
                            "dtype": "torch.float32",
                            "requires_grad": True,
                        },
                        {
                            "name": "layer.weight",
                            "shape": [2, 3],
                            "dtype": "torch.float32",
                            "requires_grad": True,
                        },
                    ],
                    "base_lr": 1e-4,
                    "static": {
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                    },
                }
            ],
        },
        "scheduler": {
            "type": "torch.optim.lr_scheduler.LambdaLR",
            "algorithm": "constant",
            "static": {"max_steps": 500_000},
        },
        "scaler": {
            "type": "torch.amp.grad_scaler.GradScaler",
            "enabled": False,
            "static": {
                "device": "cuda",
                "init_scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
            },
        },
    }


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
        environment=_environment_payload(),
        model_config=_model_config(mode),
        state_dict=_state_dict(),
        prior_stream=_prior_stream(),
        optimizer_config=_optimizer_protocol(),
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


def _formal_trainer(tmp_path, *, source_sha=None, max_checkpoint_bytes=1 << 20):
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
    environment = make_manifest("environment", _environment_payload())
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
            "--max_checkpoint_bytes",
            str(max_checkpoint_bytes),
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
    trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters(), lr=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _step: 1.0
    )
    trainer.scaler = torch.GradScaler("cuda", enabled=False)
    trainer._test_environment_manifest = environment
    return trainer


def _stub_trainer_environment(monkeypatch, trainer):
    import tabicl.train._run as run_module

    monkeypatch.setattr(
        run_module,
        "runtime_environment_manifest",
        lambda *, require_formal_runtime: trainer._test_environment_manifest,
    )


def test_formal_parser_defaults_off_and_exposes_explicit_trust_inputs():
    from tabicl.train._train_config import build_parser

    config = build_parser().parse_args([])
    assert config.formal_training is False
    assert config.max_checkpoint_bytes is None
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


def test_trainer_formal_checkpoint_persists_validated_provenance(tmp_path, monkeypatch):
    trainer = _formal_trainer(tmp_path)
    _stub_trainer_environment(monkeypatch, trainer)
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


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "100"])
def test_formal_trainer_requires_positive_overlay_checkpoint_ceiling(tmp_path, value):
    trainer = _formal_trainer(tmp_path)
    trainer.config.max_checkpoint_bytes = value
    with pytest.raises(ValueError, match="positive max_checkpoint_bytes overlay ceiling"):
        trainer.configure_formal_provenance()


def test_formal_trainer_passes_overlay_checkpoint_ceiling_to_atomic_save(
    tmp_path, monkeypatch
):
    import tabicl.train._run as run_module

    trainer = _formal_trainer(tmp_path, max_checkpoint_bytes=123_456)
    _stub_trainer_environment(monkeypatch, trainer)
    trainer.configure_formal_provenance()
    observed = {}

    def capture_save(value, path, *, max_bytes):
        observed.update(value=value, path=path, max_bytes=max_bytes)

    monkeypatch.setattr(run_module, "atomic_torch_save", capture_save)
    trainer.save_checkpoint("step-1.ckpt")

    assert observed["max_bytes"] == 123_456
    assert observed["path"].endswith("/step-1.ckpt")
    assert observed["value"]["provenance"] == trainer.formal_provenance


def test_formal_trainer_fails_closed_on_external_source_digest_drift(tmp_path):
    trainer = _formal_trainer(tmp_path, source_sha="0" * 64)
    with pytest.raises(ValueError, match="source.*external expected"):
        trainer.configure_formal_provenance()
