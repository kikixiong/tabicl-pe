from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from pe_mechanism.manifest import ArtifactDigest, FileDigest, InputDigest
from pe_mechanism.provenance import (
    RunTransaction,
    assert_manifest_matches_verified,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_file,
    verify_git_tree,
    verify_run_directory,
    verify_run_inputs,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _clean_repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("version one\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "initial")
    return _git(root, "rev-parse", "HEAD")


def _verified_inputs(root: Path):
    repository = root / "repository"
    _clean_repository(repository)
    inputs = root / "inputs"
    inputs.mkdir()
    config = inputs / "config.json"
    checkpoint = inputs / "checkpoint.bin"
    dataset = inputs / "dataset-manifest.json"
    config.write_text('{"schema_version": 2}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text('{"datasets": ["a"]}\n', encoding="utf-8")
    return verify_run_inputs(
        configuration_path=config,
        checkpoint_path=checkpoint,
        dataset_manifest_path=dataset,
        training_code_root=repository,
        model_code_root=repository,
        analysis_code_root=repository,
        command="collect",
        model_family="tabicl-v2",
        model_revision="step-180000",
        condition="stable-rope",
        sites=("row.blocks[0]",),
        seed=42,
    )


def _manifest(artifacts: tuple[ArtifactDigest, ...], verified):
    return manifest_from_verified_inputs(
        verified,
        artifacts=artifacts,
        created_at_utc="2026-08-07T12:00:00Z",
    )


def test_stream_hash_uses_actual_bytes_and_expected_hash_is_only_assertion(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"actual checkpoint bytes")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    verified = verify_file(path, expected_sha256=expected)
    assert verified.digest == FileDigest(expected, len(path.read_bytes()))
    with pytest.raises(ValueError, match="does not match"):
        verify_file(path, expected_sha256="0" * 64)

    alias = tmp_path / "checkpoint-link"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        verify_file(alias)


def test_exact_config_bytes_are_hashed_without_self_reference(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    raw = b'{"seed": 42, "provenance": {"condition": "none"}}\n'
    path.write_bytes(raw)
    verified = load_verified_json_config(path)

    assert verified.file.digest.sha256 == hashlib.sha256(raw).hexdigest()
    assert "configuration_sha256" not in json.dumps(verified.data)

    path.write_text('{"nested": {"configuration_sha256": "' + "0" * 64 + '"}}')
    with pytest.raises(ValueError, match="must not appear"):
        load_verified_json_config(path)


def test_git_head_and_cleanliness_are_measured_not_declared(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = _clean_repository(repository)
    evidence = verify_git_tree(repository, expected_sha=head)
    assert evidence.head_sha == head
    assert evidence.evidence_level == "strict"

    with pytest.raises(ValueError, match="actual HEAD"):
        verify_git_tree(repository, expected_sha="0" * 40)

    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        verify_git_tree(repository)
    downgraded = verify_git_tree(repository, allow_exploratory_legacy=True)
    assert downgraded.evidence_level == "exploratory_legacy"
    assert downgraded.legacy_reasons == ("dirty_git_tree",)


def test_verified_run_inputs_populate_manifest_from_actual_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = _clean_repository(repository)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = inputs / "config.json"
    checkpoint = inputs / "checkpoint.bin"
    dataset = inputs / "dataset-manifest.json"
    config.write_text('{"schema_version": 2}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text('{"datasets": ["a"]}\n', encoding="utf-8")

    verified = verify_run_inputs(
        configuration_path=config,
        checkpoint_path=checkpoint,
        dataset_manifest_path=dataset,
        training_code_root=repository,
        model_code_root=repository,
        analysis_code_root=repository,
        command="ablate",
        model_family="tabicl-v2",
        model_revision="step-180000",
        condition="stable-rope",
        sites=("row.blocks[0]",),
        seed=42,
    )
    artifact = ArtifactDigest("result.json", "7" * 64, 7)
    manifest = manifest_from_verified_inputs(
        verified,
        artifacts=(artifact,),
        created_at_utc="2026-08-07T12:00:00Z",
    )

    assert manifest.training_code_sha == head
    assert manifest.model_code_sha == head
    assert manifest.analysis_code_sha == head
    assert manifest.configuration.sha256 == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest.checkpoint.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest.dataset_manifest.sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match"):
        verify_run_inputs(
            configuration_path=config,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset,
            training_code_root=repository,
            model_code_root=repository,
            analysis_code_root=repository,
            command="ablate",
            model_family="tabicl-v2",
            model_revision="step-180000",
            condition="stable-rope",
            sites=("row.blocks[0]",),
            seed=42,
            expected_checkpoint_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="multiple input roles"):
        verify_run_inputs(
            configuration_path=config,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset,
            training_code_root=repository,
            model_code_root=repository,
            analysis_code_root=repository,
            command="ablate",
            model_family="tabicl-v2",
            model_revision="step-180000",
            condition="stable-rope",
            sites=("row.blocks[0]",),
            seed=42,
            additional_input_paths={"duplicate.checkpoint": checkpoint},
        )


def test_additional_inputs_are_measured_path_free_and_change_checked(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _clean_repository(repository)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = inputs / "config.json"
    checkpoint = inputs / "checkpoint.bin"
    dataset = inputs / "dataset-manifest.json"
    activations = inputs / "activations.npy"
    config.write_text('{"schema_version": 2}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text('{"datasets": ["a"]}\n', encoding="utf-8")
    activations.write_bytes(b"activation bytes")
    activation_hash = hashlib.sha256(activations.read_bytes()).hexdigest()

    verified = verify_run_inputs(
        configuration_path=config,
        checkpoint_path=checkpoint,
        dataset_manifest_path=dataset,
        training_code_root=repository,
        model_code_root=repository,
        analysis_code_root=repository,
        command="collect",
        model_family="tabicl-v2",
        model_revision="step-180000",
        condition="stable-rope",
        sites=("row.blocks[0]",),
        seed=42,
        additional_input_paths={"activations.primary": activations},
        expected_additional_sha256={"activations.primary": activation_hash},
    )
    manifest = manifest_from_verified_inputs(
        verified,
        artifacts=(ArtifactDigest("result.json", "7" * 64, 7),),
    )
    assert manifest.inputs == (
        InputDigest("activations.primary", activation_hash, len(activations.read_bytes())),
    )
    assert str(inputs) not in json.dumps(manifest.to_dict())

    activations.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        verified.assert_unchanged()


def test_configured_run_paths_and_expected_values_are_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    head = _clean_repository(repository)
    import pe_mechanism.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "__file__", str(repository / "tracked.txt"))
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"datasets":["a"]}\n', encoding="utf-8")
    activation = tmp_path / "activation.bin"
    activation.write_bytes(b"activation")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "provenance": {
                    "model_family": "tabicl-v2",
                    "model_revision": "step-180000",
                    "condition": "stable-rope",
                    "sites": ["row.blocks[0]"],
                    "checkpoint_path": str(checkpoint),
                    "dataset_manifest_path": str(dataset),
                    "training_code_root": str(repository),
                    "model_code_root": str(repository),
                    "analysis_code_root": str(repository),
                    "expected_training_code_sha": head,
                    "expected_checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    configuration = load_verified_json_config(config)
    context = verify_configured_run_inputs(
        configuration,
        command="collect",
        seed=42,
        additional_input_paths={"activations.primary": activation},
    )
    assert context.inputs.training_code.head_sha == head
    assert context.additional_file("activations.primary").path == activation

    data = dict(configuration.data)
    provenance = dict(data["provenance"])
    provenance["expected_training_code_sha"] = "0" * 40
    data["provenance"] = provenance
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="actual HEAD"):
        verify_configured_run_inputs(
            load_verified_json_config(config), command="collect", seed=42
        )


def test_transaction_atomically_commits_and_verifier_detects_tampering(tmp_path: Path) -> None:
    verified_inputs = _verified_inputs(tmp_path)
    source = verified_inputs.analysis_code.root
    final = tmp_path / "published-run"
    with RunTransaction(final, source_roots=(source,)) as transaction:
        (transaction.staging_dir / "a.json").write_text('{"value": 1}\n', encoding="utf-8")
        (transaction.staging_dir / "b.bin").write_bytes(b"binary")
        artifacts = transaction.artifact_digests(("a.json", "b.bin"))
        transaction.commit(
            _manifest(artifacts, verified_inputs), verified_inputs=verified_inputs
        )

    assert final.is_dir()
    assert not list(tmp_path.glob(".published-run.*.staging"))
    verified = verify_run_directory(final)
    assert [artifact.name for artifact in verified.artifacts] == ["a.json", "b.bin"]

    (final / "a.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        verify_run_directory(final)


def test_transaction_failure_injection_leaves_no_final_or_staging_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    final = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="injected"):
        with RunTransaction(final, source_roots=(source,)) as transaction:
            (transaction.staging_dir / "first.json").write_text("{}\n", encoding="utf-8")
            raise RuntimeError("injected failure after first artifact")

    assert not final.exists()
    assert not list(tmp_path.glob(".failed-run.*.staging"))


def test_transaction_rejects_forged_manifest_and_changed_verified_input(tmp_path: Path) -> None:
    verified_inputs = _verified_inputs(tmp_path)
    source = verified_inputs.analysis_code.root
    final = tmp_path / "forged-run"
    with pytest.raises(ValueError, match="does not match actual"):
        with RunTransaction(final, source_roots=(source,)) as transaction:
            (transaction.staging_dir / "result.json").write_text("{}\n", encoding="utf-8")
            artifacts = transaction.artifact_digests(("result.json",))
            manifest = _manifest(artifacts, verified_inputs)
            forged = replace(manifest, checkpoint=FileDigest("f" * 64, 1))
            transaction.commit(forged, verified_inputs=verified_inputs)
    assert not final.exists()

    changed_final = tmp_path / "changed-input-run"
    with pytest.raises(ValueError, match="does not match"):
        with RunTransaction(changed_final, source_roots=(source,)) as transaction:
            (transaction.staging_dir / "result.json").write_text("{}\n", encoding="utf-8")
            artifacts = transaction.artifact_digests(("result.json",))
            manifest = _manifest(artifacts, verified_inputs)
            verified_inputs.checkpoint.path.write_bytes(b"changed checkpoint")
            transaction.commit(manifest, verified_inputs=verified_inputs)
    assert not changed_final.exists()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("command", "ablate"),
        ("model_family", "other-model"),
        ("model_revision", "step-190000"),
        ("condition", "none"),
        ("sites", ("row.blocks[1]",)),
        ("seed", 43),
    ],
)
def test_manifest_contract_fields_cannot_be_changed_after_verification(
    tmp_path: Path, field: str, forged_value: object
) -> None:
    verified_inputs = _verified_inputs(tmp_path)
    manifest = _manifest(
        (ArtifactDigest("result.json", "7" * 64, 7),), verified_inputs
    )

    with pytest.raises(ValueError, match="does not match actual"):
        assert_manifest_matches_verified(
            replace(manifest, **{field: forged_value}), verified_inputs
        )


def test_transaction_rejects_extra_files_symlinks_and_existing_final(tmp_path: Path) -> None:
    verified_inputs = _verified_inputs(tmp_path)
    source = verified_inputs.analysis_code.root
    final = tmp_path / "extra-run"
    with pytest.raises(ValueError, match="differ"):
        with RunTransaction(final, source_roots=(source,)) as transaction:
            (transaction.staging_dir / "expected.json").write_text("{}\n", encoding="utf-8")
            (transaction.staging_dir / "extra.json").write_text("{}\n", encoding="utf-8")
            artifacts = transaction.artifact_digests(("expected.json",))
            transaction.commit(
                _manifest(artifacts, verified_inputs), verified_inputs=verified_inputs
            )
    assert not final.exists()

    symlink_final = tmp_path / "symlink-run"
    with pytest.raises(ValueError, match="symlink"):
        with RunTransaction(symlink_final, source_roots=(source,)) as transaction:
            target = transaction.staging_dir / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            (transaction.staging_dir / "alias.json").symlink_to(target)
            transaction.artifact_digests(("alias.json",))
    assert not symlink_final.exists()

    existing = tmp_path / "existing-run"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        with RunTransaction(existing, source_roots=(source,)):
            pass


def test_transaction_rejects_source_tree_and_source_ancestor_targets(tmp_path: Path) -> None:
    source = tmp_path / "source" / "project"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        RunTransaction(source / "output", source_roots=(source,))
    with pytest.raises(ValueError, match="ancestor"):
        RunTransaction(tmp_path / "source", source_roots=(source,))
