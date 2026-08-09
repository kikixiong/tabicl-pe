from __future__ import annotations

import json
from pathlib import Path

import pytest

from pe_mechanism.manifest import (
    ArtifactDigest,
    FileDigest,
    LegacyRunManifest,
    RunManifest,
    new_manifest,
    read_manifest,
    validate_output_dir,
    write_manifest,
)


HASH = "a" * 64
GIT_SHA = "b" * 40


def make_manifest(**overrides: object) -> RunManifest:
    values: dict[str, object] = {
        "command": "collect",
        "model_family": "tabicl-v2",
        "model_revision": "step-180000",
        "training_code_sha": GIT_SHA,
        "model_code_sha": "c" * 40,
        "analysis_code_sha": "d" * 40,
        "configuration": FileDigest("1" * 64, 101),
        "checkpoint": FileDigest("2" * 64, 202),
        "dataset_manifest": FileDigest("3" * 64, 303),
        "condition": "stable-rope",
        "sites": ("row.blocks[0]",),
        "seed": 42,
        "artifacts": (ArtifactDigest("activation-index.json", "4" * 64, 404),),
        "created_at_utc": "2026-08-07T12:00:00Z",
    }
    values.update(overrides)
    return new_manifest(**values)  # type: ignore[arg-type]


def test_output_dir_must_be_absolute_external_and_not_source_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="absolute"):
        validate_output_dir("relative/output", source_root=source)
    with pytest.raises(ValueError, match="outside"):
        validate_output_dir(source / "results", source_root=source)
    with pytest.raises(ValueError, match="ancestor"):
        validate_output_dir(tmp_path, source_root=source)
    with pytest.raises(ValueError, match="root|top-level"):
        validate_output_dir(Path(tmp_path.anchor), source_root=source)

    external = tmp_path / "external"
    assert validate_output_dir(external, source_root=source) == external.resolve()
    mount = tmp_path / "mounted-output"
    mount.mkdir()
    monkeypatch.setattr(
        "pe_mechanism.manifest.os.path.ismount", lambda path: Path(path) == mount
    )
    with pytest.raises(ValueError, match="mount point"):
        validate_output_dir(mount, source_root=source)


def test_output_dir_resolves_symlinks_before_boundary_check(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        validate_output_dir(alias / "results", source_root=source)


def test_manifest_round_trip_has_verified_inputs_artifacts_and_no_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "external"

    original = make_manifest()
    output.mkdir()
    path = output / "manifest.json"
    path.write_text(json.dumps(original.to_dict(), sort_keys=True), encoding="utf-8")
    loaded = read_manifest(path)
    assert loaded == original

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["configuration"] == {"sha256": "1" * 64, "size_bytes": 101}
    assert raw["artifacts"][0]["name"] == "activation-index.json"
    forbidden_fragments = ("path", "host", "slurm", "scheduler", "job")
    assert not any(fragment in key.lower() for key in _all_keys(raw) for fragment in forbidden_fragments)
    assert not any("/" in value or "\\" in value for value in _all_strings(raw))


def test_manifest_rejects_paths_unknown_fields_and_unsorted_artifacts() -> None:
    with pytest.raises(ValueError, match="portable identifier"):
        make_manifest(model_revision="/internal/checkpoint.ckpt")
    with pytest.raises(ValueError, match="flat portable"):
        ArtifactDigest("nested/result.json", HASH, 1)

    data = make_manifest().to_dict()
    data["job_id"] = "12345"
    with pytest.raises(ValueError, match="unknown=.*job_id"):
        RunManifest.from_dict(data)

    with pytest.raises(ValueError, match="sorted"):
        make_manifest(
            artifacts=(
                ArtifactDigest("z.json", "4" * 64, 1),
                ArtifactDigest("a.json", "5" * 64, 1),
            )
        )


def test_exploratory_legacy_downgrade_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires at least one reason"):
        make_manifest(evidence_level="exploratory_legacy")
    with pytest.raises(ValueError, match="strict evidence"):
        make_manifest(legacy_reasons=("dirty_model_code",))
    manifest = make_manifest(
        evidence_level="exploratory_legacy", legacy_reasons=("dirty_model_code",)
    )
    assert manifest.evidence_level == "exploratory_legacy"


def test_manifest_writer_requires_actual_verified_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "external"
    with pytest.raises(ValueError, match="VerifiedRunInputs"):
        write_manifest(make_manifest(), output, source_root=source)


def test_schema_v1_is_read_only_and_marked_unverified(tmp_path: Path) -> None:
    legacy_data = {
        "schema_version": 1,
        "command": "collect",
        "model_family": "tabicl-v2",
        "model_revision": "step-180000",
        "model_code_sha": GIT_SHA,
        "checkpoint_sha256": HASH,
        "dataset_manifest_sha256": HASH,
        "configuration_sha256": HASH,
        "condition": "stable-rope",
        "sites": ["row.blocks[0]"],
        "seed": 42,
        "created_at_utc": "2026-08-07T12:00:00Z",
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_data), encoding="utf-8")
    loaded = read_manifest(path)

    assert isinstance(loaded, LegacyRunManifest)
    assert loaded.legacy_unverified is True
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="read-only"):
        write_manifest(loaded, tmp_path / "external", source_root=source)  # type: ignore[arg-type]


def _all_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_all_keys(child))
    return keys


def _all_strings(value: object) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            strings.extend(_all_strings(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(_all_strings(child))
    return strings
