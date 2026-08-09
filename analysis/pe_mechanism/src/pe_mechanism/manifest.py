"""Strict, path-free provenance manifests for mechanism-analysis runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Mapping


SCHEMA_VERSION = 2
Command = Literal[
    "collect",
    "ablate",
    "localize",
    "train-repr",
    "reconstruction-sensitivity",
    "model-causal",
    "select-features",
    "confirm-features",
]
EvidenceLevel = Literal["strict", "exploratory_legacy"]

_COMMANDS = {
    "collect",
    "ablate",
    "localize",
    "train-repr",
    "reconstruction-sensitivity",
    "model-causal",
    "select-features",
    "confirm-features",
}
_LEGACY_COMMANDS = (
    _COMMANDS - {"localize", "select-features", "confirm-features"}
) | {
    "causal"
}
_EVIDENCE_LEVELS = {"strict", "exploratory_legacy"}
_PORTABLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\[\]-]{0,191}$")
_PORTABLE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def source_tree_root(start: Path | None = None) -> Path:
    """Find the containing Git source tree, with a package-root fallback."""

    starts = [start, Path(__file__), Path.cwd()]
    for candidate in starts:
        if candidate is None:
            continue
        resolved = candidate.resolve(strict=False)
        if not resolved.is_dir():
            resolved = resolved.parent
        for directory in (resolved, *resolved.parents):
            if (directory / ".git").exists():
                return directory
    return Path(__file__).resolve().parents[2]


def validate_output_dir(
    output_dir: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Normalize an external target and reject broad or source-overlapping paths."""

    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("output directory must be a path-like value")
    raw_path = Path(output_dir)
    if not raw_path.is_absolute():
        raise ValueError("output directory must be an absolute path")

    resolved_output = raw_path.resolve(strict=False)
    anchor = Path(resolved_output.anchor)
    if resolved_output == anchor or resolved_output.parent == anchor:
        raise ValueError("output directory must not be a filesystem root or top-level target")
    if resolved_output.exists() and os.path.ismount(resolved_output):
        raise ValueError("output directory must not be a mount point")

    resolved_source = (
        Path(source_root).resolve(strict=False) if source_root is not None else source_tree_root()
    )
    if _is_within(resolved_output, resolved_source):
        raise ValueError("output directory must be outside the source tree")
    if _is_within(resolved_source, resolved_output):
        raise ValueError("output directory must not be an ancestor of the source tree")
    return resolved_output


def _portable_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not _PORTABLE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a portable identifier without paths or whitespace")


def _content_hash(name: str, value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase hexadecimal content hash")


def _non_negative_size(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FileDigest:
    """Path-free digest of one verified input file."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _content_hash("sha256", self.sha256, _SHA256)
        _non_negative_size("size_bytes", self.size_bytes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileDigest":
        _require_exact_fields(data, {"sha256", "size_bytes"}, "file digest")
        return cls(sha256=data["sha256"], size_bytes=data["size_bytes"])


@dataclass(frozen=True)
class ArtifactDigest:
    """Digest of one flat, portable output artifact."""

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PORTABLE_FILENAME.fullmatch(self.name):
            raise ValueError("artifact name must be a flat portable filename")
        if self.name == "manifest.json":
            raise ValueError("manifest.json cannot digest itself")
        _content_hash("artifact sha256", self.sha256, _SHA256)
        _non_negative_size("artifact size_bytes", self.size_bytes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactDigest":
        _require_exact_fields(data, {"name", "sha256", "size_bytes"}, "artifact digest")
        return cls(name=data["name"], sha256=data["sha256"], size_bytes=data["size_bytes"])


@dataclass(frozen=True)
class InputDigest:
    """Path-free digest of one named, workflow-specific input file."""

    role: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _portable_identifier("input role", self.role)
        _content_hash("input sha256", self.sha256, _SHA256)
        _non_negative_size("input size_bytes", self.size_bytes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputDigest":
        _require_exact_fields(data, {"role", "sha256", "size_bytes"}, "input digest")
        return cls(role=data["role"], sha256=data["sha256"], size_bytes=data["size_bytes"])


@dataclass(frozen=True)
class RunManifest:
    """Schema-v2 manifest populated only from verified evidence."""

    schema_version: int
    command: Command
    model_family: str
    model_revision: str
    training_code_sha: str
    model_code_sha: str
    analysis_code_sha: str
    configuration: FileDigest
    checkpoint: FileDigest
    dataset_manifest: FileDigest
    inputs: tuple[InputDigest, ...]
    condition: str
    sites: tuple[str, ...]
    seed: int
    artifacts: tuple[ArtifactDigest, ...]
    evidence_level: EvidenceLevel
    legacy_reasons: tuple[str, ...]
    created_at_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.command not in _COMMANDS:
            raise ValueError(f"unsupported command: {self.command!r}")
        for name in ("model_family", "model_revision", "condition"):
            _portable_identifier(name, getattr(self, name))
        for site in self.sites:
            _portable_identifier("site", site)
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("sites must not contain duplicates")
        input_roles = [item.role for item in self.inputs]
        if len(set(input_roles)) != len(input_roles):
            raise ValueError("input roles must be unique")
        if input_roles != sorted(input_roles):
            raise ValueError("inputs must be sorted by role")
        for name in ("training_code_sha", "model_code_sha", "analysis_code_sha"):
            _content_hash(name, getattr(self, name), _GIT_SHA)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.artifacts:
            raise ValueError("a completed run must contain at least one artifact")
        names = [artifact.name for artifact in self.artifacts]
        if len(set(names)) != len(names):
            raise ValueError("artifact names must be unique")
        if names != sorted(names):
            raise ValueError("artifacts must be sorted by name")
        if self.evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError(f"unsupported evidence_level: {self.evidence_level!r}")
        for reason in self.legacy_reasons:
            _portable_identifier("legacy reason", reason)
        if len(set(self.legacy_reasons)) != len(self.legacy_reasons):
            raise ValueError("legacy_reasons must not contain duplicates")
        if self.evidence_level == "strict" and self.legacy_reasons:
            raise ValueError("strict evidence cannot have legacy reasons")
        if self.evidence_level == "exploratory_legacy" and not self.legacy_reasons:
            raise ValueError("exploratory legacy evidence requires at least one reason")
        _validate_timestamp(self.created_at_utc)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sites"] = list(self.sites)
        data["inputs"] = [asdict(item) for item in self.inputs]
        data["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        data["legacy_reasons"] = list(self.legacy_reasons)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        expected = set(cls.__dataclass_fields__)
        _require_exact_fields(data, expected, "schema-v2 manifest")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        values = dict(data)
        values["sites"] = _string_tuple(values["sites"], "sites")
        values["legacy_reasons"] = _string_tuple(values["legacy_reasons"], "legacy_reasons")
        inputs = values["inputs"]
        if not isinstance(inputs, list) or not all(isinstance(item, Mapping) for item in inputs):
            raise ValueError("inputs must be a list of objects")
        values["inputs"] = tuple(InputDigest.from_dict(item) for item in inputs)
        artifacts = values["artifacts"]
        if not isinstance(artifacts, list) or not all(isinstance(item, Mapping) for item in artifacts):
            raise ValueError("artifacts must be a list of objects")
        values["artifacts"] = tuple(ArtifactDigest.from_dict(item) for item in artifacts)
        for key in ("configuration", "checkpoint", "dataset_manifest"):
            if not isinstance(values[key], Mapping):
                raise ValueError(f"{key} must be a file digest object")
            values[key] = FileDigest.from_dict(values[key])
        return cls(**values)


@dataclass(frozen=True)
class LegacyRunManifest:
    """Read-only representation of a schema-v1 manifest."""

    schema_version: int
    command: str
    model_family: str
    model_revision: str
    model_code_sha: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    configuration_sha256: str
    condition: str
    sites: tuple[str, ...]
    seed: int
    created_at_utc: str
    legacy_unverified: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("legacy manifest schema_version must be 1")
        if self.command not in _LEGACY_COMMANDS:
            raise ValueError(f"unsupported legacy command: {self.command!r}")
        for name in ("model_family", "model_revision", "condition"):
            _portable_identifier(name, getattr(self, name))
        for site in self.sites:
            _portable_identifier("site", site)
        _content_hash("model_code_sha", self.model_code_sha, _GIT_SHA)
        for name in ("checkpoint_sha256", "dataset_manifest_sha256", "configuration_sha256"):
            _content_hash(name, getattr(self, name), _SHA256)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _validate_timestamp(self.created_at_utc)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sites"] = list(self.sites)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegacyRunManifest":
        expected = {
            "schema_version",
            "command",
            "model_family",
            "model_revision",
            "model_code_sha",
            "checkpoint_sha256",
            "dataset_manifest_sha256",
            "configuration_sha256",
            "condition",
            "sites",
            "seed",
            "created_at_utc",
        }
        _require_exact_fields(data, expected, "schema-v1 manifest")
        values = dict(data)
        values["sites"] = _string_tuple(values["sites"], "sites")
        return cls(**values)


def new_manifest(
    *,
    command: Command,
    model_family: str,
    model_revision: str,
    training_code_sha: str,
    model_code_sha: str,
    analysis_code_sha: str,
    configuration: FileDigest,
    checkpoint: FileDigest,
    dataset_manifest: FileDigest,
    inputs: tuple[InputDigest, ...] = (),
    condition: str,
    sites: tuple[str, ...],
    seed: int,
    artifacts: tuple[ArtifactDigest, ...],
    evidence_level: EvidenceLevel = "strict",
    legacy_reasons: tuple[str, ...] = (),
    created_at_utc: str | None = None,
) -> RunManifest:
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        command=command,
        model_family=model_family,
        model_revision=model_revision,
        training_code_sha=training_code_sha,
        model_code_sha=model_code_sha,
        analysis_code_sha=analysis_code_sha,
        configuration=configuration,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
        inputs=inputs,
        condition=condition,
        sites=sites,
        seed=seed,
        artifacts=artifacts,
        evidence_level=evidence_level,
        legacy_reasons=legacy_reasons,
        created_at_utc=timestamp,
    )


def write_manifest(
    manifest: RunManifest,
    output_dir: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] | None = None,
    verified_inputs: Any | None = None,
) -> Path:
    """Publish a schema-v2 manifest atomically without replacing an existing one."""

    if isinstance(manifest, LegacyRunManifest):
        raise ValueError("schema-v1 manifests are read-only and cannot be written")
    if not isinstance(manifest, RunManifest) or manifest.schema_version != SCHEMA_VERSION:
        raise TypeError("manifest must be a schema-v2 RunManifest")
    if verified_inputs is None:
        raise ValueError("writing a manifest requires actual VerifiedRunInputs evidence")
    # Imported lazily to avoid a module cycle: provenance owns runtime evidence,
    # while this module owns the serialized schema.
    from .provenance import assert_manifest_matches_verified

    assert_manifest_matches_verified(manifest, verified_inputs)
    verified_inputs.assert_unchanged()
    directory = validate_output_dir(output_dir, source_root=source_root)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "manifest.json"
    payload = json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_manifest(path: str | os.PathLike[str]) -> RunManifest | LegacyRunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must contain one JSON object")
    schema_version = data.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        return RunManifest.from_dict(data)
    if schema_version == 1:
        return LegacyRunManifest.from_dict(data)
    raise ValueError(f"unsupported manifest schema_version: {schema_version!r}")


def _require_exact_fields(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    received = set(data)
    if received != expected:
        missing = sorted(expected - received)
        unknown = sorted(received - expected)
        raise ValueError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def _validate_timestamp(value: str) -> None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("created_at_utc must include the UTC timezone")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ArtifactDigest",
    "FileDigest",
    "InputDigest",
    "LegacyRunManifest",
    "RunManifest",
    "SCHEMA_VERSION",
    "new_manifest",
    "read_manifest",
    "source_tree_root",
    "validate_output_dir",
    "write_manifest",
]
