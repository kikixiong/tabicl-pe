"""Verified input evidence and directory-level atomic run publication."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identifiers import require_public_label
from .manifest import (
    ArtifactDigest,
    FileDigest,
    InputDigest,
    LegacyRunManifest,
    RunManifest,
    new_manifest,
    validate_output_dir,
    write_manifest,
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SELF_REFERENTIAL_CONFIG_KEYS = {"configuration_sha256", "configuration_digest"}


@dataclass(frozen=True)
class VerifiedFile:
    """A file whose bytes were hashed while its identity remained stable."""

    digest: FileDigest
    path: Path = field(repr=False, compare=False)
    device: int = field(repr=False, compare=False)
    inode: int = field(repr=False, compare=False)
    mtime_ns: int = field(repr=False, compare=False)

    def read_bytes(self) -> bytes:
        """Read the exact verified inode and re-check its digest and identity."""

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        try:
            before = os.fstat(descriptor)
            expected_identity = (
                self.device,
                self.inode,
                self.digest.size_bytes,
                self.mtime_ns,
            )
            observed_before = (
                int(before.st_dev),
                int(before.st_ino),
                int(before.st_size),
                int(before.st_mtime_ns),
            )
            if observed_before != expected_identity or not stat.S_ISREG(before.st_mode):
                raise RuntimeError("verified input file identity changed before it was read")
            while True:
                chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        observed_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if observed_after != expected_identity:
            raise RuntimeError("verified input file changed while it was being read")
        if not hmac.compare_digest(digest.hexdigest(), self.digest.sha256):
            raise RuntimeError("verified input bytes changed after initial verification")
        return b"".join(chunks)

    def assert_unchanged(self) -> None:
        observed = verify_file(self.path, expected_sha256=self.digest.sha256)
        if (observed.device, observed.inode, observed.mtime_ns) != (
            self.device,
            self.inode,
            self.mtime_ns,
        ):
            raise RuntimeError("verified input file identity changed during the run")


@dataclass(frozen=True)
class VerifiedConfiguration:
    data: Mapping[str, Any]
    file: VerifiedFile


@dataclass(frozen=True)
class GitEvidence:
    """Actual Git HEAD plus clean/explicitly-downgraded working-tree evidence."""

    head_sha: str
    evidence_level: str
    legacy_reasons: tuple[str, ...]
    root: Path = field(repr=False, compare=False)
    status_sha256: str = field(repr=False, compare=False)

    def assert_unchanged(self) -> None:
        observed = verify_git_tree(
            self.root,
            expected_sha=self.head_sha,
            allow_exploratory_legacy=self.evidence_level == "exploratory_legacy",
        )
        if observed.status_sha256 != self.status_sha256:
            raise RuntimeError("Git working-tree state changed during the run")


@dataclass(frozen=True)
class VerifiedNamedInput:
    """One workflow-specific input bound to a portable public role."""

    role: str
    file: VerifiedFile

    def __post_init__(self) -> None:
        # Reuse the serialized-schema validation at the trust boundary.
        _ = self.digest

    @property
    def digest(self) -> InputDigest:
        return InputDigest(
            role=self.role,
            sha256=self.file.digest.sha256,
            size_bytes=self.file.digest.size_bytes,
        )


@dataclass(frozen=True)
class RunContract:
    """Public run fields fixed before any workflow input is consumed."""

    command: str
    model_family: str
    model_revision: str
    condition: str
    sites: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class VerifiedRunInputs:
    configuration: VerifiedConfiguration
    checkpoint: VerifiedFile
    dataset_manifest: VerifiedFile
    training_code: GitEvidence
    model_code: GitEvidence
    analysis_code: GitEvidence
    additional_inputs: tuple[VerifiedNamedInput, ...]
    contract: RunContract
    evidence_level: str
    legacy_reasons: tuple[str, ...]

    def assert_unchanged(self) -> None:
        self.configuration.file.assert_unchanged()
        self.checkpoint.assert_unchanged()
        self.dataset_manifest.assert_unchanged()
        for item in self.additional_inputs:
            item.file.assert_unchanged()
        self.training_code.assert_unchanged()
        self.model_code.assert_unchanged()
        self.analysis_code.assert_unchanged()


@dataclass(frozen=True)
class VerifiedRunContext:
    """Verified runtime evidence plus public, non-derived run labels."""

    inputs: VerifiedRunInputs
    model_family: str
    model_revision: str
    condition: str
    sites: tuple[str, ...]

    def additional_file(self, role: str) -> VerifiedFile:
        matches = [item.file for item in self.inputs.additional_inputs if item.role == role]
        if len(matches) != 1:
            raise KeyError(f"verified additional input role is not unique: {role!r}")
        return matches[0]


def verify_file(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> VerifiedFile:
    """Stream-hash a regular non-symlink file; an expected hash is only an assertion."""

    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("verified input files must not be symlinks")
    resolved = raw_path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("verified input must be a regular file")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError("input file changed while it was being hashed")

    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        expected = str(expected_sha256)
        if not hmac.compare_digest(actual_sha256, expected):
            raise ValueError("expected SHA-256 does not match the actual input bytes")
    return VerifiedFile(
        digest=FileDigest(actual_sha256, int(after.st_size)),
        path=resolved,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
    )


def load_verified_json_config(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> VerifiedConfiguration:
    """Hash the exact config file, parse it, and forbid a self-declared config hash."""

    verified = verify_file(path, expected_sha256=expected_sha256)
    raw = verified.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("run configuration must contain one JSON object")
    self_references = _find_keys(data, _SELF_REFERENTIAL_CONFIG_KEYS)
    if self_references:
        raise ValueError(
            "configuration hashes are computed from exact config bytes and must not appear in config"
        )
    return VerifiedConfiguration(data=data, file=verified)


def verify_git_tree(
    root: str | os.PathLike[str],
    *,
    expected_sha: str | None = None,
    allow_exploratory_legacy: bool = False,
) -> GitEvidence:
    """Read actual HEAD and require a clean tree unless explicitly downgraded."""

    raw_root = Path(root).expanduser()
    if not raw_root.is_absolute():
        raise ValueError("Git root must be an absolute path")
    resolved = raw_root.resolve(strict=True)
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel").strip()).resolve(strict=True)
    head = _git(top_level, "rev-parse", "--verify", "HEAD").strip().lower()
    if expected_sha is not None and not hmac.compare_digest(head, str(expected_sha).lower()):
        raise ValueError("expected Git SHA does not match the actual HEAD")
    status_output = _git(top_level, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status_output)
    if dirty and not allow_exploratory_legacy:
        raise RuntimeError("Git working tree is dirty; strict evidence requires a clean checkout")
    reasons = ("dirty_git_tree",) if dirty else ()
    return GitEvidence(
        head_sha=head,
        evidence_level="exploratory_legacy" if dirty else "strict",
        legacy_reasons=reasons,
        root=top_level,
        status_sha256=hashlib.sha256(status_output.encode("utf-8")).hexdigest(),
    )


def assert_git_commit_is_ancestor(
    repository: GitEvidence, ancestor_sha: str
) -> None:
    """Require ``ancestor_sha`` to be an ancestor of the verified Git HEAD.

    Parent analysis runs may legitimately come from an older clean commit, but
    they must belong to the executing analysis history.  A same-looking commit
    from an unrelated repository is therefore not accepted as lineage.
    """

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository.root),
            "merge-base",
            "--is-ancestor",
            str(ancestor_sha),
            repository.head_sha,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise ValueError(
            "parent analysis_code_sha is not an ancestor of the current analysis code"
        )
    raise ValueError("Git verification failed while checking parent analysis ancestry")


def verify_run_inputs(
    *,
    configuration_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    dataset_manifest_path: str | os.PathLike[str],
    training_code_root: str | os.PathLike[str],
    model_code_root: str | os.PathLike[str],
    analysis_code_root: str | os.PathLike[str],
    command: str,
    model_family: str,
    model_revision: str,
    condition: str,
    sites: Sequence[str],
    seed: int,
    expected_configuration_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_training_code_sha: str | None = None,
    expected_model_code_sha: str | None = None,
    expected_analysis_code_sha: str | None = None,
    additional_input_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    expected_additional_sha256: Mapping[str, str] | None = None,
    allow_exploratory_legacy: bool = False,
) -> VerifiedRunInputs:
    """Verify all fixed inputs used to construct schema-v2 manifest evidence."""

    configuration = load_verified_json_config(
        configuration_path, expected_sha256=expected_configuration_sha256
    )
    return _verify_run_inputs(
        configuration=configuration,
        checkpoint_path=checkpoint_path,
        dataset_manifest_path=dataset_manifest_path,
        training_code_root=training_code_root,
        model_code_root=model_code_root,
        analysis_code_root=analysis_code_root,
        contract=RunContract(
            command=command,
            model_family=model_family,
            model_revision=model_revision,
            condition=condition,
            sites=tuple(sites),
            seed=seed,
        ),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_training_code_sha=expected_training_code_sha,
        expected_model_code_sha=expected_model_code_sha,
        expected_analysis_code_sha=expected_analysis_code_sha,
        additional_input_paths=additional_input_paths,
        expected_additional_sha256=expected_additional_sha256,
        allow_exploratory_legacy=allow_exploratory_legacy,
    )


def verify_configured_run_inputs(
    configuration: VerifiedConfiguration,
    *,
    command: str,
    seed: int,
    additional_input_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    expected_additional_sha256: Mapping[str, str] | None = None,
) -> VerifiedRunContext:
    """Verify paths declared by the hashed config and return measured evidence.

    Expected hashes and Git SHAs are assertions only.  The returned evidence is
    always populated from bytes and repositories observed by this process.
    """

    provenance = configuration.data.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("config provenance must be an object")
    required = {
        "model_family",
        "model_revision",
        "condition",
        "sites",
        "checkpoint_path",
        "dataset_manifest_path",
        "training_code_root",
        "model_code_root",
        "analysis_code_root",
    }
    optional = {
        "expected_checkpoint_sha256",
        "expected_dataset_manifest_sha256",
        "expected_training_code_sha",
        "expected_model_code_sha",
        "expected_analysis_code_sha",
        "allow_exploratory_legacy",
    }
    missing = sorted(required - set(provenance))
    unknown = sorted(set(provenance) - required - optional)
    if missing or unknown:
        raise ValueError(f"provenance fields mismatch: missing={missing}, unknown={unknown}")
    sites = provenance["sites"]
    if not isinstance(sites, list) or not all(isinstance(site, str) for site in sites):
        raise ValueError("provenance sites must be a list of strings")
    allow_legacy = provenance.get("allow_exploratory_legacy", False)
    if not isinstance(allow_legacy, bool):
        raise ValueError("allow_exploratory_legacy must be a JSON boolean")
    for name in (
        "checkpoint_path",
        "dataset_manifest_path",
        "training_code_root",
        "model_code_root",
        "analysis_code_root",
    ):
        if not Path(str(provenance[name])).expanduser().is_absolute():
            raise ValueError(f"provenance {name} must be an absolute path")

    verified = _verify_run_inputs(
        configuration=configuration,
        checkpoint_path=str(provenance["checkpoint_path"]),
        dataset_manifest_path=str(provenance["dataset_manifest_path"]),
        training_code_root=str(provenance["training_code_root"]),
        model_code_root=str(provenance["model_code_root"]),
        analysis_code_root=str(provenance["analysis_code_root"]),
        contract=RunContract(
            command=command,
            model_family=str(provenance["model_family"]),
            model_revision=str(provenance["model_revision"]),
            condition=str(provenance["condition"]),
            sites=tuple(sites),
            seed=seed,
        ),
        expected_checkpoint_sha256=_optional_string(
            provenance, "expected_checkpoint_sha256"
        ),
        expected_dataset_manifest_sha256=_optional_string(
            provenance, "expected_dataset_manifest_sha256"
        ),
        expected_training_code_sha=_optional_string(provenance, "expected_training_code_sha"),
        expected_model_code_sha=_optional_string(provenance, "expected_model_code_sha"),
        expected_analysis_code_sha=_optional_string(provenance, "expected_analysis_code_sha"),
        additional_input_paths=additional_input_paths,
        expected_additional_sha256=expected_additional_sha256,
        allow_exploratory_legacy=allow_legacy,
    )
    executing_analysis = Path(__file__).resolve(strict=True)
    if not executing_analysis.is_relative_to(verified.analysis_code.root):
        raise RuntimeError(
            "analysis_code_root does not contain the executing pe_mechanism package"
        )
    return VerifiedRunContext(
        inputs=verified,
        model_family=str(provenance["model_family"]),
        model_revision=str(provenance["model_revision"]),
        condition=str(provenance["condition"]),
        sites=tuple(sites),
    )


def _verify_run_inputs(
    *,
    configuration: VerifiedConfiguration,
    checkpoint_path: str | os.PathLike[str],
    dataset_manifest_path: str | os.PathLike[str],
    training_code_root: str | os.PathLike[str],
    model_code_root: str | os.PathLike[str],
    analysis_code_root: str | os.PathLike[str],
    contract: RunContract,
    expected_checkpoint_sha256: str | None,
    expected_dataset_manifest_sha256: str | None,
    expected_training_code_sha: str | None,
    expected_model_code_sha: str | None,
    expected_analysis_code_sha: str | None,
    additional_input_paths: Mapping[str, str | os.PathLike[str]] | None,
    expected_additional_sha256: Mapping[str, str] | None,
    allow_exploratory_legacy: bool,
) -> VerifiedRunInputs:
    checkpoint = verify_file(checkpoint_path, expected_sha256=expected_checkpoint_sha256)
    dataset_manifest = verify_file(
        dataset_manifest_path, expected_sha256=expected_dataset_manifest_sha256
    )
    additional_paths = dict(additional_input_paths or {})
    expected_additional = dict(expected_additional_sha256 or {})
    unknown_expected = sorted(set(expected_additional) - set(additional_paths))
    if unknown_expected:
        raise ValueError(
            f"expected hashes reference unknown additional input roles: {unknown_expected}"
        )
    additional_inputs = tuple(
        VerifiedNamedInput(
            role=role,
            file=verify_file(path, expected_sha256=expected_additional.get(role)),
        )
        for role, path in sorted(additional_paths.items())
    )
    role_files = {
        "configuration": configuration.file,
        "checkpoint": checkpoint,
        "dataset_manifest": dataset_manifest,
        **{item.role: item.file for item in additional_inputs},
    }
    identities: dict[tuple[int, int], str] = {}
    for role, file in role_files.items():
        identity = (file.device, file.inode)
        previous = identities.setdefault(identity, role)
        if previous != role:
            raise ValueError(
                "one physical input file cannot satisfy multiple input roles: "
                f"{previous!r}, {role!r}"
            )
    code_specs = (
        ("training", training_code_root, expected_training_code_sha),
        ("model", model_code_root, expected_model_code_sha),
        ("analysis", analysis_code_root, expected_analysis_code_sha),
    )
    code: dict[str, GitEvidence] = {}
    legacy_reasons: list[str] = []
    for role, root, expected in code_specs:
        evidence = verify_git_tree(
            root,
            expected_sha=expected,
            allow_exploratory_legacy=allow_exploratory_legacy,
        )
        code[role] = evidence
        legacy_reasons.extend(f"{role}_{reason}" for reason in evidence.legacy_reasons)
    return VerifiedRunInputs(
        configuration=configuration,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
        training_code=code["training"],
        model_code=code["model"],
        analysis_code=code["analysis"],
        additional_inputs=additional_inputs,
        contract=contract,
        evidence_level="exploratory_legacy" if legacy_reasons else "strict",
        legacy_reasons=tuple(legacy_reasons),
    )


def manifest_from_verified_inputs(
    verified: VerifiedRunInputs,
    *,
    artifacts: Sequence[ArtifactDigest],
    created_at_utc: str | None = None,
) -> RunManifest:
    """Construct a v2 manifest without accepting caller-declared derived hashes."""

    return new_manifest(
        command=verified.contract.command,  # type: ignore[arg-type]
        model_family=verified.contract.model_family,
        model_revision=verified.contract.model_revision,
        training_code_sha=verified.training_code.head_sha,
        model_code_sha=verified.model_code.head_sha,
        analysis_code_sha=verified.analysis_code.head_sha,
        configuration=verified.configuration.file.digest,
        checkpoint=verified.checkpoint.digest,
        dataset_manifest=verified.dataset_manifest.digest,
        inputs=tuple(item.digest for item in verified.additional_inputs),
        condition=verified.contract.condition,
        sites=verified.contract.sites,
        seed=verified.contract.seed,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.name)),
        evidence_level=verified.evidence_level,  # type: ignore[arg-type]
        legacy_reasons=verified.legacy_reasons,
        created_at_utc=created_at_utc,
    )


def assert_manifest_matches_verified(
    manifest: RunManifest, verified: VerifiedRunInputs
) -> None:
    """Reject a manifest whose serialized evidence was not derived from these inputs."""

    expected = (
        verified.training_code.head_sha,
        verified.model_code.head_sha,
        verified.analysis_code.head_sha,
        verified.configuration.file.digest,
        verified.checkpoint.digest,
        verified.dataset_manifest.digest,
        tuple(item.digest for item in verified.additional_inputs),
        verified.contract.command,
        verified.contract.model_family,
        verified.contract.model_revision,
        verified.contract.condition,
        verified.contract.sites,
        verified.contract.seed,
        verified.evidence_level,
        verified.legacy_reasons,
    )
    observed = (
        manifest.training_code_sha,
        manifest.model_code_sha,
        manifest.analysis_code_sha,
        manifest.configuration,
        manifest.checkpoint,
        manifest.dataset_manifest,
        manifest.inputs,
        manifest.command,
        manifest.model_family,
        manifest.model_revision,
        manifest.condition,
        manifest.sites,
        manifest.seed,
        manifest.evidence_level,
        manifest.legacy_reasons,
    )
    if observed != expected:
        raise ValueError("manifest evidence does not match actual VerifiedRunInputs")


class RunTransaction:
    """Publish a complete flat run directory with one sibling-directory rename."""

    def __init__(
        self,
        final_output_dir: str | os.PathLike[str],
        *,
        source_roots: Sequence[str | os.PathLike[str]] = (),
    ) -> None:
        roots = tuple(source_roots)
        if roots:
            validated = None
            for source_root in roots:
                validated = validate_output_dir(final_output_dir, source_root=source_root)
            assert validated is not None
            self.final_output_dir = validated
        else:
            self.final_output_dir = validate_output_dir(final_output_dir)
        self.source_roots = roots
        self._staging_dir: Path | None = None
        self._committed = False

    @property
    def staging_dir(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError("RunTransaction has not been entered")
        return self._staging_dir

    def __enter__(self) -> "RunTransaction":
        if self._staging_dir is not None:
            raise RuntimeError("RunTransaction cannot be entered twice")
        if self.final_output_dir.exists():
            raise FileExistsError("final output directory already exists")
        parent = self.final_output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("output parent must be a real directory")
        staging = tempfile.mkdtemp(
            prefix=f".{self.final_output_dir.name}.", suffix=".staging", dir=parent
        )
        self._staging_dir = Path(staging)
        return self

    def artifact_digests(self, names: Sequence[str]) -> tuple[ArtifactDigest, ...]:
        resolved_names = sorted(str(name) for name in names)
        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("expected artifact names must be unique")
        artifacts: list[ArtifactDigest] = []
        for name in resolved_names:
            # ArtifactDigest performs the flat portable-name validation.
            placeholder = ArtifactDigest(name=name, sha256="0" * 64, size_bytes=0)
            path = self.staging_dir / placeholder.name
            _fsync_file(path)
            verified = verify_file(path)
            artifacts.append(
                ArtifactDigest(
                    name=placeholder.name,
                    sha256=verified.digest.sha256,
                    size_bytes=verified.digest.size_bytes,
                )
            )
        return tuple(artifacts)

    def commit(self, manifest: RunManifest, *, verified_inputs: VerifiedRunInputs) -> Path:
        if self._committed:
            raise RuntimeError("RunTransaction has already committed")
        staging = self.staging_dir
        assert_manifest_matches_verified(manifest, verified_inputs)
        verified_inputs.assert_unchanged()
        actual_names = _flat_artifact_names(staging)
        expected_names = [artifact.name for artifact in manifest.artifacts]
        if actual_names != expected_names:
            raise ValueError(
                f"staging artifacts differ from manifest; expected={expected_names}, actual={actual_names}"
            )
        observed = self.artifact_digests(expected_names)
        if observed != manifest.artifacts:
            raise ValueError("staging artifact bytes do not match manifest digests")
        verified_inputs_source = self.source_roots[0] if self.source_roots else None
        write_manifest(
            manifest,
            staging,
            source_root=verified_inputs_source,
            verified_inputs=verified_inputs,
        )
        _fsync_directory(staging)
        verified_inputs.assert_unchanged()
        if self.final_output_dir.exists():
            raise FileExistsError("final output directory appeared before commit")
        os.rename(staging, self.final_output_dir)
        _fsync_directory(self.final_output_dir.parent)
        self._committed = True
        return self.final_output_dir

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._committed and self._staging_dir is not None and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir)


def verify_run_directory(path: str | os.PathLike[str]) -> RunManifest:
    """Re-hash every declared artifact and reject tampering, extras, or symlinks."""

    raw_path = Path(path)
    if not raw_path.is_absolute():
        raise ValueError("run directory must be an absolute path")
    if raw_path.is_symlink():
        raise ValueError("run directory must not be a symlink")
    directory = raw_path.resolve(strict=True)
    if os.path.ismount(directory):
        raise ValueError("run directory must not be a mount point")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("run directory is missing a regular manifest.json")
    manifest_file = verify_file(manifest_path)
    manifest = load_verified_run_manifest(manifest_file)
    if isinstance(manifest, LegacyRunManifest):
        raise ValueError("legacy schema-v1 runs cannot pass strict directory verification")
    actual_names = _flat_artifact_names(directory, include_manifest=False)
    expected_names = [artifact.name for artifact in manifest.artifacts]
    if actual_names != expected_names:
        raise ValueError(
            f"run artifacts differ from manifest; expected={expected_names}, actual={actual_names}"
        )
    observed = tuple(
        ArtifactDigest(name=name, sha256=file.digest.sha256, size_bytes=file.digest.size_bytes)
        for name in actual_names
        for file in (verify_file(directory / name),)
    )
    if observed != manifest.artifacts:
        raise ValueError("run artifact hash or size mismatch")
    manifest_file.assert_unchanged()
    return manifest


def load_verified_run_manifest(
    manifest_file: VerifiedFile,
) -> RunManifest | LegacyRunManifest:
    """Parse a run manifest from the exact bytes held by ``VerifiedFile``."""

    payload = json.loads(manifest_file.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must contain one JSON object")
    schema_version = payload.get("schema_version")
    if schema_version == 2:
        return RunManifest.from_dict(payload)
    if schema_version == 1:
        return LegacyRunManifest.from_dict(payload)
    raise ValueError(f"unsupported manifest schema_version: {schema_version!r}")


def verified_dataset_roster(dataset_manifest: VerifiedFile) -> dict[str, str | None]:
    """Parse the already-hashed public dataset roster without trusting paths."""

    raw = dataset_manifest.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("dataset manifest must contain one JSON object")
    has_assignments = "assignments" in payload
    has_datasets = "datasets" in payload
    if has_assignments == has_datasets:
        raise ValueError("dataset manifest must define exactly one of assignments or datasets")
    roster: dict[str, str | None] = {}
    if has_assignments:
        assignments = payload["assignments"]
        if not isinstance(assignments, list) or not assignments:
            raise ValueError("dataset manifest assignments must be a non-empty list")
        for item in assignments:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise ValueError("each dataset assignment must contain a string name")
            name = require_public_label(item["name"], name="dataset manifest name")
            split = item.get("split")
            if not isinstance(split, str) or not split:
                raise ValueError("each dataset assignment must contain a non-empty split")
            if name in roster:
                raise ValueError(f"duplicate dataset manifest name: {name!r}")
            roster[name] = split
    else:
        datasets = payload["datasets"]
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("dataset manifest datasets must be a non-empty list")
        for raw_name in datasets:
            if not isinstance(raw_name, str):
                raise ValueError("dataset manifest names must be strings")
            name = require_public_label(raw_name, name="dataset manifest name")
            if name in roster:
                raise ValueError(f"duplicate dataset manifest name: {name!r}")
            roster[name] = None
    return roster


def assert_dataset_roster(
    dataset_manifest: VerifiedFile,
    dataset_ids: Sequence[str],
    *,
    exact: bool = False,
    required_split: str | None = None,
) -> None:
    """Bind workflow dataset IDs to the measured dataset-manifest contents."""

    requested = tuple(dataset_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("workflow dataset IDs must be a non-empty unique sequence")
    roster = verified_dataset_roster(dataset_manifest)
    missing = sorted(set(requested) - set(roster))
    extra = sorted(set(roster) - set(requested)) if exact else []
    if missing or extra:
        raise ValueError(f"dataset roster mismatch: missing={missing}, extra={extra}")
    if required_split is not None:
        splitless = sorted(name for name in requested if roster[name] is None)
        if splitless:
            raise ValueError(
                "dataset manifest lacks required split assignments for: "
                f"{splitless}"
            )
        wrong = sorted(
            name
            for name in requested
            if roster[name] != required_split
        )
        if wrong:
            raise ValueError(
                f"dataset manifest split mismatch for {required_split!r}: {wrong}"
            )


def _flat_artifact_names(directory: Path, *, include_manifest: bool = False) -> list[str]:
    names: list[str] = []
    for entry in directory.iterdir():
        if entry.name == "manifest.json" and not include_manifest:
            continue
        if entry.is_symlink():
            raise ValueError("run directories must not contain symlinks")
        if not entry.is_file():
            raise ValueError("run directories must contain only flat regular files")
        names.append(entry.name)
    return sorted(names)


def _find_keys(value: Any, forbidden: set[str]) -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_find_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_keys(child, forbidden))
    return tuple(sorted(found))


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"provenance {key} must be a string when provided")
    return value


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        raise ValueError(f"Git verification failed: {' '.join(arguments)}")
    return completed.stdout


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("run artifacts must not be symlinks")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("run artifacts must be regular files")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "GitEvidence",
    "RunContract",
    "RunTransaction",
    "VerifiedConfiguration",
    "VerifiedFile",
    "VerifiedNamedInput",
    "VerifiedRunContext",
    "VerifiedRunInputs",
    "assert_dataset_roster",
    "assert_git_commit_is_ancestor",
    "assert_manifest_matches_verified",
    "load_verified_json_config",
    "load_verified_run_manifest",
    "manifest_from_verified_inputs",
    "verified_dataset_roster",
    "verify_configured_run_inputs",
    "verify_file",
    "verify_git_tree",
    "verify_run_directory",
    "verify_run_inputs",
]
