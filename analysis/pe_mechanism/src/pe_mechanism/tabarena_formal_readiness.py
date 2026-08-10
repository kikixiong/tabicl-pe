"""Read-only production-evidence gate for formal three-arm TabArena runs.

Checkpoint lineage is necessary but not sufficient to start a formal benchmark.
This module additionally asks the *exact training commit* to revalidate the
immutable campaign and predecessor-acceptance prefix, rebuild the terminal
scheduler attestation from raw evidence and live Slurm accounting, and bind the
resulting nine-job matrix back to the checkpoint cohort selected for evaluation.

The current seed's acceptance is deliberately not an input: it can only be
published after evaluation.  A successful report therefore authorizes benchmark
execution while keeping ``campaign_acceptance_verified`` false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .provenance import GitEvidence, verify_file, verify_git_tree
from .tabarena_formal_evaluation import (
    FORMAL_ARMS,
    FORMAL_STAGES,
    FormalTabArenaInputSpec,
    _validate_materialized_spec,
)


_READINESS_FIELDS = {
    "campaign_path",
    "expected_campaign_file_sha256",
    "expected_campaign_manifest_sha256",
    "acceptance_registry",
    "terminal_attestation_path",
    "expected_terminal_file_sha256",
    "expected_terminal_manifest_sha256",
    "submission_receipt_path",
    "expected_submission_receipt_file_sha256",
    "expected_submission_receipt_manifest_sha256",
    "protocol_metadata_allowance_bytes",
}
_HEX = frozenset("0123456789abcdef")
_MAX_FORMAL_METADATA_BYTES = 128 << 20


@dataclass(frozen=True)
class FormalReadinessSpec:
    """Externally committed campaign and terminal evidence for one seed."""

    campaign_path: Path
    campaign_file_sha256: str
    campaign_manifest_sha256: str
    acceptance_registry: Path
    terminal_attestation_path: Path
    terminal_file_sha256: str
    terminal_manifest_sha256: str
    submission_receipt_path: Path
    submission_receipt_file_sha256: str
    submission_receipt_manifest_sha256: str
    protocol_metadata_allowance_bytes: int


def _exact_object(value: object, *, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def parse_formal_readiness_config(
    data: Mapping[str, Any], checkpoint_spec: FormalTabArenaInputSpec
) -> FormalReadinessSpec:
    """Parse the exact pre-benchmark campaign/terminal evidence contract."""

    _validate_materialized_spec(checkpoint_spec)
    value = _exact_object(
        data, label="formal TabArena readiness", fields=_READINESS_FIELDS
    )
    allowance = value["protocol_metadata_allowance_bytes"]
    if (
        isinstance(allowance, bool)
        or not isinstance(allowance, int)
        or allowance < 1
        or allowance > _MAX_FORMAL_METADATA_BYTES
    ):
        raise ValueError("protocol_metadata_allowance_bytes is outside its fixed bound")

    campaign = _absolute_path(value["campaign_path"], label="campaign_path")
    registry = _absolute_path(value["acceptance_registry"], label="acceptance_registry")
    terminal = _absolute_path(
        value["terminal_attestation_path"], label="terminal_attestation_path"
    )
    receipt = _absolute_path(
        value["submission_receipt_path"], label="submission_receipt_path"
    )
    if campaign != campaign.parent / "campaign.json":
        raise ValueError("campaign_path must be campaign-root/campaign.json")
    if registry != campaign.parent / "acceptances":
        raise ValueError("acceptance_registry must be campaign-root/acceptances")
    if terminal != checkpoint_spec.artifact_root / "terminal-scheduler-logs.json":
        raise ValueError("terminal attestation is outside the checkpoint namespace")
    if receipt != checkpoint_spec.artifact_root / "submission-receipt.json":
        raise ValueError("submission receipt is outside the checkpoint namespace")

    return FormalReadinessSpec(
        campaign_path=campaign,
        campaign_file_sha256=_digest(
            value["expected_campaign_file_sha256"],
            label="campaign file SHA-256",
        ),
        campaign_manifest_sha256=_digest(
            value["expected_campaign_manifest_sha256"],
            label="campaign manifest SHA-256",
        ),
        acceptance_registry=registry,
        terminal_attestation_path=terminal,
        terminal_file_sha256=_digest(
            value["expected_terminal_file_sha256"],
            label="terminal file SHA-256",
        ),
        terminal_manifest_sha256=_digest(
            value["expected_terminal_manifest_sha256"],
            label="terminal manifest SHA-256",
        ),
        submission_receipt_path=receipt,
        submission_receipt_file_sha256=_digest(
            value["expected_submission_receipt_file_sha256"],
            label="submission receipt file SHA-256",
        ),
        submission_receipt_manifest_sha256=_digest(
            value["expected_submission_receipt_manifest_sha256"],
            label="submission receipt manifest SHA-256",
        ),
        protocol_metadata_allowance_bytes=allowance,
    )


def _materialized_readiness_data(spec: FormalReadinessSpec) -> dict[str, Any]:
    return {
        "campaign_path": str(spec.campaign_path),
        "expected_campaign_file_sha256": spec.campaign_file_sha256,
        "expected_campaign_manifest_sha256": spec.campaign_manifest_sha256,
        "acceptance_registry": str(spec.acceptance_registry),
        "terminal_attestation_path": str(spec.terminal_attestation_path),
        "expected_terminal_file_sha256": spec.terminal_file_sha256,
        "expected_terminal_manifest_sha256": spec.terminal_manifest_sha256,
        "submission_receipt_path": str(spec.submission_receipt_path),
        "expected_submission_receipt_file_sha256": (
            spec.submission_receipt_file_sha256
        ),
        "expected_submission_receipt_manifest_sha256": (
            spec.submission_receipt_manifest_sha256
        ),
        "protocol_metadata_allowance_bytes": spec.protocol_metadata_allowance_bytes,
    }


def _validate_materialized_readiness(
    checkpoint_spec: FormalTabArenaInputSpec, readiness: FormalReadinessSpec
) -> None:
    if not isinstance(readiness, FormalReadinessSpec):
        raise TypeError("readiness must be a FormalReadinessSpec")
    observed = parse_formal_readiness_config(
        _materialized_readiness_data(readiness), checkpoint_spec
    )
    if observed != readiness:
        raise ValueError("materialized formal readiness spec is not canonical")


def _trusted_registry_module(
    checkpoint_spec: FormalTabArenaInputSpec, training_git: GitEvidence
) -> Any:
    """Load campaign/terminal validators only from the bound exact T checkout."""

    if training_git.evidence_level != "strict" or training_git.legacy_reasons:
        raise RuntimeError("formal training checkout lacks strict clean evidence")
    configured_root = checkpoint_spec.training_code_root.resolve(strict=True)
    verified_root = training_git.root.resolve(strict=True)
    if configured_root != verified_root:
        raise RuntimeError("training_code_root is not the verified Git root")
    expected_file = (verified_root / "scripts" / "formal_campaign_registry.py").resolve(
        strict=True
    )
    path_tag = hashlib.sha256(os.fsencode(expected_file)).hexdigest()[:16]
    module_name = f"_pe_mechanism_formal_campaign_{training_git.head_sha}_{path_tag}"
    module = sys.modules.get(module_name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(module_name, expected_file)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("cannot construct the exact-T campaign module spec")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous
    raw_file = getattr(module, "__file__", None)
    if (
        not isinstance(raw_file, str)
        or Path(raw_file).resolve(strict=True) != expected_file
    ):
        raise RuntimeError("formal campaign module is not from the bound training tree")
    for name in (
        "authorize_seed_submission",
        "read_canonical_manifest",
        "_validate_existing_terminal_evidence",
        "_validate_terminal_attestation",
    ):
        symbol = getattr(module, name, None)
        if not callable(symbol):
            raise RuntimeError(f"formal campaign module lacks canonical {name}")
        try:
            source = Path(inspect.getfile(inspect.unwrap(symbol))).resolve(strict=True)
        except (OSError, TypeError) as error:
            raise RuntimeError(
                f"formal campaign symbol {name} has no source"
            ) from error
        if source != expected_file:
            raise RuntimeError(
                f"formal campaign symbol {name} is not defined by exact T"
            )
    training_git.assert_unchanged()
    return module


def _checkpoint_report(
    value: Mapping[str, Any], checkpoint_spec: FormalTabArenaInputSpec
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint validation report must be an object")
    if (
        value.get("checkpoint_lineage_verified") is not True
        or value.get("campaign_acceptance_verified") is not False
        or value.get("terminal_scheduler_evidence_verified") is not False
        or value.get("benchmark_execution_ready") is not False
        or value.get("training_code_sha") != checkpoint_spec.training_code_sha
        or value.get("seed") != checkpoint_spec.seed
        or value.get("stage") != checkpoint_spec.stage
        or value.get("terminal_step") != checkpoint_spec.terminal_step
        or set(value.get("arms", {})) != set(FORMAL_ARMS)
    ):
        raise ValueError(
            "checkpoint validation report is not the required intake result"
        )
    ledger = value.get("transaction_ledger")
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("file_sha256") != checkpoint_spec.transaction_ledger_file_sha256
        or ledger.get("manifest_sha256")
        != checkpoint_spec.transaction_ledger_manifest_sha256
    ):
        raise ValueError("checkpoint report transaction ledger differs from its input")
    return value


def _terminal_job_matrix(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    jobs = payload.get("jobs")
    expected = tuple((arm, stage) for stage, _ in FORMAL_STAGES for arm in FORMAL_ARMS)
    if not isinstance(jobs, list) or len(jobs) != len(expected):
        raise ValueError("terminal scheduler evidence lacks the canonical nine jobs")
    matrix: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw, pair in zip(jobs, expected):
        if not isinstance(raw, Mapping) or (raw.get("arm"), raw.get("stage")) != pair:
            raise ValueError(
                "terminal scheduler job matrix/order differs from protocol"
            )
        matrix[pair] = raw
    return matrix


def _bind_terminal_checkpoints(
    *,
    checkpoint_spec: FormalTabArenaInputSpec,
    checkpoint_report: Mapping[str, Any],
    terminal_payload: Mapping[str, Any],
) -> None:
    matrix = _terminal_job_matrix(terminal_payload)
    for stage, _terminal_step in FORMAL_STAGES:
        for arm in FORMAL_ARMS:
            job = matrix[(arm, stage)]
            finalized = job.get("finalized_artifact")
            binding = (
                finalized.get("binding") if isinstance(finalized, Mapping) else None
            )
            if not isinstance(binding, Mapping):
                raise ValueError("terminal job lacks finalized checkpoint binding")
            configured = checkpoint_spec.arms[arm][stage]
            if (
                binding.get("checkpoint_sha256") != configured.checkpoint_sha256
                or binding.get("finalized_manifest_sha256")
                != configured.finalized_manifest_sha256
                or finalized.get("finalized_manifest_file_sha256")
                != configured.finalized_manifest_file_sha256
            ):
                raise ValueError(
                    f"terminal {arm} {stage} artifact differs from checkpoint intake"
                )
    target = checkpoint_report["arms"]
    for arm in FORMAL_ARMS:
        binding = matrix[(arm, checkpoint_spec.stage)]["finalized_artifact"]["binding"]
        if (
            binding["checkpoint_sha256"] != target[arm]["checkpoint_sha256"]
            or binding["finalized_manifest_sha256"]
            != target[arm]["finalized_manifest_sha256"]
        ):
            raise ValueError(f"terminal target checkpoint differs for arm {arm}")


def validate_formal_tabarena_readiness(
    checkpoint_spec: FormalTabArenaInputSpec,
    checkpoint_report: Mapping[str, Any],
    readiness: FormalReadinessSpec,
) -> dict[str, Any]:
    """Rebuild exact-T campaign/terminal evidence and authorize evaluation."""

    _validate_materialized_spec(checkpoint_spec)
    _validate_materialized_readiness(checkpoint_spec, readiness)
    intake = _checkpoint_report(checkpoint_report, checkpoint_spec)
    if checkpoint_spec.stage != "stage3":
        raise ValueError(
            "formal TabArena execution requires the complete Stage-3 cohort"
        )

    campaign_file = verify_file(
        readiness.campaign_path, expected_sha256=readiness.campaign_file_sha256
    )
    terminal_file = verify_file(
        readiness.terminal_attestation_path,
        expected_sha256=readiness.terminal_file_sha256,
    )
    receipt_file = verify_file(
        readiness.submission_receipt_path,
        expected_sha256=readiness.submission_receipt_file_sha256,
    )
    training_git = verify_git_tree(
        checkpoint_spec.training_code_root,
        expected_sha=checkpoint_spec.training_code_sha,
    )
    registry = _trusted_registry_module(checkpoint_spec, training_git)
    authorization = registry.authorize_seed_submission(
        campaign_path=readiness.campaign_path,
        campaign_expected_sha256=readiness.campaign_manifest_sha256,
        acceptance_registry=readiness.acceptance_registry,
        seed=checkpoint_spec.seed,
    )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("seed") != checkpoint_spec.seed
        or not isinstance(authorization.get("campaign"), Mapping)
        or not isinstance(authorization.get("campaign_binding"), Mapping)
    ):
        raise RuntimeError("exact-T campaign authorization returned invalid evidence")

    terminal = registry.read_canonical_manifest(
        readiness.terminal_attestation_path,
        max_bytes=registry.TERMINAL_ATTESTATION_CEILING_BYTES,
        expected_kind="formal_terminal_scheduler_logs",
        expected_sha256=readiness.terminal_manifest_sha256,
    )
    rebuilt = registry._validate_existing_terminal_evidence(
        terminal_path=readiness.terminal_attestation_path,
        terminal=terminal,
        raw_evidence={
            "artifact_root": str(checkpoint_spec.artifact_root),
            "submission_receipt_path": str(readiness.submission_receipt_path),
            "transaction_ledger_path": str(checkpoint_spec.transaction_ledger_path),
            "protocol_metadata_allowance_bytes": (
                readiness.protocol_metadata_allowance_bytes
            ),
        },
    )
    terminal_payload = registry._validate_terminal_attestation(
        rebuilt,
        campaign=authorization["campaign"],
        seed=checkpoint_spec.seed,
        expected_campaign_binding=authorization["campaign_binding"],
    )
    if not isinstance(terminal_payload, Mapping):
        raise RuntimeError("exact-T terminal validator returned invalid evidence")
    if (
        terminal_payload.get("study_id") != intake.get("study_id")
        or terminal_payload.get("seed") != checkpoint_spec.seed
        or terminal_payload.get("source_commit_sha")
        != checkpoint_spec.training_code_sha
        or terminal_payload.get("transaction_ledger_sha256")
        != checkpoint_spec.transaction_ledger_manifest_sha256
        or terminal_payload.get("submission_receipt_sha256")
        != readiness.submission_receipt_manifest_sha256
        or terminal_payload.get("campaign_binding") != authorization["campaign_binding"]
    ):
        raise ValueError("terminal evidence differs from campaign/checkpoint intake")
    campaign = authorization["campaign"]
    campaign_payload = campaign.get("payload")
    if (
        campaign.get("sha256") != readiness.campaign_manifest_sha256
        or not isinstance(campaign_payload, Mapping)
        or campaign_payload.get("campaign_id") != intake.get("campaign_id")
    ):
        raise ValueError("authorized campaign differs from checkpoint intake")
    _bind_terminal_checkpoints(
        checkpoint_spec=checkpoint_spec,
        checkpoint_report=intake,
        terminal_payload=terminal_payload,
    )

    for file in (campaign_file, terminal_file, receipt_file):
        file.assert_unchanged()
    training_git.assert_unchanged()
    report = {
        "schema_version": 1,
        "kind": "formal_tabarena_execution_readiness",
        "evidence_scope": "formal_pre_benchmark_authorization",
        "checkpoint_lineage_verified": True,
        "campaign_authorization_verified": True,
        "campaign_acceptance_verified": False,
        "terminal_scheduler_evidence_verified": True,
        "benchmark_execution_ready": True,
        "training_code_sha": checkpoint_spec.training_code_sha,
        "campaign_id": intake["campaign_id"],
        "campaign_manifest_sha256": readiness.campaign_manifest_sha256,
        "study_id": intake["study_id"],
        "seed": checkpoint_spec.seed,
        "stage": checkpoint_spec.stage,
        "stage_step": checkpoint_spec.terminal_step,
        "cumulative_training_steps": sum(step for _stage, step in FORMAL_STAGES),
        "transaction_ledger_sha256": (
            checkpoint_spec.transaction_ledger_manifest_sha256
        ),
        "terminal_attestation_sha256": readiness.terminal_manifest_sha256,
        "submission_receipt_sha256": (readiness.submission_receipt_manifest_sha256),
        "arms": {
            arm: {
                "checkpoint_sha256": intake["arms"][arm]["checkpoint_sha256"],
                "finalized_manifest_sha256": intake["arms"][arm][
                    "finalized_manifest_sha256"
                ],
            }
            for arm in FORMAL_ARMS
        },
    }
    if any(
        key.endswith(("_path", "_root"))
        for mapping in (report, *report["arms"].values())
        for key in mapping
    ):
        raise AssertionError("formal readiness report must remain path-free")
    return report


__all__ = [
    "FormalReadinessSpec",
    "parse_formal_readiness_config",
    "validate_formal_tabarena_readiness",
]
