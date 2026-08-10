from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import pe_mechanism.tabarena_formal_evaluation as intake
import pe_mechanism.tabarena_formal_readiness as readiness


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checkpoint_spec(
    tmp_path: Path, *, stage: str = "stage3"
) -> intake.FormalTabArenaInputSpec:
    training = tmp_path / "training"
    artifacts = tmp_path / "formal-artifacts"
    training.mkdir()
    artifacts.mkdir()
    required = intake.FORMAL_STAGES[
        : next(i for i, pair in enumerate(intake.FORMAL_STAGES) if pair[0] == stage) + 1
    ]
    arms: dict[str, dict[str, intake.FormalStageInput]] = {}
    for arm in intake.FORMAL_ARMS:
        chain: dict[str, intake.FormalStageInput] = {}
        for stage_name, step in required:
            root = artifacts / "arms" / arm / stage_name
            chain[stage_name] = intake.FormalStageInput(
                checkpoint_path=root / f"step-{step}.ckpt",
                checkpoint_sha256=_sha(f"checkpoint:{arm}:{stage_name}".encode()),
                finalized_manifest_path=root / "finalized-checkpoint.json",
                finalized_manifest_file_sha256=_sha(
                    f"finalized-file:{arm}:{stage_name}".encode()
                ),
                finalized_manifest_sha256=_sha(
                    f"finalized-logical:{arm}:{stage_name}".encode()
                ),
            )
        arms[arm] = chain
    return intake.FormalTabArenaInputSpec(
        seed=42,
        stage=stage,
        terminal_step=dict(intake.FORMAL_STAGES)[stage],
        training_code_root=training,
        training_code_sha="a" * 40,
        artifact_root=artifacts,
        transaction_ledger_path=artifacts / "transaction-ledger.json",
        transaction_ledger_file_sha256=_sha(b"ledger-file"),
        transaction_ledger_manifest_sha256=_sha(b"ledger-logical"),
        arms=arms,
    )


def _checkpoint_report(
    spec: intake.FormalTabArenaInputSpec,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "formal_tabarena_three_arm_input_validation",
        "evidence_scope": "formal_checkpoint_lineage_only",
        "checkpoint_lineage_verified": True,
        "campaign_acceptance_verified": False,
        "terminal_scheduler_evidence_verified": False,
        "benchmark_execution_ready": False,
        "training_code_sha": spec.training_code_sha,
        "campaign_id": "position-identity-v1",
        "study_id": "position-identity-v1-seed42",
        "seed": 42,
        "stage": spec.stage,
        "terminal_step": spec.terminal_step,
        "transaction_ledger": {
            "file_sha256": spec.transaction_ledger_file_sha256,
            "manifest_sha256": spec.transaction_ledger_manifest_sha256,
        },
        "arms": {
            arm: {
                "checkpoint_sha256": spec.arms[arm][spec.stage].checkpoint_sha256,
                "finalized_manifest_sha256": spec.arms[arm][
                    spec.stage
                ].finalized_manifest_sha256,
            }
            for arm in intake.FORMAL_ARMS
        },
    }


def _terminal_payload(spec: intake.FormalTabArenaInputSpec) -> dict[str, Any]:
    jobs = []
    for stage, _step in intake.FORMAL_STAGES:
        for arm in intake.FORMAL_ARMS:
            configured = spec.arms[arm][stage]
            jobs.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "finalized_artifact": {
                        "binding": {
                            "checkpoint_sha256": configured.checkpoint_sha256,
                            "finalized_manifest_sha256": (
                                configured.finalized_manifest_sha256
                            ),
                        },
                        "finalized_manifest_file_sha256": (
                            configured.finalized_manifest_file_sha256
                        ),
                    },
                }
            )
    return {
        "study_id": "position-identity-v1-seed42",
        "seed": 42,
        "source_commit_sha": spec.training_code_sha,
        "transaction_ledger_sha256": spec.transaction_ledger_manifest_sha256,
        "submission_receipt_sha256": _sha(b"receipt-logical"),
        "campaign_binding": {"campaign_id": "position-identity-v1"},
        "jobs": jobs,
    }


class _FakeGit:
    def __init__(self, spec: intake.FormalTabArenaInputSpec) -> None:
        self.root = spec.training_code_root.resolve()
        self.head_sha = spec.training_code_sha
        self.evidence_level = "strict"
        self.legacy_reasons: tuple[str, ...] = ()
        self.unchanged_checks = 0

    def assert_unchanged(self) -> None:
        self.unchanged_checks += 1


class _FakeRegistry:
    TERMINAL_ATTESTATION_CEILING_BYTES = 131_072

    def __init__(
        self,
        spec: intake.FormalTabArenaInputSpec,
        config: readiness.FormalReadinessSpec,
        *,
        terminal_payload: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self.payload = terminal_payload or _terminal_payload(spec)
        self.terminal = {
            "schema_version": 1,
            "kind": "formal_terminal_scheduler_logs",
            "payload": self.payload,
            "sha256": config.terminal_manifest_sha256,
        }
        self.raw_evidence: dict[str, Any] | None = None

    def authorize_seed_submission(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "campaign_path": self.config.campaign_path,
            "campaign_expected_sha256": self.config.campaign_manifest_sha256,
            "acceptance_registry": self.config.acceptance_registry,
            "seed": 42,
        }
        return {
            "seed": 42,
            "campaign": {
                "schema_version": 1,
                "kind": "formal_campaign",
                "payload": {"campaign_id": "position-identity-v1"},
                "sha256": self.config.campaign_manifest_sha256,
            },
            "campaign_binding": {"campaign_id": "position-identity-v1"},
        }

    def read_canonical_manifest(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert args == (self.config.terminal_attestation_path,)
        assert kwargs["expected_sha256"] == self.config.terminal_manifest_sha256
        return self.terminal

    def _validate_existing_terminal_evidence(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["terminal"] == self.terminal
        self.raw_evidence = kwargs["raw_evidence"]
        return self.terminal

    def _validate_terminal_attestation(
        self, terminal: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        assert terminal == self.terminal
        assert kwargs["seed"] == 42
        return self.payload


def _readiness_config(
    tmp_path: Path, spec: intake.FormalTabArenaInputSpec
) -> tuple[dict[str, Any], readiness.FormalReadinessSpec]:
    campaign_root = tmp_path / "campaign"
    acceptances = campaign_root / "acceptances"
    evaluations = campaign_root / "evaluations"
    acceptances.mkdir(parents=True)
    evaluations.mkdir()
    campaign = campaign_root / "campaign.json"
    terminal = spec.artifact_root / "terminal-scheduler-logs.json"
    receipt = spec.artifact_root / "submission-receipt.json"
    campaign.write_bytes(b"campaign-file\n")
    terminal.write_bytes(b"terminal-file\n")
    receipt.write_bytes(b"receipt-file\n")
    value = {
        "campaign_path": str(campaign),
        "expected_campaign_file_sha256": _sha(campaign.read_bytes()),
        "expected_campaign_manifest_sha256": _sha(b"campaign-logical"),
        "acceptance_registry": str(acceptances),
        "terminal_attestation_path": str(terminal),
        "expected_terminal_file_sha256": _sha(terminal.read_bytes()),
        "expected_terminal_manifest_sha256": _sha(b"terminal-logical"),
        "submission_receipt_path": str(receipt),
        "expected_submission_receipt_file_sha256": _sha(receipt.read_bytes()),
        "expected_submission_receipt_manifest_sha256": _sha(b"receipt-logical"),
        "protocol_metadata_allowance_bytes": 1 << 20,
    }
    return value, readiness.parse_formal_readiness_config(value, spec)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    spec: intake.FormalTabArenaInputSpec,
    config: readiness.FormalReadinessSpec,
    *,
    terminal_payload: dict[str, Any] | None = None,
) -> tuple[_FakeGit, _FakeRegistry]:
    git = _FakeGit(spec)
    registry = _FakeRegistry(spec, config, terminal_payload=terminal_payload)
    monkeypatch.setattr(readiness, "verify_git_tree", lambda *args, **kwargs: git)
    monkeypatch.setattr(
        readiness, "_trusted_registry_module", lambda *args, **kwargs: registry
    )
    return git, registry


def test_readiness_rebuilds_terminal_and_binds_all_nine_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _checkpoint_spec(tmp_path)
    _raw, config = _readiness_config(tmp_path, spec)
    git, registry = _install_fakes(monkeypatch, spec, config)

    report = readiness.validate_formal_tabarena_readiness(
        spec, _checkpoint_report(spec), config
    )

    assert report["checkpoint_lineage_verified"] is True
    assert report["campaign_authorization_verified"] is True
    assert report["terminal_scheduler_evidence_verified"] is True
    assert report["benchmark_execution_ready"] is True
    assert report["campaign_acceptance_verified"] is False
    assert report["stage_step"] == 10_000
    assert report["cumulative_training_steps"] == 550_000
    assert set(report["arms"]) == set(intake.FORMAL_ARMS)
    assert registry.raw_evidence == {
        "artifact_root": str(spec.artifact_root),
        "submission_receipt_path": str(config.submission_receipt_path),
        "transaction_ledger_path": str(spec.transaction_ledger_path),
        "protocol_metadata_allowance_bytes": 1 << 20,
    }
    assert git.unchanged_checks == 1
    assert not any(
        key.endswith(("_path", "_root"))
        for mapping in (report, *report["arms"].values())
        for key in mapping
    )


def test_readiness_rejects_terminal_checkpoint_or_ledger_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _checkpoint_spec(tmp_path)
    _raw, config = _readiness_config(tmp_path, spec)
    payload = _terminal_payload(spec)
    payload["jobs"][-1]["finalized_artifact"]["binding"]["checkpoint_sha256"] = "f" * 64
    _install_fakes(monkeypatch, spec, config, terminal_payload=payload)
    with pytest.raises(ValueError, match="artifact differs"):
        readiness.validate_formal_tabarena_readiness(
            spec, _checkpoint_report(spec), config
        )

    report = _checkpoint_report(spec)
    report["transaction_ledger"]["manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="transaction ledger"):
        readiness.validate_formal_tabarena_readiness(spec, report, config)


def test_parser_rejects_noncanonical_namespace_and_unbounded_metadata(
    tmp_path: Path,
) -> None:
    spec = _checkpoint_spec(tmp_path)
    raw, _config = _readiness_config(tmp_path, spec)
    broken = deepcopy(raw)
    broken["terminal_attestation_path"] = str(tmp_path / "other.json")
    with pytest.raises(ValueError, match="outside the checkpoint namespace"):
        readiness.parse_formal_readiness_config(broken, spec)

    broken = deepcopy(raw)
    broken["protocol_metadata_allowance_bytes"] = (128 << 20) + 1
    with pytest.raises(ValueError, match="fixed bound"):
        readiness.parse_formal_readiness_config(broken, spec)


def test_execution_requires_stage3_and_has_no_module_injection_seam(
    tmp_path: Path,
) -> None:
    spec = _checkpoint_spec(tmp_path, stage="stage2")
    _raw, config = _readiness_config(tmp_path, spec)
    with pytest.raises(ValueError, match="complete Stage-3"):
        readiness.validate_formal_tabarena_readiness(
            spec, _checkpoint_report(spec), config
        )
    assert tuple(
        inspect.signature(readiness.validate_formal_tabarena_readiness).parameters
    ) == ("checkpoint_spec", "checkpoint_report", "readiness")
