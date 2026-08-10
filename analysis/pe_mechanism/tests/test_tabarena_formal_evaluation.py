from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import pe_mechanism.tabarena_formal_evaluation as formal
import pe_mechanism.tabarena_evaluation as tabarena


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _logical_digest(label: str) -> str:
    return _sha256_bytes(label.encode("utf-8"))


def _fixture_config(
    tmp_path: Path, *, seed: int = 42, stage: str = "stage3"
) -> dict[str, Any]:
    training_root = tmp_path / "training"
    artifact_root = tmp_path / "artifacts"
    training_root.mkdir()
    artifact_root.mkdir()
    campaign_id = "position-identity-v1"
    study_id = f"{campaign_id}-seed{seed}"
    stages = formal.FORMAL_STAGES[:
        next(index for index, (name, _) in enumerate(formal.FORMAL_STAGES) if name == stage)
        + 1
    ]
    entries = []
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in formal.FORMAL_ARMS:
        chain: dict[str, dict[str, Any]] = {}
        for stage_name, terminal_step in stages:
            root = artifact_root / "arms" / arm / stage_name
            root.mkdir(parents=True)
            checkpoint = root / f"step-{terminal_step}.ckpt"
            checkpoint.write_bytes(f"checkpoint:{arm}:{stage_name}\n".encode())
            finalized = root / "finalized-checkpoint.json"
            logical = _logical_digest(f"finalized:{arm}:{stage_name}")
            finalized.write_text(
                json.dumps({"logical_sha256": logical}), encoding="utf-8"
            )
            chain[stage_name] = {
                "checkpoint_path": str(checkpoint),
                "expected_checkpoint_sha256": _sha256_file(checkpoint),
                "finalized_manifest_path": str(finalized),
                "expected_finalized_manifest_file_sha256": _sha256_file(finalized),
                "expected_finalized_manifest_sha256": logical,
            }
            entries.append(
                {
                    "arm": arm,
                    "stage": stage_name,
                    "upstream_identity": f"{study_id}:{arm}:{stage_name}",
                    "artifact_identity": f"{study_id}.{arm}.{stage_name}.final",
                }
            )
        arms[arm] = chain

    # The canonical ledger is stage-major even though the public config is arm-major.
    entries.sort(
        key=lambda entry: (
            [name for name, _ in formal.FORMAL_STAGES].index(entry["stage"]),
            list(formal.FORMAL_ARMS).index(entry["arm"]),
        )
    )
    ledger_manifest_sha = _logical_digest("transaction-ledger")
    ledger = {
        "schema_version": 1,
        "kind": "transaction_ledger",
        "payload": {
            "study_id": study_id,
            "campaign_binding": {
                "campaign_id": campaign_id,
                "training_commit_sha": "a" * 40,
            },
            "entries": entries,
        },
        "sha256": ledger_manifest_sha,
    }
    ledger_path = artifact_root / "transaction-ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return {
        "schema_version": 1,
        "seed": seed,
        "stage": stage,
        "terminal_step": dict(formal.FORMAL_STAGES)[stage],
        "training_code_root": str(training_root),
        "expected_training_code_sha": "a" * 40,
        "artifact_root": str(artifact_root),
        "transaction_ledger_path": str(ledger_path),
        "expected_transaction_ledger_file_sha256": _sha256_file(ledger_path),
        "expected_transaction_ledger_manifest_sha256": ledger_manifest_sha,
        "arms": arms,
    }


class _FakeGit:
    def __init__(self, root: Path, sha: str) -> None:
        self.root = root.resolve()
        self.head_sha = sha
        self.evidence_level = "strict"
        self.legacy_reasons: tuple[str, ...] = ()
        self.unchanged_checks = 0

    def assert_unchanged(self) -> None:
        self.unchanged_checks += 1


class _FakeFormalProvenance:
    def __init__(
        self,
        *,
        break_parent: tuple[str, str] | None = None,
        override: tuple[str, str, str, Any] | None = None,
    ) -> None:
        self.break_parent = break_parent
        self.override = override
        self.results: dict[tuple[str, str], Any] = {}
        self.parent_calls = 0
        self.cohort_calls = 0

    @staticmethod
    def validate_manifest(
        manifest: dict[str, Any], *, expected_kind: str, expected_sha256: str
    ) -> str:
        assert expected_kind == "transaction_ledger"
        assert manifest["kind"] == expected_kind
        assert manifest["sha256"] == expected_sha256
        return expected_sha256

    @staticmethod
    def validate_canonical_transaction_ledger(
        payload: dict[str, Any], *, artifact_root: Path
    ) -> dict[tuple[str, str], dict[str, Any]]:
        assert artifact_root.is_absolute()
        return {
            (entry["arm"], entry["stage"]): entry for entry in payload["entries"]
        }

    @staticmethod
    def ParentTrust(**kwargs: Any) -> SimpleNamespace:  # noqa: N802
        return SimpleNamespace(**kwargs)

    def validate_parent_trust(self, trust: SimpleNamespace) -> SimpleNamespace:
        self.parent_calls += 1
        terminal_step = dict(formal.FORMAL_STAGES)[trust.parent_stage]
        finalized_payload = json.loads(trust.finalized_manifest_path.read_text())
        previous = None
        if trust.parent_stage != "stage1":
            prior = "stage1" if trust.parent_stage == "stage2" else "stage2"
            previous = self.results[(trust.arm, prior)].manifest
        if self.break_parent == (trust.arm, trust.parent_stage):
            previous = {"broken": True}
        common = {
            "study_id": trust.study_id,
            "stage": trust.parent_stage,
            "terminal_step": terminal_step,
            "np_seed": 42,
            "torch_seed": 42,
            "identity_rng_seed": 42,
            "world_size": 1,
            "cuda_device_count": 1,
            "source_sha256": "1" * 64,
            "environment_sha256": "2" * 64,
            "prior_sha256": "3" * 64,
            "architecture_sha256": "4" * 64,
            "optimizer_sha256": "5" * 64,
            "scientific_sha256": "6" * 64,
            "cohort_protocol_sha256": "7" * 64,
        }
        if self.override is not None:
            arm, stage, field, value = self.override
            if (trust.arm, trust.parent_stage) == (arm, stage):
                common[field] = value
        parent = {
            **common,
            "arm": trust.arm,
            "checkpoint_sha256": _sha256_file(trust.checkpoint_path),
            "finalized_manifest_sha256": finalized_payload["logical_sha256"],
            "treatment_sha256": _logical_digest(f"treatment:{trust.arm}"),
            "arm_protocol_sha256": _logical_digest(f"protocol:{trust.arm}"),
        }
        manifest = {
            "kind": "expected_parent",
            "arm": trust.arm,
            "stage": trust.parent_stage,
            "payload": {"parent": parent},
        }
        provenance = {
            "arm": trust.arm,
            "stage": trust.parent_stage,
            "manifests": {"parent": previous},
        }
        result = SimpleNamespace(
            manifest=manifest,
            checkpoint={"provenance": provenance},
            checkpoint_sha256=parent["checkpoint_sha256"],
        )
        self.results[(trust.arm, trust.parent_stage)] = result
        return result

    def validate_cohort_provenance(
        self, bundles: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        self.cohort_calls += 1
        assert set(bundles) == set(formal.FORMAL_ARMS)
        assert {arm: bundles[arm]["arm"] for arm in bundles} == {
            arm: arm for arm in formal.FORMAL_ARMS
        }
        return {"cohort_protocol_sha256": "7" * 64}


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    spec: formal.FormalTabArenaInputSpec,
    module: _FakeFormalProvenance,
) -> _FakeGit:
    git = _FakeGit(spec.training_code_root, spec.training_code_sha)
    monkeypatch.setattr(formal, "verify_git_tree", lambda *args, **kwargs: git)
    monkeypatch.setattr(
        formal, "_trusted_provenance_module", lambda *args, **kwargs: module
    )
    return git


def test_parser_requires_exact_three_arm_canonical_chain(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    spec = formal.parse_formal_tabarena_config(config)
    assert spec.seed == 42
    assert spec.stage == "stage3"
    assert set(spec.arms) == set(formal.FORMAL_ARMS)
    assert all(tuple(chain) == ("stage1", "stage2", "stage3") for chain in spec.arms.values())

    broken = deepcopy(config)
    broken["terminal_step"] = 500_000
    with pytest.raises(ValueError, match="canonical stage budget"):
        formal.parse_formal_tabarena_config(broken)

    broken = deepcopy(config)
    broken["arms"].pop("temporary")
    with pytest.raises(ValueError, match="fields mismatch"):
        formal.parse_formal_tabarena_config(broken)

    broken = deepcopy(config)
    broken["arms"]["rope"]["stage3"]["checkpoint_path"] = str(
        tmp_path / "other.ckpt"
    )
    with pytest.raises(ValueError, match="canonical ledger layout"):
        formal.parse_formal_tabarena_config(broken)


@pytest.mark.parametrize(
    ("stage", "terminal_step", "chain_length"),
    [("stage1", 500_000, 1), ("stage2", 40_000, 2), ("stage3", 10_000, 3)],
)
def test_parser_accepts_only_the_complete_chain_to_the_requested_stage(
    tmp_path: Path, stage: str, terminal_step: int, chain_length: int
) -> None:
    spec = formal.parse_formal_tabarena_config(
        _fixture_config(tmp_path, stage=stage)
    )
    assert spec.terminal_step == terminal_step
    assert all(len(chain) == chain_length for chain in spec.arms.values())


def test_public_example_is_a_complete_stage3_input_contract() -> None:
    path = Path(__file__).parents[1] / "examples" / "tabarena-formal-inputs.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = formal.parse_formal_tabarena_config(payload)
    assert spec.stage == "stage3"
    assert spec.terminal_step == 10_000
    assert all(len(chain) == 3 for chain in spec.arms.values())


def test_checkpoint_lineage_validation_is_path_free_and_not_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = formal.parse_formal_tabarena_config(_fixture_config(tmp_path))
    module = _FakeFormalProvenance()
    git = _install_runtime_fakes(monkeypatch, spec, module)
    report = formal.validate_formal_tabarena_inputs(spec)

    assert report["checkpoint_lineage_verified"] is True
    assert report["campaign_acceptance_verified"] is False
    assert report["terminal_scheduler_evidence_verified"] is False
    assert report["benchmark_execution_ready"] is False
    assert report["evidence_scope"] == "formal_checkpoint_lineage_only"
    assert set(report["arms"]) == set(formal.FORMAL_ARMS)
    assert module.parent_calls == 9
    assert module.cohort_calls == 3
    assert git.unchanged_checks == 1
    serialized = json.dumps(report, sort_keys=True)
    assert str(spec.artifact_root) not in serialized
    assert str(spec.training_code_root) not in serialized
    report_keys = set(report) | set(report["transaction_ledger"])
    report_keys.update(
        key for arm_report in report["arms"].values() for key in arm_report
    )
    assert not any(
        key.endswith(("_path", "_root"))
        for key in report_keys
    )


def test_validation_rejects_broken_ancestor_and_cross_arm_protocol_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = formal.parse_formal_tabarena_config(_fixture_config(tmp_path))
    broken = _FakeFormalProvenance(break_parent=("temporary", "stage3"))
    _install_runtime_fakes(monkeypatch, spec, broken)
    with pytest.raises(ValueError, match="does not descend"):
        formal.validate_formal_tabarena_inputs(spec)

    drift = _FakeFormalProvenance(
        override=("none", "stage3", "scientific_sha256", "f" * 64)
    )
    _install_runtime_fakes(monkeypatch, spec, drift)
    with pytest.raises(ValueError, match="shared scientific_sha256"):
        formal.validate_formal_tabarena_inputs(spec)


def test_validation_rejects_one_inode_for_two_checkpoint_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    rope = Path(config["arms"]["rope"]["stage3"]["checkpoint_path"])
    none = Path(config["arms"]["none"]["stage3"]["checkpoint_path"])
    none.unlink()
    os.link(rope, none)
    config["arms"]["none"]["stage3"]["expected_checkpoint_sha256"] = _sha256_file(
        none
    )
    spec = formal.parse_formal_tabarena_config(config)
    _install_runtime_fakes(monkeypatch, spec, _FakeFormalProvenance())
    with pytest.raises(ValueError, match="multiple formal input roles"):
        formal.validate_formal_tabarena_inputs(spec)


def test_programmatic_spec_cannot_bypass_parser_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = formal.parse_formal_tabarena_config(_fixture_config(tmp_path))
    invalid = replace(spec, terminal_step=40_000)
    _install_runtime_fakes(monkeypatch, invalid, _FakeFormalProvenance())
    with pytest.raises(ValueError, match="terminal_step"):
        formal.validate_formal_tabarena_inputs(invalid)


def test_public_validator_has_no_module_injection_seam() -> None:
    assert tuple(inspect.signature(formal.validate_formal_tabarena_inputs).parameters) == (
        "spec",
    )


def test_shared_result_normalizer_supports_the_formal_arm_order() -> None:
    frameworks = {"r": "rope", "t": "temporary", "n": "none"}
    results = []
    for framework, error in (("r", 0.3), ("t", 0.2), ("n", 0.1)):
        results.append(
            {
                "experiment_metadata": {},
                "framework": framework,
                "memory_usage": {},
                "metric": "roc_auc",
                "metric_error": error,
                "problem_type": "binary",
                "simulation_artifacts": None,
                "task_metadata": {
                    "tid": 7,
                    "name": "task",
                    "fold": 0,
                    "repeat": 0,
                    "sample": 0,
                    "split_idx": 0,
                },
                "time_infer_s": 0.2,
                "time_train_s": 0.1,
            }
        )
    normalized = tabarena._normalize_results(
        results,
        framework_to_arm=frameworks,
        roster=("task",),
        expected_count=3,
        expected_tasks={
            "task": {"task_id": 7, "problem_type": "binary", "metric": "roc_auc"}
        },
        arm_order=formal.FORMAL_ARMS,
    )
    rows = {(row["arm"], row["dataset"]): row for row in normalized}
    assert tabarena._mean_ranks(
        rows, roster=("task",), arm_order=formal.FORMAL_ARMS
    ) == {"rope": 3.0, "temporary": 2.0, "none": 1.0}


def test_exact_t_loader_uses_private_module_name_without_importing_tabicl(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    config["expected_training_code_sha"] = "b" * 40
    provenance_file = (
        Path(config["training_code_root"])
        / "src"
        / "tabicl"
        / "train"
        / "_provenance.py"
    )
    provenance_file.parent.mkdir(parents=True)
    provenance_file.write_text(
        "\n".join(
            [
                "class ParentTrust: pass",
                "def validate_parent_trust(): pass",
                "def validate_canonical_transaction_ledger(): pass",
                "def validate_cohort_provenance(): pass",
                "def validate_manifest(): pass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = formal.parse_formal_tabarena_config(config)
    git = _FakeGit(spec.training_code_root, spec.training_code_sha)
    before = {name for name in sys.modules if name == "tabicl" or name.startswith("tabicl.")}
    module = formal._trusted_provenance_module(spec, git)
    after = {name for name in sys.modules if name == "tabicl" or name.startswith("tabicl.")}
    assert module.__name__.startswith("_pe_mechanism_formal_provenance_")
    assert Path(module.__file__).resolve() == provenance_file.resolve()
    assert after == before
