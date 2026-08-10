"""Strict input contract for future formal three-arm TabArena evaluation.

The existing :mod:`pe_mechanism.tabarena_evaluation` protocol is a frozen
exploratory comparison of RoPE, No-PE, and the released checkpoint.  This
module does not alter or execute that workflow.  It validates one formal
RoPE/Temporary/No-PE checkpoint cohort, including every finalized ancestor up
to the requested stage, before a future benchmark runner may consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .provenance import GitEvidence, VerifiedFile, verify_file, verify_git_tree


FORMAL_ARMS = ("rope", "temporary", "none")
FORMAL_STAGES = (
    ("stage1", 500_000),
    ("stage2", 40_000),
    ("stage3", 10_000),
)
FORMAL_SEEDS = frozenset({42, 43, 44})

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "seed",
    "stage",
    "terminal_step",
    "training_code_root",
    "expected_training_code_sha",
    "artifact_root",
    "transaction_ledger_path",
    "expected_transaction_ledger_file_sha256",
    "expected_transaction_ledger_manifest_sha256",
    "arms",
}
_ENTRY_FIELDS = {
    "checkpoint_path",
    "expected_checkpoint_sha256",
    "finalized_manifest_path",
    "expected_finalized_manifest_file_sha256",
    "expected_finalized_manifest_sha256",
}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class FormalStageInput:
    checkpoint_path: Path
    checkpoint_sha256: str
    finalized_manifest_path: Path
    finalized_manifest_file_sha256: str
    finalized_manifest_sha256: str


@dataclass(frozen=True)
class FormalTabArenaInputSpec:
    seed: int
    stage: str
    terminal_step: int
    training_code_root: Path
    training_code_sha: str
    artifact_root: Path
    transaction_ledger_path: Path
    transaction_ledger_file_sha256: str
    transaction_ledger_manifest_sha256: str
    arms: Mapping[str, Mapping[str, FormalStageInput]]


def _exact_object(
    value: object, *, label: str, fields: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _digest(value: object, *, label: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase {length}-character digest")
    return value


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _stage_budget(stage: str) -> int:
    matches = [budget for name, budget in FORMAL_STAGES if name == stage]
    if len(matches) != 1:
        raise ValueError("stage must be stage1, stage2, or stage3")
    return matches[0]


def _required_stages(stage: str) -> tuple[tuple[str, int], ...]:
    for index, (name, _) in enumerate(FORMAL_STAGES):
        if name == stage:
            return FORMAL_STAGES[: index + 1]
    raise ValueError("stage must be stage1, stage2, or stage3")


def parse_formal_tabarena_config(data: Mapping[str, Any]) -> FormalTabArenaInputSpec:
    """Parse an exact, path-explicit formal checkpoint-chain contract."""

    config = _exact_object(
        data, label="formal TabArena input config", fields=_TOP_LEVEL_FIELDS
    )
    if config["schema_version"] != 1:
        raise ValueError("formal TabArena input schema_version must be 1")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FORMAL_SEEDS:
        raise ValueError("seed must be exactly one of 42, 43, or 44")
    stage = config["stage"]
    if not isinstance(stage, str):
        raise ValueError("stage must be a string")
    budget = _stage_budget(stage)
    if config["terminal_step"] != budget:
        raise ValueError("terminal_step does not match the canonical stage budget")

    training_root = _absolute_path(
        config["training_code_root"], label="training_code_root"
    )
    training_sha = _digest(
        config["expected_training_code_sha"],
        label="expected_training_code_sha",
        length=40,
    )
    artifact_root = _absolute_path(config["artifact_root"], label="artifact_root")
    ledger_path = _absolute_path(
        config["transaction_ledger_path"], label="transaction_ledger_path"
    )
    if ledger_path != artifact_root / "transaction-ledger.json":
        raise ValueError("transaction ledger must be the canonical artifact-root file")

    raw_arms = _exact_object(config["arms"], label="arms", fields=set(FORMAL_ARMS))
    expected_stage_names = {name for name, _ in _required_stages(stage)}
    arms: dict[str, dict[str, FormalStageInput]] = {}
    physical_paths: set[Path] = {ledger_path}
    for arm in FORMAL_ARMS:
        raw_chain = _exact_object(
            raw_arms[arm],
            label=f"{arm} chain",
            fields=expected_stage_names,
        )
        chain: dict[str, FormalStageInput] = {}
        for stage_name, terminal_step in _required_stages(stage):
            raw = _exact_object(
                raw_chain[stage_name],
                label=f"{arm} {stage_name}",
                fields=_ENTRY_FIELDS,
            )
            checkpoint = _absolute_path(
                raw["checkpoint_path"], label=f"{arm} {stage_name} checkpoint_path"
            )
            finalized = _absolute_path(
                raw["finalized_manifest_path"],
                label=f"{arm} {stage_name} finalized_manifest_path",
            )
            expected_checkpoint = (
                artifact_root
                / "arms"
                / arm
                / stage_name
                / f"step-{terminal_step}.ckpt"
            )
            expected_finalized = (
                artifact_root
                / "arms"
                / arm
                / stage_name
                / "finalized-checkpoint.json"
            )
            if checkpoint != expected_checkpoint or finalized != expected_finalized:
                raise ValueError(
                    f"{arm} {stage_name} paths differ from the canonical ledger layout"
                )
            for path in (checkpoint, finalized):
                if path in physical_paths:
                    raise ValueError("formal input roles must use distinct files")
                physical_paths.add(path)
            chain[stage_name] = FormalStageInput(
                checkpoint_path=checkpoint,
                checkpoint_sha256=_digest(
                    raw["expected_checkpoint_sha256"],
                    label=f"{arm} {stage_name} checkpoint SHA-256",
                ),
                finalized_manifest_path=finalized,
                finalized_manifest_file_sha256=_digest(
                    raw["expected_finalized_manifest_file_sha256"],
                    label=f"{arm} {stage_name} finalized file SHA-256",
                ),
                finalized_manifest_sha256=_digest(
                    raw["expected_finalized_manifest_sha256"],
                    label=f"{arm} {stage_name} finalized manifest SHA-256",
                ),
            )
        arms[arm] = chain

    return FormalTabArenaInputSpec(
        seed=seed,
        stage=stage,
        terminal_step=budget,
        training_code_root=training_root,
        training_code_sha=training_sha,
        artifact_root=artifact_root,
        transaction_ledger_path=ledger_path,
        transaction_ledger_file_sha256=_digest(
            config["expected_transaction_ledger_file_sha256"],
            label="transaction ledger file SHA-256",
        ),
        transaction_ledger_manifest_sha256=_digest(
            config["expected_transaction_ledger_manifest_sha256"],
            label="transaction ledger manifest SHA-256",
        ),
        arms=arms,
    )


def load_formal_tabarena_config(path: str | os.PathLike[str]) -> FormalTabArenaInputSpec:
    """Load a JSON config through a no-follow, content-stable file handle."""

    file = verify_file(path)
    try:
        value = json.loads(file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal TabArena input config is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("formal TabArena input config must contain one object")
    spec = parse_formal_tabarena_config(value)
    file.assert_unchanged()
    return spec


def _validate_materialized_spec(spec: FormalTabArenaInputSpec) -> None:
    """Apply the parser's full contract to programmatically constructed specs."""

    if not isinstance(spec, FormalTabArenaInputSpec):
        raise TypeError("spec must be a FormalTabArenaInputSpec")
    data = {
        "schema_version": 1,
        "seed": spec.seed,
        "stage": spec.stage,
        "terminal_step": spec.terminal_step,
        "training_code_root": str(spec.training_code_root),
        "expected_training_code_sha": spec.training_code_sha,
        "artifact_root": str(spec.artifact_root),
        "transaction_ledger_path": str(spec.transaction_ledger_path),
        "expected_transaction_ledger_file_sha256": (
            spec.transaction_ledger_file_sha256
        ),
        "expected_transaction_ledger_manifest_sha256": (
            spec.transaction_ledger_manifest_sha256
        ),
        "arms": {
            arm: {
                stage: {
                    "checkpoint_path": str(entry.checkpoint_path),
                    "expected_checkpoint_sha256": entry.checkpoint_sha256,
                    "finalized_manifest_path": str(entry.finalized_manifest_path),
                    "expected_finalized_manifest_file_sha256": (
                        entry.finalized_manifest_file_sha256
                    ),
                    "expected_finalized_manifest_sha256": (
                        entry.finalized_manifest_sha256
                    ),
                }
                for stage, entry in spec.arms[arm].items()
            }
            for arm in spec.arms
        },
    }
    if parse_formal_tabarena_config(data) != spec:
        raise ValueError("materialized formal TabArena spec is not canonical")


def _verified_inputs(
    spec: FormalTabArenaInputSpec,
) -> tuple[VerifiedFile, dict[tuple[str, str], tuple[VerifiedFile, VerifiedFile]]]:
    ledger = verify_file(
        spec.transaction_ledger_path,
        expected_sha256=spec.transaction_ledger_file_sha256,
    )
    files: dict[tuple[str, str], tuple[VerifiedFile, VerifiedFile]] = {}
    identities = {(ledger.device, ledger.inode): "transaction_ledger"}
    for arm in FORMAL_ARMS:
        for stage, _ in _required_stages(spec.stage):
            entry = spec.arms[arm][stage]
            checkpoint = verify_file(
                entry.checkpoint_path, expected_sha256=entry.checkpoint_sha256
            )
            finalized = verify_file(
                entry.finalized_manifest_path,
                expected_sha256=entry.finalized_manifest_file_sha256,
            )
            for label, file in (
                (f"{arm}_{stage}_checkpoint", checkpoint),
                (f"{arm}_{stage}_finalized", finalized),
            ):
                identity = (file.device, file.inode)
                previous = identities.setdefault(identity, label)
                if previous != label:
                    raise ValueError(
                        "one physical file cannot satisfy multiple formal input roles"
                    )
            files[(arm, stage)] = (checkpoint, finalized)
    return ledger, files


def _strict_json(file: VerifiedFile, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _trusted_provenance_module(
    spec: FormalTabArenaInputSpec, training_git: GitEvidence
) -> Any:
    """Load the formal trust API only from the bound clean training checkout.

    The module is loaded under a private name so validation does not pre-import
    ``tabicl`` and thereby change the later benchmark runtime's import path.
    """

    if training_git.evidence_level != "strict" or training_git.legacy_reasons:
        raise RuntimeError("formal training checkout lacks strict clean evidence")
    configured_root = spec.training_code_root.resolve(strict=True)
    verified_root = training_git.root.resolve(strict=True)
    if configured_root != verified_root:
        raise RuntimeError("training_code_root is not the verified Git root")
    source_root = (training_git.root / "src").resolve(strict=True)
    expected_file = (
        source_root / "tabicl" / "train" / "_provenance.py"
    ).resolve(strict=True)
    path_tag = hashlib.sha256(os.fsencode(expected_file)).hexdigest()[:16]
    module_name = (
        f"_pe_mechanism_formal_provenance_{training_git.head_sha}_{path_tag}"
    )
    existing = sys.modules.get(module_name)
    if existing is None:
        module_spec = importlib.util.spec_from_file_location(module_name, expected_file)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("cannot construct the exact-T provenance module spec")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = existing
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or Path(raw_file).resolve(strict=True) != expected_file:
        raise RuntimeError("formal provenance module is not from the bound training tree")
    for name in (
        "ParentTrust",
        "validate_parent_trust",
        "validate_canonical_transaction_ledger",
        "validate_cohort_provenance",
        "validate_manifest",
    ):
        symbol = getattr(module, name, None)
        if not callable(symbol):
            raise RuntimeError(f"formal provenance module lacks canonical {name}")
        try:
            symbol_file = Path(inspect.getfile(inspect.unwrap(symbol))).resolve(
                strict=True
            )
        except (TypeError, OSError) as error:
            raise RuntimeError(f"formal provenance symbol {name} has no source") from error
        if symbol_file != expected_file:
            raise RuntimeError(f"formal provenance symbol {name} is not defined by exact T")
    training_git.assert_unchanged()
    return module


def _parent_payload(validated: Any) -> Mapping[str, Any]:
    manifest = getattr(validated, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("formal parent validator returned no manifest")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("parent"), Mapping):
        raise RuntimeError("formal parent validator returned an invalid parent manifest")
    return payload["parent"]


def validate_formal_tabarena_inputs(
    spec: FormalTabArenaInputSpec,
) -> dict[str, Any]:
    """Validate finalized checkpoint lineage and return path-free evidence.

    This function deliberately does not claim that the production campaign has
    been accepted or that a benchmark may run.  Those later gates also require
    the independently verified campaign-acceptance and terminal scheduler-log
    artifacts, which are not inputs to this checkpoint-lineage contract.
    """

    _validate_materialized_spec(spec)
    if spec.artifact_root.is_symlink() or not spec.artifact_root.is_dir():
        raise ValueError("artifact_root must be an existing real directory")
    if spec.training_code_root.is_symlink() or not spec.training_code_root.is_dir():
        raise ValueError("training_code_root must be an existing real directory")

    training_git = verify_git_tree(
        spec.training_code_root, expected_sha=spec.training_code_sha
    )
    module = _trusted_provenance_module(spec, training_git)
    ledger_file, files = _verified_inputs(spec)
    ledger = _strict_json(ledger_file, label="transaction ledger")
    module.validate_manifest(
        ledger,
        expected_kind="transaction_ledger",
        expected_sha256=spec.transaction_ledger_manifest_sha256,
    )
    entries = module.validate_canonical_transaction_ledger(
        ledger["payload"], artifact_root=spec.artifact_root
    )
    payload = ledger["payload"]
    campaign = payload.get("campaign_binding")
    if not isinstance(campaign, Mapping):
        raise ValueError("transaction ledger lacks campaign binding")
    if campaign.get("training_commit_sha") != training_git.head_sha:
        raise ValueError("transaction ledger campaign differs from exact training T")
    if payload.get("study_id") != f"{campaign.get('campaign_id')}-seed{spec.seed}":
        raise ValueError("transaction ledger does not belong to the requested seed")

    validated: dict[tuple[str, str], Any] = {}
    stage_reports: dict[str, Mapping[str, Any]] = {}
    for stage, terminal_step in _required_stages(spec.stage):
        for arm in FORMAL_ARMS:
            entry = entries[(arm, stage)]
            checkpoint_file, finalized_file = files[(arm, stage)]
            trust = module.ParentTrust(
                checkpoint_path=checkpoint_file.path,
                finalized_manifest_path=finalized_file.path,
                transaction_ledger_path=ledger_file.path,
                transaction_ledger_sha256=spec.transaction_ledger_manifest_sha256,
                study_id=payload["study_id"],
                arm=arm,
                parent_stage=stage,
                upstream_identity=entry["upstream_identity"],
                artifact_identity=entry["artifact_identity"],
                artifact_root=spec.artifact_root,
            )
            result = module.validate_parent_trust(trust)
            parent = _parent_payload(result)
            configured = spec.arms[arm][stage]
            if (
                getattr(result, "checkpoint_sha256", None)
                != checkpoint_file.digest.sha256
                or parent.get("checkpoint_sha256") != checkpoint_file.digest.sha256
                or parent.get("finalized_manifest_sha256")
                != configured.finalized_manifest_sha256
                or parent.get("stage") != stage
                or parent.get("terminal_step") != terminal_step
                or parent.get("arm") != arm
            ):
                raise ValueError(
                    f"validated {arm} {stage} bytes or identity differ from config"
                )
            checkpoint = getattr(result, "checkpoint", None)
            if not isinstance(checkpoint, Mapping):
                raise RuntimeError("formal parent validator returned no checkpoint")
            provenance = checkpoint.get("provenance")
            if not isinstance(provenance, Mapping):
                raise RuntimeError("validated formal checkpoint lacks provenance")
            if stage != "stage1":
                previous = "stage1" if stage == "stage2" else "stage2"
                previous_manifest = getattr(validated[(arm, previous)], "manifest", None)
                observed_parent = (
                    provenance.get("manifests", {}).get("parent")
                    if isinstance(provenance.get("manifests"), Mapping)
                    else None
                )
                if observed_parent != previous_manifest:
                    raise ValueError(
                        f"{arm} {stage} does not descend from the exact finalized {previous}"
                    )
            validated[(arm, stage)] = result

        bundles = {
            arm: validated[(arm, stage)].checkpoint["provenance"]
            for arm in FORMAL_ARMS
        }
        cohort = module.validate_cohort_provenance(bundles)
        if not isinstance(cohort, Mapping):
            raise RuntimeError("formal cohort validator returned an invalid report")
        stage_reports[stage] = cohort

    target = {
        arm: _parent_payload(validated[(arm, spec.stage)]) for arm in FORMAL_ARMS
    }
    shared_fields = (
        "study_id",
        "stage",
        "terminal_step",
        "np_seed",
        "torch_seed",
        "identity_rng_seed",
        "world_size",
        "cuda_device_count",
        "source_sha256",
        "environment_sha256",
        "prior_sha256",
        "architecture_sha256",
        "optimizer_sha256",
        "scientific_sha256",
        "cohort_protocol_sha256",
    )
    for field in shared_fields:
        if len({target[arm][field] for arm in FORMAL_ARMS}) != 1:
            raise ValueError(f"formal target arms differ in shared {field}")
    for field in ("treatment_sha256", "arm_protocol_sha256"):
        if len({target[arm][field] for arm in FORMAL_ARMS}) != len(FORMAL_ARMS):
            raise ValueError(f"formal target arms do not have distinct {field}")

    for file in (ledger_file, *(item for pair in files.values() for item in pair)):
        file.assert_unchanged()
    training_git.assert_unchanged()
    report = {
        "schema_version": 1,
        "kind": "formal_tabarena_three_arm_input_validation",
        "evidence_scope": "formal_checkpoint_lineage_only",
        "checkpoint_lineage_verified": True,
        "campaign_acceptance_verified": False,
        "terminal_scheduler_evidence_verified": False,
        "benchmark_execution_ready": False,
        "training_code_sha": training_git.head_sha,
        "campaign_id": campaign["campaign_id"],
        "study_id": payload["study_id"],
        "seed": spec.seed,
        "stage": spec.stage,
        "terminal_step": spec.terminal_step,
        "transaction_ledger": {
            "file_sha256": ledger_file.digest.sha256,
            "manifest_sha256": spec.transaction_ledger_manifest_sha256,
        },
        "cohort_protocol_sha256": stage_reports[spec.stage][
            "cohort_protocol_sha256"
        ],
        "arms": {
            arm: {
                "checkpoint_sha256": target[arm]["checkpoint_sha256"],
                "finalized_manifest_sha256": target[arm][
                    "finalized_manifest_sha256"
                ],
                "treatment_sha256": target[arm]["treatment_sha256"],
                "arm_protocol_sha256": target[arm]["arm_protocol_sha256"],
            }
            for arm in FORMAL_ARMS
        },
    }
    if any(
        key.endswith(("_path", "_root"))
        for mapping in (report, report["transaction_ledger"], *report["arms"].values())
        for key in mapping
    ):
        raise AssertionError("formal validation report must remain path-free")
    return report


__all__ = [
    "FORMAL_ARMS",
    "FORMAL_SEEDS",
    "FORMAL_STAGES",
    "FormalStageInput",
    "FormalTabArenaInputSpec",
    "load_formal_tabarena_config",
    "parse_formal_tabarena_config",
    "validate_formal_tabarena_inputs",
]
