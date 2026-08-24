from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from pe_mechanism.full_suite_pair import (
    _FIXED_CLASSIFIER_OPTIONS,
    _comparison,
    PairTaskOOM,
    aggregate_run,
    code_provenance,
    load_pair_manifest,
    load_roster,
    load_shard_plan,
    run_canary,
    run_shard,
    validate_plan_metadata,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
ROSTER = PACKAGE_ROOT / "manifests" / "beyondarena-nontext-classification-lite-v1.json"
PLAN = PACKAGE_ROOT / "manifests" / "beyondarena-nontext-classification-lite-v2-shards.json"
V1_PLAN = PACKAGE_ROOT / "manifests" / "beyondarena-nontext-classification-lite-v1-shards.json"
TABARENA_ROSTER = PACKAGE_ROOT / "manifests" / "tabarena-v0.1-classification-roster.json"
TABARENA_PLAN = PACKAGE_ROOT / "manifests" / "tabarena-v0.1-classification-lite-v2-shards.json"
TABARENA_V1_PLAN = PACKAGE_ROOT / "manifests" / "tabarena-v0.1-classification-lite-v1-shards.json"
NATIVE_LITE_FIXTURE = (
    PACKAGE_ROOT / "tests" / "fixtures" / "beyondarena-c987-lite-native-metadata.json"
)
TABARENA_NATIVE_LITE_FIXTURE = (
    PACKAGE_ROOT / "tests" / "fixtures" / "tabarena-v0.1-c987-lite-native-metadata.json"
)
RUNNER = PACKAGE_ROOT / "scripts" / "run_pair_full_suite_shard.py"
AGGREGATOR = PACKAGE_ROOT / "scripts" / "aggregate_pair_full_suite.py"
WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_pair_full_suite_shard.sh"
CANARY_WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_pair_full_suite_canary.sh"
A10_WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_pair_tabarena_a10.sh"
PYTHON_VERIFIER = PACKAGE_ROOT / "scripts" / "verify_python_environment.py"
SUBMITTER = PACKAGE_ROOT / "scripts" / "submit_pair_full_suite.py"
TABARENA_SHA = "c987d91556a14d4c9b3383c35d1b0ec68ff81883"
MODEL_SHA = "1" * 40
ANALYSIS_SHA = "2" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_stream(step: int) -> dict[str, Any]:
    schema = json.dumps(
        {
            "batch_size": 64,
            "batch_size_per_gp": 8,
            "max_classes": 10,
            "max_features": 100,
            "max_seq_len": 1024,
            "prior_type": "graph_scm",
            "schema_version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    body = {
        "schema_version": 1,
        "algorithm": "sha256-schema-seed-rank-logical-step-v1",
        "schema": schema,
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "experiment_seed": 42,
        "ddp_rank": 0,
        "world_size": 1,
        "cursor": step,
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(
            repr({key: body[key] for key in sorted(body)}).encode("utf-8")
        ).hexdigest(),
    }


def _self_hashed_document(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    core = {"kind": kind, "payload": payload, "schema_version": 1}
    return {
        **core,
        "sha256": hashlib.sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _pair(
    tmp_path: Path,
    *,
    step: int = 50_000,
    include_prior_stream: bool = True,
    formal_eligible: bool = False,
    checkpoint_stage: str = "stage1",
    legacy_receipts: bool | None = None,
    training_receipts: bool | None = None,
):
    if legacy_receipts is None:
        legacy_receipts = checkpoint_stage == "stage3" or step in {
            250_000,
            500_000,
        }
    if training_receipts is None:
        training_receipts = step == 50_000 and include_prior_stream
    arm_ids = ("rope", "fingerprint") if step == 50_000 else ("rope", "none")
    arms = []
    receipt_paths: dict[str, str] = {}
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for arm_id in arm_ids:
        checkpoint_root = tmp_path / arm_id if legacy_receipts else tmp_path
        checkpoint_root.mkdir(exist_ok=True)
        checkpoint = checkpoint_root / (
            f"step-{step}.ckpt" if legacy_receipts else f"{arm_id}.ckpt"
        )
        checkpoint_payload = {
            "curr_step": step,
            "config": {
                "embed_dim": 128,
                "col_num_blocks": 3,
                "col_nhead": 8,
                "row_num_blocks": 3,
                "row_nhead": 8,
                "icl_num_blocks": 12,
                "icl_nhead": 8,
                "row_identity_mode": "none" if arm_id == "fingerprint" else arm_id,
                "row_fingerprint": arm_id == "fingerprint",
                "row_fingerprint_dim": 16,
            },
            "state_dict": {"weight": torch.zeros(2, 3)},
        }
        if include_prior_stream:
            checkpoint_payload["prior_stream"] = _prior_stream(step)
        checkpoint_payloads[arm_id] = checkpoint_payload
        torch.save(checkpoint_payload, checkpoint)
        if legacy_receipts:
            receipt_body = {
                "schema_version": 1,
                "kind": "tabicl-legacy-pilot-checkpoint-snapshot",
                "classification": "exploratory-pilot-only",
                "continuation_id": f"unit-legacy-{step}",
                "created_at_utc": "2026-08-17T00:00:00+00:00",
                "mode": arm_id,
                "stage": checkpoint_stage,
                "step": step,
                "continuation_source_commit": MODEL_SHA,
                "source_provenance_status": "operational-history-only-not-checkpoint-bound",
                "source_checkpoint": str(checkpoint),
                "snapshot_filename": checkpoint.name,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": _sha(checkpoint),
                "limitations": ["synthetic legacy test receipt"],
            }
            receipt = dict(receipt_body)
            receipt["manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    receipt_body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            receipt_path = checkpoint_root / "snapshot-manifest.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_paths[arm_id] = str(receipt_path)
        arms.append(
            {
                "arm_id": arm_id,
                "display_name": arm_id.title(),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
            }
        )
    path = tmp_path / "pair.json"
    manifest_payload = {
                "schema_version": 1,
                "kind": "two_arm_checkpoint_pair",
                "pair_id": "unit-pair",
                "formal_eligible": formal_eligible,
                "model_source_sha": MODEL_SHA,
                "arms": arms,
                "inference": {
                    "device": "cuda",
                    "seed": 42,
                    "n_estimators": 1,
                    "classifier_options": _FIXED_CLASSIFIER_OPTIONS,
                },
            }
    if checkpoint_stage != "stage1":
        manifest_payload["checkpoint_stage"] = checkpoint_stage
    if legacy_receipts:
        manifest_payload["legacy_snapshot_receipts"] = receipt_paths
    if training_receipts:
        training_root = tmp_path / "training-receipts"
        training_root.mkdir(exist_ok=True)
        environment_sha = "e" * 64
        submission_payload = {
            "formal_eligible": False,
            "seed": 42,
            "source_commit": MODEL_SHA,
            "study": "tabiclv2-fullsize-rope-fingerprint-continuation-v1",
            "scheduler_horizon_steps": 500_000,
            "environment_sha256": environment_sha,
            "jobs": [
                {"arm": arm_id, "from_step": 35_000, "to_step": 50_000}
                for arm_id in arm_ids
            ],
        }
        submission_path = training_root / "submission.json"
        submission_path.write_text(
            json.dumps(
                _self_hashed_document(
                    "fingerprint_fullsize_continuation_release_receipt",
                    submission_payload,
                )
            ),
            encoding="utf-8",
        )
        training_paths = {"submission": str(submission_path)}
        for arm in arms:
            checkpoint_payload = checkpoint_payloads[arm["arm_id"]]
            prior_stream = checkpoint_payload["prior_stream"]
            completion_payload = {
                "formal_eligible": False,
                "seed": 42,
                "source_commit": MODEL_SHA,
                "study": "tabiclv2-fullsize-rope-fingerprint-continuation-v1",
                "scheduler_horizon_steps": 500_000,
                "environment_sha256": environment_sha,
                "arm": arm["arm_id"],
                "to_step": 50_000,
                "checkpoint": {
                    "curr_step": 50_000,
                    "sha256": arm["checkpoint_sha256"],
                    "size_bytes": arm["checkpoint_size_bytes"],
                    "prior_manifest_sha256": prior_stream["manifest_sha256"],
                    "prior_schema_sha256": prior_stream["schema_sha256"],
                },
            }
            completion_path = training_root / f"{arm['arm_id']}-completion.json"
            completion_path.write_text(
                json.dumps(
                    _self_hashed_document(
                        "fingerprint_fullsize_segment_completion",
                        completion_payload,
                    )
                ),
                encoding="utf-8",
            )
            training_paths[f"{arm['arm_id']}_completion"] = str(completion_path)
        manifest_payload["training_receipts"] = training_paths
    path.write_text(
        json.dumps(
            manifest_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_pair_manifest(path)


def _roster(tmp_path: Path):
    names = [f"dataset-{letter}" for letter in "abcdefgh"]
    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "dataset_roster",
                "suite": "BeyondArena",
                "version": "pinned",
                "task_type": "classification",
                "subset": "lite",
                "problem_types": ["binary", "multiclass"],
                "text_features": "excluded",
                "count": len(names),
                "names": names,
                "roster_hash_algorithm": "sha256_newline_join_sorted_names",
                "roster_sha256": hashlib.sha256(
                    "\n".join(names).encode("utf-8")
                ).hexdigest(),
                "source_commit": TABARENA_SHA,
                "source_url": "https://github.com/autogluon/tabarena",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_roster(path, enforce_pinned=False)


def _plan(tmp_path: Path, roster):
    records = _plan_records(roster)
    costs = {
        name: (
            record["num_instances"] * record["num_cols_after_preprocessing"]
            + record["num_instances_test"]
            * min(record["num_instances_train"], 1024)
        )
        for name, record in records.items()
    }
    assignments = [[] for _ in range(8)]
    totals = [0] * 8
    for name in sorted(roster.names, key=lambda item: (-costs[item], item)):
        index = min(range(8), key=lambda item: (totals[item], item))
        assignments[index].append(name)
        totals[index] += costs[name]
    position = {name: index for index, name in enumerate(roster.names)}
    for names in assignments:
        names.sort(key=position.__getitem__)
    name_to_shard = {
        name: index for index, names in enumerate(assignments) for name in names
    }
    assignment_lines = "\n".join(
        f"{name}\t{name_to_shard[name]}\t{costs[name]}" for name in roster.names
    )
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cost_balanced_pair_shard_plan",
                "suite": "BeyondArena",
                "version": "v2",
                "source_commit": TABARENA_SHA,
                "roster_sha256": roster.names_sha256,
                "shard_count": 8,
                "cost_model": {
                    "name": "tabicl_pair_lpt_proxy_v2",
                    "formula": (
                        "num_instances*num_cols_after_preprocessing + "
                        "floor(num_instances_test)*min(floor(num_instances_train),1024)"
                    ),
                    "assignment": (
                        "longest_processing_time_first_ties_by_dataset_then_lowest_shard"
                    ),
                },
                "canary": {
                    "names": ["dataset-a", "dataset-b", "dataset-c"],
                    "requirements": {
                        "max_rows": "dataset-a",
                        "max_dimensions": "dataset-b",
                        "grouped": "dataset-a",
                        "temporal": "dataset-c",
                        "max_lpt_workload": max(costs, key=costs.__getitem__),
                    },
                },
                "shards": [
                    {"index": index, "estimated_cost_units": totals[index], "names": names}
                    for index, names in enumerate(assignments)
                ],
                "assignment_hash_algorithm": (
                    "sha256_newline_join_roster_order_name_tab_shard_tab_cost"
                ),
                "assignment_sha256": hashlib.sha256(
                    assignment_lines.encode("utf-8")
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_shard_plan(path, roster=roster), records


def _plan_records(roster):
    records = {}
    for index, name in enumerate(roster.names):
        records[name] = {
            "num_instances": 10_000 if name == "dataset-a" else 100 + index,
            "num_cols_after_preprocessing": 1_000 if name == "dataset-b" else 10 + index,
            "num_instances_train": 80 + index,
            "num_instances_test": 20 + index,
            "split_regime": (
                "grouped" if name == "dataset-a" else "temporal" if name == "dataset-c" else "iid"
            ),
        }
    return records


def _probabilities(*, offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [[0.75 - offset, 0.25 + offset], [0.2 + offset, 0.8 - offset]],
        dtype=np.dtype("<f4"),
    )


def _metric_error(*, offset: float, metric: str) -> float:
    probabilities = _probabilities(offset=offset)
    if metric == "roc_auc":
        positive = probabilities[:, 1]
        return 0.0 if positive[1] > positive[0] else 0.5 if positive[1] == positive[0] else 1.0
    if metric == "log_loss":
        return float(-0.5 * (np.log(probabilities[0, 0]) + np.log(probabilities[1, 1])))
    raise AssertionError(metric)


def _write_prediction(path: Path, *, offset: float = 0.0) -> None:
    probabilities = _probabilities(offset=offset)
    row_fingerprints = np.arange(64, dtype=np.uint8).reshape(2, 32)
    targets = np.arange(64, 128, dtype=np.uint8).reshape(2, 32)
    class_bytes = json.dumps(
        [{"type": "int", "value": 0}, {"type": "int", "value": 1}],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            probabilities=probabilities,
            encoded_targets=np.asarray([0, 1], dtype=np.dtype("<i8")),
            row_fingerprints=row_fingerprints,
            test_target_fingerprints=targets,
            class_labels_utf8=np.frombuffer(class_bytes, dtype=np.uint8),
            train_content_sha256=np.arange(32, dtype=np.uint8),
            test_content_sha256=np.arange(32, 64, dtype=np.uint8),
        )


def _executor(task, staging, fallback):
    assert fallback["level"] in range(4)
    _write_prediction(staging / "rope" / "predictions.npz", offset=0.0)
    _write_prediction(staging / "fingerprint" / "predictions.npz", offset=0.3)
    metadata = {
        "dataset_name": task.name,
        "benchmark_dataset_id": f"{task.name}-internal",
        "task_id": task.index + 100,
        "problem_type": "binary" if task.index != 1 else "multiclass",
        "metric": "roc_auc" if task.index != 1 else "log_loss",
        "split_regime": (
            "grouped"
            if task.name == "dataset-a"
            else "temporal"
            if task.name == "dataset-c"
            else "iid"
        ),
        "fold": 0,
        "repeat": 0,
        "split_index": 0,
    }
    metric = metadata["metric"]
    results = {
        "rope": {
            "metric_error": _metric_error(offset=0.0, metric=metric),
            "time_train_s": 1.0,
            "time_infer_s": 2.0,
        },
        "fingerprint": {
            "metric_error": _metric_error(offset=0.3, metric=metric),
            "time_train_s": 1.1,
            "time_infer_s": 2.1,
        },
    }
    return metadata, results


def _provenance():
    return code_provenance(
        analysis_sha=ANALYSIS_SHA,
        model_sha=MODEL_SHA,
        tabarena_sha=TABARENA_SHA,
    )


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _runner_module(label: str):
    spec = importlib.util.spec_from_file_location(label, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FixtureNativeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def subset_tasks(self, *, dataset_names, split_indices):
        assert split_indices == "lite"
        requested = set(dataset_names)
        selected = [
            dict(row)
            for row in self.rows
            if row["dataset_name"] in requested
        ]
        available = {row["dataset_name"] for row in selected}
        if available != requested:
            raise ValueError("fixture native lite split is missing a requested dataset")
        return _FixtureNativeCollection(selected)

    def to_dataframe(self):
        return pd.DataFrame(self.rows)

    def dataset_to_tid(self):
        result = {}
        for row in self.rows:
            task_id = str(row["task_id_str"])
            parts = task_id.split("|")
            result[row["tabarena_task_name"]] = int(
                parts[1] if len(parts) > 1 else task_id
            )
        return result


class _FixtureArena:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.task_metadata_collection = _FixtureNativeCollection(rows)

    @property
    def task_metadata(self):  # pragma: no cover - forbidden legacy boundary
        raise AssertionError("the lossy legacy task_metadata bridge was accessed")


def _native_fixture_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(NATIVE_LITE_FIXTURE.read_text(encoding="utf-8"))
    return payload, payload["rows"]


def _tabarena_native_fixture_rows() -> tuple[
    dict[str, Any], list[dict[str, Any]]
]:
    payload = json.loads(
        TABARENA_NATIVE_LITE_FIXTURE.read_text(encoding="utf-8")
    )
    return payload, payload["rows"]


def test_beyondarena_roster_is_exact_nontext_classification_lite_89() -> None:
    roster = load_roster(ROSTER)
    plan = load_shard_plan(PLAN, roster=roster)
    payload = json.loads(ROSTER.read_text(encoding="utf-8"))
    assert roster.suite_id == "beyondarena"
    assert roster.source_commit == TABARENA_SHA
    assert len(roster.names) == payload["count"] == 89
    assert tuple(sorted(roster.names)) == roster.names
    assert len(set(roster.names)) == 89
    assert payload["problem_types"] == ["binary", "multiclass"]
    assert payload["subset"] == "lite"
    assert payload["text_features"] == "excluded"
    assert plan.shard_count == 8
    assert len({name for shard in plan.assignments for name in shard}) == 89
    assert plan.canary_requirements == {
        "max_rows": "amex_non_iid_1m",
        "max_dimensions": "lung_cancer_epithelial_genexp",
        "grouped": "amex_non_iid_1m",
        "temporal": "ghanas_indigenous_intel",
        "max_lpt_workload": "home_credit_default_stability_1m",
    }
    forbidden = (
        "/" + "mnt" + "/",
        "/" + "home" + "/",
        "sl" + "urm",
        "noe" + "ther",
        "jia" + "xio",
    )
    assert not any(
        token in value.casefold()
        for value in _strings(payload)
        for token in forbidden
    )


@pytest.mark.parametrize(
    ("roster_path", "plan_path", "assignment_sha256", "totals"),
    [
        (
            ROSTER,
            PLAN,
            "6abc38058806dae5e296fc4e361774e0fe1bac59b41c3f8bbe85b8b35c6223cb",
            (
                1_112_272_688,
                502_658_910,
                382_096_169,
                382_097_704,
                382_090_532,
                382_100_346,
                382_094_157,
                382_090_277,
            ),
        ),
        (
            TABARENA_ROSTER,
            TABARENA_PLAN,
            "af5b6a869cc8384deb20ca7c64c07826e4d834e61d745df274488db62de98fc7",
            (
                52_700_000,
                47_059_512,
                38_860_992,
                38_288_944,
                38_292_651,
                38_275_978,
                38_335_757,
                38_280_801,
            ),
        ),
    ],
)
def test_shard_cost_contract_declares_floor_without_changing_frozen_plan(
    roster_path: Path,
    plan_path: Path,
    assignment_sha256: str,
    totals: tuple[int, ...],
) -> None:
    roster = load_roster(roster_path)
    plan = load_shard_plan(plan_path, roster=roster)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v2"
    assert payload["cost_model"]["name"] == "tabicl_pair_lpt_proxy_v2"
    assert payload["cost_model"]["formula"] == (
        "num_instances*num_cols_after_preprocessing + "
        "floor(num_instances_test)*min(floor(num_instances_train),1024)"
    )
    assert plan.assignment_sha256 == assignment_sha256
    assert plan.estimated_cost_units == totals


@pytest.mark.parametrize(
    ("roster_path", "plan_path", "file_sha256"),
    [
        (
            ROSTER,
            V1_PLAN,
            "9fdd7da8d4fbb2e8551318d9a2bd56a77ae732fd27c77a7c6f34c0585e4c7adf",
        ),
        (
            TABARENA_ROSTER,
            TABARENA_V1_PLAN,
            "a3ee265e982678ac4f7870f8a920de0896dd0f106049b8c2863388b34b234e1a",
        ),
    ],
)
def test_ambiguous_v1_shard_plans_are_preserved_but_rejected_for_production(
    roster_path: Path, plan_path: Path, file_sha256: str
) -> None:
    assert _sha(plan_path) == file_sha256
    roster = load_roster(roster_path)
    with pytest.raises(ValueError, match="pinned eight-shard roster"):
        load_shard_plan(plan_path, roster=roster)


def test_pair_runtime_uses_exact_pinned_native_lite_schema_not_legacy_bridge() -> None:
    module = _runner_module("pair_shard_runner_native_schema")
    fixture, rows = _native_fixture_rows()
    assert fixture["source_commit"] == TABARENA_SHA
    assert fixture["source_metadata_sha256"] == (
        "04b46ef37647298991e380f972fab190ed0a26a43dbc2699f10c735cd72816fc"
    )
    runtime = module.PairTaskRuntime.__new__(module.PairTaskRuntime)
    runtime.arena = _FixtureArena(rows)
    runtime.roster = type(
        "FixtureRoster",
        (),
        {
            "names": tuple(row["dataset_name"] for row in rows),
            "suite_id": "beyondarena",
        },
    )()

    records = runtime.plan_records()
    assert records["amex_non_iid_1m"] == {
        "num_instances": 1_249_605,
        "num_cols_after_preprocessing": 198,
        "num_instances_train": 1_000_350,
        "num_instances_test": 249_255,
        "split_regime": "grouped",
    }
    assert records["ghanas_indigenous_intel"]["split_regime"] == "temporal"
    assert records["lung_cancer_epithelial_genexp"][
        "num_cols_after_preprocessing"
    ] == 22_215
    internal, evidence = runtime._task_metadata("ghanas_indigenous_intel")
    assert internal == "ghanas_indigenous_intel-ecbbda50d44e"
    assert evidence["expected"] == {
        "task_id": 4_482_953_497,
        "problem_type": "multiclass",
        "metric": "log_loss",
    }
    assert evidence["task"]["split_index"] == 0


def test_tabarena_native_lite_uses_real_fractional_counts_and_none_dimensions() -> None:
    module = _runner_module("pair_shard_runner_tabarena_native_fractional")
    fixture, rows = _tabarena_native_fixture_rows()
    assert fixture["source_commit"] == TABARENA_SHA
    assert fixture["source_metadata_sha256"] == (
        "02f35e19dead7e3795f65e91eb00fdb7fa255896ed6bde0f29b2ce0af7a46296"
    )
    records = module._native_lite_metadata(
        _FixtureArena(rows),
        names=("Bank_Customer_Churn",),
        suite_id="tabarena-v0.1",
    )
    assert records["Bank_Customer_Churn"] == {
        "dataset_name": "Bank_Customer_Churn",
        "benchmark_dataset_id": "Bank_Customer_Churn",
        "task_id": 363_619,
        "problem_type": "binary",
        "metric": "roc_auc",
        "num_instances": 10_000,
        "num_cols_after_preprocessing": 10,
        "num_instances_train": 6_666.666666666667,
        "num_instances_test": 3_333.333333333333,
        "split_regime": "iid",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_split", "not exact r0f0"),
        ("duplicate", "not one-to-one"),
        ("fractional_train", "num_instances_train is invalid"),
        ("missing_dimensions", "num_cols_after_preprocessing is not"),
    ],
)
def test_native_lite_metadata_corruption_fails_closed(
    mutation: str, message: str
) -> None:
    module = _runner_module(f"pair_shard_runner_native_corruption_{mutation}")
    _, source_rows = _native_fixture_rows()
    rows = [dict(row) for row in source_rows]
    if mutation == "wrong_split":
        rows[0]["split_index"] = "r0f1"
        rows[0]["fold"] = 1
    elif mutation == "duplicate":
        duplicate = dict(rows[0])
        duplicate["tabarena_task_name"] += "-duplicate"
        duplicate["task_id_str"] = duplicate["task_id_str"].replace(
            "6293333622", "6293333623"
        )
        rows.append(duplicate)
    elif mutation == "fractional_train":
        rows[0]["num_instances_train"] = 1_000_349.5
    elif mutation == "missing_dimensions":
        rows[0]["num_cols_after_preprocessing"] = None
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    names = tuple(dict.fromkeys(row["dataset_name"] for row in rows))
    with pytest.raises(ValueError, match=message):
        module._native_lite_metadata(
            _FixtureArena(rows), names=names, suite_id="beyondarena"
        )


def test_pair_manifest_is_exactly_two_arms_and_content_verified(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    assert pair.arm_order == ("rope", "fingerprint")
    pair.arms[0].checkpoint_path.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        load_pair_manifest(pair.path)


def test_pair_manifest_rejects_a_third_arm(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    payload = json.loads(pair.path.read_text(encoding="utf-8"))
    payload["arms"].append(dict(payload["arms"][0], arm_id="fingerprint"))
    pair.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        load_pair_manifest(pair.path)


def test_pair_manifest_requires_inference_seed42(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    payload = json.loads(pair.path.read_text(encoding="utf-8"))
    payload["inference"]["seed"] = 43
    pair.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seed=42"):
        load_pair_manifest(pair.path)


def test_checkpoint_pair_rejects_treatment_mislabelling(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    manifest = json.loads(pair.path.read_text(encoding="utf-8"))
    fingerprint_path = pair.arms[1].checkpoint_path
    payload = torch.load(fingerprint_path, map_location="cpu", weights_only=True)
    payload["config"]["row_identity_mode"] = "rope"
    torch.save(payload, fingerprint_path)
    manifest["arms"][1]["checkpoint_sha256"] = _sha(fingerprint_path)
    manifest["arms"][1]["checkpoint_size_bytes"] = fingerprint_path.stat().st_size
    pair.path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Fingerprint arm treatment"):
        load_pair_manifest(pair.path)


def test_step50k_requires_exact_prior_stream_but_step250k_can_be_legacy(
    tmp_path: Path,
) -> None:
    modern_root = tmp_path / "modern"
    modern_root.mkdir()
    with pytest.raises(ValueError, match="missing prior_stream"):
        _pair(modern_root, step=50_000, include_prior_stream=False)
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy = _pair(legacy_root, step=250_000, include_prior_stream=False)
    assert legacy.checkpoint_contract["prior_stream"]["mode"] == "same_step_legacy"
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    with pytest.raises(ValueError, match="formal_eligible=false"):
        _pair(
            formal_root,
            step=250_000,
            include_prior_stream=False,
            formal_eligible=True,
        )


def test_step500k_legacy_pair_requires_two_matching_immutable_snapshot_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-500k"
    root.mkdir()
    pair = _pair(
        root,
        step=500_000,
        include_prior_stream=False,
        legacy_receipts=True,
    )
    assert pair.formal_eligible is False
    assert pair.checkpoint_stage == "stage1"
    assert "comparison_stage" not in pair.checkpoint_contract
    assert "cumulative_training_steps" not in pair.checkpoint_contract
    assert pair.checkpoint_contract["prior_stream"]["mode"] == "same_step_legacy"
    assert set(pair.checkpoint_contract["prior_stream"]["receipts"]) == {
        "rope",
        "none",
    }
    receipt = pair.legacy_snapshot_receipts["none"]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["continuation_id"] = "different"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash|disagree"):
        load_pair_manifest(pair.path)


def test_stage3_terminal_pair_is_explicit_receipt_bound_and_exploratory(
    tmp_path: Path,
) -> None:
    pair = _pair(
        tmp_path,
        checkpoint_stage="stage3",
        step=10_000,
        include_prior_stream=False,
    )
    assert pair.formal_eligible is False
    assert pair.checkpoint_stage == "stage3"
    assert pair.arm_order == ("rope", "none")
    assert pair.checkpoint_contract["comparison_stage"] == "stage3"
    assert pair.checkpoint_contract["comparison_step"] == 10_000
    assert pair.checkpoint_contract["cumulative_training_steps"] == 550_000
    assert pair.checkpoint_contract["prior_stream"]["mode"] == "same_step_legacy"


def test_stage3_terminal_profile_fails_closed(tmp_path: Path) -> None:
    wrong_step = tmp_path / "wrong-step"
    wrong_step.mkdir()
    with pytest.raises(ValueError, match="exploratory stage3 step-10000 rope/none"):
        _pair(
            wrong_step,
            checkpoint_stage="stage3",
            step=9_999,
            include_prior_stream=False,
        )

    missing_receipts = tmp_path / "missing-receipts"
    missing_receipts.mkdir()
    with pytest.raises(ValueError, match="requires two snapshot receipts"):
        _pair(
            missing_receipts,
            checkpoint_stage="stage3",
            step=10_000,
            include_prior_stream=False,
            legacy_receipts=False,
        )

    wrong_order = tmp_path / "wrong-order"
    wrong_order.mkdir()
    pair = _pair(
        wrong_order,
        checkpoint_stage="stage3",
        step=10_000,
        include_prior_stream=False,
    )
    manifest = json.loads(pair.path.read_text(encoding="utf-8"))
    manifest["arms"].reverse()
    pair.path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exploratory stage3 step-10000 rope/none"):
        load_pair_manifest(pair.path)

    formal = tmp_path / "formal"
    formal.mkdir()
    with pytest.raises(ValueError, match="formal_eligible=false"):
        _pair(
            formal,
            checkpoint_stage="stage3",
            step=10_000,
            include_prior_stream=False,
            formal_eligible=True,
        )


def test_stage3_terminal_receipt_must_bind_stage3(tmp_path: Path) -> None:
    pair = _pair(
        tmp_path,
        checkpoint_stage="stage3",
        step=10_000,
        include_prior_stream=False,
    )
    receipt_path = pair.legacy_snapshot_receipts["none"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stage"] = "stage1"
    body = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
    receipt["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the none checkpoint"):
        load_pair_manifest(pair.path)


def test_pair_roster_and_lineage_are_fail_closed(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(ValueError, match="pair must be exactly"):
        _pair(wrong, step=60_000)

    missing_250 = tmp_path / "missing-250"
    missing_250.mkdir()
    with pytest.raises(ValueError, match="requires two snapshot receipts"):
        _pair(missing_250, step=250_000, legacy_receipts=False)

    missing_50 = tmp_path / "missing-50"
    missing_50.mkdir()
    with pytest.raises(ValueError, match="submission and completion receipts"):
        _pair(missing_50, step=50_000, training_receipts=False)


def test_prior_stream_requires_seed42_schema_and_self_hash(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    manifest = json.loads(pair.path.read_text(encoding="utf-8"))
    for arm in manifest["arms"]:
        path = Path(arm["checkpoint_path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint["prior_stream"]["experiment_seed"] = 43
        body = {
            key: value
            for key, value in checkpoint["prior_stream"].items()
            if key != "manifest_sha256"
        }
        checkpoint["prior_stream"]["manifest_sha256"] = hashlib.sha256(
            repr({key: body[key] for key in sorted(body)}).encode("utf-8")
        ).hexdigest()
        torch.save(checkpoint, path)
        arm["checkpoint_sha256"] = _sha(path)
        arm["checkpoint_size_bytes"] = path.stat().st_size
    pair.path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prior_stream lineage"):
        load_pair_manifest(pair.path)


def test_500k_receipt_source_must_equal_pair_model_sha(tmp_path: Path) -> None:
    pair = _pair(tmp_path, step=500_000, include_prior_stream=False)
    for receipt_path in pair.legacy_snapshot_receipts.values():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["continuation_source_commit"] = "3" * 40
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal pair model_source_sha"):
        load_pair_manifest(pair.path)


def test_cost_plan_tampering_and_canary_misclassification_fail_closed(
    tmp_path: Path,
) -> None:
    roster = _roster(tmp_path)
    plan, records = _plan(tmp_path, roster)
    fractional_counts = {name: dict(record) for name, record in records.items()}
    fractional_counts["dataset-a"]["num_instances_train"] += 0.75
    fractional_counts["dataset-a"]["num_instances_test"] += 0.75
    with pytest.raises(ValueError, match="must be integral"):
        validate_plan_metadata(plan, roster=roster, records=fractional_counts)
    tabarena_roster = replace(roster, suite_id="tabarena-v0.1")
    assert validate_plan_metadata(
        plan, roster=tabarena_roster, records=fractional_counts
    )["assignment_sha256"] == plan.assignment_sha256
    fractional_total = {name: dict(record) for name, record in records.items()}
    fractional_total["dataset-a"]["num_instances"] += 0.5
    with pytest.raises(ValueError, match="must be integral"):
        validate_plan_metadata(plan, roster=roster, records=fractional_total)

    payload = json.loads(plan.path.read_text(encoding="utf-8"))
    payload["shards"][0]["names"], payload["shards"][1]["names"] = (
        payload["shards"][1]["names"],
        payload["shards"][0]["names"],
    )
    swapped = tmp_path / "swapped-plan.json"
    swapped.write_text(json.dumps(payload), encoding="utf-8")
    swapped_plan = load_shard_plan(swapped, roster=roster)
    with pytest.raises(ValueError, match="canonical cost-balanced"):
        validate_plan_metadata(swapped_plan, roster=roster, records=records)

    payload = json.loads(plan.path.read_text(encoding="utf-8"))
    payload["canary"]["requirements"]["max_rows"] = "dataset-b"
    bad_canary = tmp_path / "bad-canary-plan.json"
    bad_canary.write_text(json.dumps(payload), encoding="utf-8")
    canary_plan = load_shard_plan(bad_canary, roster=roster)
    with pytest.raises(ValueError, match="maximum-row"):
        validate_plan_metadata(canary_plan, roster=roster, records=records)


def test_shards_resume_and_aggregate_only_a_complete_atomic_pair(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, records = _plan(tmp_path, roster)
    evidence = validate_plan_metadata(plan, roster=roster, records=records)
    assert evidence["assignment_sha256"] == plan.assignment_sha256
    run_root = tmp_path / "private-run"
    with pytest.raises(ValueError, match="canary"):
        run_shard(
            run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            shard_index=0,
            executor=_executor,
        )
    canary = run_canary(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=_provenance(),
        executor=_executor,
    )
    assert canary["executed"] == 3
    first = run_shard(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=_provenance(),
        shard_index=0,
        executor=_executor,
    )
    assert first["task_count"] == 1
    assert first["executed"] + first["resumed"] == 1
    for directory in (run_root / "tasks").iterdir():
        assert directory.is_dir()
        for arm in pair.arm_order:
            assert {path.name for path in (directory / arm).iterdir()} == {
                "predictions.npz",
                "result.json",
                "manifest.json",
            }
    with pytest.raises(ValueError, match="incomplete|missing"):
        aggregate_run(run_root, pair=pair, roster=roster, plan=plan)

    def must_not_execute(
        task, staging, fallback
    ):  # pragma: no cover - assertion callback
        raise AssertionError((task, staging, fallback))

    resumed = run_shard(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=_provenance(),
        shard_index=0,
        executor=must_not_execute,
    )
    assert resumed["executed"] == 0
    assert resumed["resumed"] == 1
    for index in range(1, 8):
        run_shard(
            run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            shard_index=index,
            executor=_executor,
        )
    aggregate = aggregate_run(run_root, pair=pair, roster=roster, plan=plan)
    assert aggregate["complete"] is True
    assert aggregate["task_count"] == 8
    assert aggregate["result_count"] == 16
    assert aggregate["comparison"]["left_wins"] == 8
    assert (run_root / "aggregate.json").is_file()
    rendered = (run_root / "aggregate.json").read_text(encoding="utf-8")
    assert str(pair.arms[0].checkpoint_path) not in rendered


def test_one_arm_failure_never_publishes_a_completed_task(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    run_root = tmp_path / "failed-run"

    calls = 0

    def fail_after_left(task, staging, fallback):
        nonlocal calls
        calls += 1
        del task, fallback
        _write_prediction(staging / "rope" / "predictions.npz")
        raise RuntimeError("right arm failed")

    with pytest.raises(RuntimeError, match="right arm failed"):
        run_canary(
            run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            executor=fail_after_left,
        )
    assert list((run_root / "tasks").iterdir()) == []
    assert list((run_root / ".staging").iterdir()) == []
    assert list((run_root / "shards").iterdir()) == []
    assert calls == 1


def test_right_arm_oom_retries_the_whole_pair_in_fresh_staging(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    run_root = tmp_path / "oom-retry"
    attempts: list[tuple[int, Path]] = []

    def oom_once(task, staging, fallback):
        attempts.append((fallback["level"], staging))
        if task.name == "dataset-a" and fallback["level"] == 1:
            _write_prediction(staging / "rope" / "predictions.npz")
            raise PairTaskOOM("right arm CUDA OOM")
        assert not any((staging / arm / "predictions.npz").exists() for arm in pair.arm_order)
        return _executor(task, staging, fallback)

    run_canary(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=_provenance(),
        executor=oom_once,
    )
    first_task = run_root / "tasks" / "0000-dataset-a" / "task.json"
    payload = json.loads(first_task.read_text(encoding="utf-8"))
    assert payload["fallback"] == {
        "level": 2,
        "name": "batch4-cpu",
        "batch_size": 4,
        "offload_mode": "cpu",
    }
    dataset_a_staging = [path for level, path in attempts if level in (1, 2)][:2]
    assert len(set(dataset_a_staging)) == 2
    assert list((run_root / ".staging").iterdir()) == []


def test_final_oom_level_leaves_no_success_arm_or_partial_task(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    run_root = tmp_path / "oom-final"
    levels: list[int] = []

    def always_oom(task, staging, fallback):
        del task
        levels.append(fallback["level"])
        _write_prediction(staging / "rope" / "predictions.npz")
        raise PairTaskOOM("right arm CUDA OOM")

    with pytest.raises(PairTaskOOM, match="right arm CUDA OOM"):
        run_canary(
            run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            executor=always_oom,
        )
    assert levels == [1, 2, 3]
    assert list((run_root / "tasks").iterdir()) == []
    assert list((run_root / ".staging").iterdir()) == []
    assert list((run_root / "shards").iterdir()) == []


@pytest.mark.parametrize("control", ["tasks", "shards", ".staging"])
def test_run_control_directories_reject_live_and_broken_symlinks(
    tmp_path: Path, control: str
) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    for broken in (False, True):
        run_root = tmp_path / f"run-{control.replace('.', 'dot')}-{broken}"
        run_root.mkdir()
        target = tmp_path / f"outside-{control.replace('.', 'dot')}-{broken}"
        if not broken:
            target.mkdir()
            (target / "sentinel").write_text("keep", encoding="utf-8")
        (run_root / control).symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            run_canary(
                run_root,
                pair=pair,
                roster=roster,
                plan=plan,
                provenance=_provenance(),
                executor=_executor,
            )
        if not broken:
            assert (target / "sentinel").read_text(encoding="utf-8") == "keep"


def test_run_root_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    outside = tmp_path / "outside-parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_canary(
            linked_parent / "run",
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            executor=_executor,
        )
    assert list(outside.iterdir()) == []


def test_aggregate_rejects_tampered_prediction_even_after_completion(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path)
    roster = _roster(tmp_path)
    plan, _ = _plan(tmp_path, roster)
    run_root = tmp_path / "tampered-run"
    run_canary(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=_provenance(),
        executor=_executor,
    )
    for index in range(8):
        run_shard(
            run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            provenance=_provenance(),
            shard_index=index,
            executor=_executor,
        )
    aggregate_run(run_root, pair=pair, roster=roster, plan=plan)
    prediction = next((run_root / "tasks").glob("*/rope/predictions.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="manifest|attest|prediction"):
        aggregate_run(run_root, pair=pair, roster=roster, plan=plan)


def test_prediction_capture_uses_formal_row_target_evidence(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("pair_shard_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probabilities = pd.DataFrame(
        [[0.3, 0.7], [0.8, 0.2]], index=[13, 29], columns=["no", "yes"]
    )
    output = tmp_path / "predictions.npz"
    module._capture_pair_prediction(
        probabilities,
        train_features=pd.DataFrame({"x": [1, 2]}, index=[1, 2]),
        train_targets=pd.Series(["no", "yes"], index=[1, 2]),
        test_features=pd.DataFrame({"x": [3, 4]}, index=probabilities.index),
        test_targets=pd.Series(["yes", "no"], index=probabilities.index),
        output=output,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["probabilities"].dtype == np.float32
        assert archive["probabilities"].shape == (2, 2)
        assert archive["encoded_targets"].tolist() == [1, 0]
        assert archive["row_fingerprints"].shape == (2, 32)
        assert archive["test_target_fingerprints"].shape == (2, 32)
        assert archive["train_content_sha256"].shape == (32,)
        assert archive["test_content_sha256"].shape == (32,)


def test_disk_fallback_uses_and_cleans_only_job_local_scratch(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("pair_shard_runner_disk", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fallback = {
        "level": 3,
        "name": "batch4-disk",
        "batch_size": 4,
        "offload_mode": "disk",
    }
    with module._disk_offload_attempt(
        scratch, task_index=7, fallback=fallback
    ) as attempt:
        assert attempt is not None
        assert attempt.parent == scratch.resolve()
        assert attempt.name.startswith("disk-offload-0007-")
        (attempt / "cache.bin").write_bytes(b"cache")
    assert list(scratch.iterdir()) == []

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises((RuntimeError, ValueError), match="unsafe|symlink"):
        with module._disk_offload_attempt(
            scratch, task_index=8, fallback=fallback
        ) as attempt:
            assert attempt is not None
            attempt.rmdir()
            attempt.symlink_to(outside, target_is_directory=True)
    assert outside.is_dir()


def test_paired_bootstrap_is_finite_reproducible_and_directional() -> None:
    one = [
        {
            "results": {
                "rope": {"metric_error": 0.2},
                "none": {"metric_error": 0.3},
            }
        }
    ]
    first = _comparison(one, ("rope", "none"))
    second = _comparison(one, ("rope", "none"))
    assert first == second
    raw = first["raw_metric_error_difference"]
    assert raw["direction"] == (
        "left_metric_error_minus_right_metric_error"
    )
    assert raw["negative_means_left_is_better"] is True
    assert raw["mean"] == pytest.approx(-0.1)
    assert raw["paired_bootstrap_95ci"] == pytest.approx([-0.1, -0.1])
    assert raw["paired_bootstrap_resamples"] >= 10_000
    assert all(np.isfinite(value) for value in raw["paired_bootstrap_95ci"])

    ties = [
        {
            "results": {
                "rope": {"metric_error": 0.4},
                "none": {"metric_error": 0.4},
            }
        }
        for _ in range(3)
    ]
    tied = _comparison(ties, ("rope", "none"))
    assert tied["raw_metric_error_difference"]["paired_bootstrap_95ci"] == [0.0, 0.0]
    assert tied["two_sided_exact_sign_test_p"] == 1.0
    assert tied["ties"] == 3


def test_runner_aggregator_and_slurm_wrapper_are_syntax_checked_and_bounded() -> None:
    aggregate_wrapper = PACKAGE_ROOT / "scripts" / "slurm_pair_full_suite_aggregate.sh"
    for path in (WRAPPER, CANARY_WRAPPER, A10_WRAPPER, aggregate_wrapper):
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    wrapper = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=h100",
        "#SBATCH --qos=medium",
        "#SBATCH --array=0-7%2",
        "#SBATCH --gres=gpu:1",
    ):
        assert directive in wrapper
    assert "*H100*" in wrapper
    assert "#SBATCH --mem=384G" in wrapper
    canary = CANARY_WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --qos=medium" in canary
    assert "#SBATCH --time=24:00:00" in canary
    assert "#SBATCH --mem=384G" in canary
    assert "PE_PAIR_EXPECTED_ANALYSIS_SHA PE_PAIR_EXPECTED_TABARENA_SHA" in canary
    assert "required_paths" in canary
    a10 = A10_WRAPPER.read_text(encoding="utf-8")
    assert '"NVIDIA A10"' in a10
    for source in (wrapper, canary, a10):
        assert "reject_symlink_components" in source
        assert "reject_parent_symlink_components" in source
        assert "verify_python_environment.py" in source
        assert '[[ -x "$PE_PAIR_PYTHON" && -L "$PE_PAIR_PYTHON" ]]' in source
        assert 'exec "$PE_PAIR_PYTHON" -I -B "$python_verifier"' in source
        assert '--gpu-monitor "$monitor_csv"' in source
        assert "safe_gpu_monitor" not in source
        assert "--exec-bound" not in source
        assert "monitor_python=" not in source
        assert "ensure_real_directory \"$monitor_root\"" in source
        assert 'ensure_real_directory "$runtime_root/home/tmp" runtime-tmp' in source
        assert 'export TMPDIR="$HOME/tmp"' in source
        assert 'export TMP="$TMPDIR"' in source
        assert 'export TEMP="$TMPDIR"' in source
        assert "printf 'STOP\\n'" in source
        assert '[[ "$monitor_complete" == COMPLETE ]]' in source
        assert 'IFS= read -r -t 45 monitor_complete' in source
        assert 'wait "$monitor_pid"' in source
        assert "kill -0" not in source
        assert 'kill "$monitor_pid"' not in source
        assert "local original_status=$?" in source
        assert "trap - EXIT INT TERM" in source
        assert 'exit "$original_status"' in source
        assert "trap cleanup_monitor EXIT\n" in source
        assert "trap 'cleanup_monitor_signal 2' INT" in source
        assert "trap 'cleanup_monitor_signal 15' TERM" in source
        assert 'exit "$((128 + signal_number))"' in source
        assert "trap cleanup_monitor EXIT INT TERM" not in source
        assert "monitor_pid=''" in source
        cleanup_start = source.index("cleanup_monitor_resources() {")
        cleanup_end = source.index("cleanup_monitor() {", cleanup_start)
        cleanup = source[cleanup_start:cleanup_end]
        assert cleanup.index("close_monitor_input_fd") < cleanup.index(
            'wait "$monitor_pid"'
        )
        assert cleanup.index('wait "$monitor_pid"') < cleanup.index(
            "monitor_pid=''"
        )
        assert cleanup.index("monitor_pid=''") < cleanup.index(
            "close_monitor_ready_fd"
        )
    monitor_source = PYTHON_VERIFIER.read_text(encoding="utf-8")
    assert "os.O_NOFOLLOW" in monitor_source
    assert "os.O_EXCL" in monitor_source
    assert "select.select" in monitor_source
    assert 'NVIDIA_SMI = "/usr/bin/nvidia-smi"' in monitor_source
    assert 'control != "STOP\\n"' in monitor_source
    assert 'print("COMPLETE", flush=True)' in monitor_source
    aggregate_source = aggregate_wrapper.read_text(encoding="utf-8")
    assert "verify_python_environment.py" in aggregate_source
    assert '[[ -x "$PE_PAIR_PYTHON" && -L "$PE_PAIR_PYTHON" ]]' in aggregate_source
    runner = RUNNER.read_text(encoding="utf-8")
    assert "dataset_names=[task.name]" in runner
    assert "model_artifacts_base_path=tempfile.gettempdir()" in runner
    assert runner.index("self.arena.build_jobs(") < runner.index("self.arena.run_jobs(")
    assert AGGREGATOR.is_file()


def test_wrapper_finalizer_rejects_monitor_that_exits_after_ready() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("finalize_monitor() {")
    end = source.index("\n}\n", start) + len("\n}\n")
    finalizer = source[start:end]
    harness = f"""
set -euo pipefail
{finalizer}
coproc PAIR_GPU_MONITOR {{
  printf 'READY\\n'
  exit 23
}}
monitor_pid=$PAIR_GPU_MONITOR_PID
monitor_ready_fd=${{PAIR_GPU_MONITOR[0]}}
monitor_input_fd=${{PAIR_GPU_MONITOR[1]}}
monitor_finalized=0
IFS= read -r monitor_ready <&"$monitor_ready_fd"
[[ "$monitor_ready" == READY ]]
sleep 0.1
if finalize_monitor; then
  exit 90
fi
if wait "$monitor_pid"; then
  exit 91
else
  monitor_status=$?
fi
[[ "$monitor_status" == 23 ]]
"""
    completed = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("interrupt", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_wrapper_monitor_cleanup_preserves_nonzero_signal_status(
    interrupt: signal.Signals, expected_status: int
) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("close_monitor_input_fd() {")
    end = source.index("finalize_monitor() {", start)
    cleanup_functions = source[start:end]
    harness = f"""
set -euo pipefail
{cleanup_functions}
kill() {{
  exit 99
}}
coproc PAIR_GPU_MONITOR {{
  exit 23
}}
monitor_pid=$PAIR_GPU_MONITOR_PID
monitor_ready_fd=${{PAIR_GPU_MONITOR[0]}}
monitor_input_fd=${{PAIR_GPU_MONITOR[1]}}
monitor_finalized=0
trap cleanup_monitor EXIT
trap 'cleanup_monitor_signal 2' INT
trap 'cleanup_monitor_signal 15' TERM
sleep 0.1
printf 'READY\\n'
while :; do sleep 1; done
"""
    process = subprocess.Popen(
        ["bash", "-c", harness],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == "READY\n"
        process.send_signal(interrupt)
        assert process.wait(timeout=5) == expected_status
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_wrapper_monitor_cleanup_waits_cached_child_and_preserves_exit_status() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("close_monitor_input_fd() {")
    end = source.index("finalize_monitor() {", start)
    cleanup_functions = source[start:end]
    harness = f"""
set -euo pipefail
{cleanup_functions}
kill() {{
  exit 99
}}
coproc PAIR_GPU_MONITOR {{
  exit 23
}}
monitor_pid=$PAIR_GPU_MONITOR_PID
monitor_ready_fd=${{PAIR_GPU_MONITOR[0]}}
monitor_input_fd=${{PAIR_GPU_MONITOR[1]}}
monitor_finalized=0
trap cleanup_monitor EXIT
sleep 0.1
exit 37
"""
    completed = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 37, completed.stderr


def test_submitter_holds_canary_and_git_preflight_requires_clean_detached(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("pair_submitter", SUBMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._FULL_ARRAY_BY_MODE == {
        "beyond-h100": "0-7%2",
        "tabarena-a10": "0-7%1",
    }
    wrapper = tmp_path / "worker.sh"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    command = module._sbatch_command(
        wrapper=wrapper,
        exports={"SAFE": "value"},
        output=tmp_path / "job.out",
        qos="medium",
        walltime="24:00:00",
        hold=True,
    )
    assert "--hold" in command

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "unit@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Unit"], cwd=checkout, check=True)
    (checkout / "tracked").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "unit"], cwd=checkout, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="detached"):
        module._git_checkout(checkout, expected=head, label="unit")
    subprocess.run(["git", "checkout", "-q", "--detach", head], cwd=checkout, check=True)
    assert module._git_checkout(checkout, expected=head, label="unit") == head
    (checkout / "untracked").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        module._git_checkout(checkout, expected=head, label="unit")

    help_result = subprocess.run(
        [sys.executable, str(SUBMITTER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--capacity-root" in help_result.stdout
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert submitter.index("_publish_receipt(receipt, plan_summary)") < submitter.index(
        '["scontrol", "release", canary_id]'
    )
    assert "release_failed_jobs_terminal_verified" in submitter
    assert "disk_checks" in submitter
    for name in (
        "PE_PAIR_EXPECTED_MANIFEST_SHA256",
        "PE_PAIR_EXPECTED_ROSTER_SHA256",
        "PE_PAIR_EXPECTED_SHARD_PLAN_SHA256",
        "PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256",
    ):
        assert name in submitter
    assert 'required_distributions=("autogluon.tabular",)' in submitter
    assert "python = Path(args.python)" in submitter
    assert "python_environment_contract_document_sha256" in submitter


def test_submitter_rollback_reports_only_scheduler_verified_terminal_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("pair_submitter_rollback", SUBMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def terminal_run(command, **kwargs):
        del kwargs
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(command, 0, "123|CANCELLED|\n", "")

    monkeypatch.setattr(module.subprocess, "run", terminal_run)
    verified = module._cancel_and_verify(["123"], attempts=1)
    assert verified["all_jobs_terminal"] is True
    assert verified["terminal_job_ids"] == ["123"]

    def unknown_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 1, "", "not verified")

    monkeypatch.setattr(module.subprocess, "run", unknown_run)
    unknown = module._cancel_and_verify(["124"], attempts=1)
    assert unknown["all_jobs_terminal"] is False
    assert unknown["unverified_job_ids"] == ["124"]
