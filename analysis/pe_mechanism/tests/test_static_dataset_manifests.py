from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from pe_mechanism.datasets import build_dataset_manifest, canonical_dataset_name

_MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"
_TABARENA_FILE = "tabarena-v0.1-classification-roster.json"
_TALENT_FILE = "talent-classification-split-v1.json"
_TABARENA_COMMIT = "c987d91556a14d4c9b3383c35d1b0ec68ff81883"
_TABARENA_ROSTER_SHA256 = (
    "b1f71e48085451cbe2126cbc02799c3969ecf9e7965f6aa71757db63fd6003ee"
)
_TALENT_ASSIGNMENT_SHA256 = (
    "e18e9291f9808e764e6f36874f397aa6ff64605d4997f7b0b7b4c8b9d11cf147"
)


def _load(name: str) -> dict[str, Any]:
    payload = json.loads((_MANIFESTS / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sorted_name_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


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


def test_tabarena_v01_classification_roster_is_pinned_sorted_and_hashed() -> None:
    manifest = _load(_TABARENA_FILE)
    names = manifest["names"]

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "dataset_roster"
    assert manifest["suite"] == "TabArena"
    assert manifest["version"] == "v0.1"
    assert manifest["task_type"] == "classification"
    assert manifest["source_commit"] == _TABARENA_COMMIT
    assert manifest["count"] == len(names) == 38
    assert names == sorted(names)
    assert len(set(names)) == len(names)
    assert manifest["roster_hash_algorithm"] == "sha256_newline_join_sorted_names"
    assert manifest["roster_sha256"] == _TABARENA_ROSTER_SHA256
    assert _sorted_name_digest(names) == _TABARENA_ROSTER_SHA256

    source = urlsplit(manifest["source_url"])
    assert source.scheme == "https"
    assert source.netloc == "github.com"
    assert source.path == "/autogluon/tabarena"
    assert source.username is None
    assert source.password is None


def test_talent_assignment_matches_the_frozen_selection_and_split_contract() -> None:
    tabarena = _load(_TABARENA_FILE)
    manifest = _load(_TALENT_FILE)
    assignments = manifest["assignments"]
    assigned_names = [item["name"] for item in assignments]
    excluded = manifest["excluded_tabarena_overlap"]

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "dataset_level_split"
    assert manifest["suite"] == "TALENT"
    assert manifest["task_type"] == "classification"
    assert manifest["eligibility"] == {
        "max_features": 500,
        "max_classes": 100,
        "required_splits": ["train", "val", "test"],
        "require_nonempty_splits": True,
        "require_complete_and_aligned_feature_splits": True,
        "require_safe_numeric_storage": True,
        "require_at_least_one_feature": True,
        "require_at_least_two_training_classes": True,
        "require_validation_and_test_labels_seen_in_training": True,
    }
    assert manifest["overlap_exclusion"] == {
        "suite": "TabArena",
        "version": "v0.1",
        "canonicalization": "lowercase_ascii_alphanumeric_v1",
        "roster_sha256": _TABARENA_ROSTER_SHA256,
    }
    assert manifest["scan_counts"] == {
        "discovered": 300,
        "eligible_before_overlap": 179,
        "excluded_tabarena_overlap": 16,
        "assigned": 163,
    }
    assert manifest["ineligibility_reason_counts"] == {
        "too_many_classes": 69,
        "too_many_features": 2,
        "unknown_test_labels": 77,
        "unknown_val_labels": 75,
        "unsafe_object_numeric_features": 2,
        "unsupported_task_type": 119,
    }
    assert manifest["seed"] == 20260807
    assert manifest["fractions"] == {
        "discovery": 0.70,
        "validation": 0.15,
        "held_out": 0.15,
    }
    assert manifest["counts"] == {
        "discovery": 114,
        "validation": 25,
        "held_out": 24,
    }
    assert Counter(item["split"] for item in assignments) == manifest["counts"]
    assert len(assignments) == len(assigned_names) == 163
    assert len(set(assigned_names)) == len(assigned_names)
    assert [item["rank"] for item in assignments] == list(range(163))
    assert all(
        item["canonical_name"] == canonical_dataset_name(item["name"])
        for item in assignments
    )
    assert excluded == sorted(excluded)
    assert len(excluded) == 16

    assert (
        manifest["assignment_hash_algorithm"]
        == "sha256_newline_join_sorted_assigned_names"
    )
    assert manifest["assignment_sha256"] == _TALENT_ASSIGNMENT_SHA256
    assert _sorted_name_digest(assigned_names) == _TALENT_ASSIGNMENT_SHA256

    rebuilt = build_dataset_manifest(
        [*assigned_names, *excluded],
        tabarena_names=tabarena["names"],
        seed=manifest["seed"],
    )
    assert rebuilt["assignments"] == assignments
    assert rebuilt["excluded_tabarena_overlap"] == excluded
    assert rebuilt["counts"] == manifest["counts"]
    assert rebuilt["fractions"] == manifest["fractions"]
    assert rebuilt["eligible_roster_sha256"] == manifest["assignment_sha256"]

    tabarena_canonical = {
        canonical_dataset_name(name) for name in tabarena["names"]
    }
    assert {
        canonical_dataset_name(name) for name in assigned_names
    }.isdisjoint(tabarena_canonical)


@pytest.mark.parametrize("name", [_TABARENA_FILE, _TALENT_FILE])
def test_static_dataset_manifests_do_not_contain_cluster_or_user_state(name: str) -> None:
    values = tuple(_strings(_load(name)))
    forbidden = (
        "/mnt/",
        "/" + "home" + "/",
        "/" + "users" + "/",
        "file://",
        "slu" + "rm",
        "noe" + "ther",
        "jia" + "xio",
    )
    for value in values:
        lowered = value.casefold()
        assert not any(token in lowered for token in forbidden)
