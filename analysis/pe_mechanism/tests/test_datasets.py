from __future__ import annotations

from pathlib import Path

import pytest

from pe_mechanism.datasets import (
    build_dataset_manifest,
    canonical_dataset_name,
    validate_private_source_map,
)


def test_dataset_manifest_is_deterministic_and_excludes_overlap() -> None:
    talent = [f"dataset-{index:03d}" for index in range(20)]
    first = build_dataset_manifest(
        talent, tabarena_names=["Dataset 003", "dataset-019"], seed=7
    )
    second = build_dataset_manifest(
        reversed(talent), tabarena_names=["dataset_019", "dataset-003"], seed=7
    )
    assert first == second
    assert first["counts"] == {"discovery": 12, "validation": 3, "held_out": 3}
    assert first["excluded_tabarena_overlap"] == ["dataset-003", "dataset-019"]
    assigned = {item["name"] for item in first["assignments"]}
    assert assigned.isdisjoint({"dataset-003", "dataset-019"})


def test_dataset_name_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        build_dataset_manifest(["foo-bar", "Foo Bar", "baz", "qux"])


def test_private_source_map_requires_exact_existing_directories(tmp_path: Path) -> None:
    manifest = build_dataset_manifest(["a", "b", "c"])
    sources = {}
    for name in ("a", "b", "c"):
        path = tmp_path / name
        path.mkdir()
        sources[name] = path
    resolved = validate_private_source_map(manifest, sources)
    assert set(resolved) == set(sources)
    with pytest.raises(ValueError, match="source map mismatch"):
        validate_private_source_map(manifest, {"a": sources["a"]})


def test_empty_canonical_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_dataset_name("---")


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (3, {"discovery": 1, "validation": 1, "held_out": 1}),
        (4, {"discovery": 2, "validation": 1, "held_out": 1}),
    ],
)
def test_small_rosters_keep_every_dataset_level_split_nonempty(
    count: int, expected: dict[str, int]
) -> None:
    manifest = build_dataset_manifest([f"dataset-{index}" for index in range(count)])
    assert manifest["counts"] == expected
    assert {item["split"] for item in manifest["assignments"]} == {
        "discovery",
        "validation",
        "held_out",
    }
