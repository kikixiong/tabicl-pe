"""Deterministic dataset-level splitting without embedding private file paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from .identifiers import require_public_label


_NAME_TOKEN = re.compile(r"[^a-z0-9]+")


def canonical_dataset_name(name: str) -> str:
    """Normalize a dataset label for overlap checks, not for display."""
    value = _NAME_TOKEN.sub("", name.strip().lower())
    if not value:
        raise ValueError("dataset names must contain an alphanumeric character")
    return value


def _roster_digest(names: Sequence[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_counts(total: int, fractions: Sequence[float]) -> list[int]:
    if total < len(fractions):
        raise ValueError("dataset count must be at least the number of splits")
    raw = [total * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(fractions)), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    # Largest-remainder rounding can assign zero datasets to a small positive
    # split (notably 70/15/15 with n=3).  Preserve the requested approximation
    # while guaranteeing that discovery, validation, and held-out are real.
    for receiver, count in enumerate(tuple(counts)):
        if count != 0:
            continue
        donors = [index for index, donor_count in enumerate(counts) if donor_count > 1]
        if not donors:  # Defensive: total >= number of positive splits above.
            raise RuntimeError("unable to allocate at least one dataset to each split")
        donor = max(donors, key=lambda index: (counts[index], fractions[index], -index))
        counts[donor] -= 1
        counts[receiver] = 1
    return counts


@dataclass(frozen=True)
class DatasetAssignment:
    name: str
    split: str
    rank: int
    canonical_name: str


def build_dataset_manifest(
    talent_names: Iterable[str],
    *,
    tabarena_names: Iterable[str] = (),
    seed: int = 20260807,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
) -> dict[str, object]:
    """Build a stable TALENT discovery/validation/held-out split.

    Paths are intentionally not accepted.  A private runtime source map may map
    the returned display names to local files without entering public
    provenance.
    """
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fractions must sum to one")

    display_by_canonical: dict[str, str] = {}
    for raw_name in talent_names:
        name = str(raw_name).strip()
        require_public_label(name, name="TALENT dataset name")
        canonical = canonical_dataset_name(name)
        previous = display_by_canonical.setdefault(canonical, name)
        if previous != name:
            raise ValueError(
                f"ambiguous TALENT names normalize to {canonical!r}: {previous!r}, {name!r}"
            )

    excluded_canonical = set()
    for raw_name in tabarena_names:
        display = str(raw_name).strip()
        require_public_label(display, name="TabArena dataset name")
        excluded_canonical.add(canonical_dataset_name(display))
    excluded = sorted(
        display_by_canonical[canonical]
        for canonical in display_by_canonical.keys() & excluded_canonical
    )
    eligible = [
        (canonical, display)
        for canonical, display in display_by_canonical.items()
        if canonical not in excluded_canonical
    ]
    if len(eligible) < 3:
        raise ValueError("at least three non-overlapping datasets are required")

    def rank_key(item: tuple[str, str]) -> tuple[str, str]:
        canonical, _ = item
        token = hashlib.sha256(f"{seed}\0{canonical}".encode("utf-8")).hexdigest()
        return token, canonical

    ranked = sorted(eligible, key=rank_key)
    counts = _partition_counts(len(ranked), fractions)
    split_names = ("discovery", "validation", "held_out")
    boundaries = (counts[0], counts[0] + counts[1])
    assignments: list[DatasetAssignment] = []
    for rank, (canonical, display) in enumerate(ranked):
        split_index = 0 if rank < boundaries[0] else 1 if rank < boundaries[1] else 2
        assignments.append(
            DatasetAssignment(
                name=display,
                split=split_names[split_index],
                rank=rank,
                canonical_name=canonical,
            )
        )

    public_roster = sorted(item.name for item in assignments)
    return {
        "schema_version": 1,
        "kind": "dataset_level_split",
        "seed": seed,
        "fractions": dict(zip(split_names, fractions, strict=True)),
        "counts": dict(zip(split_names, counts, strict=True)),
        "eligible_roster_sha256": _roster_digest(public_roster),
        "excluded_tabarena_overlap": excluded,
        "assignments": [asdict(item) for item in assignments],
    }


def validate_private_source_map(
    manifest: Mapping[str, object], source_map: Mapping[str, str | Path]
) -> dict[str, Path]:
    """Resolve a private name-to-path map and reject missing or extra entries."""
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("manifest assignments must be a list")
    expected = {str(item["name"]) for item in raw_assignments if isinstance(item, dict)}
    actual = set(source_map)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"source map mismatch; missing={missing}, extra={extra}")
    resolved = {name: Path(value).expanduser().resolve(strict=True) for name, value in source_map.items()}
    if any(not path.is_dir() for path in resolved.values()):
        raise ValueError("each dataset source must resolve to a directory")
    return resolved


def load_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
