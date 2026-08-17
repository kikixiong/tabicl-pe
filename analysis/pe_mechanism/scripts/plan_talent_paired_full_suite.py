#!/usr/bin/env python3
"""Create a content-bound, balanced plan for the frozen TALENT discovery set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pe_mechanism.official_tabicl import load_raw_talent_splits
from pe_mechanism.talent import assess_talent_eligibility
from pe_mechanism.talent_full_suite import (
    PLAN_KIND,
    absolute_path,
    atomic_json,
    build_shard_plan_payload,
    frozen_discovery_roster,
    require_disjoint_output,
    self_hashed_document,
    verify_clean_detached_git,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--talent-root", required=True)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def _class_count(values: np.ndarray) -> int:
    tokens = {
        f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
        for value in np.asarray(values).reshape(-1)
    }
    return len(tokens)


def build_dataset_records(
    *, analysis_root: Path, talent_root: Path
) -> list[dict[str, object]]:
    roster = frozen_discovery_roster(analysis_root)
    records: list[dict[str, object]] = []
    for ordinal, name in enumerate(roster):
        dataset_root = talent_root / name
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise ValueError(f"TALENT dataset directory is unsafe: {name}")
        eligibility = assess_talent_eligibility(
            dataset_root, max_features=500, max_classes=100
        )
        if not eligibility.eligible or not 2 <= eligibility.n_classes <= 10:
            raise ValueError(f"TALENT discovery dataset is no longer eligible: {name}")
        dataset = load_raw_talent_splits(dataset_root, trusted_pickle=True)
        n_classes = _class_count(dataset.train.y)
        if n_classes != eligibility.n_classes:
            raise RuntimeError(f"class count changed while reading {name}")
        records.append(
            {
                "ordinal": ordinal,
                "name": name,
                "n_train": int(len(dataset.train.y)),
                "n_validation": int(len(dataset.val.y)),
                "n_test": int(len(dataset.test.y)),
                "n_features": int(
                    dataset.n_numeric_features + dataset.n_categorical_features
                ),
                "n_classes": n_classes,
                "info_sha256": dataset.info_sha256,
                "input_sha256": dict(sorted(dataset.input_sha256.items())),
            }
        )
    return records


def main() -> int:
    args = _parser().parse_args()
    analysis_root = absolute_path(
        args.analysis_root, name="analysis_root", directory=True
    )
    talent_root = absolute_path(args.talent_root, name="talent_root", directory=True)
    output = absolute_path(args.output, name="output", absent=True)
    require_disjoint_output(
        output, [analysis_root, talent_root], name="TALENT shard plan output"
    )
    verify_clean_detached_git(analysis_root, expected_sha=args.expected_analysis_sha)
    records = build_dataset_records(
        analysis_root=analysis_root, talent_root=talent_root
    )
    payload = build_shard_plan_payload(records, shard_count=args.shards)
    payload["analysis_sha"] = args.expected_analysis_sha
    document = self_hashed_document(PLAN_KIND, payload)
    atomic_json(output, document)
    print(document["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
