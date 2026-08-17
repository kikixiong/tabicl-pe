#!/usr/bin/env python3
"""Fail-closed aggregation for a completed two-arm full-suite run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pe_mechanism.full_suite_pair import (
    aggregate_run,
    load_pair_manifest,
    load_roster,
    load_shard_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--run-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_absolute() or run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("run_root must be an existing absolute real directory")
    pair = load_pair_manifest(Path(args.pair_manifest), verify_checkpoints=True)
    roster = load_roster(Path(args.roster))
    plan = load_shard_plan(Path(args.shard_plan), roster=roster)
    aggregate = aggregate_run(run_root, pair=pair, roster=roster, plan=plan)
    print(
        json.dumps(
            {
                "aggregate": str(run_root / "aggregate.json"),
                "task_count": aggregate["task_count"],
                "result_count": aggregate["result_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
