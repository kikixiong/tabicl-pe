#!/usr/bin/env python3
"""Run the fail-closed CPU stratification of a completed TabArena pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pe_mechanism.tabarena_stratification import (
    analyze_completed_run,
    write_new_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--expected-tabarena-sha", required=True)
    parser.add_argument("--openml-cache-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = analyze_completed_run(
        run_root=Path(args.run_root),
        expected_aggregate_sha256=args.expected_aggregate_sha256,
        tabarena_root=Path(args.tabarena_root),
        expected_tabarena_sha=args.expected_tabarena_sha,
        openml_cache_root=Path(args.openml_cache_root),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    write_new_outputs(
        payload,
        json_path=output_json,
        markdown_path=output_markdown,
    )
    print(
        json.dumps(
            {
                "dataset_count": payload["validation"]["verified_task_count"],
                "json": str(output_json),
                "markdown": str(output_markdown),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
