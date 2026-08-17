#!/usr/bin/env python3
"""Write a fresh canonical receipt for the targeted pilot continuation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_ID = re.compile(r"^[1-9][0-9]*$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--continuation-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--rope-stage1-job", required=True)
    parser.add_argument("--rope-stage2-job", required=True)
    parser.add_argument("--none-stage2-job", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if SAFE_ID.fullmatch(args.continuation_id) is None:
            raise ValueError("unsafe continuation ID")
        if SHA40.fullmatch(args.source_commit) is None:
            raise ValueError("source commit must be a full Git SHA")
        jobs = {
            "rope-stage1-479k-to-500k": args.rope_stage1_job,
            "rope-stage2-after-500k": args.rope_stage2_job,
            "none-stage2-6200-to-40k": args.none_stage2_job,
        }
        if any(JOB_ID.fullmatch(job_id) is None for job_id in jobs.values()):
            raise ValueError("receipt contains an invalid Slurm job ID")
        if len(set(jobs.values())) != len(jobs):
            raise ValueError("receipt job IDs must be unique")

        output = args.output.absolute()
        if output.exists() or output.is_symlink():
            raise ValueError("submission receipt is fresh-only")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        body = {
            "schema_version": 1,
            "kind": "tabicl-legacy-pilot-continuation-receipt",
            "classification": "exploratory-pilot-only",
            "continuation_id": args.continuation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "repository_root": str(args.repository_root.absolute()),
            "jobs": jobs,
            "dependencies": {
                "rope-stage2-after-500k": {
                    "type": "afterok",
                    "job_id": args.rope_stage1_job,
                },
                "none-stage2-6200-to-40k": None,
            },
            "checkpoint_contract": {
                "rope_stage1_start_step": 479000,
                "rope_stage1_terminal_step": 500000,
                "none_stage2_start_step": 6200,
                "none_stage2_terminal_step": 40000,
                "rope_stage2_terminal_step": 40000,
            },
        }
        envelope = dict(body)
        envelope["receipt_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
        raw = canonical(envelope) + b"\n"

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(output, flags, 0o400)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print(str(output))
    except (OSError, ValueError) as error:
        print(f"pilot submission receipt failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
