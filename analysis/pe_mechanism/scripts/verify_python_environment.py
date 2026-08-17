#!/usr/bin/env python3
"""Fail closed unless this process matches a frozen venv entry contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from pe_mechanism.python_environment import (  # noqa: E402
    verify_python_environment_contract_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-document-sha256", required=True)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="optional Python arguments to run with the currently bound process image",
    )
    args = parser.parse_args()
    verification = {
        "entry": args.entry,
        "contract_path": args.contract,
        "expected_file_sha256": args.expected_file_sha256,
        "expected_document_sha256": args.expected_document_sha256,
    }
    verify_python_environment_contract_file(**verification)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        return 0
    completed = subprocess.run(
        [str(args.entry), *command],
        executable="/proc/self/exe",
        check=False,
    )
    # This same, still-bound verifier process performs the postcondition check.
    # A failure here overrides a successful workload so Slurm dependencies stay held.
    verify_python_environment_contract_file(**verification)
    if completed.returncode < 0:
        return 128 - completed.returncode
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
