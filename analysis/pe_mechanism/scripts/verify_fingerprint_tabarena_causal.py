#!/usr/bin/env python3
"""Verify a published 38-task fingerprint causal run and write a private receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from pe_mechanism.fingerprint_causal import _write_private_json  # noqa: E402
from pe_mechanism.fingerprint_causal_verification import (  # noqa: E402
    EXPECTED_ROSTER_FILE_SHA256,
    verify_fingerprint_causal_run,
)
from pe_mechanism.provenance import verify_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fingerprint-checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-verifier-sha", required=True)
    return parser


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _receipt_path(value: str, *, run_dir: Path) -> Path:
    receipt = Path(value)
    if not receipt.is_absolute() or receipt.exists() or receipt.is_symlink():
        raise ValueError("receipt must be a fresh absolute path")
    receipt_parent = receipt.parent.resolve(strict=True)
    run = run_dir.resolve(strict=True)
    repository = REPO_ROOT.resolve(strict=True)
    if _paths_overlap(receipt_parent, run) or _paths_overlap(
        receipt_parent, repository
    ):
        raise ValueError(
            "receipt parent must not overlap the causal run or verifier repository"
        )
    return receipt_parent / receipt.name


def _git_head(root: Path, expected: str) -> str:
    if (
        len(expected) != 40
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("expected verifier SHA must be 40 lowercase hexadecimal digits")
    repository = root.resolve(strict=True)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
    }
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    if status.stdout:
        raise ValueError("verifier repository must be clean")
    symbolic = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if symbolic.returncode == 0:
        raise ValueError("verifier repository must use a detached HEAD")
    if symbolic.returncode != 1:
        symbolic.check_returncode()
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    if head != expected:
        raise ValueError(f"verifier HEAD mismatch: expected {expected}, observed {head}")
    return head


def main() -> int:
    args = _parser().parse_args()
    scratch_value = os.environ.get("TMPDIR")
    if not scratch_value:
        raise ValueError("TMPDIR must name the verification scratch root")
    run_dir = Path(args.run_dir)
    receipt = _receipt_path(args.receipt, run_dir=run_dir)
    verifier_sha = _git_head(REPO_ROOT, args.expected_verifier_sha)
    verifier_files = {
        "script_sha256": verify_file(Path(__file__).resolve()),
        "verification_module_sha256": verify_file(
            PACKAGE_ROOT / "src/pe_mechanism/fingerprint_causal_verification.py"
        ),
        "capture_module_sha256": verify_file(
            PACKAGE_ROOT / "src/pe_mechanism/fingerprint_causal.py"
        ),
    }
    report = verify_fingerprint_causal_run(
        run_dir,
        checkpoint=Path(args.fingerprint_checkpoint),
        roster_path=PACKAGE_ROOT / "manifests/tabarena-v0.1-classification-roster.json",
        scratch_root=Path(scratch_value),
        expected_roster_sha256=EXPECTED_ROSTER_FILE_SHA256,
    )
    for file in verifier_files.values():
        file.assert_unchanged()
    _git_head(REPO_ROOT, verifier_sha)
    report["verifier_provenance"] = {
        "git_sha": verifier_sha,
        **{name: file.digest.sha256 for name, file in verifier_files.items()},
    }
    report["verifier_runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    _write_private_json(receipt, report)
    verified_receipt = verify_file(receipt)
    if receipt.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError("verification receipt is not private")
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "receipt_sha256": verified_receipt.digest.sha256,
                "status": report["status"],
                "task_count": report["task_count"],
                "result_count": report["result_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
