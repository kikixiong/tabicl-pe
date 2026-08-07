#!/usr/bin/env python3
"""Run only the fixed Trainer or checkpoint validator from exact candidate T.

This is intentionally not a generic module/command launcher.  Source archive
verification and all-loaded-``tabicl.*`` attestation happen in this process
before the fixed action begins, under isolated/no-bytecode Python.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace


def _open_directory_nofollow(path: Path, *, label: str = "archive root") -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{label} must be a normalized absolute path")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("no-follow directory traversal is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError(f"{label} contains a symlink or invalid component") from error


def _load_fixed_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load fixed exact-T module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _publish_no_replace(path: Path, raw: bytes, *, max_bytes: int) -> None:
    if max_bytes < 1 or len(raw) > max_bytes:
        raise ValueError("source attestation exceeds configured byte ceiling")
    if path.name in {"", ".", ".."}:
        raise ValueError("publication output must name a file")
    parent_fd = _open_directory_nofollow(path.parent, label="publication parent")
    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            candidate = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_fd = os.open(
                    candidate, flags, 0o600, dir_fd=parent_fd
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise FileExistsError("could not allocate a unique publication temporary")

        view = memoryview(raw)
        while view:
            written = os.write(temporary_fd, view)
            if written < 1:
                raise OSError("short write while publishing exact-T evidence")
            view = view[written:]
        os.fsync(temporary_fd)
        temporary_stat = os.fstat(temporary_fd)
        named_stat = os.stat(
            temporary_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (named_stat.st_dev, named_stat.st_ino) != (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ):
            raise RuntimeError("publication temporary identity changed")

        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (published_stat.st_dev, published_stat.st_ino) != (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ):
            raise RuntimeError("published evidence identity changed")
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        try:
            if temporary_fd is not None:
                os.close(temporary_fd)
        finally:
            try:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(parent_fd)


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    positions = [index for index, value in enumerate(argv) if value == "--"]
    if len(positions) != 1:
        raise ValueError("exact-T action requires exactly one literal -- delimiter")
    split = positions[0]
    action_args = argv[split + 1 :]
    if not action_args:
        raise ValueError("fixed exact-T action arguments must not be empty")
    return argv[:split], action_args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-tree-sha", required=True)
    parser.add_argument(
        "--action",
        required=True,
        choices=("trainer", "checkpoint-validator", "h100-validation"),
    )
    parser.add_argument("--source-attestation-output", required=True, type=Path)
    parser.add_argument("--source-attestation-max-bytes", required=True, type=int)
    parser.add_argument("--completion-receipt-output", type=Path)
    parser.add_argument("--runtime-evidence-output", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--artifact-identity-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("exact-T launcher requires Python -I -B")
    bootstrap_argv, action_argv = _split_argv(
        list(sys.argv[1:] if argv is None else argv)
    )
    args = _parser().parse_args(bootstrap_argv)
    if (args.case_id is None) != (args.artifact_identity_sha256 is None):
        raise ValueError("case ID and artifact identity digest must be supplied together")
    if args.action == "h100-validation" and args.case_id is None:
        raise ValueError("H100 validation requires an independently bound case ID")
    if (args.action == "h100-validation") != (
        args.completion_receipt_output is not None
        and args.runtime_evidence_output is not None
    ):
        raise ValueError(
            "only H100 validation requires completion and runtime evidence outputs"
        )
    if args.artifact_identity_sha256 is not None and (
        len(args.artifact_identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.artifact_identity_sha256)
    ):
        raise ValueError("artifact identity must be a lowercase SHA-256 digest")

    root = args.archive_root
    root_fd = _open_directory_nofollow(root)
    os.close(root_fd)
    verifier_path = root / "scripts" / "verify_runtime_source.py"
    launcher_path = root / "scripts" / "run_exact_tabicl.py"
    if Path(os.path.abspath(__file__)) != launcher_path:
        raise ValueError("exact-T launcher is not executing from the candidate archive")

    verifier = _load_fixed_module("_exact_t_runtime_verifier", verifier_path)
    manifest = verifier.load_and_validate_source_manifest(
        args.source_manifest,
        args.expected_manifest_sha256,
        args.expected_commit_sha,
        args.expected_tree_sha,
    )
    root = verifier.verify_archive(root, manifest)
    tracked = {entry["path"] for entry in manifest["payload"]["entries"]}
    if {
        "scripts/run_exact_tabicl.py",
        "scripts/verify_runtime_source.py",
    } - tracked:
        raise ValueError("exact-T launch chain is not fully tracked by the source manifest")
    expected_src = verifier.validate_environment(root)
    imported = verifier.import_attested_tabicl(root, expected_src)

    harness_path = root / "scripts" / "run_h100_identity_validation.py"
    if "scripts/run_h100_identity_validation.py" not in tracked:
        raise ValueError("fixed H100 validation harness is not tracked")
    harness = _load_fixed_module("_exact_t_h100_validation", harness_path)
    attestation_payload = {
        "commit_sha": manifest["payload"]["commit_sha"],
        "tree_sha": manifest["payload"]["tree_sha"],
        "source_manifest_sha256": manifest["sha256"],
        "import_relative_path": imported,
        "action": args.action,
        "case_id": args.case_id,
        "artifact_identity_sha256": args.artifact_identity_sha256,
        "python_isolated": True,
        "bytecode_disabled": True,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    }
    attestation = harness.make_source_attestation(attestation_payload)
    should_publish = os.environ.get("RANK", "0") == "0"
    if should_publish:
        _publish_no_replace(
            args.source_attestation_output,
        json.dumps(
            attestation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
            max_bytes=args.source_attestation_max_bytes,
        )

    if args.action == "trainer":
        verifier.run_attested_trainer(
            root,
            expected_src,
            manifest,
            SimpleNamespace(source_manifest=args.source_manifest),
            action_argv,
        )
    elif args.action == "checkpoint-validator":
        validator_path = (
            root / "scripts" / "validate_identity_checkpoint.py"
        ).resolve(strict=True)
        if "scripts/validate_identity_checkpoint.py" not in tracked:
            raise ValueError("fixed checkpoint validator is not tracked")
        validator = _load_fixed_module("_exact_t_checkpoint_validator", validator_path)
        verifier.attest_loaded_tabicl(root, expected_src)
        status = validator.main(action_argv)
        if status:
            raise RuntimeError(f"checkpoint validator returned status {status}")
    else:
        verifier.attest_loaded_tabicl(root, expected_src)
        harness.execute_validation_case(args.case_id, action_argv)

    # Source attestation deliberately precedes compute.  Runtime evidence and
    # the completion receipt deliberately follow compute.  For NCCL, rank 0
    # may publish only after every rank has also passed the final exact-source
    # module attestation below.
    runtime_evidence = None
    if args.action == "h100-validation" and should_publish:
        runtime_evidence = harness.capture_runtime_evidence(
            case_id=args.case_id,
            commit_sha=manifest["payload"]["commit_sha"],
            tree_sha=manifest["payload"]["tree_sha"],
            source_manifest_sha256=manifest["sha256"],
        )
    verifier.attest_loaded_tabicl(root, expected_src)
    if args.action == "h100-validation" and args.case_id == "nccl_2gpu":
        import torch.distributed as dist

        if (
            not dist.is_initialized()
            or dist.get_backend() != "nccl"
            or dist.get_world_size() != 2
        ):
            raise RuntimeError("NCCL completion barrier contract is not active")
        dist.barrier()
    if args.action == "h100-validation" and should_publish:
        assert runtime_evidence is not None
        _publish_no_replace(
            args.runtime_evidence_output,
            json.dumps(
                runtime_evidence,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
            max_bytes=args.source_attestation_max_bytes,
        )
        completion = harness.make_action_completion(
            {
                "case_id": args.case_id,
                "action": args.action,
                "artifact_identity_sha256": args.artifact_identity_sha256,
                "source_attestation_sha256": attestation["sha256"],
                "commit_sha": manifest["payload"]["commit_sha"],
                "tree_sha": manifest["payload"]["tree_sha"],
                "source_manifest_sha256": manifest["sha256"],
                "runtime_evidence_sha256": runtime_evidence["sha256"],
                "slurm_job_id": runtime_evidence["payload"]["scheduler_binding"]["job_id"],
                "requested_resource_sha256": runtime_evidence["payload"]["scheduler_binding"][
                    "requested_resource_sha256"
                ],
                "scheduler_binding_sha256": runtime_evidence["payload"]["scheduler_binding"][
                    "sha256"
                ],
                "final_tabicl_attested": True,
                "completed": True,
            }
        )
        _publish_no_replace(
            args.completion_receipt_output,
            json.dumps(
                completion,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
            max_bytes=args.source_attestation_max_bytes,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"exact-T action failed: {error}", file=sys.stderr)
        raise SystemExit(2)
