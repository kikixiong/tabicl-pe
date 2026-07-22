#!/usr/bin/env python3
"""Validate one formal identity checkpoint against external expectations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tabicl.train._provenance import (
    CheckpointExpectations,
    FinalizationTrust,
    FORMAL_MODES,
    FORMAL_STAGES,
    ParentTrust,
    canonical_json_bytes,
    finalize_identity_checkpoint,
    recover_finalized_checkpoint_manifest,
    validate_identity_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(FORMAL_MODES))
    parser.add_argument("--np-seed", required=True, type=int)
    parser.add_argument("--torch-seed", required=True, type=int)
    parser.add_argument("--identity-seed", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=sorted(FORMAL_STAGES))
    parser.add_argument("--terminal-step", required=True, type=int)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--architecture-sha256", required=True)
    parser.add_argument("--optimizer-sha256", required=True)
    parser.add_argument("--scientific-sha256", required=True)
    parser.add_argument("--cohort-protocol-sha256", required=True)
    parser.add_argument("--arm-protocol-sha256", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--cuda-device-count", required=True, type=int)

    parent = parser.add_argument_group("Stage 2/3 immutable parent trust")
    parent.add_argument("--parent-checkpoint", type=Path)
    parent.add_argument("--finalized-parent-manifest", type=Path)
    parent.add_argument("--transaction-ledger", type=Path)
    parent.add_argument("--transaction-ledger-sha256")
    parent.add_argument("--study-id")
    parent.add_argument("--parent-stage", choices=sorted(FORMAL_STAGES))
    parent.add_argument("--upstream-identity")
    parent.add_argument("--artifact-identity")
    parent.add_argument("--artifact-root", type=Path)

    finalization = parser.add_argument_group("Write-once finalized checkpoint manifest")
    finalization.add_argument("--finalize-output", type=Path)
    finalization.add_argument("--recover-finalization", action="store_true")
    finalization.add_argument("--finalization-transaction-ledger", type=Path)
    finalization.add_argument("--finalization-transaction-ledger-sha256")
    finalization.add_argument("--finalization-study-id")
    finalization.add_argument("--finalization-upstream-identity")
    finalization.add_argument("--finalization-artifact-identity")
    finalization.add_argument("--finalization-artifact-root", type=Path)
    return parser


def _parent_trust(args: argparse.Namespace) -> ParentTrust | None:
    names = (
        "parent_checkpoint",
        "finalized_parent_manifest",
        "transaction_ledger",
        "transaction_ledger_sha256",
        "study_id",
        "parent_stage",
        "upstream_identity",
        "artifact_identity",
    )
    present = [getattr(args, name) is not None for name in names]
    if not any(present):
        if args.stage != "stage1":
            raise ValueError("Stage 2/3 requires every immutable parent trust argument")
        return None
    if not all(present):
        missing = [
            name.replace("_", "-") for name in names if getattr(args, name) is None
        ]
        raise ValueError(f"incomplete parent trust arguments: {', '.join(missing)}")
    if args.stage == "stage1":
        raise ValueError("Stage 1 must not receive parent trust arguments")
    return ParentTrust(
        checkpoint_path=args.parent_checkpoint,
        finalized_manifest_path=args.finalized_parent_manifest,
        transaction_ledger_path=args.transaction_ledger,
        transaction_ledger_sha256=args.transaction_ledger_sha256,
        study_id=args.study_id,
        arm=args.mode,
        parent_stage=args.parent_stage,
        upstream_identity=args.upstream_identity,
        artifact_identity=args.artifact_identity,
        artifact_root=args.artifact_root,
    )


def _finalization_trust(args: argparse.Namespace) -> FinalizationTrust | None:
    names = (
        "finalization_transaction_ledger",
        "finalization_transaction_ledger_sha256",
        "finalization_study_id",
        "finalization_upstream_identity",
        "finalization_artifact_identity",
        "finalization_artifact_root",
    )
    present = [getattr(args, name) is not None for name in names]
    requested = args.finalize_output is not None or args.recover_finalization
    if not requested:
        if any(present):
            raise ValueError("finalization trust arguments require --finalize-output")
        return None
    if args.finalize_output is None:
        raise ValueError("--recover-finalization requires --finalize-output")
    if not all(present):
        missing = [
            name.replace("_", "-") for name in names if getattr(args, name) is None
        ]
        raise ValueError(
            f"incomplete finalization trust arguments: {', '.join(missing)}"
        )
    return FinalizationTrust(
        transaction_ledger_path=args.finalization_transaction_ledger,
        transaction_ledger_sha256=args.finalization_transaction_ledger_sha256,
        artifact_root=args.finalization_artifact_root,
        study_id=args.finalization_study_id,
        upstream_identity=args.finalization_upstream_identity,
        artifact_identity=args.finalization_artifact_identity,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    finalization_trust = _finalization_trust(args)
    expected = CheckpointExpectations(
        mode=args.mode,
        np_seed=args.np_seed,
        torch_seed=args.torch_seed,
        identity_seed=args.identity_seed,
        stage=args.stage,
        terminal_step=args.terminal_step,
        source_sha256=args.source_sha256,
        environment_sha256=args.environment_sha256,
        prior_sha256=args.prior_sha256,
        architecture_sha256=args.architecture_sha256,
        optimizer_sha256=args.optimizer_sha256,
        scientific_sha256=args.scientific_sha256,
        cohort_protocol_sha256=args.cohort_protocol_sha256,
        arm_protocol_sha256=args.arm_protocol_sha256,
        world_size=args.world_size,
        cuda_device_count=args.cuda_device_count,
        parent_trust=_parent_trust(args),
    )
    if finalization_trust is None:
        report = validate_identity_checkpoint(args.checkpoint, expected)
    elif args.recover_finalization:
        report = recover_finalized_checkpoint_manifest(
            args.checkpoint,
            expected,
            finalized_manifest_path=args.finalize_output,
            trust=finalization_trust,
        )
    else:
        report = finalize_identity_checkpoint(
            args.checkpoint,
            expected,
            finalized_manifest_path=args.finalize_output,
            trust=finalization_trust,
        )
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"checkpoint validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
