#!/bin/bash
set -euo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="$(command -v python3)"
fi
PYTHONDONTWRITEBYTECODE=1 exec "$PYTHON_BIN" -B - "$ROOT" <<'PY'
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


REPO = Path(sys.argv[1])
TEMP = Path(tempfile.mkdtemp(prefix="tabicl-task4-submit-")).resolve()
EXACT = TEMP / "exact"
EXTERNAL = TEMP / "external"
ARTIFACT_PARENT = TEMP / "artifacts"
FAKEBIN = TEMP / "fakebin"
for path in (EXACT / "scripts", EXACT / "src/tabicl", EXTERNAL, ARTIFACT_PARENT, FAKEBIN):
    path.mkdir(parents=True, exist_ok=True)


def check(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in (
    "submit_h100_identity_formal.sh",
    "verify_formal_overlay.py",
    "verify_runtime_source.py",
    "run_h100_identity_validation.py",
    "slurm_h100_identity_formal.sh",
):
    shutil.copy2(REPO / "scripts" / name, EXACT / "scripts" / name)

(EXACT / "src/tabicl/__init__.py").write_text("__version__ = 'task4-fixture'\n")
(EXACT / "scripts/check_formal_capacity.py").write_text(
    """#!/usr/bin/env python3
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--artifact-root', required=True, type=Path)
parser.add_argument('--checkpoint-ceiling-bytes', required=True)
parser.add_argument('--durable-log-allowance-bytes', required=True)
parser.add_argument('--remaining-checkpoints', required=True)
parser.add_argument('--run-log-ceiling-bytes', required=True)
parser.add_argument('--attestation-ceiling-bytes', required=True)
parser.add_argument('--manifest-ceiling-bytes', required=True)
parser.add_argument('--protocol-metadata-allowance-bytes', required=True)
args = parser.parse_args()
args.artifact_root.joinpath('capacity.called').write_text('passed\\n')
if args.artifact_root.joinpath('capacity.fail').exists():
    raise SystemExit(1)
if args.remaining_checkpoints != '15':
    raise SystemExit(2)
"""
)
for path in (EXACT / "scripts").iterdir():
    path.chmod(path.stat().st_mode | stat.S_IXUSR)

check(["/usr/bin/git", "init", "-q", str(EXACT)])
check(["/usr/bin/git", "-C", str(EXACT), "config", "user.name", "fixture"])
check(["/usr/bin/git", "-C", str(EXACT), "config", "user.email", "fixture@example.invalid"])
check(["/usr/bin/git", "-C", str(EXACT), "add", "scripts", "src"])
check(["/usr/bin/git", "-C", str(EXACT), "commit", "-q", "-m", "fixture"])
COMMIT = check(
    ["/usr/bin/git", "-C", str(EXACT), "rev-parse", "HEAD"],
    stdout=subprocess.PIPE,
).stdout.strip()
TREE = check(
    ["/usr/bin/git", "-C", str(EXACT), "rev-parse", "HEAD^{tree}"],
    stdout=subprocess.PIPE,
).stdout.strip()
check(["/usr/bin/git", "-C", str(EXACT), "checkout", "-q", "--detach", COMMIT])

overlay_module = load("task4_overlay_fixture", EXACT / "scripts/verify_formal_overlay.py")
matrix = load("task4_matrix_fixture", EXACT / "scripts/run_h100_identity_validation.py")
tracked = check(
    ["/usr/bin/git", "-C", str(EXACT), "ls-files", "-s"],
    stdout=subprocess.PIPE,
).stdout.splitlines()
entries = []
for line in tracked:
    metadata, relative = line.split("\t", 1)
    mode = metadata.split()[0]
    raw = (EXACT / relative).read_bytes()
    entries.append(
        {
            "path": relative,
            "mode": mode,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
source_manifest = overlay_module.make_manifest(
    "source",
    {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "code_roots": ["scripts", "src/tabicl"],
        "entries": sorted(entries, key=lambda item: item["path"]),
    },
)
SOURCE_PATH = EXTERNAL / "source-manifest.json"
SOURCE_PATH.write_bytes(overlay_module.canonical_json_bytes(source_manifest) + b"\n")
GPU_MODEL = "NVIDIA H100 80GB HBM3"
DRIVER = "570.00"


def envelope(kind, payload):
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    return {**body, "sha256": hashlib.sha256(matrix._canonical(body)).hexdigest()}


runtime_payload = {
    "python_version": "3.test",
    "python_implementation": "CPython",
    "platform_system": "Linux",
    "platform_release": "test",
    "platform_machine": "x86_64",
    "torch_version": "test",
    "numpy_version": "test",
    "cuda_runtime_version": "test",
    "cudnn_version": "test",
}
ONE_GPU_ENV = envelope("environment", {**runtime_payload, "visible_cuda_device_count": 1})
TWO_GPU_ENV = envelope("environment", {**runtime_payload, "visible_cuda_device_count": 2})
ENVIRONMENT = ONE_GPU_ENV["sha256"]


def case_evidence(case):
    identity = hashlib.sha256(("identity:" + case.case_id).encode()).hexdigest()
    source = matrix.make_source_attestation(
        {
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "source_manifest_sha256": source_manifest["sha256"],
            "import_relative_path": "src/tabicl/__init__.py",
            "action": "h100-validation",
            "case_id": case.case_id,
            "artifact_identity_sha256": identity,
            "python_isolated": True,
            "bytecode_disabled": True,
            "python_no_user_site": "1",
        }
    )
    environment = ONE_GPU_ENV if case.world_size == 1 else TWO_GPU_ENV
    devices = [
        {"uuid": f"GPU-fixture-{index}", "name": GPU_MODEL, "driver_version": DRIVER}
        for index in range(case.world_size)
    ]
    runtime = matrix.make_runtime_evidence(
        {
            "case_id": case.case_id,
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "source_manifest_sha256": source_manifest["sha256"],
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
        }
    )
    completion = matrix.make_action_completion(
        {
            "case_id": case.case_id,
            "action": "h100-validation",
            "artifact_identity_sha256": identity,
            "source_attestation_sha256": source["sha256"],
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "source_manifest_sha256": source_manifest["sha256"],
            "runtime_evidence_sha256": runtime["sha256"],
            "final_tabicl_attested": True,
            "completed": True,
        }
    )
    checkpoint_relative_path = (
        f"step-{case.repeat_steps}.ckpt" if case.stage is not None else None
    )
    checkpoint_bytes = b"x" if case.stage is not None else None
    checkpoint_sha256 = (
        hashlib.sha256(checkpoint_bytes).hexdigest()
        if checkpoint_bytes is not None
        else None
    )
    checkpoint_size = len(checkpoint_bytes) if checkpoint_bytes is not None else None
    case_result = envelope(
        "h100_case_result",
        {
            "case_id": case.case_id,
            "assertion": case.assertion,
            "observed_sequence_length": case.observed_sequence_length,
            "recompute": case.recompute,
            "world_size": case.world_size,
            "monotonic_start_seconds": 100.0,
            "monotonic_end_seconds": 111.0,
            "checkpoint_relative_path": checkpoint_relative_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size": checkpoint_size,
            "checkpoint_ceiling_bytes": 1,
        },
    )
    utilization = case.utilization_required
    window = {
        "gpu_uuids": ["GPU-fixture-0"] if utilization else [],
        "gpu_sample_counts": {"GPU-fixture-0": 10} if utilization else {},
        "gpu_means": {"GPU-fixture-0": 80.0} if utilization else {},
        "gpu_min_gap_seconds": 1.0 if utilization else None,
        "gpu_max_gap_seconds": 1.0 if utilization else None,
    }
    raw_hashes = {
        "source-attestation.json": hashlib.sha256(matrix._canonical(source) + b"\n").hexdigest(),
        "runtime-evidence.json": hashlib.sha256(matrix._canonical(runtime) + b"\n").hexdigest(),
        "action-completion.json": hashlib.sha256(matrix._canonical(completion) + b"\n").hexdigest(),
        "compute/case-result.json": hashlib.sha256(matrix._canonical(case_result) + b"\n").hexdigest(),
        "compute.log": hashlib.sha256(b"").hexdigest(),
    }
    if utilization:
        raw_hashes.update(
            {
                "gpu.csv": hashlib.sha256(b"gpu-csv").hexdigest(),
                "gpu.jsonl": hashlib.sha256(b"gpu-jsonl").hexdigest(),
                "gpu-summary.json": hashlib.sha256(b"gpu-summary").hexdigest(),
            }
        )
    if checkpoint_relative_path is not None:
        raw_hashes[f"compute/{checkpoint_relative_path}"] = checkpoint_sha256
    raw_sizes = {name: 0 for name in raw_hashes}
    if checkpoint_relative_path is not None:
        raw_sizes[f"compute/{checkpoint_relative_path}"] = checkpoint_size
    artifact_manifest = {
        "schema_version": 1,
        "case_id": case.case_id,
        "files": [
            {"name": name, "sha256": digest, "size": raw_sizes[name]}
            for name, digest in sorted(raw_hashes.items())
        ],
    }
    artifact = hashlib.sha256(matrix._canonical(artifact_manifest)).hexdigest()
    binding = {
        "case_id": case.case_id,
        "source_attestation_sha256": source["sha256"],
        "runtime_evidence_sha256": runtime["sha256"],
        "action_completion_sha256": completion["sha256"],
        "artifact_identity_sha256": identity,
        "artifact_sha256": artifact,
    }
    return {
        "case_id": case.case_id,
        "arm": case.arm,
        "stage": case.stage,
        "world_size": case.world_size,
        "observed_sequence_length": case.observed_sequence_length,
        "smoke_kind": case.smoke_kind,
        "recompute": case.recompute,
        "active_start_seconds": 100.0,
        "active_end_seconds": 111.0,
        "artifact_sha256": artifact,
        "artifact_manifest": artifact_manifest,
        "artifact_identity_sha256": identity,
        "case_result": case_result,
        "case_result_sha256": case_result["sha256"],
        "checkpoint_relative_path": checkpoint_relative_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "source_attestation": source,
        "source_attestation_sha256": source["sha256"],
        "runtime_evidence": runtime,
        "runtime_evidence_sha256": runtime["sha256"],
        "action_completion": completion,
        "action_completion_sha256": completion["sha256"],
        "case_binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "gpu_window_sha256": (
            hashlib.sha256(
                json.dumps(window, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if utilization
            else None
        ),
        **window,
        "checkout_commit_sha": COMMIT,
        "checkout_tree_sha": TREE,
        "clean_detached": True,
        "tabicl_attestation_count": 1,
        "python_no_user_site": "1",
    }


smoke = matrix.make_smoke_attestation(
    {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "environment_sha256": ENVIRONMENT,
        "source_manifest_sha256": source_manifest["sha256"],
        "gpu_model": GPU_MODEL,
        "driver_version": DRIVER,
        "checkpoint_ceiling_bytes": 1,
        "observed_checkpoint_max_bytes": 1,
        "cases": [case_evidence(case) for case in matrix.build_matrix()],
    }
)
SMOKE_PATH = EXTERNAL / "h100-smoke.json"
SMOKE_PATH.write_bytes(overlay_module.canonical_json_bytes(smoke) + b"\n")


def stage_values(source_sha=source_manifest["sha256"]):
    result = []
    for index, (stage, budget) in enumerate(overlay_module.STAGES, start=1):
        prior = str(index + 2) * 64
        architecture = str(index + 3) * 64
        optimizer = str(index + 4) * 64
        scientific = str(index + 5) * 64
        cohort, arms = overlay_module._expected_protocols(
            stage=stage,
            terminal_step=budget,
            source_sha256=source_sha,
            environment_sha256=ENVIRONMENT,
            prior_sha256=prior,
            architecture_sha256=architecture,
            optimizer_sha256=optimizer,
            scientific_sha256=scientific,
        )
        result.append(
            {
                "stage": stage,
                "terminal_step": budget,
                "prior_sha256": prior,
                "architecture_sha256": architecture,
                "optimizer_sha256": optimizer,
                "scientific_sha256": scientific,
                "cohort_protocol_sha256": cohort,
                "arm_protocol_sha256": arms,
            }
        )
    return result


BASE = {
    "schema_version": 1,
    "run_policy": "fresh",
    "study_id": "study-a",
    "artifact_root": "",
    "source": {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "manifest_path": str(SOURCE_PATH),
        "manifest_sha256": source_manifest["sha256"],
        "environment_sha256": ENVIRONMENT,
        "candidate_repository": str(EXACT),
    },
    "smoke": {
        "attestation_path": str(SMOKE_PATH),
        "expected_sha256": smoke["sha256"],
        "expected_gpu_model": GPU_MODEL,
    },
    "capacity": {
        "checkpoint_ceiling_bytes": 1,
        "durable_log_allowance_bytes": 30_000_000,
        "run_log_ceiling_bytes": 1_000,
        "attestation_ceiling_bytes": 1_000_000,
        "manifest_ceiling_bytes": 1_000,
        "protocol_metadata_allowance_bytes": 100_000,
    },
    "runtime": {
        "python": str(Path(sys.executable).resolve()),
        "git": "/usr/bin/git",
        "nvidia_smi": "/usr/bin/true",
        "job_work_root": str(TEMP / "job-work"),
    },
    "scheduler": {
        "partition": "h100",
        "qos": "long",
        "cpus_per_task": 32,
        "memory_mb": 131_072,
        "gpus_per_job": 1,
    },
    "stages": stage_values(),
}
(TEMP / "job-work").mkdir()

for source_name, target_name in (
    ("fake_sbatch.sh", "sbatch"),
    ("fake_scontrol.sh", "scontrol"),
    ("fake_scancel.sh", "scancel"),
):
    shutil.copy2(REPO / "tests/fixtures" / source_name, FAKEBIN / target_name)
    (FAKEBIN / target_name).chmod(0o755)

CONTROL_FILES = {
    "calls.log",
    "sbatch_count",
    "release_count",
    "fail_sbatch_at",
    "empty_at",
    "malformed_at",
    "cluster_suffix_at",
    "duplicate_at",
    "fail_release_at",
    "fail_cancel_ids",
    "ledger_path",
    "receipt_path",
    "precreate_receipt_at",
}


def reset_fake(controls=None):
    for name in CONTROL_FILES:
        try:
            (FAKEBIN / name).unlink()
        except FileNotFoundError:
            pass
    for name, value in (controls or {}).items():
        (FAKEBIN / name).write_text(str(value) + ("" if str(value).endswith("\n") else "\n"))
    try:
        (ARTIFACT_PARENT / "capacity.fail").unlink()
    except FileNotFoundError:
        pass
    try:
        (ARTIFACT_PARENT / "capacity.called").unlink()
    except FileNotFoundError:
        pass


def invoke(name, *, mutate=None, controls=None, fault=None, scheduler_path=None):
    reset_fake(controls)
    value = copy.deepcopy(BASE)
    value["study_id"] = name
    value["artifact_root"] = str(ARTIFACT_PARENT / name)
    # Rebuild safe identity-bearing paths by letting the submitter consume this
    # per-scenario study ID; protocol hashes intentionally exclude it.
    if mutate is not None:
        mutate(value)
    overlay_path = EXTERNAL / f"{name}.json"
    overlay_path.write_bytes(overlay_module.canonical_json_bytes(value) + b"\n")
    env = {
        "PATH": str(FAKEBIN) if scheduler_path is None else scheduler_path,
        "PYTHON": str(Path(sys.executable).resolve()),
        "TABICL_EXACT_ROOT": str(EXACT),
        "RUN_POLICY": "fresh",
        "PYTHONPATH": "/poison/editable/src",
        "PYTHONHOME": "/poison/home",
        "PYTHONUSERBASE": "/poison/user",
        "EVIL_AMBIENT": "must-not-reach-job",
    }
    if fault:
        env["FORMAL_FAULT_LEDGER_STAGE"] = fault
    return subprocess.run(
        ["/bin/bash", str(EXACT / "scripts/submit_h100_identity_formal.sh"), str(overlay_path)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def calls():
    path = FAKEBIN / "calls.log"
    return path.read_text().splitlines() if path.exists() else []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


success = invoke("success")
require(success.returncode == 0, f"success failed: {success.stderr}")
events = calls()
require(len(events) == 18, events)
require(all(line.startswith("sbatch ") for line in events[:9]), events)
require(all(line.startswith("scontrol <release>") for line in events[9:]), events)
for line in events[:9]:
    require("<--hold>" in line and "<--gres=gpu:1>" in line, line)
    require("--export=ALL" not in line and "gpu:2" not in line, line)
require("--dependency=afterok:1001" in events[1] and "--kill-on-invalid-dep=yes" in events[1], events[1])
require("--dependency=afterok:1002" in events[2], events[2])
require("--dependency=" not in events[3], events[3])
require("--dependency=afterok:1004" in events[4], events[4])
require("--dependency=afterok:1007" in events[7], events[7])
ledger = json.loads((ARTIFACT_PARENT / "success/transaction-ledger.json").read_text())
require(len(ledger["payload"]["entries"]) == 9, ledger)
require(all("checkpoint_sha256" not in entry for entry in ledger["payload"]["entries"]), ledger)
receipt = json.loads((ARTIFACT_PARENT / "success/submission-receipt.json").read_text())
require(receipt["kind"] == "held_submission_receipt", receipt)
require(receipt["payload"]["jobs_held_at_publication"] is True, receipt)
require(receipt["payload"]["job_ids"] == [str(1000 + i) for i in range(1, 10)], receipt)

for failure_position in range(1, 10):
    result = invoke(
        f"sbatch-fail-{failure_position}",
        controls={"fail_sbatch_at": failure_position},
    )
    require(result.returncode != 0, f"sbatch failure {failure_position} reported success")
    events = calls()
    require(sum(line.startswith("sbatch ") for line in events) == failure_position, events)
    expected = [f"scancel <{1000 + i}>" for i in range(failure_position - 1, 0, -1)]
    require([line for line in events if line.startswith("scancel ")] == expected, events)
    require(not any(line.startswith("scontrol ") for line in events), events)

for control in ("empty_at", "malformed_at", "duplicate_at"):
    result = invoke(f"bad-id-{control}", controls={control: 3})
    require(result.returncode != 0, f"{control} reported success")
    require(
        [line for line in calls() if line.startswith("scancel ")]
        == ["scancel <1002>", "scancel <1001>"],
        calls(),
    )

cluster_suffix = invoke(
    "cluster-suffix-then-fail",
    controls={"cluster_suffix_at": 3, "fail_sbatch_at": 4},
)
require(cluster_suffix.returncode != 0, "post-suffix sbatch failure reported success")
require(
    [line for line in calls() if line.startswith("scancel ")]
    == ["scancel <1003>", "scancel <1002>", "scancel <1001>"],
    calls(),
)

release = invoke("release-fail", controls={"fail_release_at": 4})
require(release.returncode != 0, "partial release reported success")
require(sum(line.startswith("scontrol ") for line in calls()) == 4, calls())
require(
    [line for line in calls() if line.startswith("scancel ")]
    == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
    calls(),
)

release_cancel = invoke(
    "release-cancel-fail",
    controls={"fail_release_at": 4, "fail_cancel_ids": "1002"},
)
require(release_cancel.returncode != 0, "release rollback failure reported success")
release_recovery = json.loads(
    (ARTIFACT_PARENT / "release-cancel-fail/rollback-incomplete.json").read_text()
)
require(release_recovery["payload"]["reason"] == "release_failed", release_recovery)
require(release_recovery["payload"]["ledger_published"] is True, release_recovery)
require(release_recovery["payload"]["remaining_job_ids"] == ["1002"], release_recovery)

receipt_collision = invoke(
    "receipt-collision", controls={"precreate_receipt_at": 9}
)
require(receipt_collision.returncode != 0, "receipt collision reported success")
require(not any(line.startswith("scontrol ") for line in calls()), calls())
require(
    [line for line in calls() if line.startswith("scancel ")]
    == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
    calls(),
)

cancel = invoke(
    "cancel-fail",
    controls={"fail_sbatch_at": 4, "fail_cancel_ids": "1002"},
)
require(cancel.returncode != 0, "incomplete rollback reported success")
recovery = json.loads((ARTIFACT_PARENT / "cancel-fail/rollback-incomplete.json").read_text())
require(recovery["kind"] == "rollback_incomplete", recovery)
require(recovery["payload"]["reason"] == "sbatch_failed", recovery)
require(recovery["payload"]["remaining_job_ids"] == ["1002"], recovery)
require(set(recovery["payload"]["cancelled_job_ids"]) == {"1001", "1003"}, recovery)

for fault in ("write", "fsync", "rename", "dir_fsync"):
    result = invoke(f"ledger-{fault}", fault=fault)
    require(result.returncode != 0, f"ledger {fault} fault reported success")
    require(not any(line.startswith("scontrol ") for line in calls()), calls())
    require(
        [line for line in calls() if line.startswith("scancel ")]
        == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
        calls(),
    )

capacity_name = "capacity-fail"
reset_fake()
(ARTIFACT_PARENT / "capacity.fail").write_text("fail\n")
value = copy.deepcopy(BASE)
value["study_id"] = capacity_name
value["artifact_root"] = str(ARTIFACT_PARENT / capacity_name)
overlay_path = EXTERNAL / f"{capacity_name}-direct.json"
overlay_path.write_bytes(overlay_module.canonical_json_bytes(value) + b"\n")
env = {
    "PATH": str(FAKEBIN),
    "PYTHON": str(Path(sys.executable).resolve()),
    "TABICL_EXACT_ROOT": str(EXACT),
    "RUN_POLICY": "fresh",
}
result = subprocess.run(
    ["/bin/bash", str(EXACT / "scripts/submit_h100_identity_formal.sh"), str(overlay_path)],
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
require(result.returncode != 0 and not calls(), f"capacity gate order failed: {calls()}")
(ARTIFACT_PARENT / "capacity.fail").unlink()

preflight_cases = []
preflight_cases.append(("wrong-gpu", lambda value: value["scheduler"].update(gpus_per_job=2)))
preflight_cases.append(("wrong-partition", lambda value: value["scheduler"].update(partition="a10")))
preflight_cases.append(("missing-smoke", lambda value: value["smoke"].update(attestation_path=str(EXTERNAL / "missing.json"))))
preflight_cases.append(("smoke-ceiling-mismatch", lambda value: value["capacity"].update(checkpoint_ceiling_bytes=2)))
preflight_cases.append(("mismatched-commit", lambda value: value["source"].update(commit_sha="f" * 40)))
preflight_cases.append(("missing-arm", lambda value: value["stages"][0]["arm_protocol_sha256"].pop("none")))
for name, mutation in preflight_cases:
    result = invoke(name, mutate=mutation)
    require(result.returncode != 0 and not calls(), f"{name} reached scheduler: {calls()}")

wrong_env = copy.deepcopy(smoke)
wrong_env["payload"]["environment_sha256"] = "f" * 64
wrong_env = matrix.make_smoke_attestation(wrong_env["payload"])
wrong_env_path = EXTERNAL / "wrong-environment-smoke.json"
wrong_env_path.write_bytes(overlay_module.canonical_json_bytes(wrong_env) + b"\n")
result = invoke(
    "wrong-environment",
    mutate=lambda value: value["smoke"].update(
        attestation_path=str(wrong_env_path), expected_sha256=wrong_env["sha256"]
    ),
)
require(result.returncode != 0 and not calls(), f"wrong environment reached scheduler: {calls()}")

bad_source = copy.deepcopy(source_manifest)
bad_source["payload"]["entries"][0]["sha256"] = "f" * 64
bad_source = overlay_module.make_manifest("source", bad_source["payload"])
bad_source_path = EXTERNAL / "bad-source.json"
bad_source_path.write_bytes(overlay_module.canonical_json_bytes(bad_source) + b"\n")
def wrong_source_mutation(value):
    value["source"]["manifest_path"] = str(bad_source_path)
    value["source"]["manifest_sha256"] = bad_source["sha256"]
    value["stages"] = stage_values(bad_source["sha256"])
result = invoke("wrong-source", mutate=wrong_source_mutation)
require(result.returncode != 0 and not calls(), f"wrong source reached scheduler: {calls()}")

(EXACT / "src/tabicl/dirty.py").write_text("raise RuntimeError('poison')\n")
try:
    result = invoke("dirty-checkout")
    require(result.returncode != 0 and not calls(), f"dirty checkout reached scheduler: {calls()}")
finally:
    (EXACT / "src/tabicl/dirty.py").unlink()

reuse = invoke("success")
require(reuse.returncode != 0 and not calls(), f"fresh namespace reuse reached scheduler: {calls()}")

(FAKEBIN / "scontrol").rename(FAKEBIN / "scontrol.disabled")
try:
    missing_scheduler = invoke("missing-scheduler")
    require(missing_scheduler.returncode != 0 and not calls(), calls())
    require(
        not (ARTIFACT_PARENT / "missing-scheduler").exists(),
        "scheduler preflight failure consumed the fresh namespace",
    )
finally:
    (FAKEBIN / "scontrol.disabled").rename(FAKEBIN / "scontrol")

relative_scheduler_path = invoke(
    "relative-scheduler-path", scheduler_path=f"{FAKEBIN}:relative"
)
require(relative_scheduler_path.returncode != 0 and not calls(), calls())
require(
    not (ARTIFACT_PARENT / "relative-scheduler-path").exists(),
    "invalid scheduler PATH consumed the fresh namespace",
)

artifact_alias = TEMP / "artifact-parent-alias"
artifact_alias.symlink_to(ARTIFACT_PARENT, target_is_directory=True)
symlinked_artifact = invoke(
    "symlinked-artifact-parent",
    mutate=lambda value: value.update(
        artifact_root=str(artifact_alias / "symlinked-artifact-parent")
    ),
)
require(symlinked_artifact.returncode != 0 and not calls(), calls())
require(
    not (ARTIFACT_PARENT / "symlinked-artifact-parent").exists(),
    "symlinked artifact parent escaped into the physical target",
)

job_work_alias = TEMP / "job-work-alias"
job_work_alias.symlink_to(TEMP / "job-work", target_is_directory=True)
symlinked_job_work = invoke(
    "symlinked-job-work",
    mutate=lambda value: value["runtime"].update(job_work_root=str(job_work_alias)),
)
require(symlinked_job_work.returncode != 0 and not calls(), calls())
require(
    not (ARTIFACT_PARENT / "symlinked-job-work").exists(),
    "symlinked job work root consumed the fresh namespace",
)

source_text = (EXACT / "scripts/submit_h100_identity_formal.sh").read_text()
require("--export=ALL" not in source_text, "submitter contains --export=ALL")
require("gpu:2" not in source_text, "production submitter contains a two-GPU request")
slurm_text = (EXACT / "scripts/slurm_h100_identity_formal.sh").read_text()
require('"$NVIDIA_SMI"' in slurm_text, "formal Slurm does not use trusted NVIDIA_SMI")
require("$(nvidia-smi" not in slurm_text, "formal Slurm uses PATH nvidia-smi")
require("--query-gpu=name,uuid" in slurm_text, "formal Slurm does not inspect GPU model")
require("H100" in slurm_text, "formal Slurm does not enforce the H100 model")
print("Task4 hermetic fake-Slurm submit matrix passed")
PY
