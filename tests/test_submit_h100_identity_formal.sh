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

import atexit
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
    "verify_filesystem_isolation.py",
    "verify_git_repository.py",
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

FAKE_GIT = FAKEBIN / "git"
FAKE_GIT.write_text(
    "#!/bin/bash\n"
    "set -euo pipefail\n"
    "if [[ \"${1:-}\" == ls-remote ]]; then\n"
    f"  [[ \"$*\" == \"ls-remote --refs https://github.com/kikixiong/tabicl-pe.git refs/heads/codex/position-identity-v1\" ]]\n"
    f"  printf '%s\\t%s\\n' '{COMMIT}' 'refs/heads/codex/position-identity-v1'\n"
    "else\n"
    "  exec /usr/bin/git \"$@\"\n"
    "fi\n"
)
FAKE_GIT.chmod(0o755)
FAKE_GIT_SHA256 = hashlib.sha256(FAKE_GIT.read_bytes()).hexdigest()

overlay_module = load("task4_overlay_fixture", EXACT / "scripts/verify_formal_overlay.py")
matrix = load("task4_matrix_fixture", EXACT / "scripts/run_h100_identity_validation.py")
repository_helper = load(
    "task4_repository_fixture", EXACT / "scripts/verify_git_repository.py"
)
REPOSITORY_BINDING = repository_helper.expected_repository_binding(
    expected_commit_sha=COMMIT,
    git_sha256=FAKE_GIT_SHA256,
)
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
    requested_resource = matrix.gate_requested_resource(case)
    requested_resource_sha256 = matrix.gate_requested_resource_sha256(case)
    job_id = str(3_000 + matrix.build_matrix().index(case))
    req_tres = (
        f"cpu={requested_resource['cpus_per_task']},mem=128G,node=1,"
        f"gres/gpu={requested_resource['gpus']}"
    )
    scontrol_argv = [
        "scontrol", "--clusters=cluster-a", "show", "job", "-o", job_id
    ]
    scheduler_binding_body = {
        "job_id": job_id,
        "requested_resource": requested_resource,
        "requested_resource_sha256": requested_resource_sha256,
        "held_plan_sha256": "7" * 64,
        "scontrol_sha256": "8" * 64,
        "scontrol_query_returncode": 0,
        "scontrol_query_argv_sha256": hashlib.sha256(
            matrix._canonical(scontrol_argv)
        ).hexdigest(),
        "scontrol_query_stdout_sha256": hashlib.sha256(
            ("JobId=" + job_id + " allocation\n").encode()
        ).hexdigest(),
        "scontrol_query_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "slurm_job_partition": requested_resource["partition"],
        "slurm_cluster_name": "cluster-a",
        "slurm_receipt_cluster": "cluster-a",
        "slurm_qos": requested_resource["qos"],
        "slurm_time_limit": requested_resource["time_limit"],
        "slurm_num_nodes": requested_resource["nodes"],
        "slurm_num_cpus": requested_resource["cpus_per_task"],
        "slurm_cpus_per_task": requested_resource["cpus_per_task"],
        "slurm_memory_per_node_mb": requested_resource["memory_mb"],
        "slurm_req_tres": req_tres,
        "slurm_alloc_tres": req_tres,
        "slurm_tres_per_node": f"gres/gpu:{requested_resource['gpus']}",
        "cuda_visible_devices": [str(index) for index in range(case.world_size)],
        "visible_gpu_uuids": [device["uuid"] for device in devices],
    }
    scheduler_binding = {
        **scheduler_binding_body,
        "sha256": hashlib.sha256(matrix._canonical(scheduler_binding_body)).hexdigest(),
    }
    runtime = matrix.make_runtime_evidence(
        {
            "case_id": case.case_id,
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "source_manifest_sha256": source_manifest["sha256"],
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
            "scheduler_binding": scheduler_binding,
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
            "slurm_job_id": job_id,
            "requested_resource_sha256": requested_resource_sha256,
            "scheduler_binding_sha256": scheduler_binding["sha256"],
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
            "slurm_job_id": job_id,
            "requested_resource_sha256": requested_resource_sha256,
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


case_evidences = [case_evidence(case) for case in matrix.build_matrix()]
receipt_sha256 = "8" * 64
sacct_sha256 = "9" * 64
terminal_observations = []
for case, evidence in zip(matrix.build_matrix(), case_evidences):
    job_id = evidence["runtime_evidence"]["payload"]["scheduler_binding"]["job_id"]
    argv = [
        "sacct",
        "--clusters=cluster-a",
        "--noheader",
        "--allocations",
        f"--jobs={job_id}",
        "--format=JobIDRaw,Cluster,State,ExitCode,DerivedExitCode",
        "--parsable2",
    ]
    terminal_observations.append(
        {
            "case_id": case.case_id,
            "job_id": job_id,
            "receipt_cluster": "cluster-a",
            "observed_cluster": "cluster-a",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "derived_exit_code": "0:0",
            "query_returncode": 0,
            "query_argv_sha256": hashlib.sha256(matrix._canonical(argv)).hexdigest(),
            "query_stdout_sha256": hashlib.sha256(
                f"{job_id}|cluster-a|COMPLETED|0:0|0:0\n".encode()
            ).hexdigest(),
            "query_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
terminal_manifest = matrix.make_scheduler_terminal_manifest(
    terminal_observations,
    submission_receipt_sha256=receipt_sha256,
    sacct_sha256=sacct_sha256,
)
smoke = matrix.make_smoke_attestation(
    {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "environment_sha256": ENVIRONMENT,
        "source_manifest_sha256": source_manifest["sha256"],
        "repository_binding": REPOSITORY_BINDING,
        "gpu_model": GPU_MODEL,
        "driver_version": DRIVER,
        "checkpoint_ceiling_bytes": 1,
        "observed_checkpoint_max_bytes": 1,
        "submission_receipt_sha256": receipt_sha256,
        "scheduler_terminal_manifest": terminal_manifest,
        "scheduler_log_manifest": matrix.make_scheduler_log_manifest(
            [
                {
                    "name": name,
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "size": 0,
                }
                for name in sorted(
                    f"{case.case_id}.{suffix}"
                    for case in matrix.build_matrix()
                    for suffix in ("err", "out")
                )
            ],
            ceiling_bytes=1,
        ),
        "cases": case_evidences,
    }
)
SMOKE_PATH = EXTERNAL / "h100-smoke.json"
SMOKE_PATH.write_bytes(overlay_module.canonical_json_bytes(smoke) + b"\n")


def stage_values(source_sha=source_manifest["sha256"], seed=42):
    result = []
    for index, (stage, budget) in enumerate(overlay_module.STAGES, start=1):
        prior = str(index + 2) * 64
        architecture = str(index + 3) * 64
        optimizer = str(index + 4) * 64
        scientific = str(index + 5) * 64
        cohort, arms = overlay_module._expected_protocols(
            seed=seed,
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


for source_name, target_name in (
    ("fake_sbatch.sh", "sbatch"),
    ("fake_scontrol.sh", "scontrol"),
    ("fake_scancel.sh", "scancel"),
    ("fake_squeue.sh", "squeue"),
    ("fake_sacct.sh", "sacct"),
):
    shutil.copy2(REPO / "tests/fixtures" / source_name, FAKEBIN / target_name)
    (FAKEBIN / target_name).chmod(0o755)


def scheduler_commands():
    return {
        name: {
            "path": str(FAKEBIN / name),
            "sha256": hashlib.sha256((FAKEBIN / name).read_bytes()).hexdigest(),
        }
        for name in ("sbatch", "scontrol", "scancel", "squeue", "sacct")
    }


BASE = {
    "schema_version": 1,
    "run_policy": "fresh",
    "seed": 42,
    "study_id": "study-a-seed42",
    "artifact_root": "",
    "source": {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "manifest_path": str(SOURCE_PATH),
        "manifest_sha256": source_manifest["sha256"],
        "environment_sha256": ENVIRONMENT,
        "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
        "candidate_ref": "refs/heads/codex/position-identity-v1",
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
        "protocol_metadata_allowance_bytes": 3_000_000,
    },
    "runtime": {
        "python": str(Path(sys.executable).resolve()),
        "git": str(FAKE_GIT),
        "git_sha256": FAKE_GIT_SHA256,
        "nvidia_smi": "/usr/bin/true",
        "job_work_root": "",
    },
    "scheduler": {
        "partition": "h100",
        "qos": "long",
        "cpus_per_task": 64,
        "memory_mb": 131_072,
        "gpus_per_job": 1,
        "time_limit_by_stage": {
            "stage1": "14-00:00:00",
            "stage2": "3-00:00:00",
            "stage3": "1-00:00:00",
        },
        "commands": scheduler_commands(),
    },
    "stages": stage_values(),
}
JOB_WORK_ROOT = None
for candidate_parent in (Path("/dev/shm"), Path("/tmp"), REPO.parent):
    if (
        candidate_parent.is_dir()
        and os.access(candidate_parent, os.W_OK | os.X_OK)
        and candidate_parent.stat().st_dev != ARTIFACT_PARENT.stat().st_dev
    ):
        JOB_WORK_ROOT = Path(
            tempfile.mkdtemp(prefix="tabicl-task4-job-work-", dir=candidate_parent)
        )
        break
if JOB_WORK_ROOT is None:
    raise RuntimeError("fake-Slurm test requires a writable second filesystem")
atexit.register(shutil.rmtree, JOB_WORK_ROOT, ignore_errors=True)
BASE["runtime"]["job_work_root"] = str(JOB_WORK_ROOT)

CONTROL_FILES = {
    "calls.log",
    "sbatch_count",
    "squeue_count",
    "sacct_count",
    "release_count",
    "fail_sbatch_at",
    "empty_at",
    "malformed_at",
    "cluster_suffix_at",
    "duplicate_at",
    "fail_release_at",
    "delete_scancel_after_release_at",
    "fail_cancel_ids",
    "signal_parent_on_cancel",
    "ledger_path",
    "receipt_path",
    "precreate_receipt_at",
    "jobs.tsv",
    "jobs.tsv.next",
    "accounting.tsv",
    "response_loss_at",
    "fail_squeue_at",
    "fail_sacct_at",
    "squeue_banner",
    "hide_squeue_until",
    "hide_sacct_until",
    "malformed_exact_name_until",
}


def reset_fake(controls=None):
    if not (FAKEBIN / "scancel").exists() and (FAKEBIN / "scancel.missing").exists():
        (FAKEBIN / "scancel.missing").rename(FAKEBIN / "scancel")
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


def study_name(name, seed=42):
    return f"{name}-seed{seed}"


def artifact_dir(name, seed=42):
    return ARTIFACT_PARENT / study_name(name, seed)


def invoke(
    name,
    *,
    mutate=None,
    controls=None,
    fault=None,
    journal_fault=None,
    commit_fault=None,
    post_commit_signal=False,
    scheduler_path=None,
):
    reset_fake(controls)
    value = copy.deepcopy(BASE)
    value["study_id"] = study_name(name, value["seed"])
    value["artifact_root"] = str(artifact_dir(name, value["seed"]))
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
        "FORMAL_SEED": "44",
        "EVIL_AMBIENT": "must-not-reach-job",
    }
    if fault:
        env["FORMAL_FAULT_LEDGER_STAGE"] = fault
    if journal_fault:
        env["FORMAL_FAULT_JOURNAL_SEAL_STAGE"] = journal_fault
    if commit_fault:
        env["FORMAL_FAULT_COMMIT_STAGE"] = commit_fault
    if post_commit_signal:
        env["FORMAL_TEST_SIGNAL_AFTER_COMMIT"] = "1"
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


def command_events(prefix):
    return [line for line in calls() if line.startswith(prefix + " ")]


def transaction_journal(name, seed=42):
    paths = list(artifact_dir(name, seed).glob("transaction-journal-*.jsonl"))
    require(len(paths) == 1, paths)
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    require(records, paths[0])
    transaction_id = records[0]["transaction_id"]
    previous = None
    for sequence, record in enumerate(records):
        require(record["sequence"] == sequence, record)
        require(record["transaction_id"] == transaction_id, record)
        require(record["previous_sha256"] == previous, record)
        body = {key: value for key, value in record.items() if key != "sha256"}
        require(
            record["sha256"]
            == hashlib.sha256(overlay_module.canonical_json_bytes(body)).hexdigest(),
            record,
        )
        previous = record["sha256"]
    require(stat.S_IMODE(paths[0].stat().st_mode) == 0o400, paths[0].stat())
    return records


def rollback_record(name, seed=42):
    paths = list(artifact_dir(name, seed).glob("rollback-incomplete-*.json"))
    require(len(paths) == 1, paths)
    return json.loads(paths[0].read_text())


success = invoke("success")
require(success.returncode == 0, f"success failed: {success.stderr}")
require(
    all((artifact_dir("success") / "arms" / arm).is_dir() for arm in ("rope", "temporary", "none")),
    "fresh controller did not precreate physical arm parents",
)
require(
    (artifact_dir("success") / "capacity.called").read_text() == "passed\n",
    "held transaction did not repeat capacity on the artifact filesystem",
)
require(
    not any((artifact_dir("success") / "arms" / arm / "stage1").exists() for arm in ("rope", "temporary", "none")),
    "submit controller created a stage namespace before its job",
)
events = calls()
submissions = command_events("sbatch")
queries = command_events("squeue")
releases = command_events("scontrol")
require(len(events) == 36, events)
require(len(submissions) == 9 and len(queries) == 18 and len(releases) == 9, events)
require(all("<release>" in line for line in releases), releases)
for line in submissions:
    require("<--hold>" in line and "<--gres=gpu:1>" in line, line)
    require("<--partition=h100>" in line and "<--qos=long>" in line, line)
    require("<--cpus-per-task=64>" in line and "<--mem=131072M>" in line, line)
    require(f"<--chdir={artifact_dir('success')}>" in line, line)
    require(f"<--output={artifact_dir('success')}/scheduler-logs/" in line, line)
    require(f"<--error={artifact_dir('success')}/scheduler-logs/" in line, line)
    require("<--open-mode=truncate>" in line, line)
    require("FORMAL_SEED=42" in line, line)
    require(f"FORMAL_SUBMISSION_EXACT_ROOT={EXACT}" in line, line)
    require("--export=ALL" not in line and "gpu:2" not in line, line)
for index, line in enumerate(submissions):
    expected_stage = ("stage1", "stage2", "stage3")[index // 3]
    expected_time = BASE["scheduler"]["time_limit_by_stage"][expected_stage]
    require(f"<--time={expected_time}>" in line, line)
    require(f"FORMAL_TIME_LIMIT={expected_time}" in line, line)
require(all("--dependency=" not in line for line in submissions[:3]), submissions[:3])
require("--dependency=afterok:1001" in submissions[3] and "--kill-on-invalid-dep=yes" in submissions[3], submissions[3])
require("--dependency=afterok:1002" in submissions[4], submissions[4])
require("--dependency=afterok:1003" in submissions[5], submissions[5])
require("--dependency=afterok:1004" in submissions[6], submissions[6])
require("--dependency=afterok:1005" in submissions[7], submissions[7])
require("--dependency=afterok:1006" in submissions[8], submissions[8])
require("-rope-s1>" in submissions[0], submissions[0])
require("-temporary-s1>" in submissions[1], submissions[1])
require("-none-s1>" in submissions[2], submissions[2])
ledger = json.loads((artifact_dir("success") / "transaction-ledger.json").read_text())
require(len(ledger["payload"]["entries"]) == 9, ledger)
require(all("checkpoint_sha256" not in entry for entry in ledger["payload"]["entries"]), ledger)
require(all(entry["np_seed"] == entry["torch_seed"] == entry["identity_rng_seed"] == 42 for entry in ledger["payload"]["entries"]), ledger)
receipt = json.loads((artifact_dir("success") / "submission-receipt.json").read_text())
require(receipt["kind"] == "held_submission_receipt", receipt)
require(receipt["payload"]["jobs_held_at_publication"] is True, receipt)
require(receipt["payload"]["job_ids"] == [str(1000 + i) for i in range(1, 10)], receipt)
require(receipt["payload"]["seed"] == 42, receipt)
require(receipt["payload"]["run_log_ceiling_bytes"] == 1_000, receipt)
require(receipt["payload"]["manifest_ceiling_bytes"] == 1_000, receipt)
require(receipt["payload"]["runtime_completion_ceiling_bytes"] == 65_536, receipt)
require(
    receipt["payload"]["terminal_log_attestation_ceiling_bytes"] == 131_072,
    receipt,
)
require(
    receipt["payload"]["terminal_log_attestation_path"]
    == str(artifact_dir("success") / "terminal-scheduler-logs.json"),
    receipt,
)
require(
    receipt["payload"]["transaction_commit_path"]
    == str(artifact_dir("success") / "transaction-committed.json"),
    receipt,
)
commit = json.loads(
    (artifact_dir("success") / "transaction-committed.json").read_text()
)
require(commit["kind"] == "formal_submission_commit", commit)
require(commit["payload"]["submission_receipt_sha256"] == receipt["sha256"], commit)
require(commit["payload"]["job_ids"] == receipt["payload"]["job_ids"], commit)
require(
    all(
        receipt["payload"]["transaction_id"] in job["job_name"]
        for job in receipt["payload"]["jobs"]
    ),
    receipt,
)
require(
    all("FORMAL_RUNTIME_COMPLETION_CEILING_BYTES=65536" in line for line in submissions),
    submissions,
)
require(receipt["payload"]["scheduler"] == {
    "partition": "h100",
    "qos": "long",
    "cpus_per_task": 64,
    "memory_mb": 131_072,
    "gpus_per_job": 1,
    "time_limit_by_stage": BASE["scheduler"]["time_limit_by_stage"],
    "sacct_path": BASE["scheduler"]["commands"]["sacct"]["path"],
    "sacct_sha256": BASE["scheduler"]["commands"]["sacct"]["sha256"],
    "scontrol_sha256": BASE["scheduler"]["commands"]["scontrol"]["sha256"],
}, receipt)
require(
    all(
        job["time_limit"]
        == BASE["scheduler"]["time_limit_by_stage"][job["stage"]]
        for job in receipt["payload"]["jobs"]
    ),
    receipt,
)
success_journal = transaction_journal("success")
require(success_journal[-1]["event"] == "transaction_released", success_journal[-1])
require(success_journal[0]["transaction_id"] == receipt["payload"]["transaction_id"], receipt)

post_commit_signal = invoke("post-commit-signal", post_commit_signal=True)
require(
    post_commit_signal.returncode == 0,
    f"post-commit signal changed successful CLI status: {post_commit_signal.stderr}",
)
require(
    (artifact_dir("post-commit-signal") / "transaction-committed.json").is_file(),
    "post-commit signal lost the authoritative commit marker",
)
require(
    not command_events("scancel"),
    f"post-commit signal cancelled a committed cohort: {calls()}",
)

for selected_seed in (43, 44):
    scenario = f"success-{selected_seed}"

    def use_selected_seed(value, selected_seed=selected_seed, scenario=scenario):
        value["seed"] = selected_seed
        value["study_id"] = study_name(scenario, selected_seed)
        value["artifact_root"] = str(artifact_dir(scenario, selected_seed))
        value["stages"] = stage_values(seed=selected_seed)

    seeded = invoke(scenario, mutate=use_selected_seed)
    require(seeded.returncode == 0, f"seed {selected_seed} failed: {seeded.stderr}")
    require(
        all(f"FORMAL_SEED={selected_seed}" in line for line in command_events("sbatch")), calls()
    )
    seeded_ledger = json.loads(
        (artifact_dir(scenario, selected_seed) / "transaction-ledger.json").read_text()
    )
    require(
        all(
            entry["np_seed"]
            == entry["torch_seed"]
            == entry["identity_rng_seed"]
            == selected_seed
            for entry in seeded_ledger["payload"]["entries"]
        ),
        seeded_ledger,
    )

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

for control in ("response_loss_at", "empty_at", "malformed_at", "duplicate_at"):
    scenario = f"reconciled-{control}"
    result = invoke(scenario, controls={control: 3})
    require(result.returncode == 0, f"{control} was not reconciled: {result.stderr}")
    require(len(command_events("sbatch")) == 9, calls())
    require(not command_events("scancel"), calls())
    reconciled = transaction_journal(scenario)
    event_names = [record["event"] for record in reconciled]
    expected_event = (
        "submit_response_id_reconciled"
        if control == "duplicate_at"
        else "submit_response_untrusted"
    )
    require(expected_event in event_names, event_names)

banner = invoke(
    "squeue-banner",
    controls={"squeue_banner": "Welcome back, worker!"},
)
require(banner.returncode == 0, f"unrelated squeue banner was fatal: {banner.stderr}")
require(transaction_journal("squeue-banner")[-1]["event"] == "transaction_released", calls())

for control in ("hide_squeue_until", "malformed_exact_name_until"):
    scenario = f"known-id-rollback-{control}"
    result = invoke(scenario, controls={control: 3})
    require(result.returncode != 0, f"{control} unexpectedly released a job")
    require(command_events("scancel") == ["scancel <1001>"], calls())
    require(not list(artifact_dir(scenario).glob("rollback-incomplete-*.json")), calls())
    known_id_journal = transaction_journal(scenario)
    require(
        "submit_response_known_id_unverified"
        in [record["event"] for record in known_id_journal],
        known_id_journal,
    )
    require(known_id_journal[-1]["event"] == "transaction_rolled_back", known_id_journal)

unreconciled_cancel = invoke(
    "cancel-accounting-hidden",
    controls={"fail_sbatch_at": 2, "hide_sacct_until": 3},
)
require(unreconciled_cancel.returncode != 0, "unreconciled cancellation reported success")
unreconciled_recovery = rollback_record("cancel-accounting-hidden")
require(
    unreconciled_recovery["payload"]["remaining_job_ids"] == ["1001"],
    unreconciled_recovery,
)
require(len(command_events("sacct")) == 3, calls())

# Enter rollback only after all nine scheduler identities are known.  An
# untrusted sbatch response with no matching scheduler row must instead retain
# an unresolved-name recovery record, so it cannot isolate signal handling.
second_signal = invoke(
    "second-signal-during-rollback",
    controls={"signal_parent_on_cancel": "yes"},
    fault="write",
)
require(second_signal.returncode != 0, "rollback with a second signal reported success")
require(
    command_events("scancel")
    == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
    calls(),
)
require(
    not list(artifact_dir("second-signal-during-rollback").glob("rollback-incomplete-*.json")),
    calls(),
)
second_signal_journal = transaction_journal("second-signal-during-rollback")
require(
    second_signal_journal[-1]["event"] == "transaction_rolled_back",
    second_signal_journal,
)
second_signal_cancellations = [
    record
    for record in second_signal_journal
    if record["event"] == "cancel_observed"
]
require(len(second_signal_cancellations) == 9, second_signal_journal)
require(
    all(
        record["payload"]["verified"] is True
        and record["payload"]["state"] == "CANCELLED"
        and record["payload"]["source"] == "sacct"
        for record in second_signal_cancellations
    ),
    second_signal_cancellations,
)

cluster_suffix = invoke(
    "cluster-suffix-then-fail",
    controls={"cluster_suffix_at": 3, "fail_sbatch_at": 4},
)
require(cluster_suffix.returncode != 0, "post-suffix sbatch failure reported success")
require(
    [line for line in calls() if line.startswith("scancel ")]
    == [
        "scancel <--clusters=cluster-a> <1003>",
        "scancel <1002>",
        "scancel <1001>",
    ],
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
release_recovery = rollback_record("release-cancel-fail")
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
recovery = rollback_record("cancel-fail")
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

for fault in ("fchmod", "fsync", "dir_fsync"):
    scenario = f"journal-seal-{fault}"
    result = invoke(scenario, journal_fault=fault)
    require(result.returncode != 0, f"journal seal {fault} fault reported success")
    require(len(command_events("scontrol")) == 9, calls())
    require(
        [line for line in calls() if line.startswith("scancel ")]
        == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
        calls(),
    )
    require(
        not (FAKEBIN / "jobs.tsv").read_text().strip(),
        "journal seal failure left a live fake job",
    )
    require(
        not list(artifact_dir(scenario).glob("rollback-incomplete-*.json")),
        "fully reconciled journal rollback published incomplete recovery",
    )

for fault in ("write", "fsync", "rename", "dir_fsync"):
    scenario = f"commit-{fault}"
    result = invoke(scenario, commit_fault=fault)
    require(result.returncode != 0, f"commit {fault} fault reported success")
    require(
        not (artifact_dir(scenario) / "transaction-committed.json").exists(),
        "failed commit publication left an acceptable success marker",
    )
    require(
        [line for line in calls() if line.startswith("scancel ")]
        == [f"scancel <{job}>" for job in range(1009, 1000, -1)],
        calls(),
    )

missing_scancel = invoke(
    "missing-scancel-after-release",
    controls={"delete_scancel_after_release_at": 1},
    commit_fault="write",
)
require(missing_scancel.returncode != 0, "missing scancel reported success")
missing_scancel_recovery = rollback_record("missing-scancel-after-release")
require(
    missing_scancel_recovery["payload"]["remaining_job_ids"]
    == [str(job) for job in range(1009, 1000, -1)],
    missing_scancel_recovery,
)

capacity_name = "capacity-fail"
reset_fake()
(ARTIFACT_PARENT / "capacity.fail").write_text("fail\n")
value = copy.deepcopy(BASE)
value["study_id"] = study_name(capacity_name)
value["artifact_root"] = str(artifact_dir(capacity_name))
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
preflight_cases.append(("missing-seed", lambda value: value.pop("seed")))
preflight_cases.append(("wrong-seed", lambda value: value.update(seed=41)))
preflight_cases.append(("wrong-seed-namespace", lambda value: value.update(study_id="wrong-seed43")))
preflight_cases.append(("wrong-gpu", lambda value: value["scheduler"].update(gpus_per_job=2)))
preflight_cases.append(("wrong-partition", lambda value: value["scheduler"].update(partition="a10")))
preflight_cases.append(("wrong-partition-case", lambda value: value["scheduler"].update(partition="H100")))
preflight_cases.append(("wrong-qos", lambda value: value["scheduler"].update(qos="short")))
preflight_cases.append(("wrong-cpus", lambda value: value["scheduler"].update(cpus_per_task=63)))
preflight_cases.append(("wrong-memory", lambda value: value["scheduler"].update(memory_mb=131_071)))
preflight_cases.append(
    (
        "missing-stage-time",
        lambda value: value["scheduler"]["time_limit_by_stage"].pop("stage3"),
    )
)
preflight_cases.append(
    (
        "noncanonical-stage-time",
        lambda value: value["scheduler"]["time_limit_by_stage"].update(
            stage1="336:00:00"
        ),
    )
)
preflight_cases.append(("missing-smoke", lambda value: value["smoke"].update(attestation_path=str(EXTERNAL / "missing.json"))))
preflight_cases.append(("smoke-ceiling-mismatch", lambda value: value["capacity"].update(checkpoint_ceiling_bytes=2)))
preflight_cases.append(("mismatched-commit", lambda value: value["source"].update(commit_sha="f" * 40)))
preflight_cases.append(("missing-arm", lambda value: value["stages"][0]["arm_protocol_sha256"].pop("none")))
for name, mutation in preflight_cases:
    result = invoke(name, mutate=mutation)
    require(result.returncode != 0 and not calls(), f"{name} reached scheduler: {calls()}")


def mismatched_seed_binding(value):
    value["seed"] = 43
    value["study_id"] = study_name("mismatched-seed-binding", 43)
    value["artifact_root"] = str(artifact_dir("mismatched-seed-binding", 43))
    # Stage protocol digests remain bound to seed 42 and must fail closed.


result = invoke("mismatched-seed-binding", mutate=mismatched_seed_binding)
require(result.returncode != 0 and not calls(), f"seed mismatch reached scheduler: {calls()}")

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
        not artifact_dir("missing-scheduler").exists(),
        "scheduler preflight failure consumed the fresh namespace",
    )
finally:
    (FAKEBIN / "scontrol.disabled").rename(FAKEBIN / "scontrol")

relative_scheduler_path = invoke(
    "relative-scheduler-path", scheduler_path=f"{FAKEBIN}:relative"
)
require(relative_scheduler_path.returncode == 0, relative_scheduler_path.stderr)
require(len(command_events("sbatch")) == 9, calls())
require(len(command_events("scontrol")) == 9, calls())
require(transaction_journal("relative-scheduler-path")[-1]["event"] == "transaction_released", calls())

artifact_alias = TEMP / "artifact-parent-alias"
artifact_alias.symlink_to(ARTIFACT_PARENT, target_is_directory=True)
symlinked_artifact = invoke(
    "symlinked-artifact-parent",
    mutate=lambda value: value.update(
        artifact_root=str(artifact_alias / study_name("symlinked-artifact-parent"))
    ),
)
require(symlinked_artifact.returncode != 0 and not calls(), calls())
require(
    not artifact_dir("symlinked-artifact-parent").exists(),
    "symlinked artifact parent escaped into the physical target",
)

job_work_alias = TEMP / "job-work-alias"
job_work_alias.symlink_to(JOB_WORK_ROOT, target_is_directory=True)
symlinked_job_work = invoke(
    "symlinked-job-work",
    mutate=lambda value: value["runtime"].update(job_work_root=str(job_work_alias)),
)
require(symlinked_job_work.returncode != 0 and not calls(), calls())
require(
    not artifact_dir("symlinked-job-work").exists(),
    "symlinked job work root consumed the fresh namespace",
)

same_device_work = TEMP / "same-device-job-work"
same_device_work.mkdir()
same_device_job_work = invoke(
    "same-device-job-work",
    mutate=lambda value: value["runtime"].update(job_work_root=str(same_device_work)),
)
require(same_device_job_work.returncode != 0 and not calls(), calls())
require(
    "different filesystem" in same_device_job_work.stderr,
    same_device_job_work.stderr,
)
require(
    not artifact_dir("same-device-job-work").exists(),
    "same-device job work root consumed the fresh namespace",
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
