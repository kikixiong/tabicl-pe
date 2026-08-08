from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
CONTROLLER_FILES = (
    "scripts/submit_h100_identity_validation.py",
    "scripts/submit_h100_identity_validation.sh",
    "scripts/run_h100_identity_validation.py",
    "scripts/run_h100_identity_maxseq_smoke.sh",
    "scripts/run_slurm_h100_identity_case.sh",
    "scripts/formal_run_with_gpu_monitor.sh",
    "scripts/run_with_durable_log.sh",
    "scripts/exec_digest_bound_git.py",
    "scripts/exec_digest_bound_nvidia_smi.py",
    "scripts/slurm_h100_identity_maxseq_smoke.sh",
    "scripts/slurm_h100_identity_nccl_smoke.sh",
    "scripts/verify_filesystem_isolation.py",
    "scripts/verify_formal_environment.py",
    "scripts/verify_formal_environment_transaction.py",
    "scripts/verify_git_repository.py",
    "scripts/verify_runtime_source.py",
)


def load_submit_module():
    path = ROOT / "scripts/submit_h100_identity_validation.py"
    spec = importlib.util.spec_from_file_location(
        "_gate_submit_direct_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def make_manifest(kind: str, payload: dict) -> dict:
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    return {**body, "sha256": hashlib.sha256(canonical(body)).hexdigest()}


def write_environment_transaction(fixture, name: str) -> tuple[dict, dict]:
    root = fixture["external"] / "environments" / name
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "visible_distribution_multiset": [],
        "effective_formal_runtime_distributions": [],
    }
    installed_sha256 = hashlib.sha256(canonical(fingerprint)).hexdigest()
    base_environment = {
        "environment_fingerprint_schema_version": 2,
        "installed_distributions_sha256": installed_sha256,
    }
    one = make_manifest(
        "environment", {**base_environment, "visible_cuda_device_count": 1}
    )
    two = make_manifest(
        "environment", {**base_environment, "visible_cuda_device_count": 2}
    )
    environment_map = {"1": one["sha256"], "2": two["sha256"]}
    inventory = make_manifest(
        "formal_environment_inventory",
        {
            "source_commit_sha": fixture["commit"],
            "source_tree_sha": fixture["tree"],
            "git_sha256": fixture["fake_git_sha256"],
            "capture_visible_cuda_device_count": 0,
            "installed_distributions_sha256": installed_sha256,
            "environment_sha256_by_world_size": environment_map,
            "fingerprint_preimage": fingerprint,
        },
    )
    values = (
        ("one_gpu", "one-gpu.json", one),
        ("two_gpu", "two-gpu.json", two),
        ("inventory", "inventory.json", inventory),
    )
    outputs = []
    for role, filename, value in values:
        raw = canonical(value) + b"\n"
        (root / filename).write_bytes(raw)
        outputs.append(
            {
                "role": role,
                "name": filename,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    descriptor = {
        "schema_version": 1,
        "source_commit_sha": fixture["commit"],
        "source_tree_sha": fixture["tree"],
        "outputs": outputs,
    }
    transaction_sha256 = hashlib.sha256(canonical(descriptor) + b"\n").hexdigest()
    completion = {
        **descriptor,
        "kind": "formal_environment_generation_completion",
        "transaction_sha256": transaction_sha256,
    }
    completion_raw = canonical(completion) + b"\n"
    completion_path = root / "environment-complete.json"
    completion_path.write_bytes(completion_raw)
    return environment_map, {
        "completion_path": str(completion_path),
        "completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "transaction_sha256": transaction_sha256,
        "inventory_sha256": inventory["sha256"],
        "completion_max_bytes": 1 << 20,
        "manifest_max_bytes": 1 << 20,
        "inventory_max_bytes": 128 << 20,
    }


def run(argv, **kwargs):
    return subprocess.run(argv, text=True, check=True, **kwargs)


def write_executable(path: Path, body: str) -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        f"FAKE_EXECUTABLE_PATH = {str(path)!r}\n"
        f"FAKE_EXECUTABLE_NAME = {path.name!r}\n"
        + body
    )
    path.chmod(0o755)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    temp = tmp_path_factory.mktemp("h100-gate-submit")
    exact = temp / "exact"
    external = temp / "external"
    fakebin = temp / "fakebin"
    case_work = temp / "case-work"
    for path in (exact / "scripts", exact / "src/tabicl", external, fakebin, case_work):
        path.mkdir(parents=True, exist_ok=True)
    for relative in CONTROLLER_FILES:
        target = exact / relative
        shutil.copy2(ROOT / relative, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    (exact / "src/tabicl/__init__.py").write_text("__version__ = 'gate-fixture'\n")

    run(["/usr/bin/git", "init", "-q", str(exact)])
    run(["/usr/bin/git", "-C", str(exact), "config", "user.name", "fixture"])
    run(
        [
            "/usr/bin/git",
            "-C",
            str(exact),
            "config",
            "user.email",
            "fixture@example.invalid",
        ]
    )
    run(["/usr/bin/git", "-C", str(exact), "add", "scripts", "src"])
    run(["/usr/bin/git", "-C", str(exact), "commit", "-q", "-m", "fixture"])
    commit = run(
        ["/usr/bin/git", "-C", str(exact), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree = run(
        ["/usr/bin/git", "-C", str(exact), "rev-parse", "HEAD^{tree}"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    run(["/usr/bin/git", "-C", str(exact), "checkout", "-q", "--detach", commit])

    fake_git = fakebin / "git"
    write_executable(
        fake_git,
        f"""
import os
import sys
from pathlib import Path
state = Path(FAKE_EXECUTABLE_PATH).parent
is_remote_query = sys.argv[1:] == [
    "ls-remote", "--refs",
    "https://github.com/kikixiong/tabicl-pe.git",
    "refs/heads/codex/position-identity-v1",
]
if not is_remote_query:
    local_count_path = state / "git-local-count"
    local_count = int(local_count_path.read_text()) + 1 if local_count_path.exists() else 1
    local_count_path.write_text(str(local_count))
    if (state / "swap-git-path-on-first-local").exists() and local_count == 1:
        current = Path(FAKE_EXECUTABLE_PATH)
        saved = current.with_name(current.name + ".verified-inode")
        current.rename(saved)
        current.write_text(
            "#!" + sys.executable + "\\n"
            + "from pathlib import Path\\n"
            + "Path(" + repr(str(state / "replacement-executed")) + ").write_text('git')\\n"
            + "raise SystemExit(97)\\n"
        )
        current.chmod(0o755)
    if (state / "swap-git-path-on-first-local").exists() and local_count == 4:
        current = Path(FAKE_EXECUTABLE_PATH)
        saved = current.with_name(current.name + ".verified-inode")
        current.unlink()
        saved.rename(current)
if is_remote_query:
    count_path = state / "git-query-count"
    count = int(count_path.read_text()) + 1 if count_path.exists() else 1
    count_path.write_text(str(count))
    if (state / "git-ref-drift-after-preflight").exists() and count >= 2:
        print("{commit}\\trefs/heads/other")
    else:
        print("{commit}\\trefs/heads/codex/position-identity-v1")
else:
    os.execv("/usr/bin/git", ["/usr/bin/git", *sys.argv[1:]])
""",
    )
    fake_git_sha256 = hashlib.sha256(fake_git.read_bytes()).hexdigest()

    entries = []
    tracked = run(
        ["/usr/bin/git", "-C", str(exact), "ls-files", "-s"],
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    for line in tracked:
        metadata, relative = line.split("\t", 1)
        raw = (exact / relative).read_bytes()
        entries.append(
            {
                "path": relative,
                "mode": metadata.split()[0],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    source_body = {
        "schema_version": 1,
        "kind": "source",
        "payload": {
            "commit_sha": commit,
            "tree_sha": tree,
            "code_roots": ["scripts", "src/tabicl"],
            "entries": sorted(entries, key=lambda item: item["path"]),
        },
    }
    source = {
        **source_body,
        "sha256": hashlib.sha256(canonical(source_body)).hexdigest(),
    }
    source_path = external / "source.json"
    source_path.write_bytes(canonical(source) + b"\n")

    common = """
import json
import hashlib
from pathlib import Path
import sys

state = Path(FAKE_EXECUTABLE_PATH).parent
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"cmd": FAKE_EXECUTABLE_NAME, "argv": sys.argv[1:]}, separators=(",", ":")) + "\\n")

def swap_scheduler_paths():
    for name in ("sbatch", "scontrol", "scancel", "squeue", "sacct"):
        current = state / name
        saved = state / (name + ".verified-inode")
        current.rename(saved)
        current.write_text(
            "#!" + sys.executable + "\\n"
            + "from pathlib import Path\\n"
            + "Path(" + repr(str(state / "replacement-executed")) + ").write_text(" + repr(name) + ")\\n"
            + "raise SystemExit(97)\\n"
        )
        current.chmod(0o755)

def restore_scheduler_paths():
    for name in ("sbatch", "scontrol", "scancel", "squeue", "sacct"):
        current = state / name
        saved = state / (name + ".verified-inode")
        if saved.exists():
            current.unlink(missing_ok=True)
            saved.rename(current)
"""
    write_executable(
        fakebin / "sbatch",
        common
        + """
count_path = state / "sbatch-count"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
wrapper_swap = state / "swap-spool-wrappers-on-first-sbatch"
if wrapper_swap.exists() and count == 1:
    exact_root = Path(wrapper_swap.read_text())
    for filename in (
        "slurm_h100_identity_maxseq_smoke.sh",
        "slurm_h100_identity_nccl_smoke.sh",
    ):
        current = exact_root / "scripts" / filename
        saved = current.with_name(filename + ".verified-inode")
        current.rename(saved)
        current.write_text(
            "#!/bin/bash\\n"
            + "printf malicious > "
            + repr(str(state / "replacement-wrapper-executed"))
            + "\\nexit 97\\n"
        )
        current.chmod(0o755)
(state / f"spooled-wrapper-{count}.sha256").write_text(
    hashlib.sha256(Path(sys.argv[-1]).read_bytes()).hexdigest()
)
if (state / "swap-scheduler-paths-on-first-sbatch").exists() and count == 1:
    swap_scheduler_paths()
fail = state / "fail-sbatch-at"
if fail.exists() and count == int(fail.read_text()):
    raise SystemExit(41)
job_id = str(1000 + count)
job_name_arg = next((item for item in sys.argv[1:] if item.startswith('--job-name=')), None)
if job_name_arg is None:
    raise SystemExit(92)
job_name = job_name_arg.split('=', 1)[1]
(state / f"job-{job_id}.json").write_text(json.dumps({"job_id": job_id, "job_name": job_name}))
malformed = state / "malformed-at"
if malformed.exists() and count == int(malformed.read_text()):
    print("not-a-job")
elif (state / "signal-sbatch-at").exists() and count == int((state / "signal-sbatch-at").read_text()):
    import os, signal, time
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(5)
elif (state / "signal-hup-sbatch-at").exists() and count == int((state / "signal-hup-sbatch-at").read_text()):
    import os, signal, time
    os.kill(os.getppid(), signal.SIGHUP)
    time.sleep(5)
elif (state / "sleep-sbatch-at").exists() and count == int((state / "sleep-sbatch-at").read_text()):
    import time
    time.sleep(60)
elif (state / "response-loss-nonzero-at").exists() and count == int((state / "response-loss-nonzero-at").read_text()):
    raise SystemExit(41)
elif (state / "duplicate-at").exists() and count == int((state / "duplicate-at").read_text()):
    print(1001)
elif (state / "cluster-suffix").exists():
    print(f"{job_id};cluster-a")
else:
    print(job_id)
""",
    )
    write_executable(
        fakebin / "scontrol",
        common
        + """
count_path = state / "release-count"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
fail = state / "fail-release-at"
if fail.exists() and count == int(fail.read_text()):
    raise SystemExit(42)
if len(sys.argv) < 3 or sys.argv[-2] != "release" or not sys.argv[-1].isdigit():
    raise SystemExit(91)
job_id = sys.argv[-1]
(state / f"released-{job_id}").write_text("1")
delete_cancel = state / "delete-scancel-after-release-at"
if delete_cancel.exists() and count == int(delete_cancel.read_text()):
    cancel = state / "scancel"
    backup = state / "scancel.deleted-backup"
    backup.write_bytes(cancel.read_bytes())
    backup.chmod(cancel.stat().st_mode)
    cancel.unlink()
    (state / "scancel-path-deleted").write_text("1")
collision = state / "precreate-receipt-at-release"
if collision.exists() and count == int(collision.read_text()):
    Path((state / "receipt-path").read_text()).write_text('{"invalid":true}' + chr(10))
if (state / "swap-scheduler-paths-on-first-sbatch").exists() and count == 12:
    restore_scheduler_paths()
""",
    )
    write_executable(
        fakebin / "scancel",
        common
        + """
if len(sys.argv) < 2 or not sys.argv[-1].isdigit():
    raise SystemExit(91)
failed = state / "fail-cancel-ids"
if failed.exists() and sys.argv[-1] in failed.read_text().splitlines():
    raise SystemExit(43)
sleeping = state / "sleep-cancel-ids"
if sleeping.exists() and sys.argv[-1] in sleeping.read_text().splitlines():
    import time
    time.sleep(60)
(state / f"cancelled-{sys.argv[-1]}").write_text("1")
if sys.argv[-1] == "1001":
    backup = state / "scancel.deleted-backup"
    if backup.exists():
        current = state / "scancel"
        current.write_bytes(backup.read_bytes())
        current.chmod(backup.stat().st_mode)
        backup.unlink()
    restore_scheduler_paths()
""",
    )
    write_executable(
        fakebin / "squeue",
        common
        + """
job_arg = next((item for item in sys.argv[1:] if item.startswith('--jobs=')), None)
name_arg = next((item for item in sys.argv[1:] if item.startswith('--name=')), None)
if (state / "squeue-banner").exists():
    print("Welcome back, fixture!")
if job_arg is not None:
    job_id = job_arg.split('=', 1)[1]
    if (state / "squeue-hide-id").exists():
        raise SystemExit(0)
    if (state / f"cancelled-{job_id}").exists():
        if (state / "cancel-stays-active").exists():
            print(f"{job_id}|RUNNING|None")
        elif (state / "cancel-delay-once").exists() and not (state / f"cancel-seen-{job_id}").exists():
            (state / f"cancel-seen-{job_id}").write_text("1")
            print(f"{job_id}|RUNNING|None")
        else:
            print(f"{job_id}|CANCELLED|Cancelled")
    elif (state / f"released-{job_id}").exists():
        if (state / "release-stays-held").exists():
            print(f"{job_id}|PENDING|JobHeldUser")
        elif (state / "release-delay-once").exists() and not (state / f"release-seen-{job_id}").exists():
            (state / f"release-seen-{job_id}").write_text("1")
            print(f"{job_id}|PENDING|JobHeldUser")
        else:
            print(f"{job_id}|PENDING|Resources")
    else:
        print(f"{job_id}|PENDING|JobHeldUser")
elif name_arg is not None:
    if (state / "squeue-hide-name").exists():
        raise SystemExit(0)
    job_name = name_arg.split('=', 1)[1]
    matches = []
    for path in state.glob('job-*.json'):
        item = json.loads(path.read_text())
        if item['job_name'] == job_name:
            matches.append(item)
    if (state / "ambiguous-name").exists() and matches:
        matches.append({"job_id": "9001", "job_name": job_name})
    for item in sorted(matches, key=lambda value: int(value['job_id'])):
        print(f"{item['job_id']}|{item['job_name']}|PENDING|JobHeldUser")
else:
    raise SystemExit(91)
""",
    )
    write_executable(
        fakebin / "sacct",
        common
        + """
job_arg = next((item for item in sys.argv[1:] if item.startswith('--jobs=')), None)
name_arg = next((item for item in sys.argv[1:] if item.startswith('--name=')), None)
if job_arg is not None:
    job_id = job_arg.split('=', 1)[1]
    if (state / f"cancelled-{job_id}").exists():
        print(f"{job_id}|CANCELLED|")
    elif (state / "accounting-running").exists():
        print(f"{job_id}|RUNNING|")
    elif (state / "accounting-pending").exists():
        print(f"{job_id}|PENDING|")
elif name_arg is not None:
    job_name = name_arg.split('=', 1)[1]
    for path in sorted(state.glob('job-*.json')):
        item = json.loads(path.read_text())
        if item['job_name'] == job_name:
            print(f"{item['job_id']}|{item['job_name']}|PENDING|")
else:
    raise SystemExit(91)
""",
    )
    return {
        "temp": temp,
        "exact": exact,
        "external": external,
        "fakebin": fakebin,
        "case_work": case_work,
        "commit": commit,
        "tree": tree,
        "source": source,
        "source_path": source_path,
        "fake_git": fake_git,
        "fake_git_sha256": fake_git_sha256,
    }


def reset_fake(fixture, controls=None):
    fakebin = fixture["fakebin"]
    for filename in (
        "slurm_h100_identity_maxseq_smoke.sh",
        "slurm_h100_identity_nccl_smoke.sh",
    ):
        current = fixture["exact"] / "scripts" / filename
        saved = current.with_name(filename + ".verified-inode")
        if saved.exists():
            current.unlink(missing_ok=True)
            saved.rename(current)
    for command in ("git", "sbatch", "scontrol", "scancel", "squeue", "sacct"):
        current = fakebin / command
        saved = fakebin / f"{command}.verified-inode"
        if saved.exists():
            current.unlink(missing_ok=True)
            saved.rename(current)
    deleted_cancel = fakebin / "scancel.deleted-backup"
    if deleted_cancel.exists():
        (fakebin / "scancel").unlink(missing_ok=True)
        deleted_cancel.rename(fakebin / "scancel")
    if not (fakebin / "scancel").exists() and (fakebin / "scancel.missing").exists():
        (fakebin / "scancel.missing").rename(fakebin / "scancel")
    for name in (
        "calls.jsonl",
        "sbatch-count",
        "release-count",
        "fail-sbatch-at",
        "malformed-at",
        "signal-sbatch-at",
        "signal-hup-sbatch-at",
        "sleep-sbatch-at",
        "sleep-cancel-ids",
        "response-loss-nonzero-at",
        "duplicate-at",
        "cluster-suffix",
        "fail-release-at",
        "fail-cancel-ids",
        "delete-scancel-after-release-at",
        "scancel-path-deleted",
        "precreate-receipt-at-release",
        "release-stays-held",
        "release-delay-once",
        "cancel-stays-active",
        "cancel-delay-once",
        "squeue-banner",
        "squeue-hide-id",
        "squeue-hide-name",
        "accounting-running",
        "accounting-pending",
        "ambiguous-name",
        "receipt-path",
        "git-query-count",
        "git-local-count",
        "git-ref-drift-after-preflight",
        "swap-git-path-on-first-local",
        "swap-scheduler-paths-on-first-sbatch",
        "swap-spool-wrappers-on-first-sbatch",
        "replacement-executed",
        "replacement-wrapper-executed",
    ):
        (fakebin / name).unlink(missing_ok=True)
    for path in fakebin.glob("released-*"):
        path.unlink()
    for path in fakebin.glob("cancelled-*"):
        path.unlink()
    for path in fakebin.glob("cancel-seen-*"):
        path.unlink()
    for path in fakebin.glob("release-seen-*"):
        path.unlink()
    for path in fakebin.glob("job-*.json"):
        path.unlink()
    for path in fakebin.glob("spooled-wrapper-*.sha256"):
        path.unlink()
    for name, value in (controls or {}).items():
        (fakebin / name).write_text(str(value))


def overlay(fixture, name):
    def command(name):
        path = fixture["fakebin"] / name
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    environment_map, environment_transaction = write_environment_transaction(
        fixture, name
    )
    return {
        "schema_version": 1,
        "kind": "h100_identity_gate_submit",
        "run_policy": "fresh",
        "validation_id": name,
        "artifact_root": str(fixture["temp"] / "artifacts" / name),
        "source": {
            "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
            "candidate_ref": "refs/heads/codex/position-identity-v1",
            "commit_sha": fixture["commit"],
            "tree_sha": fixture["tree"],
            "manifest_path": str(fixture["source_path"]),
            "manifest_sha256": fixture["source"]["sha256"],
        },
        "environment_sha256_by_world_size": environment_map,
        "environment_transaction": environment_transaction,
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "git": str(fixture["fake_git"]),
            "git_sha256": fixture["fake_git_sha256"],
            "nvidia_smi": "/usr/bin/true",
            "nvidia_smi_sha256": hashlib.sha256(
                Path("/usr/bin/true").read_bytes()
            ).hexdigest(),
            "case_work_root": str(fixture["case_work"]),
        },
        "scheduler_commands": {
            name: command(name)
            for name in ("sbatch", "scontrol", "scancel", "squeue", "sacct")
        },
        "limits": {
            "checkpoint_ceiling_bytes": 300_000_000,
            "run_log_ceiling_bytes": 10_000_000,
            "gpu_monitor_ceiling_bytes": 10_000_000,
            "attestation_ceiling_bytes": 1_000_000,
            "receipt_ceiling_bytes": 1_000_000,
            "scheduler_log_ceiling_bytes": 1_000_000,
        },
    }


def invoke(
    fixture,
    name,
    controls=None,
    *,
    capacity_delta=0,
    mutate=None,
    real_filesystem_isolation=False,
    filesystem_isolation_drift=False,
    capacity_unit_drift=False,
    post_commit_signal=False,
    post_held_plan_signal=False,
    publication_interrupt="",
    publication_uncertain="",
):
    reset_fake(fixture, controls)
    value = overlay(fixture, name)
    value["artifact_root"] = str(fixture["temp"] / "artifacts" / name)
    if mutate is not None:
        mutate(value)
    Path(value["artifact_root"]).parent.mkdir(parents=True, exist_ok=True)
    overlay_path = fixture["external"] / f"{name}.json"
    overlay_path.write_bytes(canonical(value) + b"\n")
    env = {
        "PATH": str(fixture["fakebin"]),
        "PYTHONPATH": str(fixture["exact"] / "src"),
        "PYTHONNOUSERSITE": "1",
        "EVIL_AMBIENT": "must-not-reach-job",
        "TEST_CAPACITY_DELTA": str(capacity_delta),
        "TEST_REAL_FILESYSTEM_ISOLATION": (
            "1" if real_filesystem_isolation else "0"
        ),
        "TEST_FILESYSTEM_ISOLATION_DRIFT": (
            "1" if filesystem_isolation_drift else "0"
        ),
        "TEST_CAPACITY_UNIT_DRIFT": "1" if capacity_unit_drift else "0",
        "TEST_POST_COMMIT_SIGNAL": "1" if post_commit_signal else "0",
        "TEST_POST_HELD_PLAN_SIGNAL": "1" if post_held_plan_signal else "0",
        "TEST_PUBLICATION_INTERRUPT": publication_interrupt,
        "TEST_PUBLICATION_UNCERTAIN": publication_uncertain,
    }
    program = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys

script, overlay, exact = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("_gate_submit_test_controller", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
real_capacity = module.capacity_preflight
capacity_calls = 0

def capacity(parent, limits):
    global capacity_calls
    capacity_calls += 1
    allocation_unit = (
        2
        if os.environ["TEST_CAPACITY_UNIT_DRIFT"] == "1" and capacity_calls > 1
        else 1
    )
    budget = module.gate_capacity_budget(
        limits, allocation_unit_bytes=allocation_unit
    )
    available = budget["required_free_bytes"] + int(os.environ["TEST_CAPACITY_DELTA"])
    if available % allocation_unit:
        available += allocation_unit - (available % allocation_unit)
    class VFS:
        f_bavail = available // allocation_unit
        f_frsize = allocation_unit
    return real_capacity(
        parent,
        limits,
        statvfs_fn=lambda path: VFS(),
        stat_fn=os.stat,
    )

module.capacity_preflight = capacity
real_publish = module._publish_no_replace
def publish(path, *args, **kwargs):
    result = real_publish(path, *args, **kwargs)
    if (
        os.environ["TEST_POST_HELD_PLAN_SIGNAL"] == "1"
        and Path(path).name == "held-plan.json"
    ):
        os.kill(os.getpid(), module.signal.SIGTERM)
    if Path(path).name == os.environ["TEST_PUBLICATION_INTERRUPT"]:
        raise module._PublicationInterrupted(
            Path(path), KeyboardInterrupt("injected durable publication interrupt")
        )
    if Path(path).name == os.environ["TEST_PUBLICATION_UNCERTAIN"]:
        raise module._PublicationUncertain(Path(path))
    if (
        os.environ["TEST_POST_COMMIT_SIGNAL"] == "1"
        and Path(path).name == "submission-receipt.json"
    ):
        os.kill(os.getpid(), module.signal.SIGTERM)
    return result
module._publish_no_replace = publish
module.SCHEDULER_COMMAND_TIMEOUT_SECONDS = 1.0
if os.environ["TEST_REAL_FILESYSTEM_ISOLATION"] != "1":
    isolation_calls = 0
    def isolation(*, exact_root, work_root, artifact_directory):
        global isolation_calls
        isolation_calls += 1
        return {
            "schema_version": 1,
            "work_root": str(work_root),
            "work_device": 101,
            "artifact_root": str(artifact_directory),
            "artifact_device": (
                303
                if os.environ["TEST_FILESYSTEM_ISOLATION_DRIFT"] == "1"
                and isolation_calls > 1
                else 202
            ),
        }
    module._require_work_artifact_filesystem_isolation = isolation
receipt = module.submit(
    overlay,
    exact_root=exact,
    process_scoped_signals=True,
)
sys.stdout.buffer.write(module._canonical(receipt) + b"\n")
'''
    return subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            "-c",
            program,
            str(fixture["exact"] / "scripts/submit_h100_identity_validation.py"),
            str(overlay_path),
            str(fixture["exact"]),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )


def calls(fixture):
    path = fixture["fakebin"] / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def exports(argv):
    raw = next(item for item in argv if item.startswith("--export="))[len("--export=") :]
    return dict(item.split("=", 1) for item in raw.split(","))


def test_success_is_exact_held_12_case_transaction_with_static_resources(fixture):
    result = invoke(fixture, "success", {"cluster-suffix": "1"})
    assert result.returncode == 0, result.stderr
    events = calls(fixture)
    sbatches = [event for event in events if event["cmd"] == "sbatch"]
    releases = [event for event in events if event["cmd"] == "scontrol"]
    assert len(sbatches) == len(releases) == 12
    assert not [event for event in events if event["cmd"] == "scancel"]

    matrix_spec = importlib.util.spec_from_file_location(
        "gate_submit_test_matrix", ROOT / "scripts/run_h100_identity_validation.py"
    )
    matrix = importlib.util.module_from_spec(matrix_spec)
    sys.modules[matrix_spec.name] = matrix
    matrix_spec.loader.exec_module(matrix)
    case_ids = [case.case_id for case in matrix.build_matrix()]
    assert [exports(event["argv"])["VALIDATION_CASE_ID"] for event in sbatches] == case_ids
    artifact_root = Path(overlay(fixture, "success")["artifact_root"])

    for event in sbatches:
        argv = event["argv"]
        exported = exports(argv)
        case_id = exported["VALIDATION_CASE_ID"]
        assert "--parsable" in argv and "--hold" in argv
        assert "--partition=h100" in argv and "--qos=short" in argv
        assert "--time=03:00:00" in argv and "--mem=131072M" in argv
        assert f"--chdir={artifact_root / 'job-cwd'}" in argv
        assert f"--output={artifact_root / 'scheduler-logs' / (case_id + '.out')}" in argv
        assert f"--error={artifact_root / 'scheduler-logs' / (case_id + '.err')}" in argv
        assert "--open-mode=truncate" in argv
        assert not any(item == "--export=ALL" or item.startswith("--export=ALL,") for item in argv)
        assert "EVIL_AMBIENT" not in next(item for item in argv if item.startswith("--export="))
        assert exported["PATH"] == "/usr/bin:/bin"
        for key in (
            "FORMAL_SOURCE_COMMIT_SHA",
            "FORMAL_SOURCE_TREE_SHA",
            "FORMAL_SOURCE_SHA256",
            "FORMAL_EXPECTED_ENVIRONMENT_SHA256",
            "CHECKPOINT_CEILING_BYTES",
            "VALIDATION_ARTIFACT_IDENTITY_SHA256",
        ):
            assert exported[key]
        if case_id == "nccl_2gpu":
            assert "--gres=gpu:2" in argv and "--cpus-per-task=64" in argv
            assert exported["FORMAL_EXPECTED_GPUS"] == "2"
        else:
            assert "--gres=gpu:1" in argv and "--cpus-per-task=32" in argv
            assert exported["FORMAL_EXPECTED_GPUS"] == "1"
        assert re.fullmatch(r"/proc/self/fd/[0-9]+", argv[-1]) is not None

    job_names = [
        next(item for item in event["argv"] if item.startswith("--job-name="))
        for event in sbatches
    ]
    assert len(set(job_names)) == 12
    assert all(name.startswith("--job-name=tabicl-gate-") for name in job_names)

    assert [event["argv"] for event in releases] == [
        ["--clusters=cluster-a", "release", str(1000 + index)]
        for index in range(1, 13)
    ]
    receipt_path = artifact_root / "submission-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["kind"] == "h100_gate_submission_receipt"
    assert receipt["payload"]["all_jobs_released"] is True
    transaction_id = receipt["payload"]["transaction_id"]
    assert job_names == [
        f"--job-name=tabicl-gate-{transaction_id}-{position:02d}"
        for position in range(1, 13)
    ]
    assert [item["job_id"] for item in receipt["payload"]["cases"]] == [
        str(1000 + index) for index in range(1, 13)
    ]
    assert {item["cluster"] for item in receipt["payload"]["cases"]} == {"cluster-a"}
    assert set(receipt["payload"]["scheduler_command_sha256"]) == {
        "sbatch", "scontrol", "scancel", "squeue", "sacct"
    }
    assert (artifact_root / "transaction-journal.jsonl").is_file()
    receipt_text = receipt_path.read_text()
    assert str(fixture["temp"]) not in receipt_text
    assert str(fixture["exact"]) not in receipt_text
    assert "candidate_repository" not in receipt_text
    assert receipt["payload"]["repository_binding"]["repository_url"] == (
        "https://github.com/kikixiong/tabicl-pe.git"
    )
    assert receipt["payload"]["repository_binding"]["repository_ref"] == (
        "refs/heads/codex/position-identity-v1"
    )
    expected_environment = overlay(fixture, "success")
    environment_summary = receipt["payload"]["environment_transaction"]
    assert environment_summary["completion_raw_sha256"] == (
        expected_environment["environment_transaction"]["completion_sha256"]
    )
    assert environment_summary["transaction_sha256"] == (
        expected_environment["environment_transaction"]["transaction_sha256"]
    )
    assert environment_summary["one_gpu_environment_sha256"] == (
        expected_environment["environment_sha256_by_world_size"]["1"]
    )
    assert environment_summary["two_gpu_environment_sha256"] == (
        expected_environment["environment_sha256_by_world_size"]["2"]
    )
    assert set(environment_summary["output_raw_sha256_by_role"]) == {
        "one_gpu",
        "two_gpu",
        "inventory",
    }
    validated = matrix.validate_gate_submission_receipt(
        receipt_path,
        expected_commit_sha=fixture["commit"],
        expected_tree_sha=fixture["tree"],
        expected_source_manifest_sha256=fixture["source"]["sha256"],
        expected_checkpoint_ceiling_bytes=300_000_000,
        max_bytes=1_000_000,
    )
    assert set(validated["artifact_identities"]) == set(case_ids)
    assert validated["environment_transaction"] == environment_summary


def test_local_checkout_git_uses_verified_inode_across_real_path_swap(fixture):
    result = invoke(
        fixture,
        "git-inode-swap",
        {"swap-git-path-on-first-local": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert (fixture["fakebin"] / "git-local-count").read_text() == "4"
    assert not (fixture["fakebin"] / "replacement-executed").exists()


def test_environment_verifier_executes_digest_bound_inode_after_path_swap(
    tmp_path,
):
    module = load_submit_module()
    verifier = tmp_path / "verifier.py"
    observed = tmp_path / "observed.txt"
    replacement_observed = tmp_path / "replacement-observed.txt"
    verifier.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('verified-inode')\n"
    )
    expected_sha256 = hashlib.sha256(verifier.read_bytes()).hexdigest()
    command, descriptor = module._open_digest_bound_regular_file(
        verifier,
        expected_sha256,
        "test environment verifier",
    )
    saved = verifier.with_suffix(".verified-inode")
    verifier.rename(saved)
    verifier.write_text(
        "from pathlib import Path\n"
        f"Path({str(replacement_observed)!r}).write_text('replacement')\n"
        "raise SystemExit(97)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", command, str(observed)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert completed.returncode == 0, completed.stderr
    assert observed.read_text() == "verified-inode"
    assert not replacement_observed.exists()
    with pytest.raises(ValueError, match="digest mismatch"):
        module._open_digest_bound_regular_file(
            verifier,
            expected_sha256,
            "test environment verifier",
        )


def test_scheduler_transaction_uses_verified_inodes_across_real_path_swap(
    fixture,
):
    result = invoke(
        fixture,
        "scheduler-inode-swap-success",
        {
            "swap-scheduler-paths-on-first-sbatch": "1",
            "squeue-hide-id": "1",
            "accounting-running": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert len([event for event in events if event["cmd"] == "scontrol"]) == 12
    assert len([event for event in events if event["cmd"] == "squeue"]) == 12
    assert len([event for event in events if event["cmd"] == "sacct"]) == 12
    assert not (fixture["fakebin"] / "replacement-executed").exists()


def test_sbatch_spools_digest_bound_wrapper_inode_after_exact_path_swap(fixture):
    try:
        result = invoke(
            fixture,
            "wrapper-inode-swap-success",
            {"swap-spool-wrappers-on-first-sbatch": fixture["exact"]},
        )
        assert result.returncode == 0, result.stderr
        entries = {
            entry["path"]: entry["sha256"]
            for entry in fixture["source"]["payload"]["entries"]
        }
        sbatches = [event for event in calls(fixture) if event["cmd"] == "sbatch"]
        assert len(sbatches) == 12
        for position, event in enumerate(sbatches, start=1):
            exported = exports(event["argv"])
            wrapper = (
                "scripts/slurm_h100_identity_nccl_smoke.sh"
                if exported["VALIDATION_CASE_ID"] == "nccl_2gpu"
                else "scripts/slurm_h100_identity_maxseq_smoke.sh"
            )
            assert re.fullmatch(r"/proc/self/fd/[0-9]+", event["argv"][-1])
            observed = (
                fixture["fakebin"] / f"spooled-wrapper-{position}.sha256"
            ).read_text()
            assert observed == entries[wrapper]
        assert not (fixture["fakebin"] / "replacement-wrapper-executed").exists()
    finally:
        reset_fake(fixture)


def test_scheduler_rollback_uses_verified_inodes_across_real_path_swap(fixture):
    result = invoke(
        fixture,
        "scheduler-inode-swap-rollback",
        {
            "swap-scheduler-paths-on-first-sbatch": "1",
            "fail-release-at": "5",
        },
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert len([event for event in events if event["cmd"] == "scontrol"]) == 5
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job)] for job in range(1012, 1000, -1)
    ]
    root = fixture["temp"] / "artifacts" / "scheduler-inode-swap-rollback"
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["remaining_job_ids"] == []
    assert not (fixture["fakebin"] / "replacement-executed").exists()


def test_post_commit_signal_cannot_turn_published_receipt_into_cli_failure(
    fixture,
):
    result = invoke(
        fixture,
        "post-commit-signal",
        post_commit_signal=True,
    )
    assert result.returncode == 0, result.stderr
    receipt_path = (
        Path(overlay(fixture, "post-commit-signal")["artifact_root"])
        / "submission-receipt.json"
    )
    assert receipt_path.is_file()
    assert not [event for event in calls(fixture) if event["cmd"] == "scancel"]


def test_held_plan_publication_interrupt_is_repropagated_after_truthful_rollback(
    fixture,
):
    name = "held-plan-publication-interrupt"
    result = invoke(fixture, name, publication_interrupt="held-plan.json")
    assert result.returncode != 0
    root = fixture["temp"] / "artifacts" / name
    assert (root / "held-plan.json").is_file()
    assert not (root / "submission-receipt.json").exists()
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["held_plan_published"] is True
    assert recovery["payload"]["held_plan_visibility"] == "durable"
    assert recovery["payload"]["submission_receipt_visibility"] == "absent"
    assert recovery["payload"]["remaining_job_ids"] == []
    assert [event["argv"] for event in calls(fixture) if event["cmd"] == "scancel"] == [
        [str(job)] for job in range(1012, 1000, -1)
    ]


def test_signal_after_held_plan_return_observes_committed_flags_before_rollback(
    fixture,
):
    name = "held-plan-post-return-signal"
    result = invoke(fixture, name, post_held_plan_signal=True)
    assert result.returncode != 0
    root = fixture["temp"] / "artifacts" / name
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["held_plan_published"] is True
    assert recovery["payload"]["held_plan_visibility"] == "durable"
    assert recovery["payload"]["submission_receipt_visibility"] == "absent"
    assert recovery["payload"]["remaining_job_ids"] == []
    assert (root / "held-plan.json").is_file()
    assert not (root / "submission-receipt.json").exists()


def test_durable_receipt_publication_interrupt_is_repropagated_without_rollback(
    fixture,
):
    name = "receipt-publication-interrupt"
    result = invoke(
        fixture,
        name,
        publication_interrupt="submission-receipt.json",
    )
    assert result.returncode != 0
    root = fixture["temp"] / "artifacts" / name
    assert (root / "submission-receipt.json").is_file()
    assert not (root / "rollback-recovery.json").exists()
    assert not [event for event in calls(fixture) if event["cmd"] == "scancel"]


def test_uncertain_receipt_visibility_never_rolls_back_released_jobs(fixture):
    name = "receipt-publication-uncertain"
    result = invoke(
        fixture,
        name,
        publication_uncertain="submission-receipt.json",
    )
    assert result.returncode != 0
    assert "receipt visibility is uncertain" in result.stderr
    root = fixture["temp"] / "artifacts" / name
    assert (root / "submission-receipt.json").is_file()
    assert not (root / "rollback-recovery.json").exists()
    assert not [event for event in calls(fixture) if event["cmd"] == "scancel"]


def test_imported_submit_default_never_rewrites_host_signal_handlers(tmp_path):
    module = load_submit_module()
    before = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }
    with pytest.raises(Exception):
        module.submit(tmp_path / "missing.json", exact_root=tmp_path)
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    } == before


@pytest.mark.parametrize(
    "fault", ("after_link", "after_link_fsync", "after_temp_unlink")
)
def test_write_once_publication_heals_each_post_link_failure(
    tmp_path, fault
):
    module = load_submit_module()
    value = module._make_envelope("publication_test", {"fault": fault})
    path = tmp_path / "published.json"
    assert module._publish_no_replace(
        path, value, max_bytes=1 << 20, fault=fault
    ) == path
    assert path.read_bytes() == module._canonical(value) + b"\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_unhealable_directory_fsync_failure_revokes_link_before_error(
    tmp_path, monkeypatch
):
    module = load_submit_module()
    value = module._make_envelope("publication_test", {"fault": "fsync"})
    path = tmp_path / "published.json"
    real_fsync = module.os.fsync
    calls = 0

    def failing_fsync(fd):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("injected persistent directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", failing_fsync)
    with pytest.raises(module._PublicationUncertain):
        module._publish_no_replace(path, value, max_bytes=1 << 20)
    assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_process_interrupt_after_link_is_repropagated_after_durable_publication(
    tmp_path, monkeypatch
):
    module = load_submit_module()
    value = module._make_envelope("publication_test", {"fault": "interrupt"})
    path = tmp_path / "published.json"
    real_fsync = module.os.fsync
    calls = 0

    def interrupt_once(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("injected publication interrupt")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", interrupt_once)
    with pytest.raises(module._PublicationInterrupted) as caught:
        module._publish_no_replace(path, value, max_bytes=1 << 20)
    assert isinstance(caught.value.interruption, KeyboardInterrupt)
    assert path.read_bytes() == module._canonical(value) + b"\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_process_interrupt_between_link_syscall_and_flag_is_truthfully_recovered(
    tmp_path, monkeypatch
):
    module = load_submit_module()
    value = module._make_envelope("publication_test", {"fault": "link-window"})
    path = tmp_path / "published.json"
    real_link = module.os.link

    def link_then_interrupt(*args, **kwargs):
        real_link(*args, **kwargs)
        raise KeyboardInterrupt("injected post-link pre-flag interrupt")

    monkeypatch.setattr(module.os, "link", link_then_interrupt)
    with pytest.raises(module._PublicationInterrupted) as caught:
        module._publish_no_replace(path, value, max_bytes=1 << 20)
    assert isinstance(caught.value.interruption, KeyboardInterrupt)
    assert path.read_bytes() == module._canonical(value) + b"\n"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("cleanup_error", (KeyboardInterrupt, OSError))
def test_durable_publication_cleanup_cannot_be_misreported_as_rollback(
    tmp_path, monkeypatch, cleanup_error
):
    module = load_submit_module()
    value = module._make_envelope("publication_test", {"fault": "final-cleanup"})
    path = tmp_path / "published.json"
    real_open_directory = module._open_directory_nofollow
    real_close = module.os.close
    captured = {}

    def capture_directory(*args, **kwargs):
        descriptor = real_open_directory(*args, **kwargs)
        captured["parent_fd"] = descriptor
        return descriptor

    def close_then_fail(descriptor):
        real_close(descriptor)
        if descriptor == captured.get("parent_fd"):
            raise cleanup_error("injected durable-final cleanup failure")

    monkeypatch.setattr(module, "_open_directory_nofollow", capture_directory)
    monkeypatch.setattr(module.os, "close", close_then_fail)
    if cleanup_error is KeyboardInterrupt:
        with pytest.raises(module._PublicationInterrupted):
            module._publish_no_replace(path, value, max_bytes=1 << 20)
    else:
        assert module._publish_no_replace(path, value, max_bytes=1 << 20) == path
    assert path.read_bytes() == module._canonical(value) + b"\n"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure_position", range(1, 13))
def test_each_sbatch_failure_rolls_back_only_this_batch_in_reverse(fixture, failure_position):
    result = invoke(
        fixture,
        f"sbatch-fail-{failure_position}",
        {"fail-sbatch-at": failure_position},
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == failure_position
    assert not [event for event in events if event["cmd"] == "scontrol"]
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(1000 + index)] for index in range(failure_position - 1, 0, -1)
    ]


def test_release_failure_cancels_all_and_incomplete_cancel_writes_truthful_recovery(fixture):
    result = invoke(
        fixture,
        "release-cancel-fail",
        {"fail-release-at": 5, "fail-cancel-ids": "1004\n"},
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert len([event for event in events if event["cmd"] == "scontrol"]) == 5
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job)] for job in range(1012, 1000, -1)
    ]
    root = Path(overlay(fixture, "release-cancel-fail")["artifact_root"])
    assert not (root / "submission-receipt.json").exists()
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["kind"] == "h100_gate_rollback_recovery"
    assert recovery["payload"]["reason"] == "release_failed"
    assert recovery["payload"]["remaining_job_ids"] == ["1004"]
    assert recovery["payload"]["held_plan_published"] is True
    assert recovery["payload"]["submission_receipt_published"] is False


def test_deleted_scancel_path_still_runs_verified_inode_during_rollback(fixture):
    root = Path(overlay(fixture, "missing-scancel-exec")["artifact_root"])
    result = invoke(
        fixture,
        "missing-scancel-exec",
        {
            "delete-scancel-after-release-at": "1",
            "precreate-receipt-at-release": "12",
        },
    )
    assert result.returncode != 0
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["remaining_job_ids"] == []
    assert recovery["payload"]["cancelled_job_ids"] == [
        str(job) for job in range(1012, 1000, -1)
    ]
    assert (fixture["fakebin"] / "scancel-path-deleted").read_text() == "1"
    assert recovery["payload"]["submission_receipt_published"] is False


def test_malformed_success_output_records_unidentified_acceptance(fixture):
    result = invoke(fixture, "malformed", {"malformed-at": 4})
    assert result.returncode != 0
    root = Path(overlay(fixture, "malformed")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["untrusted_sbatch_result_position"] == 4
    assert [item["job_id"] for item in recovery["payload"]["accepted_jobs"]] == [
        "1001", "1002", "1003", "1004"
    ]
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"
    assert recovery["payload"]["cancelled_job_ids"] == [
        "1004", "1003", "1002", "1001"
    ]


def test_duplicate_job_id_is_untrusted_and_records_recovery(fixture):
    result = invoke(fixture, "duplicate", {"duplicate-at": 4})
    assert result.returncode != 0
    root = Path(overlay(fixture, "duplicate")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["untrusted_sbatch_result_position"] == 4
    assert [item["job_id"] for item in recovery["payload"]["accepted_jobs"]] == [
        "1001", "1002", "1003", "1004"
    ]
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"


@pytest.mark.parametrize(
    ("scenario", "control"),
    (
        ("signal-inflight", "signal-sbatch-at"),
        ("sighup-inflight", "signal-hup-sbatch-at"),
    ),
)
def test_signal_during_inflight_sbatch_uses_rollback_and_records_unknown_result(
    fixture, scenario, control
):
    result = invoke(fixture, scenario, {control: 4})
    assert result.returncode != 0
    root = Path(overlay(fixture, scenario)["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["reason"] == "sbatch_interrupted_with_unknown_result"
    assert recovery["payload"]["untrusted_sbatch_result_position"] == 4
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"
    assert recovery["payload"]["cancelled_job_ids"] == [
        "1004", "1003", "1002", "1001"
    ]
    assert recovery["payload"]["remaining_job_ids"] == []


def test_sbatch_timeout_discovers_and_cancels_the_uncertain_job(fixture):
    result = invoke(fixture, "sbatch-timeout", {"sleep-sbatch-at": 2})
    assert result.returncode != 0
    root = Path(overlay(fixture, "sbatch-timeout")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"
    assert recovery["payload"]["cancelled_job_ids"] == ["1002", "1001"]
    assert recovery["payload"]["remaining_job_ids"] == []


def test_scancel_timeout_continues_rollback_and_publishes_recovery(fixture):
    result = invoke(
        fixture,
        "scancel-timeout",
        {"fail-sbatch-at": 3, "sleep-cancel-ids": "1002"},
    )
    assert result.returncode != 0
    root = Path(overlay(fixture, "scancel-timeout")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["remaining_job_ids"] == ["1002"]
    assert recovery["payload"]["cancelled_job_ids"] == ["1001"]


def test_nonzero_response_loss_discovers_and_cancels_the_accepted_job(fixture):
    result = invoke(
        fixture,
        "nonzero-response-loss",
        {"response-loss-nonzero-at": 4},
    )
    assert result.returncode != 0
    root = Path(overlay(fixture, "nonzero-response-loss")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["reason"] == "sbatch_nonzero_with_unknown_result"
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"
    assert recovery["payload"]["cancelled_job_ids"] == [
        "1004", "1003", "1002", "1001"
    ]


def test_response_loss_discovery_falls_back_to_bound_accounting(fixture):
    result = invoke(
        fixture,
        "accounting-response-loss",
        {"response-loss-nonzero-at": 4, "squeue-hide-name": "1"},
    )
    assert result.returncode != 0
    root = Path(overlay(fixture, "accounting-response-loss")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["response_loss_resolution"] == "unique_job_absorbed"
    assert recovery["payload"]["response_loss_discovery"] == [
        {
            "job_id": "1004",
            "cluster": None,
            "state": "PENDING",
            "reason": "",
            "source": "sacct",
        }
    ]


def test_ambiguous_response_loss_remains_fail_closed_and_is_not_auto_cancelled(fixture):
    result = invoke(
        fixture,
        "ambiguous-response-loss",
        {"response-loss-nonzero-at": 4, "ambiguous-name": "1"},
    )
    assert result.returncode != 0
    assert "unresolved scheduler state" in result.stderr
    root = Path(overlay(fixture, "ambiguous-response-loss")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["response_loss_resolution"] == "ambiguous_jobs_visible"
    assert [item["job_id"] for item in recovery["payload"]["response_loss_discovery"]] == [
        "1004", "9001"
    ]
    assert recovery["payload"]["cancelled_job_ids"] == ["1003", "1002", "1001"]


def test_release_must_reconcile_out_of_hold_and_cancel_must_reconcile_terminal(fixture):
    held = invoke(fixture, "release-held", {"release-stays-held": "1"})
    assert held.returncode != 0
    held_root = Path(overlay(fixture, "release-held")["artifact_root"])
    held_recovery = json.loads((held_root / "rollback-recovery.json").read_text())
    assert held_recovery["payload"]["reason"] == "release_failed"
    assert held_recovery["payload"]["remaining_job_ids"] == []

    active = invoke(
        fixture,
        "cancel-active",
        {"fail-release-at": 1, "cancel-stays-active": "1"},
    )
    assert active.returncode != 0
    active_root = Path(overlay(fixture, "cancel-active")["artifact_root"])
    active_recovery = json.loads((active_root / "rollback-recovery.json").read_text())
    assert active_recovery["payload"]["remaining_job_ids"] == [
        str(job) for job in range(1012, 1000, -1)
    ]


def test_release_and_cancel_reconciliation_tolerate_one_propagation_snapshot(fixture):
    released = invoke(fixture, "release-delay", {"release-delay-once": "1"})
    assert released.returncode == 0, released.stderr
    assert len([event for event in calls(fixture) if event["cmd"] == "squeue"]) == 24

    cancelled = invoke(
        fixture,
        "cancel-delay",
        {"fail-release-at": 1, "cancel-delay-once": "1"},
    )
    assert cancelled.returncode != 0
    root = Path(overlay(fixture, "cancel-delay")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["remaining_job_ids"] == []
    assert recovery["payload"]["cancelled_job_ids"] == [
        str(job) for job in range(1012, 1000, -1)
    ]


def test_squeue_banner_is_ignored_but_exact_reason_bearing_row_is_authoritative(fixture):
    result = invoke(fixture, "banner-success", {"squeue-banner": "1"})
    assert result.returncode == 0, result.stderr
    root = Path(overlay(fixture, "banner-success")["artifact_root"])
    assert (root / "submission-receipt.json").is_file()


def test_accounting_pending_without_reason_cannot_prove_release(fixture):
    result = invoke(
        fixture,
        "accounting-pending",
        {"squeue-hide-id": "1", "accounting-pending": "1"},
    )
    assert result.returncode != 0
    root = Path(overlay(fixture, "accounting-pending")["artifact_root"])
    recovery = json.loads((root / "rollback-recovery.json").read_text())
    assert recovery["payload"]["reason"] == "release_failed"
    assert not (root / "submission-receipt.json").exists()


def test_cluster_suffix_is_used_for_rollback_commands(fixture):
    result = invoke(
        fixture,
        "cluster-rollback",
        {"cluster-suffix": "1", "fail-release-at": 2},
    )
    assert result.returncode != 0
    cancellations = [
        event["argv"] for event in calls(fixture) if event["cmd"] == "scancel"
    ]
    assert cancellations == [
        ["--clusters=cluster-a", str(job)] for job in range(1012, 1000, -1)
    ]


def test_receipt_publication_collision_after_all_releases_rolls_back_batch(fixture):
    root = Path(overlay(fixture, "receipt-collision")["artifact_root"])
    result = invoke(
        fixture,
        "receipt-collision",
        {
            "precreate-receipt-at-release": 12,
            "receipt-path": root / "submission-receipt.json",
        },
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "scontrol"]) == 12
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job)] for job in range(1012, 1000, -1)
    ]
    assert json.loads((root / "submission-receipt.json").read_text()) == {"invalid": True}


def test_capacity_boundary_is_checked_before_namespace_or_scheduler_calls(fixture):
    blocked = invoke(fixture, "capacity-minus-one", capacity_delta=-1)
    assert blocked.returncode != 0
    assert calls(fixture) == []
    assert not Path(overlay(fixture, "capacity-minus-one")["artifact_root"]).exists()
    exact = invoke(fixture, "capacity-exact", capacity_delta=0)
    assert exact.returncode == 0, exact.stderr
    receipt = json.loads(
        (Path(overlay(fixture, "capacity-exact")["artifact_root"]) / "submission-receipt.json").read_text()
    )
    capacity = receipt["payload"]["capacity"]
    held = json.loads(
        (
            Path(overlay(fixture, "capacity-exact")["artifact_root"])
            / "held-plan.json"
        ).read_text()
    )
    assert held["payload"]["capacity"] == capacity
    assert capacity["available_bytes"] == capacity["required_free_bytes"]
    assert capacity["allocation_unit_bytes"] == 1
    assert capacity["available_blocks"] == capacity["available_bytes"]
    assert capacity["total_file_slots"] == 116
    assert capacity["directory_slot_count"] == 28
    assert capacity["atomic_extra_dirent_slots"] == 13
    assert capacity["total_dirent_slots"] == 157
    assert capacity["total_directory_physical_bytes"] == 185


def test_capacity_formula_fake_statvfs_boundary_plus_minus_one(fixture):
    spec = importlib.util.spec_from_file_location(
        "gate_capacity_unit", ROOT / "scripts/submit_h100_identity_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    limits = overlay(fixture, "unused")["limits"]
    budget = module.gate_capacity_budget(limits, allocation_unit_bytes=1)
    assert budget["final_attestation_bytes"] == limits["attestation_ceiling_bytes"]
    assert budget["worst_case_gate_bytes"] == sum(
        budget[key]
        for key in (
            "checkpoint_bytes",
            "case_log_bytes",
            "case_attestation_bytes",
            "final_attestation_bytes",
            "case_result_bytes",
            "gpu_raw_bytes",
            "gpu_summary_bytes",
            "controller_receipt_bytes",
            "scheduler_log_bytes",
        )
    )
    assert budget["total_file_slots"] == 116
    assert budget["total_file_physical_bytes"] == budget["worst_case_gate_bytes"]
    assert budget["directory_slot_count"] == 28
    assert budget["atomic_extra_dirent_slots"] == 13
    assert budget["total_dirent_slots"] == 157
    assert budget["total_directory_physical_bytes"] == 185
    assert budget["worst_case_gate_physical_bytes"] == (
        budget["total_file_physical_bytes"]
        + budget["total_directory_physical_bytes"]
    )
    snapshot = os.stat(fixture["external"], follow_symlinks=False)

    def stat_fn(path, *, follow_symlinks=False):
        assert follow_symlinks is False
        return snapshot

    def probe(available):
        vfs = type("VFS", (), {"f_bavail": available, "f_frsize": 1})()
        return module.capacity_preflight(
            fixture["external"],
            limits,
            statvfs_fn=lambda path: vfs,
            stat_fn=stat_fn,
        )

    required = budget["required_free_bytes"]
    assert probe(required)["available_bytes"] == required
    assert probe(required + 1)["available_bytes"] == required + 1
    with pytest.raises(ValueError, match="insufficient space"):
        probe(required - 1)


def test_capacity_rounds_each_tiny_file_slot_and_directory_entry_physically(fixture):
    spec = importlib.util.spec_from_file_location(
        "gate_capacity_rounding_unit",
        ROOT / "scripts/submit_h100_identity_validation.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    limits = {
        "checkpoint_ceiling_bytes": 1,
        "run_log_ceiling_bytes": 1,
        "gpu_monitor_ceiling_bytes": 1,
        "attestation_ceiling_bytes": 1,
        "receipt_ceiling_bytes": 1,
        "scheduler_log_ceiling_bytes": 1,
    }
    unit = 4096
    budget = module.gate_capacity_budget(
        limits, allocation_unit_bytes=unit
    )
    slots = budget["file_slots_by_category"]
    assert slots == {
        "checkpoint": 9,
        "case_log": 12,
        "case_attestation": 36,
        "final_attestation": 1,
        "case_result": 12,
        "gpu_raw": 12,
        "gpu_summary": 6,
        "controller_receipt": 4,
        "scheduler_log": 24,
    }
    for category in slots:
        if category == "case_result":
            assert budget[f"{category}_physical_bytes"] == 12 * (1 << 20)
        else:
            assert budget[f"{category}_physical_bytes"] == slots[category] * unit
    assert budget["checkpoint_bytes"] == 9
    assert budget["checkpoint_physical_bytes"] == 9 * unit
    assert budget["base_directory_physical_bytes"] == 28 * unit
    assert budget["atomic_extra_dirent_slots"] == 13
    assert budget["dirent_physical_bytes"] == 157 * unit
    assert budget["total_directory_physical_bytes"] == 185 * unit
    assert budget["required_free_bytes"] % unit == 0

    limits["checkpoint_ceiling_bytes"] = unit + 1
    rounded = module.gate_capacity_budget(
        limits, allocation_unit_bytes=unit
    )
    assert rounded["checkpoint_physical_bytes"] == 9 * 2 * unit


def test_capacity_physical_block_boundary_exact_and_one_block_short(fixture):
    spec = importlib.util.spec_from_file_location(
        "gate_capacity_block_boundary",
        ROOT / "scripts/submit_h100_identity_validation.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    limits = overlay(fixture, "unused-physical-boundary")["limits"]
    unit = 4096
    budget = module.gate_capacity_budget(
        limits, allocation_unit_bytes=unit
    )
    required_blocks = budget["required_free_bytes"] // unit
    assert required_blocks * unit == budget["required_free_bytes"]
    snapshot = os.stat(fixture["external"], follow_symlinks=False)

    def stat_fn(path, *, follow_symlinks=False):
        assert follow_symlinks is False
        return snapshot

    def probe(blocks):
        vfs = type("VFS", (), {"f_bavail": blocks, "f_frsize": unit})()
        return module.capacity_preflight(
            fixture["external"],
            limits,
            statvfs_fn=lambda path: vfs,
            stat_fn=stat_fn,
        )

    exact = probe(required_blocks)
    assert exact["allocation_unit_bytes"] == unit
    assert exact["available_bytes"] == exact["required_free_bytes"]
    with pytest.raises(ValueError, match="insufficient space"):
        probe(required_blocks - 1)


def test_environment_transaction_is_consumed_before_namespace_or_scheduler(fixture):
    def remove_marker(value):
        Path(value["environment_transaction"]["completion_path"]).unlink()

    def tamper_one_gpu_output(value):
        completion = Path(value["environment_transaction"]["completion_path"])
        marker = json.loads(completion.read_text())
        one_gpu = completion.parent / marker["outputs"][0]["name"]
        one_gpu.write_bytes(one_gpu.read_bytes() + b"tamper")

    scenarios = (
        ("environment-marker-missing", remove_marker),
        ("environment-output-tampered", tamper_one_gpu_output),
        (
            "environment-marker-digest-mismatch",
            lambda value: value["environment_transaction"].update(
                completion_sha256="0" * 64
            ),
        ),
        (
            "environment-transaction-digest-mismatch",
            lambda value: value["environment_transaction"].update(
                transaction_sha256="0" * 64
            ),
        ),
        (
            "environment-inventory-digest-mismatch",
            lambda value: value["environment_transaction"].update(
                inventory_sha256="0" * 64
            ),
        ),
        (
            "environment-noncanonical-ceiling",
            lambda value: value["environment_transaction"].update(
                inventory_max_bytes=(128 << 20) - 1
            ),
        ),
        (
            "environment-marker-inside-source",
            lambda value: value["environment_transaction"].update(
                completion_path=str(fixture["exact"] / "marker.json")
            ),
        ),
    )
    for name, mutation in scenarios:
        result = invoke(fixture, name, mutate=mutation)
        assert result.returncode != 0
        assert calls(fixture) == []
        assert not (fixture["temp"] / "artifacts" / name).exists()


def test_source_manifest_must_include_environment_transaction_verifier(fixture):
    def omit_verifier(value):
        source = json.loads(json.dumps(fixture["source"]))
        source["payload"]["entries"] = [
            entry
            for entry in source["payload"]["entries"]
            if entry["path"]
            != "scripts/verify_formal_environment_transaction.py"
        ]
        body = {
            key: source[key] for key in ("schema_version", "kind", "payload")
        }
        source["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
        path = fixture["external"] / "source-without-environment-verifier.json"
        path.write_bytes(canonical(source) + b"\n")
        value["source"].update(
            manifest_path=str(path), manifest_sha256=source["sha256"]
        )

    result = invoke(
        fixture,
        "source-omits-environment-verifier",
        mutate=omit_verifier,
    )
    assert result.returncode != 0
    assert (
        "unexpected code file under tracked code root" in result.stderr
        or "omits H100 gate controller files" in result.stderr
        or "omits required runtime files" in result.stderr
    )
    assert "verify_formal_environment_transaction.py" in result.stderr
    assert calls(fixture) == []
    assert not (
        fixture["temp"] / "artifacts" / "source-omits-environment-verifier"
    ).exists()


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/submit_h100_identity_validation.sh",
        "scripts/run_h100_identity_maxseq_smoke.sh",
        "scripts/run_slurm_h100_identity_case.sh",
        "scripts/formal_run_with_gpu_monitor.sh",
        "scripts/run_with_durable_log.sh",
        "scripts/slurm_h100_identity_maxseq_smoke.sh",
        "scripts/slurm_h100_identity_nccl_smoke.sh",
    ),
)
def test_source_manifest_must_include_every_h100_launch_shell(fixture, relative):
    def omit_shell(value):
        source = json.loads(json.dumps(fixture["source"]))
        source["payload"]["entries"] = [
            entry
            for entry in source["payload"]["entries"]
            if entry["path"] != relative
        ]
        body = {key: source[key] for key in ("schema_version", "kind", "payload")}
        source["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
        path = fixture["external"] / f"source-without-{Path(relative).name}.json"
        path.write_bytes(canonical(source) + b"\n")
        value["source"].update(
            manifest_path=str(path), manifest_sha256=source["sha256"]
        )

    result = invoke(
        fixture,
        f"source-omits-{Path(relative).stem}",
        mutate=omit_shell,
    )
    assert result.returncode != 0
    assert Path(relative).name in result.stderr
    assert calls(fixture) == []


def test_source_manifest_must_scan_scripts_code_root(fixture):
    def omit_scripts_root(value):
        source = json.loads(json.dumps(fixture["source"]))
        source["payload"]["code_roots"] = ["src/tabicl"]
        body = {key: source[key] for key in ("schema_version", "kind", "payload")}
        source["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
        path = fixture["external"] / "source-without-scripts-code-root.json"
        path.write_bytes(canonical(source) + b"\n")
        value["source"].update(
            manifest_path=str(path), manifest_sha256=source["sha256"]
        )

    result = invoke(
        fixture,
        "source-omits-scripts-code-root",
        mutate=omit_scripts_root,
    )
    assert result.returncode != 0
    assert "source manifest omits required code roots: ['scripts']" in result.stderr
    assert calls(fixture) == []


def test_preflight_negatives_never_reach_scheduler(fixture):
    scenarios = (
        (
            "bad-env",
            lambda value: value.update(
                environment_sha256_by_world_size={"1": "1" * 64, "2": "1" * 64}
            ),
        ),
        (
            "unsafe-export",
            lambda value: value["source"].update(candidate_repository="bad,repository"),
        ),
        (
            "leading-dash-repository",
            lambda value: value["source"].update(candidate_repository="--upload-pack=evil"),
        ),
        (
            "local-repository",
            lambda value: value["source"].update(candidate_repository="/tmp/local.git"),
        ),
        (
            "file-url-repository",
            lambda value: value["source"].update(candidate_repository="file:///tmp/local.git"),
        ),
        (
            "repository-ref-drift",
            lambda value: value["source"].update(candidate_ref="refs/heads/other"),
        ),
        (
            "scheduler-digest-mismatch",
            lambda value: value["scheduler_commands"]["sbatch"].update(
                sha256="0" * 64
            ),
        ),
        (
            "nvidia-smi-digest-mismatch",
            lambda value: value["runtime"].update(
                nvidia_smi_sha256="0" * 64
            ),
        ),
        (
            "case-work-overlaps-source",
            lambda value: value["runtime"].update(
                case_work_root=str(fixture["exact"])
            ),
        ),
        (
            "case-work-contains-artifacts",
            lambda value: value["runtime"].update(
                case_work_root=str(Path(value["artifact_root"]).parent)
            ),
        ),
        (
            "hostile-resource-field",
            lambda value: value.update(scheduler={"gpus": 99}),
        ),
    )
    for name, mutation in scenarios:
        result = invoke(fixture, name, mutate=mutation)
        assert result.returncode != 0
        assert calls(fixture) == []
        assert not Path(overlay(fixture, name)["artifact_root"]).exists()


def test_same_work_and_artifact_filesystem_fails_before_namespace_or_scheduler(
    fixture,
):
    result = invoke(
        fixture,
        "same-work-artifact-filesystem",
        real_filesystem_isolation=True,
    )
    assert result.returncode != 0
    assert "different filesystem" in result.stderr
    assert calls(fixture) == []
    assert not Path(
        overlay(fixture, "same-work-artifact-filesystem")["artifact_root"]
    ).exists()


def test_filesystem_device_drift_while_held_rolls_back_before_release(fixture):
    result = invoke(
        fixture,
        "filesystem-device-drift",
        filesystem_isolation_drift=True,
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert not [event for event in events if event["cmd"] == "scontrol"]
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job_id)] for job_id in range(1012, 1000, -1)
    ]


def test_filesystem_allocation_unit_drift_while_held_rolls_back_before_release(
    fixture,
):
    result = invoke(
        fixture,
        "filesystem-allocation-unit-drift",
        capacity_unit_drift=True,
    )
    assert result.returncode != 0
    assert "physical allocation model changed" in result.stderr
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert not [event for event in events if event["cmd"] == "scontrol"]
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job_id)] for job_id in range(1012, 1000, -1)
    ]


def test_public_ref_drift_while_held_rolls_back_before_release(fixture):
    result = invoke(
        fixture,
        "repository-ref-drift-held",
        {"git-ref-drift-after-preflight": "1"},
    )
    assert result.returncode != 0
    events = calls(fixture)
    assert len([event for event in events if event["cmd"] == "sbatch"]) == 12
    assert not [event for event in events if event["cmd"] == "scontrol"]
    assert [event["argv"] for event in events if event["cmd"] == "scancel"] == [
        [str(job_id)] for job_id in range(1012, 1000, -1)
    ]
    root = Path(overlay(fixture, "repository-ref-drift-held")["artifact_root"])
    assert not (root / "submission-receipt.json").exists()


def test_receipt_validator_rejects_forged_repository_binding(fixture):
    result = invoke(fixture, "receipt-repository-tamper")
    assert result.returncode == 0, result.stderr
    root = Path(overlay(fixture, "receipt-repository-tamper")["artifact_root"])
    receipt_path = root / "submission-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["payload"]["repository_binding"]["repository_url"] = (
        "file:///tmp/forged.git"
    )
    body = {
        key: receipt[key] for key in ("schema_version", "kind", "payload")
    }
    receipt["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    receipt_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    matrix_spec = importlib.util.spec_from_file_location(
        "gate_repository_tamper_matrix",
        ROOT / "scripts/run_h100_identity_validation.py",
    )
    matrix = importlib.util.module_from_spec(matrix_spec)
    sys.modules[matrix_spec.name] = matrix
    matrix_spec.loader.exec_module(matrix)
    with pytest.raises(ValueError, match="repository binding"):
        matrix.validate_gate_submission_receipt(
            receipt_path,
            expected_commit_sha=fixture["commit"],
            expected_tree_sha=fixture["tree"],
            expected_source_manifest_sha256=fixture["source"]["sha256"],
            expected_checkpoint_ceiling_bytes=300_000_000,
            max_bytes=1_000_000,
        )


def test_receipt_validator_rejects_forged_environment_transaction_binding(fixture):
    result = invoke(fixture, "receipt-environment-transaction-tamper")
    assert result.returncode == 0, result.stderr
    root = Path(
        overlay(fixture, "receipt-environment-transaction-tamper")["artifact_root"]
    )
    held_path = root / "held-plan.json"
    receipt_path = root / "submission-receipt.json"
    held = json.loads(held_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    for value in (held, receipt):
        value["payload"]["environment_transaction"][
            "one_gpu_environment_sha256"
        ] = "0" * 64
    held_body = {
        key: held[key] for key in ("schema_version", "kind", "payload")
    }
    held["sha256"] = hashlib.sha256(canonical(held_body)).hexdigest()
    receipt["payload"]["held_plan_sha256"] = held["sha256"]
    receipt_body = {
        key: receipt[key] for key in ("schema_version", "kind", "payload")
    }
    receipt["sha256"] = hashlib.sha256(canonical(receipt_body)).hexdigest()
    for path, value in ((held_path, held), (receipt_path, receipt)):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.write_bytes(canonical(value) + b"\n")

    matrix_spec = importlib.util.spec_from_file_location(
        "gate_environment_transaction_tamper_matrix",
        ROOT / "scripts/run_h100_identity_validation.py",
    )
    matrix = importlib.util.module_from_spec(matrix_spec)
    sys.modules[matrix_spec.name] = matrix
    matrix_spec.loader.exec_module(matrix)
    with pytest.raises(ValueError, match="environment transaction.*binding mismatch"):
        matrix.validate_gate_submission_receipt(
            receipt_path,
            expected_commit_sha=fixture["commit"],
            expected_tree_sha=fixture["tree"],
            expected_source_manifest_sha256=fixture["source"]["sha256"],
            expected_checkpoint_ceiling_bytes=300_000_000,
            max_bytes=1_000_000,
        )


def test_receipt_validator_rejects_forged_physical_capacity_components(fixture):
    result = invoke(fixture, "receipt-capacity-tamper")
    assert result.returncode == 0, result.stderr
    root = Path(overlay(fixture, "receipt-capacity-tamper")["artifact_root"])
    held_path = root / "held-plan.json"
    receipt_path = root / "submission-receipt.json"
    held = json.loads(held_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    held["payload"]["capacity"]["allocation_unit_bytes"] = 2
    receipt["payload"]["capacity"]["allocation_unit_bytes"] = 2
    held_body = {
        key: held[key] for key in ("schema_version", "kind", "payload")
    }
    held["sha256"] = hashlib.sha256(canonical(held_body)).hexdigest()
    receipt["payload"]["held_plan_sha256"] = held["sha256"]
    receipt_body = {
        key: receipt[key] for key in ("schema_version", "kind", "payload")
    }
    receipt["sha256"] = hashlib.sha256(canonical(receipt_body)).hexdigest()
    for path, value in ((held_path, held), (receipt_path, receipt)):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.write_bytes(canonical(value) + b"\n")

    matrix_spec = importlib.util.spec_from_file_location(
        "gate_capacity_tamper_matrix",
        ROOT / "scripts/run_h100_identity_validation.py",
    )
    matrix = importlib.util.module_from_spec(matrix_spec)
    sys.modules[matrix_spec.name] = matrix
    matrix_spec.loader.exec_module(matrix)
    with pytest.raises(ValueError, match="physical capacity formula"):
        matrix.validate_gate_submission_receipt(
            receipt_path,
            expected_commit_sha=fixture["commit"],
            expected_tree_sha=fixture["tree"],
            expected_source_manifest_sha256=fixture["source"]["sha256"],
            expected_checkpoint_ceiling_bytes=300_000_000,
            max_bytes=1_000_000,
        )


def test_preexisting_namespace_dirty_or_attached_checkout_and_missing_scheduler_fail_closed(fixture):
    root = Path(overlay(fixture, "preexisting")["artifact_root"])
    root.mkdir(parents=True)
    preexisting = invoke(fixture, "preexisting")
    assert preexisting.returncode != 0 and calls(fixture) == []

    dirty_path = fixture["exact"] / "UNTRACKED"
    dirty_path.write_text("dirty\n")
    try:
        dirty = invoke(fixture, "dirty-exact")
        assert dirty.returncode != 0 and calls(fixture) == []
    finally:
        dirty_path.unlink()

    run(["/usr/bin/git", "-C", str(fixture["exact"]), "switch", "-q", "-c", "attached-test"])
    try:
        attached = invoke(fixture, "attached-exact")
        assert attached.returncode != 0 and calls(fixture) == []
    finally:
        run(
            [
                "/usr/bin/git",
                "-C",
                str(fixture["exact"]),
                "checkout",
                "-q",
                "--detach",
                fixture["commit"],
            ]
        )
        run(["/usr/bin/git", "-C", str(fixture["exact"]), "branch", "-D", "attached-test"], stdout=subprocess.PIPE)

    missing = invoke(
        fixture,
        "missing-scheduler",
        mutate=lambda value: value["scheduler_commands"]["scontrol"].update(
            path=str(fixture["fakebin"] / "missing-scontrol")
        ),
    )
    assert missing.returncode != 0 and calls(fixture) == []
    assert not Path(overlay(fixture, "missing-scheduler")["artifact_root"]).exists()


def test_receipt_validator_rejects_mutated_durable_journal(fixture):
    result = invoke(fixture, "journal-tamper")
    assert result.returncode == 0, result.stderr
    root = Path(overlay(fixture, "journal-tamper")["artifact_root"])
    journal = root / "transaction-journal.jsonl"
    journal.write_bytes(journal.read_bytes() + b"{}\n")
    matrix_spec = importlib.util.spec_from_file_location(
        "gate_tamper_matrix",
        ROOT / "scripts/run_h100_identity_validation.py",
    )
    matrix = importlib.util.module_from_spec(matrix_spec)
    sys.modules[matrix_spec.name] = matrix
    matrix_spec.loader.exec_module(matrix)
    with pytest.raises(ValueError, match="journal digest"):
        matrix.validate_gate_submission_receipt(
            root / "submission-receipt.json",
            expected_commit_sha=fixture["commit"],
            expected_tree_sha=fixture["tree"],
            expected_source_manifest_sha256=fixture["source"]["sha256"],
            expected_checkpoint_ceiling_bytes=300_000_000,
            max_bytes=1_000_000,
        )


def test_gate_shell_entrypoints_are_syntax_valid_and_static_gpu_contracts_are_split():
    for relative in (
        "scripts/submit_h100_identity_validation.sh",
        "scripts/run_h100_identity_maxseq_smoke.sh",
        "scripts/run_slurm_h100_identity_case.sh",
        "scripts/slurm_h100_identity_maxseq_smoke.sh",
        "scripts/slurm_h100_identity_nccl_smoke.sh",
    ):
        subprocess.run(["/bin/bash", "-n", str(ROOT / relative)], check=True)
    one = (ROOT / "scripts/slurm_h100_identity_maxseq_smoke.sh").read_text()
    two = (ROOT / "scripts/slurm_h100_identity_nccl_smoke.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in one and "nccl_2gpu must use" in one
    assert "#SBATCH --gres=gpu:2" in two and "reserved for nccl_2gpu" in two
