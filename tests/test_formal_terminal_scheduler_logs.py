from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "finalize_formal_scheduler_logs.py"
)
REPOSITORY_HELPER = (
    Path(__file__).parents[1] / "scripts" / "verify_git_repository.py"
)
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
TIME_LIMIT_BY_STAGE = {
    "stage1": "14-00:00:00",
    "stage2": "3-00:00:00",
    "stage3": "1-00:00:00",
}


def _load():
    spec = importlib.util.spec_from_file_location("formal_terminal_logs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _manifest(kind: str, payload: dict) -> dict:
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _write(path: Path, value: dict) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _repository_binding() -> dict:
    spec = importlib.util.spec_from_file_location(
        "formal_terminal_repository_helper", REPOSITORY_HELPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expected_repository_binding(
        expected_commit_sha="a" * 40,
        git_sha256="d" * 64,
    )


def _manifest_sha256(kind: str, payload: dict) -> str:
    return _manifest(kind, payload)["sha256"]


def _seed_sha256(seed: int) -> str:
    return _manifest_sha256(
        "seed",
        {
            "np_seed": seed,
            "torch_seed": seed,
            "identity_rng_seed": seed,
            "world_size": 1,
        },
    )


def _treatment_sha256(arm: str, seed: int) -> str:
    body = {
        "schema_version": 1,
        "row_identity_mode": arm,
        "identity_rng_seed": seed,
        "seed_policy": "sha256-domain-separated-base-seed-and-rank-v1",
        "sampler_version": (
            "tabicl-temporary-identity/randperm-cpu-v1"
            if arm == "temporary"
            else None
        ),
        "world_size": 1,
    }
    return _manifest_sha256(
        "treatment",
        {**body, "manifest_sha256": hashlib.sha256(_canonical(body)).hexdigest()},
    )


def _rewrite_payload(path: Path, mutate) -> dict:
    value = json.loads(path.read_text())
    mutate(value["payload"])
    rewritten = _manifest(value["kind"], value["payload"])
    _write(path, rewritten)
    return rewritten


def _fixture(
    tmp_path: Path,
    *,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    derived_exit_code: str = "0:0",
):
    seed = 42
    artifact = tmp_path / "formal-seed42"
    logs = artifact / "scheduler-logs"
    completions = artifact / "runtime-completions"
    logs.mkdir(parents=True)
    completions.mkdir()
    transaction_id = "0123456789abcdef0123456789abcdef"
    jobs = []
    ledger_entries = []
    artifact_bindings = {}
    finalized_paths = {}
    checkpoint_paths = {}
    rows = []
    job_id = 1000
    parent_by_arm: dict[str, str] = {}
    source_sha256 = hashlib.sha256(b"source").hexdigest()
    environment_sha256 = hashlib.sha256(b"environment").hexdigest()
    seed_sha256 = _seed_sha256(seed)
    treatment_by_arm = {arm: _treatment_sha256(arm, seed) for arm in ARMS}
    repository_binding = _repository_binding()
    for stage, terminal_step in STAGES:
        stage_digests = {
            name: hashlib.sha256(f"{name}-{stage}".encode()).hexdigest()
            for name in (
                "prior",
                "architecture",
                "optimizer",
                "scientific",
            )
        }
        cohort_sha256 = _manifest_sha256(
            "cohort_protocol",
            {
                "stage": stage,
                "terminal_step": terminal_step,
                "source_sha256": source_sha256,
                "environment_sha256": environment_sha256,
                "architecture_sha256": stage_digests["architecture"],
                "prior_sha256": stage_digests["prior"],
                "optimizer_sha256": stage_digests["optimizer"],
                "seed_sha256": seed_sha256,
                "scientific_config_sha256": stage_digests["scientific"],
            },
        )
        for arm in ARMS:
            job_id += 1
            identifier = str(job_id)
            job_name = f"tabicl-{transaction_id}-{arm}-s{STAGES.index((stage, terminal_step)) + 1}"
            stdout = logs / f"{arm}-{stage}.out"
            stderr = logs / f"{arm}-{stage}.err"
            stdout.write_text(f"bootstrap {arm} {stage}\n")
            stderr.write_bytes(b"")
            completion = completions / f"{arm}-{stage}.json"
            stage_dir = artifact / "arms" / arm / stage
            stage_dir.mkdir(parents=True)
            checkpoint = stage_dir / f"step-{terminal_step}.ckpt"
            checkpoint.write_bytes(f"checkpoint:{arm}:{stage}".encode())
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            finalized = stage_dir / "finalized-checkpoint.json"
            arm_protocol_sha256 = _manifest_sha256(
                "arm_protocol",
                {
                    "cohort_protocol_sha256": cohort_sha256,
                    "mode": arm,
                    "treatment_sha256": treatment_by_arm[arm],
                },
            )
            entry = {
                "arm": arm,
                "stage": stage,
                "terminal_step": terminal_step,
                "upstream_identity": f"formal-seed42:{arm}:{stage}",
                "artifact_identity": f"formal-seed42.{arm}.{stage}.final",
                "checkpoint_relpath": f"arms/{arm}/{stage}/step-{terminal_step}.ckpt",
                "finalized_manifest_relpath": f"arms/{arm}/{stage}/finalized-checkpoint.json",
                "np_seed": seed,
                "torch_seed": seed,
                "identity_rng_seed": seed,
                "world_size": 1,
                "cuda_device_count": 1,
                "max_checkpoint_bytes": 10_000,
                "source_sha256": source_sha256,
                "environment_sha256": environment_sha256,
                "prior_sha256": stage_digests["prior"],
                "architecture_sha256": stage_digests["architecture"],
                "optimizer_sha256": stage_digests["optimizer"],
                "scientific_sha256": stage_digests["scientific"],
                "cohort_protocol_sha256": cohort_sha256,
                "arm_protocol_sha256": arm_protocol_sha256,
            }
            finalized_manifest = _manifest(
                "finalized_checkpoint",
                {
                    "study_id": "formal-seed42",
                    "arm": arm,
                    "stage": stage,
                    "terminal_step": terminal_step,
                    "upstream_identity": entry["upstream_identity"],
                    "artifact_identity": entry["artifact_identity"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_size": checkpoint.stat().st_size,
                    "provenance_sha256": hashlib.sha256(
                        f"provenance:{arm}:{stage}".encode()
                    ).hexdigest(),
                    "source_sha256": source_sha256,
                    "environment_sha256": environment_sha256,
                    "prior_sha256": stage_digests["prior"],
                    "architecture_sha256": stage_digests["architecture"],
                    "optimizer_sha256": stage_digests["optimizer"],
                    "seed_sha256": seed_sha256,
                    "treatment_sha256": treatment_by_arm[arm],
                    "scientific_sha256": stage_digests["scientific"],
                    "cohort_protocol_sha256": cohort_sha256,
                    "arm_protocol_sha256": arm_protocol_sha256,
                    "cuda_device_count": 1,
                    "max_checkpoint_bytes": 10_000,
                },
            )
            _write(finalized, finalized_manifest)
            binding = {
                "finalized_manifest_path": str(finalized),
                "finalized_manifest_sha256": finalized_manifest["sha256"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_size": checkpoint.stat().st_size,
                "provenance_sha256": finalized_manifest["payload"][
                    "provenance_sha256"
                ],
                "seed_sha256": seed_sha256,
                "treatment_sha256": treatment_by_arm[arm],
            }
            jobs.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "terminal_step": terminal_step,
                    "time_limit": TIME_LIMIT_BY_STAGE[stage],
                    "job_id": identifier,
                    "cluster": None,
                    "job_name": job_name,
                    "parent_job_id": parent_by_arm.get(arm),
                    "scheduler_stdout": str(stdout),
                    "scheduler_stderr": str(stderr),
                    "completion_path": str(completion),
                }
            )
            parent_by_arm[arm] = identifier
            ledger_entries.append(entry)
            artifact_bindings[(arm, stage)] = binding
            finalized_paths[(arm, stage)] = finalized
            checkpoint_paths[(arm, stage)] = checkpoint
            rows.append(
                f"{identifier}|{job_name}|{state}|{exit_code}|{derived_exit_code}|"
            )

    sacct = tmp_path / "sacct"
    sacct.write_text("#!/bin/bash\nprintf '%s\\n' " + " ".join(repr(row) for row in rows) + "\n")
    sacct.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    sacct_sha256 = hashlib.sha256(sacct.read_bytes()).hexdigest()
    ledger = _manifest(
        "transaction_ledger",
        {"study_id": "formal-seed42", "entries": ledger_entries},
    )
    ledger_path = artifact / "transaction-ledger.json"
    _write(ledger_path, ledger)
    receipt = _manifest(
        "held_submission_receipt",
        {
            "study_id": "formal-seed42",
            "seed": seed,
            "transaction_id": transaction_id,
            "transaction_ledger_sha256": ledger["sha256"],
            "source_commit_sha": "a" * 40,
            "source_tree_sha": "b" * 40,
            "repository_binding": repository_binding,
            "jobs_held_at_publication": True,
            "run_log_ceiling_bytes": 10_000,
            "manifest_ceiling_bytes": 2_000_000,
            "runtime_completion_ceiling_bytes": 65_536,
            "terminal_log_attestation_path": str(
                artifact / "terminal-scheduler-logs.json"
            ),
            "transaction_commit_path": str(
                artifact / "transaction-committed.json"
            ),
            "terminal_log_attestation_ceiling_bytes": 131_072,
            "scheduler": {
                "partition": "h100",
                "qos": "long",
                "cpus_per_task": 64,
                "memory_mb": 131_072,
                "gpus_per_job": 1,
                "time_limit_by_stage": dict(TIME_LIMIT_BY_STAGE),
                "sacct_path": str(sacct),
                "sacct_sha256": sacct_sha256,
                "scontrol_sha256": "c" * 64,
            },
            "job_ids": [job["job_id"] for job in jobs],
            "jobs": jobs,
        },
    )
    receipt_path = artifact / "submission-receipt.json"
    _write(receipt_path, receipt)
    commit = _manifest(
        "formal_submission_commit",
        {
            "study_id": "formal-seed42",
            "transaction_id": transaction_id,
            "submission_receipt_sha256": receipt["sha256"],
            "transaction_ledger_sha256": ledger["sha256"],
            "job_ids": [job["job_id"] for job in jobs],
        },
    )
    commit_path = artifact / "transaction-committed.json"
    _write(commit_path, commit)
    for job in jobs:
        stdout_size = Path(job["scheduler_stdout"]).stat().st_size
        stderr_size = Path(job["scheduler_stderr"]).stat().st_size
        stage_index = [name for name, _step in STAGES].index(job["stage"])
        if stage_index == 0:
            parent_lineage = None
        else:
            parent_stage = STAGES[stage_index - 1][0]
            parent_job = next(
                item
                for item in jobs
                if item["arm"] == job["arm"] and item["stage"] == parent_stage
            )
            parent_lineage = {
                **artifact_bindings[(job["arm"], parent_stage)],
                "job_id": parent_job["job_id"],
                "stage": parent_stage,
            }
        completion = _manifest(
            "formal_job_completion",
            {
                "study_id": "formal-seed42",
                "transaction_id": transaction_id,
                "submission_receipt_sha256": receipt["sha256"],
                "transaction_ledger_sha256": ledger["sha256"],
                "source_commit_sha": "a" * 40,
                "source_tree_sha": "b" * 40,
                "repository_binding": repository_binding,
                "job_id": job["job_id"],
                "job_name": job["job_name"],
                "arm": job["arm"],
                "stage": job["stage"],
                "seed": seed,
                "scheduler": {
                    "partition": "h100",
                    "qos": "long",
                    "cpus_per_task": 64,
                    "memory_mb": 131_072,
                    "gpus_per_job": 1,
                    "time_limit": TIME_LIMIT_BY_STAGE[job["stage"]],
                    "query_sha256": hashlib.sha256(
                        f"scontrol:{job['job_id']}".encode()
                    ).hexdigest(),
                },
                "cuda_visible_devices": "0",
                "gpu_name": "NVIDIA H100",
                "gpu_uuid": "GPU-fixture",
                "scheduler_stdout_path": job["scheduler_stdout"],
                "scheduler_stderr_path": job["scheduler_stderr"],
                "scheduler_log_ceiling_bytes": 10_000,
                "manifest_ceiling_bytes": 2_000_000,
                "scheduler_stdout_observed_size_at_completion": stdout_size,
                "scheduler_stderr_observed_size_at_completion": stderr_size,
                "scheduler_logs_terminal_verified": False,
                "finalized_artifact": artifact_bindings[
                    (job["arm"], job["stage"])
                ],
                "parent_lineage": parent_lineage,
                "stage_exit_code": 0,
                "completed": True,
            },
        )
        _write(Path(job["completion_path"]), completion)
    return {
        "artifact": artifact,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "ledger": ledger,
        "ledger_path": ledger_path,
        "commit_path": commit_path,
        "jobs": jobs,
        "ledger_entries": ledger_entries,
        "artifact_bindings": artifact_bindings,
        "finalized_paths": finalized_paths,
        "checkpoint_paths": checkpoint_paths,
    }


def _finalize(module, fixture):
    return module.finalize(
        submission_receipt=fixture["receipt_path"],
        transaction_ledger=fixture["ledger_path"],
        artifact_root=fixture["artifact"],
        max_metadata_bytes=3_000_000,
    )


def _rewrite_ledger_and_receipt(fixture, mutate) -> None:
    ledger = _rewrite_payload(fixture["ledger_path"], mutate)
    receipt = _rewrite_payload(
        fixture["receipt_path"],
        lambda payload: payload.update(
            transaction_ledger_sha256=ledger["sha256"]
        ),
    )
    _rewrite_payload(
        fixture["commit_path"],
        lambda payload: payload.update(
            transaction_ledger_sha256=ledger["sha256"],
            submission_receipt_sha256=receipt["sha256"],
        ),
    )


def _rewrite_receipt_and_commit(fixture, mutate) -> None:
    receipt = _rewrite_payload(fixture["receipt_path"], mutate)
    _rewrite_payload(
        fixture["commit_path"],
        lambda payload: payload.update(
            submission_receipt_sha256=receipt["sha256"]
        ),
    )


def test_terminal_finalizer_binds_all_jobs_completions_and_stable_spool_bytes(tmp_path):
    module = _load()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    fixture = _fixture(tmp_path)
    attestation = _finalize(module, fixture)
    assert attestation["kind"] == "formal_terminal_scheduler_logs"
    payload = attestation["payload"]
    assert payload["terminal_verified"] is True
    assert payload["submission_receipt_sha256"] == fixture["receipt"]["sha256"]
    assert payload["repository_binding"] == fixture["receipt"]["payload"][
        "repository_binding"
    ]
    assert payload["transaction_ledger_sha256"] == fixture["ledger"]["sha256"]
    assert payload["path_format"] == "artifact_root_relative_posix_v1"
    assert len(payload["jobs"]) == 9
    assert all(job["state"] == "COMPLETED" for job in payload["jobs"])
    assert all(job["exit_code"] == "0:0" for job in payload["jobs"])
    assert all(job["derived_exit_code"] == "0:0" for job in payload["jobs"])
    assert all(job["completion"]["seed"] == 42 for job in payload["jobs"])
    assert all(
        job["completion"]["scheduler"]["gpus_per_job"] == 1
        and "H100" in job["completion"]["gpu_name"]
        and job["completion"]["gpu_uuid"].startswith("GPU-")
        for job in payload["jobs"]
    )
    assert all(
        job["finalized_artifact"]["independent_stable_validations"] == 2
        and len(job["finalized_artifact"]["finalized_manifest_file_sha256"])
        == 64
        for job in payload["jobs"]
    )
    assert all(
        job["parent_lineage"] is None
        for job in payload["jobs"]
        if job["stage"] == "stage1"
    )
    assert all(
        job["parent_lineage"]["stage"]
        == ("stage1" if job["stage"] == "stage2" else "stage2")
        for job in payload["jobs"]
        if job["stage"] != "stage1"
    )
    assert all(
        log["fatal_signature_scan_passed"] and log["stable_reads"] == 2
        for job in payload["jobs"]
        for log in job["scheduler_logs"]
    )
    output = fixture["artifact"] / "terminal-scheduler-logs.json"
    assert json.loads(output.read_text()) == attestation
    encoded_paths = []
    for job in payload["jobs"]:
        encoded_paths.append(job["completion"]["path"])
        encoded_paths.extend(log["path"] for log in job["scheduler_logs"])
        encoded_paths.extend(
            (
                job["finalized_artifact"]["binding"]["finalized_manifest_path"],
                job["finalized_artifact"]["binding"]["checkpoint_path"],
            )
        )
        if job["parent_lineage"] is not None:
            encoded_paths.extend(
                (
                    job["parent_lineage"]["finalized_manifest_path"],
                    job["parent_lineage"]["checkpoint_path"],
                )
            )
    assert all(
        not Path(value).is_absolute()
        and ".." not in Path(value).parts
        and Path(value).as_posix() == value
        for value in encoded_paths
    )
    assert str(fixture["artifact"]) not in output.read_text()
    assert output.stat().st_size <= module.TERMINAL_LOG_ATTESTATION_CEILING_BYTES
    with pytest.raises(FileExistsError, match="write-once"):
        _finalize(module, fixture)


def test_terminal_attestation_size_is_independent_of_a_long_artifact_parent(tmp_path):
    module = _load()
    long_parent = tmp_path
    for index in range(14):
        long_parent /= f"parent-{index:02d}-" + "x" * 220
    fixture = _fixture(long_parent)

    attestation = _finalize(module, fixture)
    output = fixture["artifact"] / "terminal-scheduler-logs.json"
    raw = output.read_bytes()

    assert len(raw) <= module.TERMINAL_LOG_ATTESTATION_CEILING_BYTES
    assert str(long_parent).encode() not in raw
    assert attestation["payload"]["path_format"] == (
        "artifact_root_relative_posix_v1"
    )


def test_terminal_finalizer_cli_requires_and_runs_in_isolated_mode(tmp_path):
    fixture = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--submission-receipt",
            str(fixture["receipt_path"]),
            "--transaction-ledger",
            str(fixture["ledger_path"]),
            "--artifact-root",
            str(fixture["artifact"]),
            "--max-metadata-bytes",
            "3000000",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["kind"] == "formal_terminal_scheduler_logs"


@pytest.mark.parametrize(
    "state,exit_code,derived_exit_code",
    [
        ("RUNNING", "0:0", "0:0"),
        ("FAILED", "1:0", "1:0"),
        ("COMPLETED", "1:0", "0:0"),
        ("COMPLETED", "0:0", "1:0"),
    ],
)
def test_terminal_finalizer_rejects_nonterminal_or_failed_allocations(
    tmp_path, state, exit_code, derived_exit_code
):
    module = _load()
    fixture = _fixture(
        tmp_path,
        state=state,
        exit_code=exit_code,
        derived_exit_code=derived_exit_code,
    )
    with pytest.raises(ValueError, match="not successfully terminal"):
        _finalize(module, fixture)
    assert not (fixture["artifact"] / "terminal-scheduler-logs.json").exists()


@pytest.mark.parametrize(
    "fatal_text", ["OOM while allocating\n", "ENOSPC\n", "Traceback (most recent call last):\n"]
)
def test_terminal_finalizer_rejects_fatal_scheduler_log_signatures(
    tmp_path, fatal_text
):
    module = _load()
    fixture = _fixture(tmp_path)
    Path(fixture["jobs"][0]["scheduler_stderr"]).write_text(fatal_text)
    with pytest.raises(ValueError, match="fatal signatures"):
        _finalize(module, fixture)
    assert not (fixture["artifact"] / "terminal-scheduler-logs.json").exists()


def test_terminal_finalizer_rejects_symlinked_scheduler_log(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["scheduler_stdout"])
    outside = tmp_path / "outside.log"
    outside.write_text(target.read_text())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="physical regular|following links"):
        _finalize(module, fixture)


def test_terminal_finalizer_rejects_symlinked_runtime_completion(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["completion_path"])
    outside = tmp_path / "outside-completion.json"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="physical regular|following links"):
        _finalize(module, fixture)


@pytest.mark.parametrize(
    "directory,mutation",
    [
        ("scheduler-logs", "extra"),
        ("scheduler-logs", "missing"),
        ("runtime-completions", "extra"),
        ("runtime-completions", "missing"),
    ],
)
def test_terminal_finalizer_requires_exact_spool_and_completion_file_sets(
    tmp_path, directory, mutation
):
    module = _load()
    fixture = _fixture(tmp_path)
    root = fixture["artifact"] / directory
    if mutation == "extra":
        (root / "unknown.entry").write_text("unexpected\n")
    else:
        next(root.iterdir()).unlink()
    with pytest.raises(ValueError, match="missing or unknown entry"):
        _finalize(module, fixture)


def test_completion_file_and_manifest_digests_are_revalidated_before_publish(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["completion_path"])
    expected_file_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    original = module._read_stable_regular
    reads = 0

    def count_completion_reads(path, *, max_bytes, where, executable=False):
        nonlocal reads
        if path == target:
            reads += 1
        return original(
            path, max_bytes=max_bytes, where=where, executable=executable
        )

    monkeypatch.setattr(module, "_read_stable_regular", count_completion_reads)
    attestation = _finalize(module, fixture)
    first = next(
        item
        for item in attestation["payload"]["jobs"]
        if item["job_id"] == fixture["jobs"][0]["job_id"]
    )
    assert reads == 2
    assert first["completion"]["file_sha256"] == expected_file_sha256


def test_terminal_finalizer_rejects_change_between_post_hash_stable_reads(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["scheduler_stdout"])
    original = module._read_stable_regular
    mutated = False

    def mutate_after_first(path, *, max_bytes, where, executable=False):
        nonlocal mutated
        result = original(
            path, max_bytes=max_bytes, where=where, executable=executable
        )
        if path == target and where == "scheduler stdout log" and not mutated:
            mutated = True
            target.write_bytes(result[0] + b"late scheduler byte\n")
        return result

    monkeypatch.setattr(module, "_read_stable_regular", mutate_after_first)
    with pytest.raises(ValueError, match="changed after its terminal hash"):
        _finalize(module, fixture)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["entries"][0].update(np_seed=43),
            "entry binding",
        ),
        (
            lambda payload: payload["entries"][0].update(
                checkpoint_relpath="arms/rope/stage1/other.ckpt"
            ),
            "entry binding",
        ),
        (
            lambda payload: payload["entries"][0].update(source_sha256="x" * 64),
            "digest",
        ),
        (
            lambda payload: payload["entries"][0].update(source_sha256=7),
            "digest",
        ),
        (
            lambda payload: payload["entries"][0].update(
                cohort_protocol_sha256="f" * 64
            ),
            "cohort protocol digest is not derived|differs within a stage",
        ),
        (
            lambda payload: payload["entries"][0].update(
                arm_protocol_sha256="e" * 64
            ),
            "arm protocol digest is not derived",
        ),
        (
            lambda payload: payload["entries"][0].update(unexpected=True),
            "schema mismatch",
        ),
        (
            lambda payload: payload["entries"].__setitem__(
                slice(0, 2), list(reversed(payload["entries"][:2]))
            ),
            "entry binding",
        ),
    ],
)
def test_terminal_finalizer_requires_exact_canonical_ledger(
    tmp_path, mutate, match
):
    module = _load()
    fixture = _fixture(tmp_path)
    _rewrite_ledger_and_receipt(fixture, mutate)
    with pytest.raises(ValueError, match=match):
        _finalize(module, fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("scheduler", {}),
        (
            "scheduler",
            {
                "partition": "h100",
                "qos": "long",
                "cpus_per_task": 32,
                "memory_mb": 131_072,
                "gpus_per_job": 1,
                "query_sha256": "a" * 64,
            },
        ),
        ("cuda_visible_devices", "0,1"),
        ("gpu_name", "NVIDIA A100"),
        ("gpu_uuid", "not-a-gpu"),
    ],
)
def test_terminal_finalizer_rejects_completion_seed_scheduler_or_gpu_drift(
    tmp_path, field, value
):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["completion_path"])
    _rewrite_payload(target, lambda payload: payload.update({field: value}))
    with pytest.raises(ValueError, match="scheduler schema mismatch|binding mismatch"):
        _finalize(module, fixture)


@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_terminal_finalizer_requires_exact_completion_parent_lineage(
    tmp_path, stage
):
    module = _load()
    fixture = _fixture(tmp_path)
    job = next(
        item
        for item in fixture["jobs"]
        if item["arm"] == "rope" and item["stage"] == stage
    )
    target = Path(job["completion_path"])
    _rewrite_payload(
        target,
        lambda payload: payload.update(
            parent_lineage=(
                {"unexpected": True}
                if stage == "stage1"
                else {**payload["parent_lineage"], "job_id": "999999"}
            )
        ),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        _finalize(module, fixture)


def test_terminal_finalizer_requires_completion_artifact_exact_equality(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    target = Path(fixture["jobs"][0]["completion_path"])
    _rewrite_payload(
        target,
        lambda payload: payload["finalized_artifact"].update(unexpected=True),
    )
    with pytest.raises(ValueError, match="finalized artifact schema mismatch"):
        _finalize(module, fixture)


def test_terminal_finalizer_rehashes_every_checkpoint_twice(tmp_path, monkeypatch):
    module = _load()
    fixture = _fixture(tmp_path)
    original = module._hash_relative_bounded_regular
    calls = []

    def count(root, relative, *, max_bytes, where):
        calls.append(relative)
        return original(root, relative, max_bytes=max_bytes, where=where)

    monkeypatch.setattr(module, "_hash_relative_bounded_regular", count)
    _finalize(module, fixture)
    assert sorted(calls) == sorted(
        entry["checkpoint_relpath"]
        for entry in fixture["ledger_entries"]
        for _ in range(2)
    )


def test_terminal_finalizer_rejects_symlinked_finalized_artifact_component(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    stage_dir = fixture["artifact"] / "arms" / "rope" / "stage1"
    outside = tmp_path / "outside-stage"
    stage_dir.rename(outside)
    stage_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="component.*following links"):
        _finalize(module, fixture)


def test_terminal_finalizer_rejects_checkpoint_or_finalized_manifest_drift(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    fixture["checkpoint_paths"][("rope", "stage1")].write_bytes(b"replacement")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        _finalize(module, fixture)


def test_terminal_finalizer_uses_receipt_manifest_ceiling_not_metadata_allowance(
    tmp_path,
):
    module = _load()
    fixture = _fixture(tmp_path)
    _rewrite_receipt_and_commit(
        fixture,
        lambda payload: payload.update(manifest_ceiling_bytes=1),
    )
    with pytest.raises(ValueError, match="bounded physical regular file"):
        _finalize(module, fixture)


def test_terminal_finalizer_rejects_finalized_manifest_claim_drift(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    _rewrite_payload(
        fixture["finalized_paths"][("rope", "stage1")],
        lambda payload: payload.update(provenance_sha256="f" * 64),
    )
    with pytest.raises(ValueError, match="completion binding mismatch"):
        _finalize(module, fixture)


def test_terminal_finalizer_revalidates_artifacts_after_terminal_log_scan(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _fixture(tmp_path)
    target_checkpoint = fixture["checkpoint_paths"][("rope", "stage1")]
    target_finalized = fixture["finalized_paths"][("rope", "stage1")]
    last_stderr = Path(fixture["jobs"][-1]["scheduler_stderr"])
    original = module._stable_log_evidence
    mutated = False

    def mutate_after_last_log(path, *, max_bytes, stream):
        nonlocal mutated
        evidence = original(path, max_bytes=max_bytes, stream=stream)
        if path == last_stderr and not mutated:
            mutated = True
            target_checkpoint.write_bytes(b"late but internally consistent checkpoint")
            checkpoint_sha256 = hashlib.sha256(target_checkpoint.read_bytes()).hexdigest()
            _rewrite_payload(
                target_finalized,
                lambda payload: payload.update(
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_size=target_checkpoint.stat().st_size,
                ),
            )
        return evidence

    monkeypatch.setattr(module, "_stable_log_evidence", mutate_after_last_log)
    with pytest.raises(ValueError, match="changed before publication"):
        _finalize(module, fixture)


def test_terminal_finalizer_revalidates_completions_after_terminal_log_scan(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _fixture(tmp_path)
    target_completion = Path(fixture["jobs"][0]["completion_path"])
    last_stderr = Path(fixture["jobs"][-1]["scheduler_stderr"])
    original = module._stable_log_evidence
    mutated = False

    def mutate_after_last_log(path, *, max_bytes, stream):
        nonlocal mutated
        evidence = original(path, max_bytes=max_bytes, stream=stream)
        if path == last_stderr and not mutated:
            mutated = True
            _rewrite_payload(
                target_completion,
                lambda payload: payload.update(gpu_name="NVIDIA H100 late rewrite"),
            )
        return evidence

    monkeypatch.setattr(module, "_stable_log_evidence", mutate_after_last_log)
    with pytest.raises(ValueError, match="completion changed before publication"):
        _finalize(module, fixture)


def test_terminal_finalizer_rejects_noncanonical_receipt_parent_job(tmp_path):
    module = _load()
    fixture = _fixture(tmp_path)
    _rewrite_receipt_and_commit(
        fixture,
        lambda payload: payload["jobs"][3].update(parent_job_id=payload["job_ids"][1]),
    )
    with pytest.raises(ValueError, match="stage lineage is not canonical"):
        _finalize(module, fixture)
