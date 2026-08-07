from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
MATRIX_SCRIPT = ROOT / "scripts" / "run_h100_identity_validation.py"
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_formal_gpu_usage.py"
EXACT_SCRIPT = ROOT / "scripts" / "run_exact_tabicl.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_runtime_source.py"
REPOSITORY_SCRIPT = ROOT / "scripts" / "verify_git_repository.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _repository_binding(matrix, *, commit: str = "1" * 40, git_sha256: str = "9" * 64):
    helper = matrix._load_module("h100_test_git_repository", REPOSITORY_SCRIPT)
    return helper.expected_repository_binding(
        expected_commit_sha=commit,
        git_sha256=git_sha256,
    )


@pytest.fixture(scope="module")
def matrix():
    return _load("h100_identity_matrix", MATRIX_SCRIPT)


def test_canonical_matrix_is_exactly_twelve_cases_with_frozen_composition(matrix):
    cases = matrix.build_matrix()
    ids = [case.case_id for case in cases]

    assert len(ids) == len(set(ids)) == 12
    assert set(ids) == {
        "stage1_rope_one_step",
        "stage1_temporary_one_step",
        "stage1_none_one_step",
        "stage2_rope_maxseq10240",
        "stage2_temporary_maxseq10240",
        "stage2_none_maxseq10240",
        "stage3_rope_maxseq60000_recompute",
        "stage3_temporary_maxseq60000_recompute",
        "stage3_none_maxseq60000_recompute",
        "temporary_cuda_rng_resume",
        "nccl_2gpu",
        "prior_dataloader_resume",
    }
    assert [case.category for case in cases].count("stage1_one_step") == 3
    assert [case.category for case in cases].count("stage2_maxseq") == 3
    assert [case.category for case in cases].count("stage3_maxseq") == 3
    assert [case.category for case in cases].count("cuda_rng_resume") == 1
    assert [case.category for case in cases].count("nccl") == 1
    assert [case.category for case in cases].count("prior_resume") == 1


def test_matrix_freezes_lengths_recompute_utilization_and_world_size(matrix):
    cases = matrix.build_matrix()
    for case in cases:
        assert case.python_no_user_site == "1"
        assert case.clean_detached_required is True
        assert case.tabicl_attestation_count == 1
        if case.category == "stage2_maxseq":
            assert case.repeat_steps == 16
            assert (case.min_seq_len, case.max_seq_len) == (10240, 10240)
            assert case.observed_sequence_length == 10240
            assert case.utilization_required is True
            assert case.recompute is False
        elif case.category == "stage3_maxseq":
            assert case.repeat_steps == 16
            assert (case.min_seq_len, case.max_seq_len) == (60000, 60000)
            assert case.observed_sequence_length == 60000
            assert case.utilization_required is True
            assert case.recompute is True
        else:
            assert case.utilization_required is False
        if case.case_id == "nccl_2gpu":
            assert case.world_size == 2
            assert case.assertion == "nccl_all_reduce_exact"
        else:
            assert case.world_size == 1
    assert next(c for c in cases if c.case_id == "temporary_cuda_rng_resume").assertion == (
        "temporary_cuda_rng_isolation_and_resume_exact"
    )
    assert next(c for c in cases if c.case_id == "prior_dataloader_resume").assertion == (
        "full_prior_dataloader_uninterrupted_equals_resume"
    )


def test_all_training_shapes_disable_random_or_log_length_replay(matrix):
    for case in matrix.build_matrix():
        if case.stage is not None:
            assert case.log_seq_len is False
            assert case.replay_small is False


def test_all_dry_run_argv_use_the_exact_t_outer_runner(matrix):
    for case in matrix.build_matrix():
        argv = matrix.execution_argv(case, ROOT)
        assert argv == [
            str(ROOT / "scripts" / "run_h100_identity_maxseq_smoke.sh"),
            case.case_id,
        ]
        assert "--execute-special" not in argv


def test_prior_dataloader_special_uses_real_graph_workers_and_reconstructed_rng(matrix):
    # The executable H100 gate itself uses multiprocessing; running that
    # nested worker pool inside pytest is not portable on macOS spawn.  The
    # underlying byte-exact graph stream is exercised by
    # test_prior_stream_reproducibility.py; here we freeze the H100 contract.
    source = inspect.getsource(matrix._assert_prior_dataloader_resume)
    for contract in (
        'prior_type="graph_scm"',
        "num_workers=2",
        "prefetch_factor=4",
        "num_workers=1",
        "prefetch_factor=1",
        "initial_generator_state",
        "logical_stream_state_dict(cursor=2)",
        "actual != expected[2:]",
    ):
        assert contract in source


def test_stage1_one_step_uses_formal_graph_prior_and_worker_topology(matrix):
    source = inspect.getsource(matrix._training_config)
    stage1 = source.split('elif case.category == "stage2_maxseq":', 1)[0]
    for contract in (
        '"--prior_type", "graph_scm"',
        '"--n_jobs", "16"',
        '"--batch_size_per_gp", "4"',
        '"--min_features", "1"',
        '"--max_features", "100"',
        '"--min_train_size", "0.3"',
        '"--max_train_size", "0.9"',
        '"--seq_len_per_gp", "true"',
        '"--filter_unpredictable_graphs", "true"',
        '"--filter_unpredictable_datasets", "true"',
        '"--min_n_nodes", "2"',
        '"--max_n_nodes", "32"',
    ):
        assert contract in stage1
    assert '"--prior_type", "dummy"' not in stage1


def test_stage2_and_stage3_maxseq_keep_fixed_dummy_prior(matrix):
    source = inspect.getsource(matrix._training_config)
    assert source.count('"--prior_type", "dummy"') == 2
    assert source.count('"--min_seq_len", str(case.observed_sequence_length)') == 2
    assert source.count('"--max_seq_len", str(case.observed_sequence_length)') == 3


@pytest.mark.parametrize("case_index", range(9))
def test_training_config_binds_external_checkpoint_writer_ceiling(
    tmp_path, matrix, case_index
):
    case = matrix.build_matrix()[case_index]
    config = matrix._training_config(case, tmp_path, 12_345)
    assert config.max_checkpoint_bytes == 12_345


def _case_evidence(
    case,
    matrix,
    *,
    commit="1" * 40,
    tree="2" * 40,
    source_manifest="4" * 64,
):
    artifact_identity = hashlib.sha256(("identity:" + case.case_id).encode()).hexdigest()
    def canonical(value):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()

    def envelope(kind, payload):
        body = {"schema_version": 1, "kind": kind, "payload": payload}
        return {**body, "sha256": hashlib.sha256(canonical(body)).hexdigest()}

    source_attestation = matrix.make_source_attestation(
        {
            "commit_sha": commit,
            "tree_sha": tree,
            "source_manifest_sha256": source_manifest,
            "import_relative_path": "src/tabicl/__init__.py",
            "action": "h100-validation",
            "case_id": case.case_id,
            "artifact_identity_sha256": artifact_identity,
            "python_isolated": True,
            "bytecode_disabled": True,
            "python_no_user_site": "1",
        }
    )
    environment_payload = {
        "python_version": "3.11.0",
        "python_implementation": "CPython",
        "platform_system": "Linux",
        "platform_release": "test",
        "platform_machine": "x86_64",
        "torch_version": "2.7.0",
        "numpy_version": "2.0.0",
        "cuda_runtime_version": "12.8",
        "cudnn_version": 9000,
        "visible_cuda_device_count": case.world_size,
    }
    environment = envelope("environment", environment_payload)
    devices = [
        {
            "uuid": f"GPU-{index}",
            "name": "NVIDIA H100 80GB HBM3",
            "driver_version": "570.00",
        }
        for index in range(case.world_size)
    ]
    requested_resource = matrix.gate_requested_resource(case)
    requested_resource_sha256 = matrix.gate_requested_resource_sha256(case)
    job_id = str(2_000 + matrix.build_matrix().index(case))
    req_tres = (
        f"cpu={requested_resource['cpus_per_task']},"
        "mem=128G,node=1,"
        f"gres/gpu={requested_resource['gpus']}"
    )
    alloc_tres = req_tres
    tres_per_node = f"gres/gpu:{requested_resource['gpus']}"
    scontrol_argv = [
        "scontrol",
        "--clusters=cluster-a",
        "show",
        "job",
        "-o",
        job_id,
    ]
    scheduler_binding_body = {
        "job_id": job_id,
        "requested_resource": requested_resource,
        "requested_resource_sha256": requested_resource_sha256,
        "held_plan_sha256": "7" * 64,
        "scontrol_sha256": "8" * 64,
        "scontrol_query_returncode": 0,
        "scontrol_query_argv_sha256": hashlib.sha256(
            canonical(scontrol_argv)
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
        "slurm_alloc_tres": alloc_tres,
        "slurm_tres_per_node": tres_per_node,
        "cuda_visible_devices": [str(index) for index in range(case.world_size)],
        "visible_gpu_uuids": [device["uuid"] for device in devices],
    }
    scheduler_binding = {
        **scheduler_binding_body,
        "sha256": hashlib.sha256(canonical(scheduler_binding_body)).hexdigest(),
    }
    runtime_evidence = matrix.make_runtime_evidence(
        {
            "case_id": case.case_id,
            "commit_sha": commit,
            "tree_sha": tree,
            "source_manifest_sha256": source_manifest,
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
            "scheduler_binding": scheduler_binding,
        }
    )
    action_completion = matrix.make_action_completion(
        {
            "case_id": case.case_id,
            "action": "h100-validation",
            "artifact_identity_sha256": artifact_identity,
            "source_attestation_sha256": source_attestation["sha256"],
            "commit_sha": commit,
            "tree_sha": tree,
            "source_manifest_sha256": source_manifest,
            "runtime_evidence_sha256": runtime_evidence["sha256"],
            "slurm_job_id": job_id,
            "requested_resource_sha256": requested_resource_sha256,
            "scheduler_binding_sha256": scheduler_binding["sha256"],
            "final_tabicl_attested": True,
            "completed": True,
        }
    )
    checkpoint_path = f"step-{case.repeat_steps}.ckpt" if case.stage else None
    checkpoint_sha = (
        hashlib.sha256(("checkpoint:" + case.case_id).encode()).hexdigest()
        if case.stage
        else None
    )
    checkpoint_size = 500 if case.stage else None
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
            "checkpoint_relative_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size": checkpoint_size,
            "checkpoint_ceiling_bytes": 1_000,
            "slurm_job_id": job_id,
            "requested_resource_sha256": requested_resource_sha256,
        },
    )
    utilization = case.utilization_required
    gpu_window = {
        "gpu_uuids": ["GPU-0"] if utilization else [],
        "gpu_sample_counts": {"GPU-0": 10} if utilization else {},
        "gpu_means": {"GPU-0": 80.0} if utilization else {},
        "gpu_min_gap_seconds": 1.0 if utilization else None,
        "gpu_max_gap_seconds": 1.0 if utilization else None,
    }
    raws = {
        "source-attestation.json": canonical(source_attestation) + b"\n",
        "runtime-evidence.json": canonical(runtime_evidence) + b"\n",
        "action-completion.json": canonical(action_completion) + b"\n",
        "compute/case-result.json": canonical(case_result) + b"\n",
        "compute.log": b"ok",
    }
    if case.stage:
        raws[f"compute/{checkpoint_path}"] = b"x" * checkpoint_size
    if utilization:
        raws.update(
            {"gpu.csv": b"csv", "gpu.jsonl": b"jsonl", "gpu-summary.json": b"summary"}
        )
    artifact_manifest = {
        "schema_version": 1,
        "case_id": case.case_id,
        "files": [
            {
                "name": name,
                "sha256": checkpoint_sha
                if name == f"compute/{checkpoint_path}"
                else hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
            for name, raw in sorted(raws.items())
        ],
    }
    artifact = hashlib.sha256(canonical(artifact_manifest)).hexdigest()
    case_binding = {
        "case_id": case.case_id,
        "source_attestation_sha256": source_attestation["sha256"],
        "runtime_evidence_sha256": runtime_evidence["sha256"],
        "action_completion_sha256": action_completion["sha256"],
        "artifact_identity_sha256": artifact_identity,
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
        "artifact_identity_sha256": artifact_identity,
        "case_result": case_result,
        "case_result_sha256": case_result["sha256"],
        "checkpoint_relative_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size": checkpoint_size,
        "source_attestation": source_attestation,
        "source_attestation_sha256": source_attestation["sha256"],
        "runtime_evidence": runtime_evidence,
        "runtime_evidence_sha256": runtime_evidence["sha256"],
        "action_completion": action_completion,
        "action_completion_sha256": action_completion["sha256"],
        "case_binding_sha256": hashlib.sha256(
            json.dumps(case_binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "gpu_window_sha256": hashlib.sha256(
            json.dumps(gpu_window, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if utilization
        else None,
        **gpu_window,
        "checkout_commit_sha": commit,
        "checkout_tree_sha": tree,
        "clean_detached": True,
        "tabicl_attestation_count": 1,
        "python_no_user_site": "1",
    }


def _attestation(matrix):
    commit = "1" * 40
    tree = "2" * 40
    cases = [
        _case_evidence(case, matrix, commit=commit, tree=tree)
        for case in matrix.build_matrix()
    ]
    scheduler_log_manifest = matrix.make_scheduler_log_manifest(
        [
            {
                "name": name,
                "sha256": hashlib.sha256(("scheduler:" + name).encode()).hexdigest(),
                "size": len(name),
            }
            for name in sorted(
                f"{case.case_id}.{suffix}"
                for case in matrix.build_matrix()
                for suffix in ("err", "out")
            )
        ],
        ceiling_bytes=1_000,
    )
    receipt_sha = "5" * 64
    sacct_sha = "6" * 64
    terminal_observations = []
    for case, item in zip(matrix.build_matrix(), cases):
        job_id = item["runtime_evidence"]["payload"]["scheduler_binding"]["job_id"]
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
        submission_receipt_sha256=receipt_sha,
        sacct_sha256=sacct_sha,
    )
    payload = {
        "commit_sha": commit,
        "tree_sha": tree,
        "environment_sha256": cases[0]["runtime_evidence"]["payload"]["environment"]["sha256"],
        "source_manifest_sha256": "4" * 64,
        "repository_binding": _repository_binding(matrix, commit=commit),
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.00",
        "checkpoint_ceiling_bytes": 1_000,
        "observed_checkpoint_max_bytes": 500,
        "submission_receipt_sha256": receipt_sha,
        "scheduler_terminal_manifest": terminal_manifest,
        "scheduler_log_manifest": scheduler_log_manifest,
        "cases": cases,
    }
    return matrix.make_smoke_attestation(payload)


def test_runtime_evidence_is_bound_to_receipt_job_cluster_and_resource(matrix):
    case = matrix.build_matrix()[0]
    item = _case_evidence(case, matrix)
    scheduler = item["runtime_evidence"]["payload"]["scheduler_binding"]
    expected = {
        "job_id": scheduler["job_id"],
        "cluster": scheduler["slurm_cluster_name"],
        "requested_resource": scheduler["requested_resource"],
        "requested_resource_sha256": scheduler["requested_resource_sha256"],
        "held_plan_sha256": scheduler["held_plan_sha256"],
        "scontrol_sha256": scheduler["scontrol_sha256"],
    }
    matrix.validate_runtime_evidence(
        item["runtime_evidence"],
        case=case,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        source_manifest_sha256="4" * 64,
        expected_job_binding=expected,
    )
    for field, value in (
        ("job_id", "9999"),
        ("cluster", "cluster-b"),
        ("held_plan_sha256", "a" * 64),
        ("scontrol_sha256", "b" * 64),
    ):
        wrong = dict(expected)
        wrong[field] = value
        with pytest.raises(ValueError, match="submission receipt"):
            matrix.validate_runtime_evidence(
                item["runtime_evidence"],
                case=case,
                commit_sha="1" * 40,
                tree_sha="2" * 40,
                source_manifest_sha256="4" * 64,
                expected_job_binding=wrong,
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("slurm_job_partition", "cpu"),
        ("slurm_qos", "normal"),
        ("slurm_time_limit", "02:59:59"),
        ("slurm_num_nodes", 2),
        ("slurm_num_cpus", 31),
        ("slurm_cpus_per_task", 31),
        ("slurm_memory_per_node_mb", 130048),
        ("slurm_alloc_tres", "cpu=32,mem=128G,node=1,gres/gpu=0"),
        ("slurm_tres_per_node", "gres/gpu:0"),
        ("scontrol_query_returncode", 1),
        ("scontrol_query_argv_sha256", "f" * 64),
        ("scontrol_query_stderr_sha256", "f" * 64),
    ],
)
def test_runtime_evidence_rejects_noncanonical_actual_resource_proof(
    matrix, field, value
):
    case = matrix.build_matrix()[0]
    runtime = copy.deepcopy(_case_evidence(case, matrix)["runtime_evidence"])
    scheduler = runtime["payload"]["scheduler_binding"]
    scheduler[field] = value
    scheduler_body = {key: item for key, item in scheduler.items() if key != "sha256"}
    scheduler["sha256"] = hashlib.sha256(matrix._canonical(scheduler_body)).hexdigest()
    runtime_body = {
        key: runtime[key] for key in ("schema_version", "kind", "payload")
    }
    runtime["sha256"] = hashlib.sha256(matrix._canonical(runtime_body)).hexdigest()
    with pytest.raises(ValueError, match="scheduler/resource binding"):
        matrix.validate_runtime_evidence(
            runtime,
            case=case,
            commit_sha="1" * 40,
            tree_sha="2" * 40,
            source_manifest_sha256="4" * 64,
        )


def _scontrol_allocation_line(resource, *, job_id="7001"):
    tres = (
        f"cpu={resource['cpus_per_task']},mem=128G,node=1,"
        f"gres/gpu={resource['gpus']}"
    )
    return (
        f"JobId={job_id} Partition={resource['partition']} QOS={resource['qos']} "
        f"TimeLimit={resource['time_limit']} NumNodes={resource['nodes']} "
        f"NumCPUs={resource['cpus_per_task']} CPUs/Task={resource['cpus_per_task']} "
        f"MinMemoryNode=128G ReqTRES={tres} AllocTRES={tres} "
        f"TresPerNode=gres/gpu:{resource['gpus']}\n"
    )


def _install_runtime_scheduler_fixture(
    tmp_path, matrix, monkeypatch, case, *, scontrol_digest=None
):
    resource = matrix.gate_requested_resource(case)
    state = tmp_path / "scontrol-state.txt"
    state.write_text(_scontrol_allocation_line(resource))
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    scontrol = fakebin / "scontrol"
    scontrol.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        f"print(Path({str(state)!r}).read_text(), end='')\n"
    )
    scontrol.chmod(0o755)
    actual_scontrol_digest = hashlib.sha256(scontrol.read_bytes()).hexdigest()
    artifact_identity = "9" * 64
    planned_cases = []
    for index, expected_case in enumerate(matrix.build_matrix()):
        planned_cases.append(
            {
                "case_id": expected_case.case_id,
                "world_size": expected_case.world_size,
                "artifact_identity_sha256": (
                    artifact_identity
                    if expected_case.case_id == case.case_id
                    else hashlib.sha256(expected_case.case_id.encode()).hexdigest()
                ),
                "requested_resource": matrix.gate_requested_resource(expected_case),
                "requested_resource_sha256": matrix.gate_requested_resource_sha256(
                    expected_case
                ),
                "job_id": "7001" if expected_case.case_id == case.case_id else str(7100 + index),
                "cluster": "cluster-a",
            }
        )
    scheduler = matrix._scheduler_contract()
    payload = {
        "commit_sha": "1" * 40,
        "tree_sha": "2" * 40,
        "source_manifest_sha256": "4" * 64,
        "repository_binding": _repository_binding(matrix),
        "jobs_held_at_publication": True,
        "scheduler": scheduler,
        "scheduler_command_sha256": {
            "sbatch": "a" * 64,
            "scontrol": scontrol_digest or actual_scontrol_digest,
            "scancel": "b" * 64,
            "squeue": "c" * 64,
            "sacct": "d" * 64,
        },
        "cases": planned_cases,
    }
    body = {"schema_version": 1, "kind": "h100_gate_held_plan", "payload": payload}
    held = {**body, "sha256": hashlib.sha256(matrix._canonical(body)).hexdigest()}
    validation_root = tmp_path / "artifacts" / "cases"
    validation_root.mkdir(parents=True)
    _write_canonical(validation_root.parent / "held-plan.json", held)
    environment = {
        "FORMAL_REQUESTED_RESOURCE_SHA256": matrix.gate_requested_resource_sha256(case),
        "FORMAL_REQUESTED_PARTITION": resource["partition"],
        "FORMAL_REQUESTED_QOS": resource["qos"],
        "FORMAL_REQUESTED_TIME_LIMIT": resource["time_limit"],
        "FORMAL_REQUESTED_NODES": str(resource["nodes"]),
        "FORMAL_REQUESTED_GPUS": str(resource["gpus"]),
        "FORMAL_REQUESTED_CPUS": str(resource["cpus_per_task"]),
        "FORMAL_REQUESTED_MEMORY_MB": str(resource["memory_mb"]),
        "SLURM_JOB_ID": "7001",
        "CUDA_VISIBLE_DEVICES": "0",
        "FORMAL_VISIBLE_GPU_TOKENS": "0",
        "FORMAL_VISIBLE_GPU_UUIDS": "GPU-0",
        "SLURM_JOB_PARTITION": resource["partition"],
        "SLURM_CLUSTER_NAME": "cluster-a",
        "SLURM_CPUS_PER_TASK": str(resource["cpus_per_task"]),
        "SLURM_MEM_PER_NODE": str(resource["memory_mb"]),
        "H100_VALIDATION_ROOT": str(validation_root),
        "FORMAL_ATTESTATION_CEILING_BYTES": str(1 << 20),
        "VALIDATION_ARTIFACT_IDENTITY_SHA256": artifact_identity,
        "CANDIDATE_REPOSITORY": "https://github.com/kikixiong/tabicl-pe.git",
        "CANDIDATE_REPOSITORY_REF": "refs/heads/codex/position-identity-v1",
        "FORMAL_GIT_SHA256": "9" * 64,
        "FORMAL_REPOSITORY_IDENTITY_SHA256": _repository_binding(matrix)[
            "repository_identity_sha256"
        ],
        "FORMAL_REPOSITORY_QUERY_SHA256": _repository_binding(matrix)[
            "query_sha256"
        ],
        "SCONTROL": str(scontrol),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return resource, state, actual_scontrol_digest, held["sha256"]


def _capture_scheduler_fixture(matrix, case):
    return matrix.capture_scheduler_binding(
        case,
        [{"uuid": "GPU-0", "name": "NVIDIA H100", "driver_version": "570"}],
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        source_manifest_sha256="4" * 64,
    )


def test_runtime_scheduler_capture_binds_actual_scontrol_allocation_and_digests(
    tmp_path, matrix, monkeypatch
):
    case = matrix.build_matrix()[0]
    resource, _state, scontrol_digest, held_digest = _install_runtime_scheduler_fixture(
        tmp_path, matrix, monkeypatch, case
    )
    binding = _capture_scheduler_fixture(matrix, case)
    assert binding["job_id"] == "7001"
    assert binding["requested_resource"] == resource
    assert binding["held_plan_sha256"] == held_digest
    assert binding["scontrol_sha256"] == scontrol_digest
    assert binding["slurm_qos"] == "short"
    assert binding["slurm_time_limit"] == "03:00:00"
    assert binding["slurm_num_nodes"] == 1
    assert binding["slurm_num_cpus"] == resource["cpus_per_task"]
    assert binding["slurm_memory_per_node_mb"] == 131072
    expected_argv = [
        "scontrol",
        "--clusters=cluster-a",
        "show",
        "job",
        "-o",
        "7001",
    ]
    assert binding["scontrol_query_argv_sha256"] == hashlib.sha256(
        matrix._canonical(expected_argv)
    ).hexdigest()


def test_runtime_scheduler_capture_rejects_repository_ref_export_drift(
    tmp_path, matrix, monkeypatch
):
    case = matrix.build_matrix()[0]
    _install_runtime_scheduler_fixture(tmp_path, matrix, monkeypatch, case)
    monkeypatch.setenv("CANDIDATE_REPOSITORY_REF", "refs/heads/other")
    with pytest.raises(RuntimeError, match="held plan"):
        _capture_scheduler_fixture(matrix, case)


@pytest.mark.parametrize(
    "old,new",
    [
        ("QOS=short", "QOS=normal"),
        ("TimeLimit=03:00:00", "TimeLimit=02:59:59"),
        ("NumNodes=1", "NumNodes=2"),
        ("Partition=h100", "Partition=cpu"),
        ("NumCPUs=32", "NumCPUs=31"),
        ("MinMemoryNode=128G", "MinMemoryNode=127G"),
        ("AllocTRES=cpu=32,mem=128G,node=1,gres/gpu=1", "AllocTRES=cpu=32,mem=128G,node=1,gres/gpu=0"),
    ],
)
def test_runtime_scheduler_capture_rejects_actual_allocation_drift(
    tmp_path, matrix, monkeypatch, old, new
):
    case = matrix.build_matrix()[0]
    _resource, state, _command_digest, _held_digest = _install_runtime_scheduler_fixture(
        tmp_path, matrix, monkeypatch, case
    )
    state.write_text(state.read_text().replace(old, new))
    with pytest.raises(RuntimeError, match="actual scontrol allocation"):
        _capture_scheduler_fixture(matrix, case)


def test_runtime_scheduler_capture_rejects_exported_allocation_drift(
    tmp_path, matrix, monkeypatch
):
    case = matrix.build_matrix()[0]
    resource, _state, _command_digest, _held_digest = _install_runtime_scheduler_fixture(
        tmp_path, matrix, monkeypatch, case
    )

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(resource["cpus_per_task"] - 1))
    with pytest.raises(RuntimeError, match="allocation differs"):
        _capture_scheduler_fixture(matrix, case)


def test_runtime_scheduler_capture_rejects_held_plan_command_digest_tamper(
    tmp_path, matrix, monkeypatch
):
    case = matrix.build_matrix()[0]
    _install_runtime_scheduler_fixture(
        tmp_path,
        matrix,
        monkeypatch,
        case,
        scontrol_digest="f" * 64,
    )
    with pytest.raises(ValueError, match="scontrol.*digest mismatch"):
        _capture_scheduler_fixture(matrix, case)


def _write_canonical(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _write_assembler_cases(root: Path, matrix) -> None:
    scheduler_logs = root.parent / "scheduler-logs"
    scheduler_logs.mkdir()
    for case in matrix.build_matrix():
        for suffix in ("out", "err"):
            (scheduler_logs / f"{case.case_id}.{suffix}").write_bytes(
                f"{case.case_id}:{suffix}\n".encode()
            )
    for case in matrix.build_matrix():
        item = _case_evidence(case, matrix)
        case_root = root / case.case_id
        compute_root = case_root / "compute"
        compute_root.mkdir(parents=True)
        if case.stage is not None:
            checkpoint = b"checkpoint:" + case.case_id.encode()
            checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
            checkpoint_path = f"step-{case.repeat_steps}.ckpt"
            (compute_root / checkpoint_path).write_bytes(checkpoint)
            result_payload = dict(item["case_result"]["payload"])
            result_payload.update(
                {
                    "checkpoint_relative_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_size": len(checkpoint),
                }
            )
            body = {
                "schema_version": 1,
                "kind": "h100_case_result",
                "payload": result_payload,
            }
            item["case_result"] = {
                **body,
                "sha256": hashlib.sha256(matrix._canonical(body)).hexdigest(),
            }
        _write_canonical(case_root / "source-attestation.json", item["source_attestation"])
        _write_canonical(case_root / "runtime-evidence.json", item["runtime_evidence"])
        _write_canonical(case_root / "action-completion.json", item["action_completion"])
        _write_canonical(compute_root / "case-result.json", item["case_result"])
        (case_root / "compute.log").write_bytes(b"completed\n")
        if case.utilization_required:
            records = [
                {
                    "kind": "training_start",
                    "monotonic_seconds": 100.0,
                    "expected_gpu_uuids": ["GPU-0"],
                    "expected_gpu_count": 1,
                }
            ]
            records.extend(
                {
                    "kind": "sample",
                    "monotonic_seconds": 100.5 + index,
                    "gpu_uuid": "GPU-0",
                    "utilization_percent": 80.0,
                }
                for index in range(10)
            )
            records.append(
                {
                    "kind": "training_end",
                    "monotonic_seconds": 111.0,
                    "expected_gpu_uuids": ["GPU-0"],
                    "expected_gpu_count": 1,
                }
            )
            csv_rows = []
            for record in records:
                if record["kind"] == "sample":
                    csv_rows.append(
                        f"sample,{record['monotonic_seconds']:.9f},GPU-0,80.0"
                    )
                else:
                    csv_rows.append(
                        f"{record['kind']},{record['monotonic_seconds']:.9f}"
                    )
            (case_root / "gpu.csv").write_text("\n".join(csv_rows) + "\n")
            (case_root / "gpu.jsonl").write_text(
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in records
                )
            )
            _write_canonical(
                case_root / "gpu-summary.json",
                {
                    "schema_version": 1,
                    "utilization_required": True,
                    "utilization_claimed": True,
                    "sample_counts": {"GPU-0": 10},
                    "means": {"GPU-0": 80.0},
                },
            )


def _assembler_scheduler_contract(root: Path, matrix):
    state = {}
    bindings = {}
    for case in matrix.build_matrix():
        scheduler = _case_evidence(case, matrix)["runtime_evidence"]["payload"][
            "scheduler_binding"
        ]
        job_id = scheduler["job_id"]
        state[job_id] = [f"{job_id}|cluster-a|COMPLETED|0:0|0:0"]
        bindings[case.case_id] = {
            "job_id": job_id,
            "cluster": "cluster-a",
            "requested_resource": scheduler["requested_resource"],
            "requested_resource_sha256": scheduler["requested_resource_sha256"],
            "held_plan_sha256": scheduler["held_plan_sha256"],
            "scontrol_sha256": scheduler["scontrol_sha256"],
        }
    state_path = root.parent / "terminal-state.json"
    if not state_path.exists():
        state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    bin_root = root.parent / "bin"
    bin_root.mkdir(exist_ok=True)
    sacct = bin_root / "sacct"
    if not sacct.exists():
        sacct.write_text(
            f"#!{sys.executable}\n"
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n"
            f"state = json.loads(Path({str(state_path)!r}).read_text())\n"
            "job = next(value.split('=', 1)[1] for value in sys.argv[1:] if value.startswith('--jobs='))\n"
            "for row in state.get(job, []): print(row)\n"
        )
        sacct.chmod(0o755)
    return sacct, bindings


def _assemble(
    root: Path,
    output: Path,
    matrix,
    *,
    final_ceiling: int = 8 << 20,
    max_output: int = 8 << 20,
):
    sacct, bindings = _assembler_scheduler_contract(root, matrix)
    return matrix.assemble_smoke_attestation(
        root,
        output=output,
        expected_commit_sha="1" * 40,
        expected_tree_sha="2" * 40,
        expected_source_manifest_sha256="4" * 64,
        expected_checkpoint_ceiling_bytes=1_000,
        expected_scheduler_log_ceiling_bytes=1_000,
        expected_final_attestation_ceiling_bytes=final_ceiling,
        expected_submission_receipt_sha256="5" * 64,
        sacct_path=sacct,
        expected_sacct_sha256=hashlib.sha256(sacct.read_bytes()).hexdigest(),
        max_input_bytes=1 << 20,
        max_output_bytes=max_output,
        expected_job_bindings=bindings,
        expected_repository_binding=_repository_binding(matrix),
    )


def test_smoke_attestation_requires_external_expected_digest_and_full_schema(matrix):
    attestation = _attestation(matrix)
    report = matrix.validate_smoke_attestation(
        attestation,
        expected_sha256=attestation["sha256"],
        expected_commit_sha="1" * 40,
        expected_tree_sha="2" * 40,
        expected_environment_sha256=attestation["payload"]["environment_sha256"],
        expected_source_manifest_sha256="4" * 64,
        expected_gpu_model="NVIDIA H100 80GB HBM3",
        expected_checkpoint_ceiling_bytes=1_000,
    )
    assert report["case_count"] == 12

    tampered = copy.deepcopy(attestation)
    tampered["payload"]["cases"][0]["artifact_sha256"] = "f" * 64
    tampered = matrix.make_smoke_attestation(tampered["payload"])
    with pytest.raises(ValueError, match="external expected digest"):
        matrix.validate_smoke_attestation(
            tampered,
            expected_sha256=attestation["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=attestation["payload"]["environment_sha256"],
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
        )


def test_smoke_checkpoint_ceiling_is_external_and_observed_max_is_exact(matrix):
    attestation = _attestation(matrix)
    kwargs = {
        "expected_sha256": attestation["sha256"],
        "expected_commit_sha": "1" * 40,
        "expected_tree_sha": "2" * 40,
        "expected_environment_sha256": attestation["payload"]["environment_sha256"],
        "expected_source_manifest_sha256": "4" * 64,
        "expected_gpu_model": "NVIDIA H100 80GB HBM3",
        "expected_checkpoint_ceiling_bytes": 999,
    }
    with pytest.raises(ValueError, match="external commitment"):
        matrix.validate_smoke_attestation(attestation, **kwargs)

    payload = copy.deepcopy(attestation["payload"])
    payload["observed_checkpoint_max_bytes"] = 499
    malformed = matrix.make_smoke_attestation(payload)
    kwargs.update(
        expected_sha256=malformed["sha256"],
        expected_checkpoint_ceiling_bytes=1_000,
    )
    with pytest.raises(ValueError, match="observed checkpoint maximum"):
        matrix.validate_smoke_attestation(malformed, **kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "gpu_model",
        "driver_version",
        "repository_binding",
        "submission_receipt_sha256",
        "scheduler_terminal_manifest",
        "scheduler_log_manifest",
        "cases",
    ],
)
def test_smoke_attestation_rejects_missing_required_payload_field(matrix, field):
    attestation = _attestation(matrix)
    payload = copy.deepcopy(attestation["payload"])
    del payload[field]
    malformed = matrix.make_smoke_attestation(payload)
    with pytest.raises(ValueError, match="keys mismatch"):
        matrix.validate_smoke_attestation(
            malformed,
            expected_sha256=malformed["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=malformed["payload"].get(
                "environment_sha256", attestation["payload"]["environment_sha256"]
            ),
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
        )


def test_assembler_consumes_all_raw_cases_and_publishes_once(tmp_path, matrix):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    output = tmp_path / "smoke.json"

    attestation = _assemble(cases, output, matrix)

    assert output.read_bytes() == matrix._canonical(attestation) + b"\n"
    assert attestation["payload"]["checkpoint_ceiling_bytes"] == 1_000
    assert attestation["payload"]["observed_checkpoint_max_bytes"] > 0
    assert len(attestation["payload"]["cases"]) == 12
    scheduler_logs = attestation["payload"]["scheduler_log_manifest"]
    assert scheduler_logs["payload"]["ceiling_bytes"] == 1_000
    assert len(scheduler_logs["payload"]["files"]) == 24
    with pytest.raises((FileExistsError, ValueError), match="no-replace|exist"):
        _assemble(cases, output, matrix)


def test_final_attestation_output_is_same_namespace_and_capacity_bounded(
    tmp_path, matrix
):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    with pytest.raises(ValueError, match="submission budget"):
        _assemble(
            cases,
            tmp_path / "oversized.json",
            matrix,
            final_ceiling=1_000,
            max_output=1_001,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="gate namespace"):
        _assemble(cases, outside / "smoke.json", matrix)


@pytest.mark.parametrize("mutation", ["missing", "extra", "oversize", "symlink"])
def test_assembler_requires_exact_bounded_physical_scheduler_logs(
    tmp_path, matrix, mutation
):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    logs = tmp_path / "scheduler-logs"
    target = logs / "stage1_rope_one_step.out"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (logs / "unknown.out").write_text("unexpected\n")
    elif mutation == "oversize":
        target.write_bytes(b"x" * 1_001)
    else:
        target.unlink()
        outside = tmp_path / "outside.log"
        outside.write_text("untrusted\n")
        target.symlink_to(outside)
    with pytest.raises(ValueError, match="scheduler log|artifact"):
        _assemble(cases, tmp_path / f"{mutation}.json", matrix)


def test_assembler_rejects_fatal_scheduler_spool_signatures(tmp_path, matrix):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    target = tmp_path / "scheduler-logs" / "stage1_rope_one_step.err"
    target.write_text("CUDA out of memory\nENOSPC\nTraceback (most recent call last)\n")
    with pytest.raises(ValueError, match="rejected signatures"):
        _assemble(cases, tmp_path / "fatal.json", matrix)


@pytest.mark.parametrize(
    "rows",
    (
        lambda job: [f"{job}|cluster-a|RUNNING|0:0|0:0"],
        lambda job: [f"{job}|cluster-a|COMPLETING|0:0|0:0"],
        lambda job: [f"{job}|cluster-a|FAILED|1:0|1:0"],
        lambda job: [f"{job}|cluster-a|COMPLETED|1:0|0:0"],
        lambda job: [f"{job}|cluster-a|COMPLETED|0:0|1:0"],
        lambda job: [],
        lambda job: [
            f"{job}|cluster-a|COMPLETED|0:0|0:0",
            f"{job}|cluster-a|COMPLETED|0:0|0:0",
        ],
        lambda job: ["Welcome back, forged banner", f"{job}|cluster-a|COMPLETED|0:0|0:0"],
        lambda job: [f"{job}.batch|cluster-a|COMPLETED|0:0|0:0"],
        lambda job: [f"{job}|cluster-b|COMPLETED|0:0|0:0"],
    ),
    ids=(
        "running",
        "completing",
        "failed",
        "nonzero-exit",
        "nonzero-derived-exit",
        "missing",
        "duplicate",
        "banner-spoof",
        "step-only",
        "wrong-cluster",
    ),
)
def test_terminal_finalizer_rejects_noncanonical_sacct_rows(
    tmp_path, matrix, rows
):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    _sacct, _bindings = _assembler_scheduler_contract(cases, matrix)
    state_path = tmp_path / "terminal-state.json"
    state = json.loads(state_path.read_text())
    job_id = _case_evidence(matrix.build_matrix()[0], matrix)["runtime_evidence"][
        "payload"
    ]["scheduler_binding"]["job_id"]
    state[job_id] = rows(job_id)
    state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="sacct"):
        _assemble(cases, tmp_path / "terminal-invalid.json", matrix)


def test_terminal_finalizer_rejects_sacct_path_and_digest_tamper(tmp_path, matrix):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    sacct, bindings = _assembler_scheduler_contract(cases, matrix)
    expected = hashlib.sha256(sacct.read_bytes()).hexdigest()
    sacct.write_bytes(sacct.read_bytes() + b"# tampered\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        matrix.capture_scheduler_terminal_manifest(
            sacct_path=sacct,
            expected_sacct_sha256=expected,
            submission_receipt_sha256="5" * 64,
            expected_job_bindings=bindings,
        )

    outside = tmp_path / "outside-sacct"
    outside.write_bytes(sacct.read_bytes())
    outside.chmod(0o755)
    sacct.unlink()
    sacct.symlink_to(outside)
    with pytest.raises(ValueError, match="without following links|identity"):
        matrix.capture_scheduler_terminal_manifest(
            sacct_path=sacct,
            expected_sacct_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
            submission_receipt_sha256="5" * 64,
            expected_job_bindings=bindings,
        )


def test_assembler_rejects_scheduler_log_change_after_terminal_query(
    tmp_path, matrix, monkeypatch
):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    target = tmp_path / "scheduler-logs" / "stage1_rope_one_step.out"
    real_capture = matrix._capture_scheduler_log_manifest
    calls = 0

    def capture(*args, **kwargs):
        nonlocal calls
        result = real_capture(*args, **kwargs)
        calls += 1
        if calls == 1:
            target.write_bytes(target.read_bytes() + b"late benign append\n")
        return result

    monkeypatch.setattr(matrix, "_capture_scheduler_log_manifest", capture)
    with pytest.raises(ValueError, match="changed after terminal"):
        _assemble(cases, tmp_path / "changed.json", matrix)


def test_smoke_attestation_binds_receipt_and_terminal_observations(matrix):
    attestation = _attestation(matrix)
    with pytest.raises(ValueError, match="smoke submission receipt mismatch"):
        matrix.validate_smoke_attestation(
            attestation,
            expected_sha256=attestation["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=attestation["payload"]["environment_sha256"],
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
            expected_submission_receipt_sha256="f" * 64,
        )
    payload = copy.deepcopy(attestation["payload"])
    payload["submission_receipt_sha256"] = "7" * 64
    malformed = matrix.make_smoke_attestation(payload)
    with pytest.raises(ValueError, match="receipt mismatch"):
        matrix.validate_smoke_attestation(
            malformed,
            expected_sha256=malformed["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=payload["environment_sha256"],
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
        )

    payload = copy.deepcopy(attestation["payload"])
    terminal = payload["scheduler_terminal_manifest"]
    observations = copy.deepcopy(terminal["payload"]["observations"])
    observations[0]["state"] = "RUNNING"
    payload["scheduler_terminal_manifest"] = matrix.make_scheduler_terminal_manifest(
        observations,
        submission_receipt_sha256=payload["submission_receipt_sha256"],
        sacct_sha256=terminal["payload"]["sacct_sha256"],
    )
    malformed = matrix.make_smoke_attestation(payload)
    with pytest.raises(ValueError, match="COMPLETED"):
        matrix.validate_smoke_attestation(
            malformed,
            expected_sha256=malformed["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=payload["environment_sha256"],
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
        )


def test_smoke_validation_rejects_tampered_scheduler_log_manifest(matrix):
    attestation = _attestation(matrix)
    payload = copy.deepcopy(attestation["payload"])
    payload["scheduler_log_manifest"]["payload"]["files"][0]["size"] += 1
    malformed = matrix.make_smoke_attestation(payload)
    with pytest.raises(ValueError, match="scheduler log manifest.*hash"):
        matrix.validate_smoke_attestation(
            malformed,
            expected_sha256=malformed["sha256"],
            expected_commit_sha="1" * 40,
            expected_tree_sha="2" * 40,
            expected_environment_sha256=malformed["payload"]["environment_sha256"],
            expected_source_manifest_sha256="4" * 64,
            expected_gpu_model="NVIDIA H100 80GB HBM3",
            expected_checkpoint_ceiling_bytes=1_000,
        )


def test_scheduler_log_byte_change_changes_final_attestation_digest(tmp_path, matrix):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    first = _assemble(cases, tmp_path / "first.json", matrix)
    target = tmp_path / "scheduler-logs" / "stage1_rope_one_step.out"
    target.write_bytes(target.read_bytes() + b"tampered\n")
    second = _assemble(cases, tmp_path / "second.json", matrix)
    assert (
        first["payload"]["scheduler_log_manifest"]["sha256"]
        != second["payload"]["scheduler_log_manifest"]["sha256"]
    )
    assert first["sha256"] != second["sha256"]


def test_assembler_rejects_missing_completion_and_unbound_nonutil_gpu_file(
    tmp_path, matrix
):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    missing = cases / "stage1_rope_one_step" / "action-completion.json"
    missing.unlink()
    with pytest.raises(ValueError, match="file set"):
        _assemble(cases, tmp_path / "missing.json", matrix)

    _write_canonical(
        missing,
        _case_evidence(matrix.build_matrix()[0], matrix)["action_completion"],
    )
    (cases / "stage1_rope_one_step" / "gpu.csv").write_bytes(b"")
    with pytest.raises(ValueError, match="file set"):
        _assemble(cases, tmp_path / "extra.json", matrix)


def test_assembler_rejects_wrong_case_and_reused_artifact_identity(tmp_path, matrix):
    cases = tmp_path / "cases"
    cases.mkdir()
    _write_assembler_cases(cases, matrix)
    first, second = matrix.build_matrix()[:2]
    first_source = json.loads(
        (cases / first.case_id / "source-attestation.json").read_text()
    )
    _write_canonical(cases / second.case_id / "source-attestation.json", first_source)
    with pytest.raises(ValueError, match="exact case"):
        _assemble(cases, tmp_path / "wrong.json", matrix)

    # Restore a correctly case-bound source but deliberately reuse identity.
    second_item = _case_evidence(second, matrix)
    source_payload = dict(second_item["source_attestation"]["payload"])
    source_payload["artifact_identity_sha256"] = first_source["payload"][
        "artifact_identity_sha256"
    ]
    source = matrix.make_source_attestation(source_payload)
    completion_payload = dict(second_item["action_completion"]["payload"])
    completion_payload["artifact_identity_sha256"] = source_payload[
        "artifact_identity_sha256"
    ]
    completion_payload["source_attestation_sha256"] = source["sha256"]
    completion = matrix.make_action_completion(completion_payload)
    _write_canonical(cases / second.case_id / "source-attestation.json", source)
    _write_canonical(cases / second.case_id / "action-completion.json", completion)
    with pytest.raises(ValueError, match="reuse.*artifact identity"):
        _assemble(cases, tmp_path / "reused.json", matrix)


def test_gpu_summary_reuses_monitor_active_window_contract(tmp_path):
    summary = _load("formal_gpu_summary", SUMMARY_SCRIPT)
    csv_path = tmp_path / "gpu.csv"
    rows = ["sample,90,GPU-a,0", "training_start,99.5"]
    # The low sample before initialization is excluded, but the low sample
    # inside the explicit active window is retained in the unfiltered mean.
    rows.extend(
        f"sample,{100 + index},GPU-a,{0 if index == 0 else 100}"
        for index in range(10)
    )
    rows.append("training_end,110")
    rows.append("sample,111,GPU-a,0")
    csv_path.write_text("\n".join(rows) + "\n")

    report = summary.summarize(
        csv_path,
        utilization_required=True,
        expected_gpu_uuids=("GPU-a",),
        expected_gpu_count=1,
    )
    assert report["means"] == {"GPU-a": 90.0}
    assert report["sample_counts"] == {"GPU-a": 10}


@pytest.mark.parametrize("count,gap", [(9, 1.0), (10, 0.49), (10, 1.51)])
def test_gpu_summary_rejects_short_or_wrong_cadence_unfiltered_windows(
    tmp_path, count, gap
):
    summary = _load(f"formal_gpu_summary_{count}_{gap}", SUMMARY_SCRIPT)
    csv_path = tmp_path / "gpu.csv"
    rows = ["training_start,99.5"]
    rows.extend(f"sample,{100 + index * gap},GPU-a,100" for index in range(count))
    rows.append("training_end,120")
    csv_path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="GPU utilization window failed"):
        summary.summarize(
            csv_path,
            utilization_required=True,
            expected_gpu_uuids=("GPU-a",),
            expected_gpu_count=1,
        )


def test_functional_only_summary_makes_no_utilization_claim(tmp_path):
    summary = _load("formal_gpu_summary_functional", SUMMARY_SCRIPT)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    report = summary.summarize(
        empty,
        utilization_required=False,
        expected_gpu_uuids=(),
        expected_gpu_count=0,
    )
    assert report == {
        "schema_version": 1,
        "utilization_required": False,
        "utilization_claimed": False,
    }


def test_gpu_recorder_binds_expected_uuid_set_and_checks_bytes_before_write():
    summary = _load("formal_gpu_summary_ceiling", SUMMARY_SCRIPT)
    marker = {
        "kind": "training_start",
        "monotonic_seconds": 1.0,
        "expected_gpu_uuids": ["GPU-a"],
        "expected_gpu_count": 1,
    }
    csv_record, jsonl_record = summary._encoded_record(marker)
    exact = len(csv_record.encode()) + len(jsonl_record.encode())
    csv_handle, jsonl_handle = io.StringIO(), io.StringIO()
    summary._write_record(
        csv_handle, jsonl_handle, marker, max_bytes=exact
    )
    assert "expected_gpu_uuids" in jsonl_handle.getvalue()

    csv_short, jsonl_short = io.StringIO(), io.StringIO()
    with pytest.raises(ValueError, match="ceiling"):
        summary._write_record(
            csv_short, jsonl_short, marker, max_bytes=exact - 1
        )
    assert csv_short.getvalue() == jsonl_short.getvalue() == ""


def test_gpu_recorder_waits_for_harness_active_signal_and_keeps_every_sample(
    tmp_path, monkeypatch
):
    summary = _load("formal_gpu_summary_active_signal", SUMMARY_SCRIPT)
    csv_path, jsonl_path = tmp_path / "gpu.csv", tmp_path / "gpu.jsonl"
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    start, end = tmp_path / "start", tmp_path / "end"
    current = 10.0
    sample_calls = 0

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout):
            self.stdout = stdout

    def query(argv, **_kwargs):
        nonlocal current, sample_calls
        assert "--id=0" in argv
        if "--query-gpu=uuid" in argv:
            return Result("GPU-0\n")
        sample_calls += 1
        current += 1.0
        if sample_calls == 10:
            end.write_text("21\n")
        return Result("GPU-0, 80\n")

    def sleep(_seconds):
        assert sample_calls == 0
        start.write_text("10\n")

    monkeypatch.setenv("NVIDIA_SMI", sys.executable)
    monkeypatch.setenv("FORMAL_VISIBLE_GPU_TOKENS", "0")
    summary.record_gpu_window(
        csv_path,
        jsonl_path,
        ready_path=ready,
        stop_path=stop,
        active_start_path=start,
        active_end_path=end,
        max_bytes=1 << 20,
        expected_gpu_count=1,
        monotonic_fn=lambda: current,
        sleep_fn=sleep,
        query_fn=query,
    )
    records, invalid = summary._monitor_module().parse_gpu_records(
        jsonl_path.read_text()
    )
    assert invalid == 0
    assert [record.kind for record in records].count("sample") == 10
    assert records[0].kind == "training_start"
    assert records[-1].kind == "training_end"


def test_strict_attestation_loader_rejects_symlink_duplicate_and_oversize(
    tmp_path, matrix
):
    attestation = _attestation(matrix)
    trusted = tmp_path / "trusted.json"
    _write_canonical(trusted, attestation)
    alias = tmp_path / "alias.json"
    alias.symlink_to(trusted)
    with pytest.raises(ValueError, match="safely"):
        matrix._read_regular_bytes(alias, max_bytes=1 << 20)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.json"
    _write_canonical(nested, attestation)
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|invalid component"):
        matrix._read_regular_bytes(
            parent_alias / "nested.json", max_bytes=1 << 20
        )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        matrix._strict_json(b'{"a":1,"a":2}\n', where="duplicate")
    with pytest.raises(ValueError, match="exceeds ceiling"):
        matrix._read_regular_bytes(trusted, max_bytes=1)


def test_case_result_directory_fsync_failure_cannot_be_republished(
    tmp_path, matrix, monkeypatch
):
    output = tmp_path / "case-result.json"
    real_fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(matrix.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory fsync failed"):
        matrix._publish_case_result(output, {"case_id": "incomplete"})
    assert output.is_file()
    with pytest.raises(ValueError, match="no-replace"):
        matrix._publish_case_result(output, {"case_id": "replacement"})


def test_runtime_gpu_query_rejects_path_lookup(monkeypatch, matrix):
    monkeypatch.setenv("NVIDIA_SMI", "nvidia-smi")
    with pytest.raises(RuntimeError, match="absolute executable"):
        matrix._query_visible_gpu_devices()


def test_nccl_completion_publish_is_after_all_rank_final_barrier():
    source = (ROOT / "scripts" / "run_exact_tabicl.py").read_text()
    final_attestation = source.index("verifier.attest_loaded_tabicl(root, expected_src)\n    if args.action")
    barrier = source.index("dist.barrier()", final_attestation)
    completion = source.index("completion = harness.make_action_completion", barrier)
    assert final_attestation < barrier < completion


def test_exact_bootstrap_rejects_parent_symlink_and_wrong_editable_import(
    tmp_path, monkeypatch
):
    exact = _load("exact_t_bootstrap_contract", EXACT_SCRIPT)
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|invalid component"):
        exact._open_directory_nofollow(alias)

    verifier = _load("exact_t_runtime_verifier_contract", VERIFY_SCRIPT)
    candidate = tmp_path / "candidate"
    expected_package = candidate / "src" / "tabicl"
    expected_package.mkdir(parents=True)
    (expected_package / "__init__.py").write_text("")
    wrong_package = tmp_path / "raw" / "src" / "tabicl"
    wrong_package.mkdir(parents=True)
    wrong_init = wrong_package / "__init__.py"
    wrong_init.write_text("")
    fake = SimpleNamespace(__file__=str(wrong_init), __path__=[str(wrong_package)])
    monkeypatch.setitem(sys.modules, "tabicl", fake)
    with pytest.raises(ValueError, match="not the exact attested checkout"):
        verifier.attest_loaded_tabicl(candidate, candidate / "src")


def test_exact_bootstrap_no_replace_and_isolation_guard(tmp_path):
    exact = _load("exact_t_bootstrap_no_replace", EXACT_SCRIPT)
    output = tmp_path / "source.json"
    exact._publish_no_replace(output, b"authority\n", max_bytes=100)
    with pytest.raises(FileExistsError):
        exact._publish_no_replace(output, b"replacement\n", max_bytes=100)
    assert output.read_bytes() == b"authority\n"
    if not sys.flags.isolated:
        with pytest.raises(ValueError, match="requires Python -I -B"):
            exact.main([])


def test_exact_publisher_requires_precreated_physical_parent(tmp_path):
    exact = _load("exact_t_physical_publish_parent", EXACT_SCRIPT)
    missing_parent = tmp_path / "missing"
    with pytest.raises(ValueError, match="publication parent.*invalid component"):
        exact._publish_no_replace(
            missing_parent / "source.json", b"authority\n", max_bytes=100
        )
    assert not missing_parent.exists()

    physical_parent = tmp_path / "physical"
    physical_parent.mkdir()
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(physical_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="publication parent.*symlink"):
        exact._publish_no_replace(
            parent_alias / "source.json", b"authority\n", max_bytes=100
        )
    assert not (physical_parent / "source.json").exists()

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside\n")
    destination_alias = physical_parent / "source.json"
    destination_alias.symlink_to(outside)
    with pytest.raises(FileExistsError):
        exact._publish_no_replace(
            destination_alias, b"replacement\n", max_bytes=100
        )
    assert destination_alias.is_symlink()
    assert outside.read_bytes() == b"outside\n"


def test_exact_publisher_uses_dirfd_nofollow_link_and_never_creates_parent():
    source = EXACT_SCRIPT.read_text()
    publisher = source.split("def _publish_no_replace", 1)[1].split(
        "\ndef _split_argv", 1
    )[0]
    assert ".mkdir(" not in publisher
    assert "tempfile" not in publisher
    assert "os.O_NOFOLLOW" in publisher
    assert "dir_fd=parent_fd" in publisher
    assert "src_dir_fd=parent_fd" in publisher
    assert "dst_dir_fd=parent_fd" in publisher
    assert "os.fsync(parent_fd)" in publisher
