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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _case_evidence(
    case,
    matrix,
    *,
    commit="1" * 40,
    tree="2" * 40,
    source_manifest="4" * 64,
):
    artifact_identity = hashlib.sha256(("identity:" + case.case_id).encode()).hexdigest()
    canonical = lambda value: json.dumps(
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
    runtime_evidence = matrix.make_runtime_evidence(
        {
            "case_id": case.case_id,
            "commit_sha": commit,
            "tree_sha": tree,
            "source_manifest_sha256": source_manifest,
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
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
    payload = {
        "commit_sha": commit,
        "tree_sha": tree,
        "environment_sha256": cases[0]["runtime_evidence"]["payload"]["environment"]["sha256"],
        "source_manifest_sha256": "4" * 64,
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.00",
        "checkpoint_ceiling_bytes": 1_000,
        "observed_checkpoint_max_bytes": 500,
        "cases": cases,
    }
    return matrix.make_smoke_attestation(payload)


def _write_canonical(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _write_assembler_cases(root: Path, matrix) -> None:
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


def _assemble(root: Path, output: Path, matrix):
    return matrix.assemble_smoke_attestation(
        root,
        output=output,
        expected_commit_sha="1" * 40,
        expected_tree_sha="2" * 40,
        expected_source_manifest_sha256="4" * 64,
        expected_checkpoint_ceiling_bytes=1_000,
        max_input_bytes=1 << 20,
        max_output_bytes=8 << 20,
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


@pytest.mark.parametrize("field", ["gpu_model", "driver_version", "cases"])
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
    with pytest.raises((FileExistsError, ValueError), match="no-replace|exist"):
        _assemble(cases, output, matrix)


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
        if argv[1] == "--query-gpu=uuid":
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
