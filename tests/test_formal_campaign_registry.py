from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "formal_campaign_registry.py"
REPOSITORY_HELPER = Path(__file__).parents[1] / "scripts" / "verify_git_repository.py"
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
TIME_LIMITS = {
    "stage1": "14-00:00:00",
    "stage2": "3-00:00:00",
    "stage3": "1-00:00:00",
}


def _load():
    spec = importlib.util.spec_from_file_location("formal_campaign_registry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flags", [("-B",), ("-I",)])
def test_campaign_cli_requires_isolated_no_bytecode_python(flags):
    completed = subprocess.run(
        [
            sys.executable,
            *flags,
            str(SCRIPT),
            "capacity",
            "--fragment-size",
            "4096",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires Python -I -B" in completed.stderr


def test_campaign_cli_accepts_isolated_no_bytecode_python():
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "capacity",
            "--fragment-size",
            "4096",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["fragment_size"] == 4096


def test_campaign_cli_rejects_fifo_spec_without_blocking(tmp_path):
    spec = tmp_path / "campaign-spec.json"
    os.mkfifo(spec)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "publish-campaign",
            "--spec",
            str(spec),
            "--spec-max-bytes",
            "4096",
            "--output",
            str(tmp_path / "campaign.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode != 0
    assert "bounded singly-linked regular file" in completed.stderr


def test_campaign_fifo_git_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "git"
    os.mkfifo(fifo)
    code = (
        "import importlib.util,pathlib,sys\n"
        "spec=importlib.util.spec_from_file_location('fifo_campaign',sys.argv[1])\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "module._open_trusted_executable(pathlib.Path(sys.argv[2]),"
        "expected_sha256='0'*64,where='campaign Git')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SCRIPT), str(fifo)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode != 0
    assert "bounded executable regular file" in completed.stderr


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_repository_git(path: Path, *, commit_sha: str) -> str:
    state = path.parent
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"state = Path({str(state)!r})\n"
        "expected = [\n"
        "    'ls-remote', '--refs',\n"
        "    'https://github.com/kikixiong/tabicl-pe.git',\n"
        "    'refs/heads/codex/position-identity-v1',\n"
        "]\n"
        "if sys.argv[1:] != expected:\n"
        "    raise SystemExit(91)\n"
        "count_path = state / 'repository-query-count'\n"
        "count = int(count_path.read_text()) + 1 if count_path.exists() else 1\n"
        "count_path.write_text(str(count))\n"
        "(state / 'repository-query-marker').write_text(str(count))\n"
        "if (state / 'repository-ref-missing').exists():\n"
        "    raise SystemExit(0)\n"
        f"commit = {commit_sha!r}\n"
        "if (state / 'repository-ref-moved').exists():\n"
        "    commit = 'f' * 40\n"
        "print(commit + '\\trefs/heads/codex/position-identity-v1')\n"
    )
    path.chmod(0o755)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> dict:
    return {
        "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
        "candidate_ref": "refs/heads/codex/position-identity-v1",
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
        "source_manifest_sha256": "1" * 64,
        "environment_sha256": "2" * 64,
    }


def _repository_binding() -> dict:
    spec = importlib.util.spec_from_file_location(
        "formal_campaign_repository_helper", REPOSITORY_HELPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expected_repository_binding(
        expected_commit_sha="a" * 40,
        git_sha256="d" * 64,
    )


def _manifest_sha256(module, kind: str, payload: dict) -> str:
    return module.make_manifest(kind, payload)["sha256"]


def _seed_sha256(module, seed: int) -> str:
    return _manifest_sha256(
        module,
        "seed",
        {
            "np_seed": seed,
            "torch_seed": seed,
            "identity_rng_seed": seed,
            "world_size": 1,
        },
    )


def _treatment_sha256(module, arm: str, seed: int) -> str:
    identity = {
        "schema_version": 1,
        "row_identity_mode": arm,
        "identity_rng_seed": seed,
        "seed_policy": "sha256-domain-separated-base-seed-and-rank-v1",
        "sampler_version": (
            "tabicl-temporary-identity/randperm-cpu-v1" if arm == "temporary" else None
        ),
        "world_size": 1,
    }
    return _manifest_sha256(
        module,
        "treatment",
        {**identity, "manifest_sha256": module.canonical_sha256(identity)},
    )


def _fake_digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _rehash(module, manifest: dict) -> None:
    body = {key: manifest[key] for key in ("schema_version", "kind", "payload")}
    manifest["sha256"] = module.canonical_sha256(body)


def _stages() -> list[dict]:
    return [
        {
            "stage": stage,
            "terminal_step": terminal_step,
            "time_limit": TIME_LIMITS[stage],
            "prior_sha256": "3" * 64,
            "architecture_sha256": "4" * 64,
            "optimizer_sha256": "5" * 64,
            "scientific_sha256": "6" * 64,
        }
        for stage, terminal_step in STAGES
    ]


def _campaign(module):
    smoke_payload = {
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
        "source_manifest_sha256": "1" * 64,
        "environment_sha256": "2" * 64,
        "checkpoint_ceiling_bytes": 300_000_000,
        "observed_checkpoint_max_bytes": 220_700_000,
        "nvidia_smi_sha256": "7" * 64,
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.00",
    }
    smoke = module.make_manifest("h100_identity_smoke", smoke_payload)
    draft = module.build_campaign_manifest(
        campaign_id="position-identity-v1",
        training_source=_source(),
        h100_attestation=smoke,
        expected_h100_sha256=smoke["sha256"],
        checkpoint_ceiling_bytes=300_000_000,
        expected_gpu_model="NVIDIA H100 80GB HBM3",
        stages=_stages(),
    )
    evidence = {
        "exact_root": "/exact/candidate",
        "source_manifest_path": "/external/source.json",
        "h100_attestation_path": "/external/h100-smoke.json",
        "h100_submission_receipt_path": "/external/submission-receipt.json",
        "environment_completion_path": "/external/environment-complete.json",
        "git_path": "/usr/bin/git",
        "git_sha256": "7" * 64,
        "h100_attestation_max_bytes": module.H100_ATTESTATION_CEILING_BYTES,
        "h100_submission_receipt_max_bytes": (
            module.H100_SUBMISSION_RECEIPT_CEILING_BYTES
        ),
        "source_manifest_max_bytes": module.SOURCE_MANIFEST_CEILING_BYTES,
        "environment_completion_max_bytes": (
            module.ENVIRONMENT_COMPLETION_CEILING_BYTES
        ),
        "environment_manifest_max_bytes": module.ENVIRONMENT_MANIFEST_CEILING_BYTES,
        "environment_inventory_max_bytes": (module.ENVIRONMENT_INVENTORY_CEILING_BYTES),
        "h100_submission_receipt_sha256": "8" * 64,
        "h100_validation_report_sha256": "9" * 64,
        "environment_transaction_sha256": "a" * 64,
        "environment_transaction_completion_raw_sha256": "b" * 64,
        "two_gpu_environment_sha256": "c" * 64,
        "sacct_sha256": "d" * 64,
        "repository_identity_sha256": "e" * 64,
        "repository_query_sha256": "f" * 64,
    }
    return module._formal_campaign_from_draft(draft, evidence)


def _layout(tmp_path: Path, module):
    root = tmp_path / "campaign"
    acceptances = root / "acceptances"
    evaluations = root / "evaluations"
    acceptances.mkdir(parents=True)
    evaluations.mkdir()
    campaign = _campaign(module)
    # Registry-ordering and acceptance unit tests use a schema-complete campaign
    # fixture. Dedicated production-publisher tests below exercise the real
    # exact-T H100/receipt revalidation path without this local test double.
    module._revalidate_campaign_evidence = module.validate_campaign_manifest
    module._validate_existing_terminal_evidence = (
        lambda *, terminal_path, terminal, raw_evidence: terminal
    )
    campaign_path = root / "campaign.json"
    module.publish_manifest_no_replace(
        campaign_path,
        campaign,
        max_bytes=module.CAMPAIGN_MANIFEST_CEILING_BYTES,
    )
    return root, campaign_path, acceptances, evaluations, campaign


def _production_publish_fixture(tmp_path: Path, module):
    matrix = _load_path(
        f"campaign_h100_matrix_{tmp_path.name}",
        SCRIPT.with_name("run_h100_identity_validation.py"),
    )
    matrix_tests = _load_path(
        f"campaign_h100_fixture_{tmp_path.name}",
        Path(__file__).with_name("test_h100_identity_matrix.py"),
    )
    attestation = matrix_tests._attestation(matrix)
    payload = attestation["payload"]
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    fake_git = evidence_root / "git"
    fake_git_sha256 = _write_fake_repository_git(
        fake_git, commit_sha=payload["commit_sha"]
    )
    repository_contract = _load_path(
        f"campaign_repository_contract_{tmp_path.name}", REPOSITORY_HELPER
    )
    repository = repository_contract.expected_repository_binding(
        expected_commit_sha=payload["commit_sha"],
        git_sha256=fake_git_sha256,
    )
    payload["repository_binding"] = repository
    attestation = matrix.make_smoke_attestation(payload)
    environments: dict[str, str] = {}
    artifact_identities: dict[str, str] = {}
    job_bindings: dict[str, dict] = {}
    for item in payload["cases"]:
        case_id = item["case_id"]
        runtime = item["runtime_evidence"]["payload"]
        scheduler = runtime["scheduler_binding"]
        environments[str(item["world_size"])] = runtime["environment"]["sha256"]
        artifact_identities[case_id] = item["artifact_identity_sha256"]
        job_bindings[case_id] = {
            "job_id": scheduler["job_id"],
            "cluster": scheduler["slurm_receipt_cluster"],
            "requested_resource": scheduler["requested_resource"],
            "requested_resource_sha256": scheduler["requested_resource_sha256"],
            "held_plan_sha256": scheduler["held_plan_sha256"],
            "scontrol_sha256": scheduler["scontrol_sha256"],
        }
    assert set(environments) == {"1", "2"}
    environment_transaction = {
        "schema_version": 1,
        "kind": "formal_environment_transaction_verification",
        "completion_raw_sha256": "1" * 64,
        "transaction_sha256": "2" * 64,
        "source_commit_sha": payload["commit_sha"],
        "source_tree_sha": payload["tree_sha"],
        "git_sha256": repository["git_sha256"],
        "one_gpu_environment_sha256": environments["1"],
        "two_gpu_environment_sha256": environments["2"],
        "inventory_sha256": "3" * 64,
        "installed_distributions_sha256": "d" * 64,
        "capture_visible_cuda_device_count": 2,
        "output_raw_sha256_by_role": {
            "one_gpu": "4" * 64,
            "two_gpu": "5" * 64,
            "inventory": "6" * 64,
        },
    }
    receipt = {
        "sha256": payload["submission_receipt_sha256"],
        "repository_binding": repository,
        "environment_sha256_by_world_size": environments,
        "environment_transaction": environment_transaction,
        "artifact_identities": artifact_identities,
        "job_bindings": job_bindings,
        "nvidia_smi_sha256": payload["nvidia_smi_sha256"],
        "sacct_sha256": payload["scheduler_terminal_manifest"]["payload"][
            "sacct_sha256"
        ],
    }
    attestation_path = evidence_root / "h100-smoke.json"
    attestation_path.write_bytes(module.canonical_json_bytes(attestation) + b"\n")
    receipt_path = evidence_root / "submission-receipt.json"
    receipt_path.write_text("receipt placeholder\n")
    completion_path = evidence_root / "environment-complete.json"
    completion_path.write_text("environment placeholder\n")
    source_path = evidence_root / "source.json"
    source_path.write_text("source placeholder\n")
    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    campaign_root = tmp_path / "campaign"
    (campaign_root / "acceptances").mkdir(parents=True)
    (campaign_root / "evaluations").mkdir()

    module._attest_exact_campaign_source = lambda **_kwargs: {}
    module._load_exact_h100_contract = lambda _root: matrix
    module._load_exact_repository_contract = lambda _root: repository_contract
    module._test_repository_query_state = evidence_root
    environment_verification_calls = []

    def verify_environment(**call):
        environment_verification_calls.append(call)
        return dict(environment_transaction)

    module._load_exact_environment_transaction_contract = lambda _root: SimpleNamespace(
        _verify=verify_environment
    )
    module._test_environment_verification_calls = environment_verification_calls
    matrix.validate_gate_submission_receipt = lambda *_args, **_kwargs: receipt
    source = {
        "candidate_repository": module.CANONICAL_REPOSITORY,
        "candidate_ref": module.CANONICAL_TRAINING_REF,
        "commit_sha": payload["commit_sha"],
        "tree_sha": payload["tree_sha"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "environment_sha256": payload["environment_sha256"],
    }
    kwargs = {
        "output_path": campaign_root / "campaign.json",
        "campaign_id": "position-identity-v1",
        "training_source": source,
        "exact_root": exact_root,
        "source_manifest_path": source_path,
        "h100_attestation_path": attestation_path,
        "expected_h100_sha256": attestation["sha256"],
        "h100_submission_receipt_path": receipt_path,
        "environment_completion_path": completion_path,
        "environment_completion_max_bytes": (
            module.ENVIRONMENT_COMPLETION_CEILING_BYTES
        ),
        "environment_manifest_max_bytes": module.ENVIRONMENT_MANIFEST_CEILING_BYTES,
        "environment_inventory_max_bytes": (module.ENVIRONMENT_INVENTORY_CEILING_BYTES),
        "git_path": fake_git,
        "git_sha256": fake_git_sha256,
        "checkpoint_ceiling_bytes": payload["checkpoint_ceiling_bytes"],
        "expected_gpu_model": payload["gpu_model"],
        "stages": _stages(),
        "h100_attestation_max_bytes": 8 << 20,
        "h100_submission_receipt_max_bytes": 1 << 20,
        "source_manifest_max_bytes": 1 << 20,
    }
    return matrix, attestation, receipt, kwargs


def _raw_environment_fixture(tmp_path: Path, module):
    contract = _load_path(
        f"campaign_environment_contract_{tmp_path.name}",
        SCRIPT.with_name("verify_formal_environment_transaction.py"),
    )
    root = tmp_path / "raw-environment"
    root.mkdir()
    fingerprint = {
        "visible_distribution_multiset": [],
        "effective_formal_runtime_distributions": [],
    }
    installed_sha256 = module.canonical_sha256(fingerprint)
    base_environment = {
        "environment_fingerprint_schema_version": 2,
        "installed_distributions_sha256": installed_sha256,
    }
    one = module.make_manifest(
        "environment", {**base_environment, "visible_cuda_device_count": 1}
    )
    two = module.make_manifest(
        "environment", {**base_environment, "visible_cuda_device_count": 2}
    )
    environments = {"1": one["sha256"], "2": two["sha256"]}
    source = {
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
    }
    git_sha256 = "c" * 64
    inventory = module.make_manifest(
        "formal_environment_inventory",
        {
            "source_commit_sha": source["commit_sha"],
            "source_tree_sha": source["tree_sha"],
            "git_sha256": git_sha256,
            "capture_visible_cuda_device_count": 0,
            "installed_distributions_sha256": installed_sha256,
            "environment_sha256_by_world_size": environments,
            "fingerprint_preimage": fingerprint,
        },
    )
    outputs = []
    paths = {}
    for role, name, value in (
        ("one_gpu", "one-gpu.json", one),
        ("two_gpu", "two-gpu.json", two),
        ("inventory", "inventory.json", inventory),
    ):
        raw = module.canonical_json_bytes(value) + b"\n"
        path = root / name
        path.write_bytes(raw)
        paths[role] = path
        outputs.append(
            {
                "role": role,
                "name": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    descriptor = {
        "schema_version": 1,
        "source_commit_sha": source["commit_sha"],
        "source_tree_sha": source["tree_sha"],
        "outputs": outputs,
    }
    transaction_sha256 = hashlib.sha256(
        module.canonical_json_bytes(descriptor) + b"\n"
    ).hexdigest()
    completion = {
        **descriptor,
        "kind": "formal_environment_generation_completion",
        "transaction_sha256": transaction_sha256,
    }
    completion_raw = module.canonical_json_bytes(completion) + b"\n"
    completion_path = root / "environment-complete.json"
    completion_path.write_bytes(completion_raw)
    verify_kwargs = {
        "completion_path": completion_path,
        "expected_completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "expected_transaction_sha256": transaction_sha256,
        "expected_one_gpu_environment_sha256": environments["1"],
        "expected_two_gpu_environment_sha256": environments["2"],
        "expected_inventory_sha256": inventory["sha256"],
        "expected_source_commit_sha": source["commit_sha"],
        "expected_source_tree_sha": source["tree_sha"],
        "expected_git_sha256": git_sha256,
        "completion_max_bytes": module.ENVIRONMENT_COMPLETION_CEILING_BYTES,
        "manifest_max_bytes": module.ENVIRONMENT_MANIFEST_CEILING_BYTES,
        "inventory_max_bytes": module.ENVIRONMENT_INVENTORY_CEILING_BYTES,
    }
    summary = contract._verify(**verify_kwargs)
    receipt = {
        "environment_transaction": summary,
        "environment_sha256_by_world_size": environments,
        "repository_binding": {"git_sha256": git_sha256},
    }
    module._load_exact_environment_transaction_contract = lambda _root: contract
    return {
        "completion": completion_path,
        "one_gpu": paths["one_gpu"],
        "inventory": paths["inventory"],
        "receipt": receipt,
        "source": source,
    }


def _terminal(module, campaign, seed: int, predecessor_map: dict[str, str]):
    binding = module.campaign_binding(
        campaign,
        predecessor_acceptance_sha256_by_seed=predecessor_map,
    )
    repository_binding = _repository_binding()
    transaction_id = f"{seed:032x}"
    manifest_ceiling = 2_000_000
    protocol_metadata_allowance = module.FORMAL_METADATA_CEILING_BYTES
    jobs = []
    jobs_by_pair = {}
    for position, (stage, terminal_step) in enumerate(STAGES):
        for arm_index, arm in enumerate(ARMS):
            job_id = f"{seed}{position + 1}{arm_index + 1}"
            slug = f"{arm}-{stage}"
            finalized_binding = {
                "finalized_manifest_path": (
                    f"arms/{arm}/{stage}/finalized-checkpoint.json"
                ),
                "finalized_manifest_sha256": _fake_digest(
                    f"finalized-manifest-{seed}-{arm}-{stage}"
                ),
                "checkpoint_path": f"arms/{arm}/{stage}/step-{terminal_step}.ckpt",
                "checkpoint_sha256": _fake_digest(f"checkpoint-{seed}-{arm}-{stage}"),
                "checkpoint_size": 220_000_000,
                "provenance_sha256": _fake_digest(f"provenance-{seed}-{arm}-{stage}"),
                "seed_sha256": _seed_sha256(module, seed),
                "treatment_sha256": _treatment_sha256(module, arm, seed),
            }
            stable_validation = _fake_digest(f"stable-validation-{seed}-{arm}-{stage}")
            finalized_artifact = {
                "binding": finalized_binding,
                "finalized_manifest_file_sha256": _fake_digest(
                    f"finalized-file-{seed}-{arm}-{stage}"
                ),
                "finalized_manifest_size": 4_096,
                "independent_stable_validations": 2,
                "initial_validation_sha256": stable_validation,
                "publication_validation_sha256": stable_validation,
            }
            job = {
                "arm": arm,
                "stage": stage,
                "job_id": job_id,
                "cluster": None,
                "job_name": f"tabicl-{transaction_id}-{arm}-s{position + 1}",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "derived_exit_code": "0:0",
                "completion": {
                    "path": f"runtime-completions/{slug}.json",
                    "manifest_sha256": _fake_digest(
                        f"completion-manifest-{seed}-{arm}-{stage}"
                    ),
                    "file_sha256": _fake_digest(
                        f"completion-file-{seed}-{arm}-{stage}"
                    ),
                    "stdout_observed_size": 90,
                    "stderr_observed_size": 10,
                    "seed": seed,
                    "scheduler": {
                        "partition": "h100",
                        "qos": "long",
                        "cpus_per_task": 64,
                        "memory_mb": 131_072,
                        "gpus_per_job": 1,
                        "time_limit": TIME_LIMITS[stage],
                        "query_sha256": _fake_digest(f"scontrol-{seed}-{arm}-{stage}"),
                    },
                    "cuda_visible_devices": "0",
                    "gpu_name": "NVIDIA H100 80GB HBM3",
                    "gpu_uuid": f"GPU-terminal-{seed}-{position}-{arm_index}",
                    "gpu_driver_version": "570.00",
                    "repository_binding": repository_binding,
                    "manifest_ceiling_bytes": manifest_ceiling,
                    "protocol_metadata_allowance_bytes": (protocol_metadata_allowance),
                },
                "scheduler_logs": [
                    {
                        "stream": "stdout",
                        "path": f"scheduler-logs/{slug}.out",
                        "size": 100,
                        "sha256": _fake_digest(f"stdout-{seed}-{arm}-{stage}"),
                        "fatal_signature_scan_passed": True,
                        "stable_reads": 2,
                    },
                    {
                        "stream": "stderr",
                        "path": f"scheduler-logs/{slug}.err",
                        "size": 20,
                        "sha256": _fake_digest(f"stderr-{seed}-{arm}-{stage}"),
                        "fatal_signature_scan_passed": True,
                        "stable_reads": 2,
                    },
                ],
                "finalized_artifact": finalized_artifact,
                "parent_lineage": None,
            }
            jobs.append(job)
            jobs_by_pair[(arm, stage)] = job
    for arm in ARMS:
        for stage_index, (stage, _terminal_step) in enumerate(STAGES):
            if stage_index == 0:
                continue
            parent_stage = STAGES[stage_index - 1][0]
            parent = jobs_by_pair[(arm, parent_stage)]
            jobs_by_pair[(arm, stage)]["parent_lineage"] = {
                **parent["finalized_artifact"]["binding"],
                "job_id": parent["job_id"],
                "stage": parent_stage,
            }
    job_ids = [job["job_id"] for job in jobs]
    terminal_query_argv = [
        "--noheader",
        "--allocations",
        f"--jobs={','.join(job_ids)}",
        "--format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode",
        "--parsable2",
    ]
    return module.make_manifest(
        "formal_terminal_scheduler_logs",
        {
            "study_id": f"position-identity-v1-seed{seed}",
            "seed": seed,
            "transaction_id": f"{seed:032x}",
            "submission_receipt_sha256": hashlib.sha256(
                f"receipt-{seed}".encode()
            ).hexdigest(),
            "transaction_ledger_sha256": hashlib.sha256(
                f"ledger-{seed}".encode()
            ).hexdigest(),
            "source_commit_sha": "a" * 40,
            "source_tree_sha": "b" * 40,
            "repository_binding": repository_binding,
            "campaign_binding": binding,
            "runtime_tools": {"nvidia_smi_sha256": "7" * 64},
            "manifest_ceiling_bytes": manifest_ceiling,
            "protocol_metadata_allowance_bytes": protocol_metadata_allowance,
            "sacct_sha256": "8" * 64,
            "path_format": "artifact_root_relative_posix_v1",
            "terminal_verified": True,
            "terminal_queries": [
                {
                    "cluster": None,
                    "job_ids": job_ids,
                    "query_argv_sha256": module.canonical_sha256(terminal_query_argv),
                    "stdout_sha256": _fake_digest(f"sacct-stdout-{seed}"),
                    "stderr_sha256": _fake_digest(f"sacct-stderr-{seed}"),
                }
            ],
            "jobs": jobs,
        },
    )


def _evaluation(module, campaign, terminal, seed, predecessor_map):
    binding = module.campaign_binding(
        campaign,
        predecessor_acceptance_sha256_by_seed=predecessor_map,
    )
    return module.make_manifest(
        "minimum_evaluation_sanity",
        {
            "campaign_id": "position-identity-v1",
            "campaign_manifest_sha256": campaign["sha256"],
            "campaign_binding": binding,
            "seed": seed,
            "study_id": f"position-identity-v1-seed{seed}",
            "formal_terminal_attestation_sha256": terminal["sha256"],
            "formal_submission_receipt_sha256": terminal["payload"][
                "submission_receipt_sha256"
            ],
            "training_source": _source(),
            "evaluation_source": {
                "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
                "commit_sha": "c" * 40,
                "tree_sha": "d" * 40,
                "source_manifest_sha256": "9" * 64,
                "descendant_of_training_commit_sha": "a" * 40,
                "ancestry_verified": True,
                "ancestry_query_sha256": "e" * 64,
            },
            "evaluation_protocol_sha256": "f" * 64,
            "matched_dataset_sha256": "0" * 64,
            "results_by_arm": {
                arm: {
                    "evaluated_datasets": 1,
                    "evaluated_examples": 32,
                    "prediction_manifest_sha256": hashlib.sha256(
                        f"{seed}-{arm}".encode()
                    ).hexdigest(),
                    "metrics": {"accuracy": 0.5, "log_loss": 1.0},
                }
                for arm in ARMS
            },
            "acceptance": {
                "all_three_arms_present": True,
                "all_values_finite": True,
                "minimum_sanity_passed": True,
            },
        },
    )


def _publish_seed_evidence(
    tmp_path: Path,
    module,
    campaign,
    evaluations: Path,
    seed: int,
    predecessor_map: dict[str, str],
):
    terminal = _terminal(module, campaign, seed, predecessor_map)
    terminal_root = tmp_path / f"formal-seed-{seed}"
    terminal_root.mkdir()
    terminal_path = terminal_root / "terminal-scheduler-logs.json"
    module.publish_manifest_no_replace(
        terminal_path,
        terminal,
        max_bytes=module.TERMINAL_ATTESTATION_CEILING_BYTES,
    )
    evaluation = _evaluation(module, campaign, terminal, seed, predecessor_map)
    evaluation_path = evaluations / f"seed-{seed}.json"
    module.publish_manifest_no_replace(
        evaluation_path,
        evaluation,
        max_bytes=module.EVALUATION_RECEIPT_CEILING_BYTES,
    )
    return terminal_path, evaluation_path, terminal, evaluation


def _terminal_raw_kwargs(terminal_path: Path, module) -> dict[str, object]:
    terminal_root = terminal_path.parent
    return {
        "formal_artifact_root": terminal_root,
        "submission_receipt_path": terminal_root / "submission-receipt.json",
        "transaction_ledger_path": terminal_root / "transaction-ledger.json",
        "terminal_metadata_max_bytes": module.FORMAL_METADATA_CEILING_BYTES,
    }


def _accept(
    tmp_path: Path,
    module,
    campaign_path: Path,
    acceptances: Path,
    evaluations: Path,
    campaign,
    seed: int,
    predecessor_map: dict[str, str],
):
    terminal_path, evaluation_path, _terminal_value, _evaluation_value = (
        _publish_seed_evidence(
            tmp_path,
            module,
            campaign,
            evaluations,
            seed,
            predecessor_map,
        )
    )
    return module.publish_seed_acceptance(
        campaign_path=campaign_path,
        campaign_expected_sha256=campaign["sha256"],
        acceptance_registry=acceptances,
        seed=seed,
        terminal_attestation_path=terminal_path,
        evaluation_receipt_path=evaluation_path,
        **_terminal_raw_kwargs(terminal_path, module),
    )


def test_campaign_freezes_seed_independent_static_protocol_and_h100_contract():
    module = _load()
    campaign = _campaign(module)
    validated = module.validate_campaign_manifest(campaign)
    assert validated["training_source"] == _source()
    assert validated["h100_gate"] == {
        "attestation_sha256": campaign["payload"]["h100_gate"]["attestation_sha256"],
        "checkpoint_ceiling_bytes": 300_000_000,
        "observed_checkpoint_max_bytes": 220_700_000,
        "nvidia_smi_sha256": "7" * 64,
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.00",
    }
    assert [item["terminal_step"] for item in validated["stages"]] == [
        500_000,
        40_000,
        10_000,
    ]
    assert all("seed" not in item for item in validated["stages"])


def test_campaign_source_trust_chain_includes_terminal_acceptance_contract():
    module = _load()
    assert module._CAMPAIGN_REQUIRED_SOURCE_PATHS == frozenset(
        {
            "scripts/finalize_formal_scheduler_logs.py",
            "scripts/formal_campaign_registry.py",
            "scripts/run_h100_identity_validation.py",
            "scripts/verify_formal_environment_transaction.py",
            "scripts/verify_git_repository.py",
            "scripts/verify_runtime_source.py",
        }
    )


@pytest.mark.parametrize("code_roots", [("scripts",), ("src/tabicl",)])
def test_campaign_source_coverage_rejects_omitted_code_root(code_roots):
    module = _load()
    verifier = _load_path(
        f"campaign_source_coverage_{code_roots[0].replace('/', '_')}",
        SCRIPT.with_name("verify_runtime_source.py"),
    )
    manifest = {
        "payload": {
            "entries": [
                {"path": path}
                for path in sorted(module._CAMPAIGN_REQUIRED_SOURCE_PATHS)
            ],
            "code_roots": list(code_roots),
        }
    }

    with pytest.raises(ValueError, match="omits required code roots"):
        module._require_campaign_manifest_coverage(verifier, manifest)


def test_campaign_source_coverage_accepts_both_required_code_roots():
    module = _load()
    verifier = _load_path(
        "campaign_source_coverage_complete",
        SCRIPT.with_name("verify_runtime_source.py"),
    )
    manifest = {
        "payload": {
            "entries": [
                {"path": path}
                for path in sorted(module._CAMPAIGN_REQUIRED_SOURCE_PATHS)
            ],
            "code_roots": ["scripts", "src/tabicl"],
        }
    }

    assert module._require_campaign_manifest_coverage(verifier, manifest) is manifest


def test_offline_draft_or_hand_published_minimal_smoke_cannot_authorize(tmp_path):
    module = _load()
    smoke = module.make_manifest(
        "h100_identity_smoke",
        {
            "commit_sha": "a" * 40,
            "tree_sha": "b" * 40,
            "source_manifest_sha256": "1" * 64,
            "environment_sha256": "2" * 64,
            "checkpoint_ceiling_bytes": 300_000_000,
            "observed_checkpoint_max_bytes": 220_700_000,
            "nvidia_smi_sha256": "7" * 64,
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "driver_version": "570.00",
        },
    )
    draft = module.build_campaign_manifest(
        campaign_id="position-identity-v1",
        training_source=_source(),
        h100_attestation=smoke,
        expected_h100_sha256=smoke["sha256"],
        checkpoint_ceiling_bytes=300_000_000,
        expected_gpu_model="NVIDIA H100 80GB HBM3",
        stages=_stages(),
    )
    assert draft["kind"] == "formal_campaign_draft"
    with pytest.raises(ValueError, match="schema or kind"):
        module.validate_campaign_manifest(draft)

    root = tmp_path / "campaign"
    (root / "acceptances").mkdir(parents=True)
    (root / "evaluations").mkdir()
    module.publish_manifest_no_replace(
        root / "campaign.json",
        draft,
        max_bytes=module.CAMPAIGN_MANIFEST_CEILING_BYTES,
    )
    with pytest.raises(ValueError, match="schema or kind"):
        module.authorize_seed_submission(
            campaign_path=root / "campaign.json",
            campaign_expected_sha256=draft["sha256"],
            acceptance_registry=root / "acceptances",
            seed=42,
        )


def test_production_campaign_publisher_consumes_full_h100_validator_and_receipt(
    tmp_path,
):
    module = _load()
    _matrix, attestation, receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    campaign = module.publish_campaign(**kwargs)
    assert len(module._test_environment_verification_calls) == 2
    query_state = module._test_repository_query_state
    assert (query_state / "repository-query-count").read_text() == "2"
    assert (query_state / "repository-query-marker").read_text() == "2"
    assert campaign["kind"] == "formal_campaign"
    validated = module.validate_campaign_manifest(campaign)
    evidence = validated["validation_evidence"]
    assert evidence["h100_submission_receipt_sha256"] == receipt["sha256"]
    assert evidence["environment_transaction_sha256"] == module.canonical_sha256(
        receipt["environment_transaction"]
    )
    assert (
        validated["h100_gate"]["driver_version"]
        == attestation["payload"]["driver_version"]
    )
    authorized = module.authorize_seed_submission(
        campaign_path=kwargs["output_path"],
        campaign_expected_sha256=campaign["sha256"],
        acceptance_registry=Path(kwargs["output_path"]).parent / "acceptances",
        seed=42,
    )
    assert authorized["campaign"]["sha256"] == campaign["sha256"]
    assert len(module._test_environment_verification_calls) == 3
    assert (query_state / "repository-query-count").read_text() == "3"
    assert (query_state / "repository-query-marker").read_text() == "3"


@pytest.mark.parametrize("control", ["repository-ref-moved", "repository-ref-missing"])
def test_production_publisher_requires_fresh_advertised_ref(tmp_path, control):
    module = _load()
    _matrix, _attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    query_state = module._test_repository_query_state
    (query_state / control).write_text("1")

    with pytest.raises(ValueError, match="advertise the exact candidate commit"):
        module.publish_campaign(**kwargs)
    assert (query_state / "repository-query-count").read_text() == "1"
    assert (query_state / "repository-query-marker").read_text() == "1"
    assert not Path(kwargs["output_path"]).exists()


@pytest.mark.parametrize("control", ["repository-ref-moved", "repository-ref-missing"])
def test_authorization_requeries_and_rejects_advertised_ref_drift(tmp_path, control):
    module = _load()
    _matrix, _attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    campaign = module.publish_campaign(**kwargs)
    query_state = module._test_repository_query_state
    assert (query_state / "repository-query-count").read_text() == "2"
    (query_state / control).write_text("1")

    with pytest.raises(ValueError, match="advertise the exact candidate commit"):
        module.authorize_seed_submission(
            campaign_path=kwargs["output_path"],
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=Path(kwargs["output_path"]).parent / "acceptances",
            seed=42,
        )
    assert (query_state / "repository-query-count").read_text() == "3"
    assert (query_state / "repository-query-marker").read_text() == "3"


def test_raw_environment_evidence_rebuilds_receipt_summary(tmp_path):
    module = _load()
    fixture = _raw_environment_fixture(tmp_path, module)

    rebuilt = module._revalidate_raw_environment_transaction(
        exact_root=tmp_path,
        completion_path=fixture["completion"],
        completion_max_bytes=module.ENVIRONMENT_COMPLETION_CEILING_BYTES,
        manifest_max_bytes=module.ENVIRONMENT_MANIFEST_CEILING_BYTES,
        inventory_max_bytes=module.ENVIRONMENT_INVENTORY_CEILING_BYTES,
        receipt=fixture["receipt"],
        source=fixture["source"],
    )

    assert rebuilt == fixture["receipt"]["environment_transaction"]


@pytest.mark.parametrize(
    "mutation",
    ["marker_deleted", "marker_replaced", "manifest_tampered", "inventory_tampered"],
)
def test_raw_environment_evidence_is_reopened_fail_closed(tmp_path, mutation):
    module = _load()
    fixture = _raw_environment_fixture(tmp_path, module)
    if mutation == "marker_deleted":
        fixture["completion"].unlink()
    elif mutation == "marker_replaced":
        fixture["completion"].write_bytes(b"{}\n")
    elif mutation == "manifest_tampered":
        fixture["one_gpu"].write_bytes(b"{}\n")
    else:
        fixture["inventory"].write_bytes(b"{}\n")

    with pytest.raises(ValueError):
        module._revalidate_raw_environment_transaction(
            exact_root=tmp_path,
            completion_path=fixture["completion"],
            completion_max_bytes=module.ENVIRONMENT_COMPLETION_CEILING_BYTES,
            manifest_max_bytes=module.ENVIRONMENT_MANIFEST_CEILING_BYTES,
            inventory_max_bytes=module.ENVIRONMENT_INVENTORY_CEILING_BYTES,
            receipt=fixture["receipt"],
            source=fixture["source"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "environment_completion_max_bytes",
        "environment_manifest_max_bytes",
        "environment_inventory_max_bytes",
    ],
)
def test_production_publisher_rejects_environment_ceiling_mismatch(tmp_path, field):
    module = _load()
    _matrix, _attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    kwargs[field] -= 1

    with pytest.raises(ValueError, match="fixed maximum"):
        module.publish_campaign(**kwargs)
    assert not Path(kwargs["output_path"]).exists()


def test_acceptance_path_revalidates_campaign_environment_before_seed_evidence(
    tmp_path,
):
    module = _load()
    _matrix, _attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    campaign = module.publish_campaign(**kwargs)
    calls_after_publication = len(module._test_environment_verification_calls)
    query_state = module._test_repository_query_state
    queries_after_publication = int(
        (query_state / "repository-query-count").read_text()
    )
    formal_root = tmp_path / "formal-seed-42"
    formal_root.mkdir()

    with pytest.raises(FileNotFoundError):
        module.publish_seed_acceptance(
            campaign_path=kwargs["output_path"],
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=Path(kwargs["output_path"]).parent / "acceptances",
            seed=42,
            formal_artifact_root=formal_root,
            terminal_attestation_path=formal_root / "terminal-scheduler-logs.json",
            submission_receipt_path=formal_root / "submission-receipt.json",
            transaction_ledger_path=formal_root / "transaction-ledger.json",
            terminal_metadata_max_bytes=module.FORMAL_METADATA_CEILING_BYTES,
            evaluation_receipt_path=(
                Path(kwargs["output_path"]).parent / "evaluations/seed-42.json"
            ),
        )
    assert len(module._test_environment_verification_calls) == (
        calls_after_publication + 1
    )
    assert int((query_state / "repository-query-count").read_text()) == (
        queries_after_publication + 1
    )
    assert (query_state / "repository-query-marker").read_text() == str(
        queries_after_publication + 1
    )


def test_acceptance_requeries_and_rejects_advertised_ref_drift(tmp_path):
    module = _load()
    _matrix, _attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    campaign = module.publish_campaign(**kwargs)
    query_state = module._test_repository_query_state
    (query_state / "repository-ref-moved").write_text("1")
    formal_root = tmp_path / "formal-seed-42"
    formal_root.mkdir()

    with pytest.raises(ValueError, match="advertise the exact candidate commit"):
        module.publish_seed_acceptance(
            campaign_path=kwargs["output_path"],
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=Path(kwargs["output_path"]).parent / "acceptances",
            seed=42,
            formal_artifact_root=formal_root,
            terminal_attestation_path=formal_root / "terminal-scheduler-logs.json",
            submission_receipt_path=formal_root / "submission-receipt.json",
            transaction_ledger_path=formal_root / "transaction-ledger.json",
            terminal_metadata_max_bytes=module.FORMAL_METADATA_CEILING_BYTES,
            evaluation_receipt_path=(
                Path(kwargs["output_path"]).parent / "evaluations/seed-42.json"
            ),
        )
    assert (query_state / "repository-query-count").read_text() == "3"
    assert (query_state / "repository-query-marker").read_text() == "3"


def test_production_publisher_rejects_minimal_self_hashed_smoke(tmp_path):
    module = _load()
    _matrix, attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    minimal = module.make_manifest(
        "h100_identity_smoke",
        {
            key: attestation["payload"][key]
            for key in (
                "commit_sha",
                "tree_sha",
                "source_manifest_sha256",
                "environment_sha256",
                "checkpoint_ceiling_bytes",
                "observed_checkpoint_max_bytes",
                "nvidia_smi_sha256",
                "gpu_model",
                "driver_version",
            )
        },
    )
    Path(kwargs["h100_attestation_path"]).write_bytes(
        module.canonical_json_bytes(minimal) + b"\n"
    )
    kwargs["expected_h100_sha256"] = minimal["sha256"]
    with pytest.raises(ValueError, match="smoke payload"):
        module.publish_campaign(**kwargs)
    assert not Path(kwargs["output_path"]).exists()


def test_production_publisher_rejects_rehashed_driver_drift(tmp_path):
    module = _load()
    matrix, attestation, _receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    payload = json.loads(json.dumps(attestation["payload"]))
    payload["driver_version"] = "999.0"
    changed = matrix.make_smoke_attestation(payload)
    Path(kwargs["h100_attestation_path"]).write_bytes(
        module.canonical_json_bytes(changed) + b"\n"
    )
    kwargs["expected_h100_sha256"] = changed["sha256"]
    with pytest.raises(ValueError, match="top-level driver"):
        module.publish_campaign(**kwargs)
    assert not Path(kwargs["output_path"]).exists()


def test_authorization_reopens_and_rejects_environment_transaction_drift(tmp_path):
    module = _load()
    _matrix, _attestation, receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    campaign = module.publish_campaign(**kwargs)
    receipt["environment_transaction"] = {
        **receipt["environment_transaction"],
        "inventory_sha256": "0" * 64,
    }
    with pytest.raises(
        ValueError,
        match="raw environment transaction differs|validation evidence changed",
    ):
        module.authorize_seed_submission(
            campaign_path=kwargs["output_path"],
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=Path(kwargs["output_path"]).parent / "acceptances",
            seed=42,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda _attestation, receipt, _kwargs: receipt[
                "environment_sha256_by_world_size"
            ].update({"1": "0" * 64}),
            "training environment differs",
        ),
        (
            lambda _attestation, receipt, _kwargs: receipt["repository_binding"].update(
                {"query_sha256": "0" * 64}
            ),
            "repository binding mismatch|repository query",
        ),
        (
            lambda _attestation, receipt, _kwargs: receipt[
                "artifact_identities"
            ].update({next(iter(receipt["artifact_identities"])): "0" * 64}),
            "artifact identity differs",
        ),
        (
            lambda _attestation, _receipt, kwargs: kwargs.update(
                checkpoint_ceiling_bytes=999
            ),
            "checkpoint.ceiling|checkpoint_ceiling",
        ),
        (
            lambda _attestation, _receipt, kwargs: kwargs["training_source"].update(
                commit_sha="f" * 40
            ),
            "commit.mismatch|commit_sha differs",
        ),
        (
            lambda _attestation, _receipt, kwargs: kwargs["training_source"].update(
                candidate_repository="https://example.invalid/forged.git"
            ),
            "repository is not canonical",
        ),
        (
            lambda _attestation, _receipt, kwargs: kwargs.update(
                expected_gpu_model="NVIDIA H100 forged"
            ),
            "GPU model mismatch|gpu_model differs",
        ),
    ],
)
def test_production_publisher_rejects_h100_receipt_or_candidate_tamper(
    tmp_path, mutation, match
):
    module = _load()
    _matrix, attestation, receipt, kwargs = _production_publish_fixture(
        tmp_path, module
    )
    mutation(attestation, receipt, kwargs)
    with pytest.raises(ValueError, match=match):
        module.publish_campaign(**kwargs)
    assert not Path(kwargs["output_path"]).exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda campaign: campaign["payload"]["training_source"].update(
                commit_sha="c" * 40
            ),
            "self-hash",
        ),
        (
            lambda campaign: campaign["payload"]["h100_gate"].update(
                checkpoint_ceiling_bytes=220_000_000
            ),
            "self-hash",
        ),
        (
            lambda campaign: campaign["payload"]["stages"][0].update(
                time_limit="13-00:00:00"
            ),
            "self-hash",
        ),
        (
            lambda campaign: campaign["payload"]["stages"][1].update(
                static_protocol_sha256="0" * 64
            ),
            "self-hash",
        ),
    ],
)
def test_campaign_rejects_unrehashable_source_gate_or_protocol_drift(mutate, match):
    module = _load()
    campaign = _campaign(module)
    mutate(campaign)
    with pytest.raises(ValueError, match=match):
        module.validate_campaign_manifest(campaign)


def test_campaign_rejects_rehashed_static_protocol_or_seed_set_drift():
    module = _load()
    campaign = _campaign(module)
    campaign["payload"]["stages"][0]["time_limit"] = "13-00:00:00"
    body = {key: campaign[key] for key in ("schema_version", "kind", "payload")}
    campaign["sha256"] = module.canonical_sha256(body)
    with pytest.raises(ValueError, match="static protocol"):
        module.validate_campaign_manifest(campaign)

    campaign = _campaign(module)
    campaign["payload"]["supported_seeds"] = [42, 44]
    body = {key: campaign[key] for key in ("schema_version", "kind", "payload")}
    campaign["sha256"] = module.canonical_sha256(body)
    with pytest.raises(ValueError, match="supported seeds"):
        module.validate_campaign_manifest(campaign)


def test_seed42_authorization_requires_campaign_and_empty_registry(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, _evaluations, campaign = _layout(
        tmp_path, module
    )
    result = module.authorize_seed_submission(
        campaign_path=campaign_path,
        campaign_expected_sha256=campaign["sha256"],
        acceptance_registry=acceptances,
        seed=42,
    )
    assert result["campaign_binding"]["predecessor_acceptance_sha256_by_seed"] == {}
    assert result["campaign_binding"]["training_commit_sha"] == "a" * 40


def test_seeds_43_and_44_require_exact_write_once_acceptance_prefix(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(tmp_path, module)
    with pytest.raises(ValueError, match="exact predecessor prefix"):
        module.authorize_seed_submission(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=43,
        )

    accepted42 = _accept(
        tmp_path,
        module,
        campaign_path,
        acceptances,
        evaluations,
        campaign,
        42,
        {},
    )
    seed43 = module.authorize_seed_submission(
        campaign_path=campaign_path,
        campaign_expected_sha256=campaign["sha256"],
        acceptance_registry=acceptances,
        seed=43,
    )
    assert seed43["campaign_binding"]["predecessor_acceptance_sha256_by_seed"] == {
        "42": accepted42["sha256"]
    }

    with pytest.raises(ValueError, match="exact predecessor prefix"):
        module.authorize_seed_submission(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=44,
        )
    accepted43 = _accept(
        tmp_path,
        module,
        campaign_path,
        acceptances,
        evaluations,
        campaign,
        43,
        {"42": accepted42["sha256"]},
    )
    seed44 = module.authorize_seed_submission(
        campaign_path=campaign_path,
        campaign_expected_sha256=campaign["sha256"],
        acceptance_registry=acceptances,
        seed=44,
    )
    assert seed44["campaign_binding"]["predecessor_acceptance_sha256_by_seed"] == {
        "42": accepted42["sha256"],
        "43": accepted43["sha256"],
    }


def test_acceptance_cannot_be_published_twice_or_out_of_order(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(tmp_path, module)
    terminal43, evaluation43, _t, _e = _publish_seed_evidence(
        tmp_path, module, campaign, evaluations, 43, {}
    )
    with pytest.raises(ValueError, match="out of order"):
        module.publish_seed_acceptance(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=43,
            terminal_attestation_path=terminal43,
            evaluation_receipt_path=evaluation43,
            **_terminal_raw_kwargs(terminal43, module),
        )

    accepted42 = _accept(
        tmp_path,
        module,
        campaign_path,
        acceptances,
        evaluations,
        campaign,
        42,
        {},
    )
    assert accepted42["payload"]["accepted"] is True
    terminal42 = Path(accepted42["payload"]["formal_terminal_attestation"]["path"])
    evaluation42 = Path(accepted42["payload"]["minimum_evaluation_sanity"]["path"])
    with pytest.raises(ValueError, match="out of order|overwrite"):
        module.publish_seed_acceptance(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=42,
            terminal_attestation_path=terminal42,
            evaluation_receipt_path=evaluation42,
            **_terminal_raw_kwargs(terminal42, module),
        )


def test_registry_rejects_unknown_or_future_acceptance_files(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, _evaluations, campaign = _layout(
        tmp_path, module
    )
    (acceptances / "README").write_text("not evidence\n")
    with pytest.raises(ValueError, match="exact predecessor prefix"):
        module.authorize_seed_submission(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=42,
        )


def test_registry_reopens_and_rejects_replaced_terminal_evidence(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(tmp_path, module)
    accepted = _accept(
        tmp_path,
        module,
        campaign_path,
        acceptances,
        evaluations,
        campaign,
        42,
        {},
    )
    terminal_path = Path(accepted["payload"]["formal_terminal_attestation"]["path"])
    os.unlink(terminal_path)
    changed = _terminal(module, campaign, 42, {})
    changed["payload"]["terminal_verified"] = False
    body = {key: changed[key] for key in ("schema_version", "kind", "payload")}
    changed["sha256"] = module.canonical_sha256(body)
    module.publish_manifest_no_replace(
        terminal_path,
        changed,
        max_bytes=module.TERMINAL_ATTESTATION_CEILING_BYTES,
    )
    with pytest.raises(ValueError, match="externally committed digest"):
        module.authorize_seed_submission(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=43,
        )


def test_registry_rejects_symlinked_evidence_and_parent_components(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(tmp_path, module)
    terminal_path, evaluation_path, _terminal_value, _evaluation_value = (
        _publish_seed_evidence(tmp_path, module, campaign, evaluations, 42, {})
    )
    real_terminal = terminal_path.with_name("real-terminal.json")
    terminal_path.rename(real_terminal)
    terminal_path.symlink_to(real_terminal)
    with pytest.raises((OSError, ValueError)):
        module.publish_seed_acceptance(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=42,
            terminal_attestation_path=terminal_path,
            evaluation_receipt_path=evaluation_path,
            **_terminal_raw_kwargs(terminal_path, module),
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda receipt: receipt["payload"]["evaluation_source"].update(
                commit_sha="a" * 40
            ),
            "distinct attested descendant",
        ),
        (
            lambda receipt: receipt["payload"]["evaluation_source"].update(
                ancestry_verified=False
            ),
            "distinct attested descendant",
        ),
        (
            lambda receipt: receipt["payload"]["results_by_arm"]["rope"].update(
                evaluated_examples=31
            ),
            "not matched",
        ),
        (
            lambda receipt: receipt["payload"]["results_by_arm"]["none"][
                "metrics"
            ].update(accuracy=True),
            "non-finite or non-numeric",
        ),
        (
            lambda receipt: receipt["payload"]["acceptance"].update(
                minimum_sanity_passed=False
            ),
            "flags",
        ),
    ],
)
def test_evaluation_sanity_schema_rejects_lineage_mismatch_or_bad_results(
    mutate, match
):
    module = _load()
    campaign = _campaign(module)
    terminal = _terminal(module, campaign, 42, {})
    receipt = _evaluation(module, campaign, terminal, 42, {})
    mutate(receipt)
    body = {key: receipt[key] for key in ("schema_version", "kind", "payload")}
    receipt["sha256"] = module.canonical_sha256(body)
    with pytest.raises(ValueError, match=match):
        module.validate_evaluation_sanity_receipt(
            receipt,
            campaign=campaign,
            seed=42,
            terminal_attestation_sha256=terminal["sha256"],
            expected_campaign_binding=module.campaign_binding(
                campaign, predecessor_acceptance_sha256_by_seed={}
            ),
        )


def test_terminal_failure_or_wrong_campaign_binding_is_not_acceptable(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(tmp_path, module)
    terminal = _terminal(module, campaign, 42, {})
    terminal["payload"]["jobs"][0]["state"] = "FAILED"
    body = {key: terminal[key] for key in ("schema_version", "kind", "payload")}
    terminal["sha256"] = module.canonical_sha256(body)
    terminal_root = tmp_path / "failed-formal"
    terminal_root.mkdir()
    terminal_path = terminal_root / "terminal-scheduler-logs.json"
    module.publish_manifest_no_replace(
        terminal_path,
        terminal,
        max_bytes=module.TERMINAL_ATTESTATION_CEILING_BYTES,
    )
    evaluation = _evaluation(module, campaign, terminal, 42, {})
    evaluation_path = evaluations / "seed-42.json"
    module.publish_manifest_no_replace(
        evaluation_path,
        evaluation,
        max_bytes=module.EVALUATION_RECEIPT_CEILING_BYTES,
    )
    with pytest.raises(ValueError, match="not independently successful"):
        module.publish_seed_acceptance(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=42,
            terminal_attestation_path=terminal_path,
            evaluation_receipt_path=evaluation_path,
            **_terminal_raw_kwargs(terminal_path, module),
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["payload"].update(transaction_id="bad"),
            "campaign binding mismatch",
        ),
        (
            lambda value: value["payload"]["repository_binding"].update(
                query_sha256="0" * 64
            ),
            "repository binding mismatch",
        ),
        (
            lambda value: value["payload"].update(terminal_queries=[]),
            "query set is incomplete",
        ),
        (
            lambda value: value["payload"]["terminal_queries"][0].update(
                job_ids=list(
                    reversed(value["payload"]["terminal_queries"][0]["job_ids"])
                )
            ),
            "query binding mismatch",
        ),
        (
            lambda value: value["payload"]["terminal_queries"][0].update(
                query_argv_sha256="0" * 64
            ),
            "query binding mismatch",
        ),
        (
            lambda value: value["payload"]["terminal_queries"][0].update(
                stdout_sha256="bad"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["payload"]["jobs"][0].update(unexpected=True),
            "job schema mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][1].update(
                job_id=value["payload"]["jobs"][0]["job_id"]
            ),
            "scheduler identity",
        ),
        (
            lambda value: value["payload"]["jobs"][0].update(job_name="wrong"),
            "scheduler identity",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"].update(
                path="runtime-completions/wrong.json"
            ),
            "completion path",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"].update(
                manifest_sha256="bad"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"]["scheduler"].update(
                time_limit="00:01:00"
            ),
            "scheduler binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"][
                "repository_binding"
            ].update(query_sha256="0" * 64),
            "completion binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"].update(
                gpu_name="NVIDIA A100"
            ),
            "completion binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["completion"].update(
                protocol_metadata_allowance_bytes=1
            ),
            "completion binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["scheduler_logs"][0].update(
                path="scheduler-logs/wrong.out"
            ),
            "scheduler log binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["scheduler_logs"][0].update(
                sha256="bad"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["scheduler_logs"][0].update(
                stable_reads=1
            ),
            "scheduler log binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["finalized_artifact"][
                "binding"
            ].update(checkpoint_path="arms/rope/stage1/wrong.ckpt"),
            "artifact path",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["finalized_artifact"][
                "binding"
            ].update(seed_sha256="0" * 64),
            "artifact binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["finalized_artifact"][
                "binding"
            ].update(treatment_sha256="0" * 64),
            "artifact binding mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["finalized_artifact"][
                "binding"
            ].update(checkpoint_size=300_000_001),
            "immutable ceiling",
        ),
        (
            lambda value: value["payload"]["jobs"][0]["finalized_artifact"].update(
                publication_validation_sha256="0" * 64
            ),
            "artifact validation mismatch",
        ),
        (
            lambda value: value["payload"]["jobs"][3]["parent_lineage"].update(
                stage="stage3"
            ),
            "parent lineage",
        ),
        (
            lambda value: value["payload"]["jobs"].__setitem__(
                slice(0, 2), list(reversed(value["payload"]["jobs"][:2]))
            ),
            "matrix/order",
        ),
        (
            lambda value: value["payload"].update(submission_receipt_sha256="bad"),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["payload"].update(sacct_sha256="bad"),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["payload"].update(
                protocol_metadata_allowance_bytes=(128 << 20) + 1
            ),
            "evidence ceiling is unbounded",
        ),
    ],
    ids=[
        "transaction",
        "repository",
        "query-set",
        "query-jobs",
        "query-argv",
        "query-digest",
        "job-schema",
        "job-id",
        "job-name",
        "completion-path",
        "completion-digest",
        "completion-scheduler",
        "completion-repository",
        "completion-gpu",
        "completion-metadata-allowance",
        "log-path",
        "log-digest",
        "log-stability",
        "artifact-path",
        "artifact-seed",
        "artifact-treatment",
        "artifact-ceiling",
        "artifact-validation",
        "parent-lineage",
        "job-order",
        "receipt-digest",
        "sacct-digest",
        "metadata-allowance-ceiling",
    ],
)
def test_terminal_acceptance_rejects_exact_schema_or_binding_tampering(mutate, match):
    module = _load()
    campaign = _campaign(module)
    binding = module.campaign_binding(
        campaign, predecessor_acceptance_sha256_by_seed={}
    )
    terminal = _terminal(module, campaign, 42, {})
    mutate(terminal)
    _rehash(module, terminal)
    with pytest.raises(ValueError, match=match):
        module._validate_terminal_attestation(
            terminal,
            campaign=campaign,
            seed=42,
            expected_campaign_binding=binding,
        )


def test_canonical_reader_rejects_duplicate_keys_noncanonical_json_and_hardlinks(
    tmp_path,
):
    module = _load()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"kind":"formal_campaign","kind":"formal_campaign","payload":{},'
        '"schema_version":1,"sha256":"' + "0" * 64 + '"}\n'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        module.read_canonical_manifest(
            duplicate, max_bytes=1000, expected_kind="formal_campaign"
        )

    campaign = _campaign(module)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(campaign, indent=2) + "\n")
    with pytest.raises(ValueError, match="not canonical"):
        module.read_canonical_manifest(
            noncanonical, max_bytes=100_000, expected_kind="formal_campaign"
        )

    canonical = tmp_path / "canonical.json"
    module.publish_manifest_no_replace(canonical, campaign, max_bytes=100_000)
    os.link(canonical, tmp_path / "hardlink.json")
    with pytest.raises(ValueError, match="singly-linked"):
        module.read_canonical_manifest(
            canonical, max_bytes=100_000, expected_kind="formal_campaign"
        )


def test_write_once_publication_never_replaces_existing_destination(tmp_path):
    module = _load()
    campaign = _campaign(module)
    path = tmp_path / "campaign.json"
    module.publish_manifest_no_replace(path, campaign, max_bytes=100_000)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.publish_manifest_no_replace(path, campaign, max_bytes=100_000)
    assert path.read_bytes() == before


def test_campaign_registry_capacity_charges_fragments_files_dirs_and_temp_dirents():
    module = _load()
    report = module.campaign_registry_capacity(
        fragment_size=4096,
        campaign_manifest_ceiling_bytes=4097,
        acceptance_ceiling_bytes=1,
        evaluation_receipt_ceiling_bytes=4096,
    )
    assert report["durable_file_slots"] == 7
    assert report["directory_objects"] == 3
    assert report["directory_entry_slots"] == 17
    assert report["atomic_temporary_entry_slots"] == 7
    assert report["physical_file_bytes"] == 8_192 + 3 * 4_096 + 3 * 4_096
    assert report["physical_directory_bytes"] == 20 * 4_096
    assert report["required_physical_bytes"] == (
        report["physical_file_bytes"] + report["physical_directory_bytes"]
    )
