from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "formal_campaign_registry.py"
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


def _source() -> dict:
    return {
        "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
        "candidate_ref": "refs/heads/codex/position-identity-v1",
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
        "source_manifest_sha256": "1" * 64,
        "environment_sha256": "2" * 64,
    }


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
        "gpu_model": "NVIDIA H100 80GB HBM3",
    }
    smoke = module.make_manifest("h100_identity_smoke", smoke_payload)
    campaign = module.build_campaign_manifest(
        campaign_id="position-identity-v1",
        training_source=_source(),
        h100_attestation=smoke,
        expected_h100_sha256=smoke["sha256"],
        checkpoint_ceiling_bytes=300_000_000,
        expected_gpu_model="NVIDIA H100 80GB HBM3",
        stages=_stages(),
    )
    return campaign


def _layout(tmp_path: Path, module):
    root = tmp_path / "campaign"
    acceptances = root / "acceptances"
    evaluations = root / "evaluations"
    acceptances.mkdir(parents=True)
    evaluations.mkdir()
    campaign = _campaign(module)
    campaign_path = root / "campaign.json"
    module.publish_manifest_no_replace(
        campaign_path,
        campaign,
        max_bytes=module.CAMPAIGN_MANIFEST_CEILING_BYTES,
    )
    return root, campaign_path, acceptances, evaluations, campaign


def _terminal(module, campaign, seed: int, predecessor_map: dict[str, str]):
    binding = module.campaign_binding(
        campaign,
        predecessor_acceptance_sha256_by_seed=predecessor_map,
    )
    repository_binding = {"identity": "test"}
    jobs = []
    for stage, _terminal_step in STAGES:
        for arm in ARMS:
            jobs.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "derived_exit_code": "0:0",
                    "finalized_artifact": {"checkpoint_sha256": "7" * 64},
                    "completion": {
                        "seed": seed,
                        "repository_binding": repository_binding,
                    },
                }
            )
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
            "manifest_ceiling_bytes": 2_000_000,
            "sacct_sha256": "8" * 64,
            "path_format": "artifact_root_relative_posix_v1",
            "terminal_verified": True,
            "terminal_queries": [],
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
    )


def test_campaign_freezes_seed_independent_static_protocol_and_h100_contract():
    module = _load()
    campaign = _campaign(module)
    validated = module.validate_campaign_manifest(campaign)
    assert validated["training_source"] == _source()
    assert validated["h100_gate"] == {
        "attestation_sha256": campaign["payload"]["h100_gate"][
            "attestation_sha256"
        ],
        "checkpoint_ceiling_bytes": 300_000_000,
        "observed_checkpoint_max_bytes": 220_700_000,
        "gpu_model": "NVIDIA H100 80GB HBM3",
    }
    assert [item["terminal_step"] for item in validated["stages"]] == [
        500_000,
        40_000,
        10_000,
    ]
    assert all("seed" not in item for item in validated["stages"])


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
    assert result["campaign_binding"][
        "predecessor_acceptance_sha256_by_seed"
    ] == {}
    assert result["campaign_binding"]["training_commit_sha"] == "a" * 40


def test_seeds_43_and_44_require_exact_write_once_acceptance_prefix(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(
        tmp_path, module
    )
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
    assert seed43["campaign_binding"][
        "predecessor_acceptance_sha256_by_seed"
    ] == {"42": accepted42["sha256"]}

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
    assert seed44["campaign_binding"][
        "predecessor_acceptance_sha256_by_seed"
    ] == {"42": accepted42["sha256"], "43": accepted43["sha256"]}


def test_acceptance_cannot_be_published_twice_or_out_of_order(tmp_path):
    module = _load()
    _root, campaign_path, acceptances, evaluations, campaign = _layout(
        tmp_path, module
    )
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
    terminal42 = (
        Path(accepted42["payload"]["formal_terminal_attestation"]["path"])
    )
    evaluation42 = (
        Path(accepted42["payload"]["minimum_evaluation_sanity"]["path"])
    )
    with pytest.raises(ValueError, match="out of order|overwrite"):
        module.publish_seed_acceptance(
            campaign_path=campaign_path,
            campaign_expected_sha256=campaign["sha256"],
            acceptance_registry=acceptances,
            seed=42,
            terminal_attestation_path=terminal42,
            evaluation_receipt_path=evaluation42,
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
    _root, campaign_path, acceptances, evaluations, campaign = _layout(
        tmp_path, module
    )
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
    terminal_path = Path(
        accepted["payload"]["formal_terminal_attestation"]["path"]
    )
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
    _root, campaign_path, acceptances, evaluations, campaign = _layout(
        tmp_path, module
    )
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
    _root, campaign_path, acceptances, evaluations, campaign = _layout(
        tmp_path, module
    )
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
