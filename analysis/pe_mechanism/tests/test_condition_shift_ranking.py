from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pe_mechanism.representation as representation_module
import pytest
import torch
from torch import nn

from pe_mechanism.condition_shift_ranking import (
    RankingParameters,
    rank_condition_shift,
    run,
)
from pe_mechanism.causal import raw_space_decoder_feature_norms
from pe_mechanism.representation import (
    MeanRMSNormalizer,
    PCARepresentation,
    run as run_representation,
)
from test_representation import (
    _digest,
    _mutate_activation_and_reseal,
    _provenance,
    _strict_collect_run,
    _strict_lineage_config,
)


class _IdentityRepresentation(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(dimension, dimension, bias=False)
        self.decoder = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(dimension))
            self.decoder.weight.copy_(torch.eye(dimension))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


def _pure_activations() -> dict[str, dict[str, torch.Tensor]]:
    shifts = torch.arange(16.0, 0.0, -1.0)
    return {
        "none": {
            name: torch.zeros(rows, 16)
            for name, rows in (("rank-a", 3), ("rank-b", 11), ("rank-c", 29))
        },
        "rope": {
            name: shifts.mul(scale).repeat(rows, 1)
            for (name, rows), scale in zip(
                (("rank-a", 3), ("rank-b", 11), ("rank-c", 29)),
                (1.0, 2.0, 4.0),
                strict=True,
            )
        },
    }


def test_rank_condition_shift_is_equal_dataset_deterministic_and_top_four() -> None:
    parameters = RankingParameters(
        dataset_ids=("rank-a", "rank-b", "rank-c"),
        minimum_nonzero_datasets=3,
        target_count=4,
        candidate_pool_size=8,
        random_seed=42,
    )
    normalizer = MeanRMSNormalizer(torch.zeros(16), torch.ones(16))

    first = rank_condition_shift(
        _pure_activations(),
        model=_IdentityRepresentation(16),
        normalizer=normalizer,
        parameters=parameters,
    )
    second = rank_condition_shift(
        _pure_activations(),
        model=_IdentityRepresentation(16),
        normalizer=normalizer,
        parameters=parameters,
    )

    assert first.target_features == (0, 1, 2, 3)
    assert first.control_features == second.control_features
    assert len(set(first.control_features)) == 4
    assert not set(first.target_features) & set(first.control_features)
    np.testing.assert_array_equal(first.median_scores, second.median_scores)
    np.testing.assert_array_equal(first.nonzero_dataset_counts, np.full(16, 3))
    assert set(first.dataset_score_sha256) == {"rank-a", "rank-b", "rank-c"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_condition", "exactly none and rope"),
        ("mismatched_roster", "activation roster"),
        ("mismatched_shape", "activation shape differs"),
        ("nonfinite", "activations must be finite"),
    ],
)
def test_rank_condition_shift_rejects_unpaired_inputs(
    mutation: str, message: str
) -> None:
    activations = _pure_activations()
    if mutation == "missing_condition":
        activations.pop("rope")
    elif mutation == "mismatched_roster":
        activations["rope"].pop("rank-c")
    elif mutation == "mismatched_shape":
        activations["rope"]["rank-a"] = torch.ones(2, 16)
    else:
        activations["rope"]["rank-a"][0, 0] = float("nan")
    parameters = RankingParameters(
        dataset_ids=("rank-a", "rank-b", "rank-c"),
        minimum_nonzero_datasets=1,
        target_count=4,
        candidate_pool_size=8,
        random_seed=42,
    )

    with pytest.raises(ValueError, match=message):
        rank_condition_shift(
            activations,
            model=_IdentityRepresentation(16),
            normalizer=MeanRMSNormalizer(torch.zeros(16), torch.ones(16)),
            parameters=parameters,
        )


def test_rank_condition_shift_refuses_silently_truncated_control_pool() -> None:
    parameters = RankingParameters(
        dataset_ids=("rank-a", "rank-b", "rank-c"),
        minimum_nonzero_datasets=1,
        target_count=4,
        candidate_pool_size=10,
        random_seed=42,
    )
    with pytest.raises(ValueError, match="would be truncated"):
        rank_condition_shift(
            _pure_activations(),
            model=_IdentityRepresentation(16),
            normalizer=MeanRMSNormalizer(torch.zeros(16), torch.ones(16)),
            parameters=parameters,
        )


def test_rank_condition_shift_uses_denormalized_raw_space_decoder_norms() -> None:
    parameters = RankingParameters(
        dataset_ids=("rank-a", "rank-b", "rank-c"),
        minimum_nonzero_datasets=3,
        target_count=4,
        candidate_pool_size=8,
        random_seed=42,
    )
    raw_scales = torch.arange(1.0, 17.0)

    ranking = rank_condition_shift(
        _pure_activations(),
        model=_IdentityRepresentation(16),
        normalizer=MeanRMSNormalizer(torch.zeros(16), raw_scales),
        parameters=parameters,
    )

    np.testing.assert_allclose(ranking.decoder_norms, raw_scales.numpy())
    assert ranking.target_features == (0, 1, 2, 3)


def test_pca_decoder_norms_are_also_measured_in_raw_space() -> None:
    model = PCARepresentation(4, 4)
    with torch.no_grad():
        model.components.copy_(torch.eye(4))
        model.is_fitted.fill_(True)
    raw_scales = torch.tensor([0.5, 2.0, 3.0, 7.0])
    normalizer = MeanRMSNormalizer(torch.zeros(4), raw_scales)

    observed = raw_space_decoder_feature_norms(model, normalizer)

    torch.testing.assert_close(observed, raw_scales.to(torch.float64))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_protocol(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "protocol_id": "unit-whole-row-causal-split-v1",
        "feature_ranking_datasets": ["train-a"],
        "causal_test_datasets": ["unused-causal-test"],
        "feature_protocol": {
            "target_count": 1,
            "score": (
                "median across feature-ranking datasets of RMS RoPE-minus-No-PE "
                "latent difference times decoder-direction norm divided by raw "
                "activation RMS"
            ),
            "minimum_nonzero_ranking_datasets": 1,
            "matched_control": (
                "activation-frequency and log-decoder-norm nearest-neighbour pool "
                "of size 1, sampled once with seed 42 without replacement"
            ),
            "same_features_both_directions": True,
        },
        "causal_protocol": {
            "directions": ["rope_to_none", "none_to_rope"],
            "maximum_symmetric_donor_shift_rms_ratio": 1.25,
            "checkpoint_scope": "exploratory_pilot",
            "formal_claim": "forbidden",
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _dose_protocol(path: Path, split_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "protocol_id": "tabicl-step250k-whole-row-causal-dose-amendment-v1",
        "frozen_at": "2026-08-09T22:36:19+01:00",
        "parent_split_protocol_id": "unit-whole-row-causal-split-v1",
        "parent_split_protocol_sha256": _file_sha256(split_path),
        "scope": "exploratory_pilot_ranking_bound_paired_row_interactor_only",
        "formal_claim": "forbidden",
        "trigger": (
            "A decoded-dose balance gate failed before any target, control, "
            "or donor intervention prediction was published."
        ),
        "model_outcomes_observed_before_freeze": False,
        "matched_control_dose": (
            "per_call_decoded_rms_clip_to_smaller_without_amplification"
        ),
        "reference_activation": "recipient_no_op_reconstruction",
        "matching_unit": "official_raw_model_call",
        "dose_metric": (
            "root_mean_square_of_decoded_activation_edit_after_cast_to_live_"
            "activation_dtype_minus_recipient_no_op_reconstruction_after_same_cast"
        ),
        "adjustment": (
            "Set the common requested dose to the smaller full-edit dose, retain "
            "the smaller latent edit and decoded activation byte-for-byte, shrink "
            "only the larger latent edit by their ratio, decode the changed edit "
            "again, cast both edit and no-op reconstruction to the live activation "
            "dtype, reject any actual per-side dose increase, and gate the actual "
            "injected values."
        ),
        "amplification_allowed": False,
        "zero_or_non_finite_dose_policy": "fail_closed",
        "maximum_post_adjustment_symmetric_rms_ratio": 1.25,
        "interpretation": (
            "The executed interventions are dose-matched partial edits, not "
            "complete deletion or complete transplantation. Target ablation and "
            "donor patch families are matched only within their own target/control "
            "pair and are not dose-comparable to each other."
        ),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _completed_parent_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, tuple[Path, dict[str, str]]]]:
    representation_config, runs = _strict_lineage_config(
        tmp_path,
        monkeypatch,
        site="row_interactor",
        extra_dataset_assignments=(("unused-causal-test", "discovery"),),
    )
    representation_config["model"] = {
        "type": "pca",
        "latent_dim": 4,
        "solver": "full_svd",
    }
    representation_config["training"] = {
        "epochs": 1,
        "batch_size": 8,
        "seed": 42,
        "device": "cpu",
    }
    representation_path = tmp_path / "representation-config.json"
    representation_path.write_text(
        json.dumps(representation_config), encoding="utf-8"
    )
    parent_dir = tmp_path / "completed-representation"
    assert run_representation(
        SimpleNamespace(config=representation_path, output_dir=parent_dir)
    ) == 0
    split_path = _split_protocol(tmp_path / "split-protocol.json")
    dose_path = _dose_protocol(tmp_path / "dose-protocol.json", split_path)
    rank_config: dict[str, Any] = {
        "representation_run_dir": str(parent_dir),
        "expected_parent_manifest_sha256": _file_sha256(
            parent_dir / "manifest.json"
        ),
        "activation_sources_by_condition": copy.deepcopy(
            representation_config["training_by_condition"]
        ),
        "split_protocol": {
            "path": str(split_path),
            "expected_sha256": _file_sha256(split_path),
        },
        "dose_protocol": {
            "path": str(dose_path),
            "expected_sha256": _file_sha256(dose_path),
        },
        "ranking": {
            "dataset_ids": ["train-a"],
            "minimum_nonzero_datasets": 1,
            "target_count": 1,
            "random_candidate_pool_size": 1,
            "random_seed": 42,
        },
        "provenance": copy.deepcopy(representation_config["provenance"]),
    }
    return parent_dir, rank_config, runs


def _write_rank_config(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "rank-config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _assert_path_free(value: Any) -> None:
    if isinstance(value, str):
        assert "/" not in value
        assert "\\" not in value
    elif isinstance(value, list):
        for item in value:
            _assert_path_free(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_path_free(key)
            _assert_path_free(item)


def test_parent_bound_source_loader_does_not_open_unselected_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = _provenance(tmp_path, monkeypatch)
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    run_dir, artifacts = _strict_collect_run(
        tmp_path,
        provenance,
        condition="none",
        train_values=values,
        validation_values=values + 1.0,
    )
    unselected = (run_dir / artifacts["valid-a"]).resolve()
    original_verify = representation_module.verify_file

    def guarded_verify(path: Any, *, expected_sha256: str | None = None):
        if Path(path).resolve() == unselected:
            raise AssertionError("unselected activation shard was opened")
        return original_verify(path, expected_sha256=expected_sha256)

    monkeypatch.setattr(representation_module, "verify_file", guarded_verify)
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    collect_runs: dict[Path, Any] = {}
    specifications = representation_module._condition_input_specs(
        {
            "none": {
                "train-a": {
                    "run_dir": str(run_dir),
                    "artifact": artifacts["train-a"],
                    "key": "activations",
                }
            }
        },
        split="training",
        paths=paths,
        expected_hashes=expected,
        collect_runs=collect_runs,
        verify_complete_runs=False,
    )

    assert set(specifications["none"]) == {"train-a"}
    assert unselected not in paths.values()


def test_run_publishes_bound_path_free_selection_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    config_path = _write_rank_config(tmp_path, config)
    first_output = tmp_path / "rank-output-first"
    second_output = tmp_path / "rank-output-second"

    assert run(SimpleNamespace(config=config_path, output_dir=first_output)) == 0
    assert run(SimpleNamespace(config=config_path, output_dir=second_output)) == 0
    assert {path.name for path in first_output.iterdir()} == {
        "manifest.json",
        "selection.json",
    }
    assert (first_output / "selection.json").read_bytes() == (
        second_output / "selection.json"
    ).read_bytes()

    manifest = json.loads((first_output / "manifest.json").read_text())
    selection = json.loads((first_output / "selection.json").read_text())
    assert manifest["command"] == "rank-condition-shift"
    assert manifest["evidence_level"] == "strict"
    assert selection["target_features"]
    assert selection["control_features"]
    assert not set(selection["target_features"]) & set(selection["control_features"])
    roles = {item["role"] for item in manifest["inputs"]}
    assert {
        "ranking.split_protocol",
        "ranking.dose_protocol",
        "representation.parent_manifest",
        "representation.model",
    } <= roles
    assert any(role.startswith("source.activation.") for role in roles)
    assert set(selection["activation_sha256_by_condition"]) == {"none", "rope"}
    assert selection["decoder_norm_space"] == "raw_activation_after_denormalize"
    assert selection["maximum_symmetric_donor_shift_rms_ratio"] == 1.25
    assert selection["matched_control_dose"] == (
        "per_call_decoded_rms_clip_to_smaller_without_amplification"
    )
    assert selection["dose_protocol_sha256"] == _file_sha256(
        Path(config["dose_protocol"]["path"])
    )
    _assert_path_free(selection)

    with pytest.raises(FileExistsError):
        run(SimpleNamespace(config=config_path, output_dir=first_output))


def test_run_rejects_wrong_parent_digest_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    config["expected_parent_manifest_sha256"] = "0" * 64
    output = tmp_path / "rank-output"
    with pytest.raises(ValueError, match="expected SHA-256"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_source_not_bound_by_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, runs = _completed_parent_and_config(tmp_path, monkeypatch)
    rope_run, rope_artifacts = runs["rope"]
    _mutate_activation_and_reseal(
        rope_run,
        rope_artifacts["train-a"],
        lambda arrays: arrays["activations"].__iadd__(1.0),
    )
    output = tmp_path / "rank-output"
    with pytest.raises(ValueError, match="bound by train-repr"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_split_protocol_parameter_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    config["ranking"]["random_candidate_pool_size"] = 2
    output = tmp_path / "rank-output"
    with pytest.raises(ValueError, match="frozen feature protocol"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amplification_allowed", True),
        ("protocol_id", "renamed-dose-amendment"),
    ],
)
def test_run_rejects_dose_protocol_drift_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    dose_path = Path(config["dose_protocol"]["path"])
    dose = json.loads(dose_path.read_text(encoding="utf-8"))
    dose[field] = value
    dose_path.write_text(json.dumps(dose, sort_keys=True), encoding="utf-8")
    config["dose_protocol"]["expected_sha256"] = _file_sha256(dose_path)
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="frozen contract"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_dose_protocol_schema_extension_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    dose_path = Path(config["dose_protocol"]["path"])
    dose = json.loads(dose_path.read_text(encoding="utf-8"))
    dose["unregistered_override"] = True
    dose_path.write_text(json.dumps(dose, sort_keys=True), encoding="utf-8")
    config["dose_protocol"]["expected_sha256"] = _file_sha256(dose_path)
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="dose protocol fields mismatch"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_resealed_dose_metric_drift_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    dose_path = Path(config["dose_protocol"]["path"])
    dose = json.loads(dose_path.read_text(encoding="utf-8"))
    dose["dose_metric"] = "a different but non-empty metric"
    dose_path.write_text(json.dumps(dose, sort_keys=True), encoding="utf-8")
    config["dose_protocol"]["expected_sha256"] = _file_sha256(dose_path)
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="frozen contract"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_ranking_and_causal_test_roster_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    split_path = Path(config["split_protocol"]["path"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["causal_test_datasets"] = ["train-a"]
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    config["split_protocol"]["expected_sha256"] = _file_sha256(split_path)
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="must be disjoint"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_undeclared_causal_test_source_before_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    forbidden = {
        "run_dir": str(tmp_path / "must-not-be-read"),
        "artifact": "forbidden.npz",
        "key": "activations",
    }
    for condition in ("none", "rope"):
        config["activation_sources_by_condition"][condition][
            "unused-causal-test"
        ] = forbidden
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="declared ranking dataset roster"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not (tmp_path / "must-not-be-read").exists()
    assert not output.exists()


def test_run_rejects_parent_model_digest_or_size_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    with (parent / "model.pt").open("ab") as stream:
        stream.write(b"tamper")
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="artifact hash or size mismatch"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()


def test_run_rejects_unqualified_representation_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, config, _runs = _completed_parent_and_config(tmp_path, monkeypatch)
    model_path = parent / "model.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    payload["metrics"]["explained_variance"] = 0.5
    torch.save(payload, model_path)
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _digest(model_path)
    model_artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "model.pt"
    )
    model_artifact["sha256"] = digest.sha256
    model_artifact["size_bytes"] = digest.size_bytes
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config["expected_parent_manifest_sha256"] = _file_sha256(manifest_path)
    output = tmp_path / "rank-output"

    with pytest.raises(ValueError, match="representation qualification"):
        run(
            SimpleNamespace(
                config=_write_rank_config(tmp_path, config), output_dir=output
            )
        )
    assert not output.exists()
