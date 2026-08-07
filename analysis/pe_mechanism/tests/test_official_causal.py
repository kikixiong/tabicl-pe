from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import pe_mechanism.official_causal as official_causal
from pe_mechanism.adapters.tabicl import TabICLAdapter
from pe_mechanism.causal import model_decoder_feature_norms
from pe_mechanism.manifest import (
    ArtifactDigest,
    FileDigest,
    InputDigest,
    new_manifest,
)
from pe_mechanism.official_causal import run_official_tabicl_causal_edits
from pe_mechanism.official_tabicl import (
    OfficialTabICLDriver,
    official_inference_contract_sha256,
)
from pe_mechanism.provenance import (
    VerifiedRunContext,
    verify_file,
    verify_git_tree,
    verify_run_directory,
    verify_run_inputs,
)
from pe_mechanism.representation import (
    DenseAutoencoder,
    MeanRMSNormalizer,
    PCARepresentation,
    TopKSparseAutoencoder,
)
from test_official_tabicl import FakeOfficialClassifier, make_driver


class ExactOvercompleteAutoencoder(nn.Module):
    input_dim = 1
    latent_dim = 3

    def __init__(self, *, negate_decode: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 3, bias=False)
        self.decoder = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[1.0], [0.5], [-0.25]]))
            sign = -1.0 if negate_decode else 1.0
            self.decoder.weight.copy_(torch.tensor([[sign, 0.0, 0.0]]))

    def encode(self, values):
        return self.encoder(values)

    def decode(self, latents):
        return self.decoder(latents)


@pytest.fixture
def causal_inputs():
    X = np.array(
        [[0.1, 0.6, -0.3, 0.8], [0.9, -0.2, 0.4, 0.3]],
        dtype=np.float32,
    )
    y = np.array([0, 1])
    return X, y


def qualification(passed=True):
    return {
        "metric_split": "validation",
        "activation_fidelity_passed": passed,
        "held_out_explained_variance": 0.99,
        "model_sha": "a" * 40,
        "checkpoint_sha": "b" * 64,
        "site": "row_interactor",
    }


def test_official_causal_runner_preserves_ensemble_and_rng(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    driver = make_driver(classifier)
    generator = classifier.model_.row_interactor._identity_generator
    entry_state = generator.get_state().clone()
    direct = classifier.predict_proba(X.copy())
    expected_final_state = generator.get_state().clone()
    generator.set_state(entry_state)

    evaluation = run_official_tabicl_causal_edits(
        driver,
        X,
        y,
        site="row_interactor",
        autoencoder=ExactOvercompleteAutoencoder(),
        normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
        target_features=(0,),
        dataset_id="toy-official",
        sample_ids=("row-0", "row-1"),
        representation_qualification=qualification(),
        random_candidate_pool_size=2,
    )

    assert evaluation.capture_exact_to_direct
    assert np.array_equal(evaluation.native_prediction.probabilities, direct)
    assert torch.equal(generator.get_state(), expected_final_state)
    assert [
        call.metadata.raw_input_shape[0]
        for call in evaluation.native_forward_calls
    ] == [2, 1, 2]
    assert set(evaluation.conditions) == {
        "no_op_reconstruction",
        "target_baseline_edit",
        "matched_random_edit",
        "roundtrip_restore_control",
    }
    assert evaluation.no_op_reconstruction_mse == 0.0
    assert evaluation.no_op_accuracy_drop == 0.0
    assert np.array_equal(
        evaluation.conditions["no_op_reconstruction"].prediction.probabilities,
        evaluation.native_prediction.probabilities,
    )
    assert not np.array_equal(
        evaluation.conditions["target_baseline_edit"].prediction.probabilities,
        evaluation.native_prediction.probabilities,
    )
    assert np.array_equal(
        evaluation.conditions["roundtrip_restore_control"].prediction.probabilities,
        evaluation.conditions["no_op_reconstruction"].prediction.probabilities,
    )
    assert evaluation.mechanistic_rescue_passed is False
    assert evaluation.mechanistic_rescue_status == "paired_rescue_not_run"
    for condition in evaluation.conditions.values():
        assert condition.delta_log_loss_vs_model_baseline.shape == (2,)
        assert condition.delta_log_loss_vs_reconstruction.shape == (2,)
    assert len(evaluation.matched_control_features) == 1


def test_qualification_fails_before_any_model_or_rng_change(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="qualification"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(False),
        )

    assert torch.equal(generator.get_state(), initial)


def test_qualification_identity_mismatch_fails_before_inference(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()
    mismatched = qualification()
    mismatched["checkpoint_sha"] = "0" * 64

    with pytest.raises(RuntimeError, match="not bound"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=mismatched,
        )

    assert torch.equal(generator.get_state(), initial)


def test_no_op_native_accuracy_gate_is_enforced_and_rng_rolls_back(
    causal_inputs,
):
    X, _ = causal_inputs
    y = np.array([0, 0])
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="absolute accuracy difference"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(negate_decode=True),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
            max_no_op_reconstruction_mse=10.0,
            max_no_op_probability_deviation=1.0,
            max_no_op_accuracy_drop=0.005,
        )

    assert torch.equal(generator.get_state(), initial)


def test_no_op_probability_gate_is_enforced_and_rng_rolls_back(
    causal_inputs,
):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()
    autoencoder = ExactOvercompleteAutoencoder()
    with torch.no_grad():
        autoencoder.decoder.weight[0, 0] = 0.99

    with pytest.raises(RuntimeError, match="probability difference"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=autoencoder,
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
            max_no_op_reconstruction_mse=10.0,
            max_no_op_probability_deviation=0.0,
            max_no_op_accuracy_drop=1.0,
        )

    assert torch.equal(generator.get_state(), initial)


def test_cross_condition_schedule_drift_fails_closed_and_rolls_back(
    causal_inputs,
):
    X, y = causal_inputs

    class DriftingClassifier(FakeOfficialClassifier):
        def __init__(self):
            super().__init__(temporary=True)
            self.predict_calls = 0

        def predict_proba(self, values):
            result = super().predict_proba(values)
            self.predict_calls += 1
            if self.predict_calls == 2:
                current = self.ensemble_generator_.feature_shuffles_["none"]
                self.ensemble_generator_.feature_shuffles_["none"] = list(
                    reversed(current)
                )
            return result

    classifier = DriftingClassifier()
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="feature permutations differ"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
        )

    assert torch.equal(generator.get_state(), initial)


@pytest.mark.parametrize("model_kind", ["dense", "topk", "pca"])
def test_official_causal_matched_controls_support_all_representation_models(
    causal_inputs, model_kind: str
) -> None:
    X, y = causal_inputs

    class ThreeDimensionalColumnEmbedder(nn.Module):
        feature_group = "same"
        feature_group_size = 2

        def forward(self, values, **_kwargs):
            return torch.stack((values, values * 0.5, -values), dim=-1)

    classifier = FakeOfficialClassifier(temporary=True)
    classifier.model_.col_embedder = ThreeDimensionalColumnEmbedder()
    if model_kind == "dense":
        model = DenseAutoencoder(3, 3, activation="linear")
    elif model_kind == "topk":
        model = TopKSparseAutoencoder(3, latent_dim=3, top_k=2)
    else:
        model = PCARepresentation(3, 3).fit(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    evaluation = run_official_tabicl_causal_edits(
        make_driver(classifier),
        X,
        y,
        site="row_interactor",
        autoencoder=model,
        normalizer=MeanRMSNormalizer(torch.zeros(3), torch.ones(3)),
        target_features=(0,),
        dataset_id="toy-official",
        sample_ids=(0, 1),
        representation_qualification=qualification(),
        random_candidate_pool_size=2,
        max_no_op_reconstruction_mse=100.0,
        max_no_op_probability_deviation=1.0,
        max_no_op_accuracy_drop=1.0,
    )

    assert len(evaluation.matched_control_features) == 1
    assert model_decoder_feature_norms(model).shape == (3,)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _clean_repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("strict source\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "initial")
    return _git(root, "rev-parse", "HEAD")


def _write_numeric_talent(root: Path) -> None:
    root.mkdir()
    (root / "info.json").write_text(
        json.dumps({"task_type": "binclass", "num_classes": 2}),
        encoding="utf-8",
    )
    arrays = {
        "train": np.array(
            [
                [0.2, 0.3, 0.7, 1.1],
                [1.0, -0.4, 0.5, 0.2],
                [-0.2, 0.8, 0.4, -0.1],
                [0.6, 0.9, -0.5, 0.3],
            ],
            dtype=np.float32,
        ),
        "val": np.array([[0.3, 0.4, 0.1, -0.2]], dtype=np.float32),
        "test": np.array(
            [[0.1, 0.6, -0.3, 0.8], [0.9, -0.2, 0.4, 0.3]],
            dtype=np.float32,
        ),
    }
    labels = {
        "train": np.array([0, 1, 0, 1]),
        "val": np.array([1]),
        "test": np.array([0, 1]),
    }
    for split in ("train", "val", "test"):
        np.save(root / f"N_{split}.npy", arrays[split])
        np.save(root / f"y_{split}.npy", labels[split])


def _workflow_fixture(tmp_path: Path) -> SimpleNamespace:
    repository = tmp_path / "repository"
    head = _clean_repository(repository)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"temporary official checkpoint")
    reference_checkpoint = tmp_path / "reference-checkpoint.ckpt"
    reference_checkpoint.write_bytes(b"rope reference checkpoint")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "assignments": [
                    {"name": "selection-dataset", "split": "validation"},
                    {"name": "toy-official", "split": "held_out"},
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_dataset = tmp_path / "raw-talent"
    _write_numeric_talent(raw_dataset)
    sample_roster = tmp_path / "sample-roster.json"
    sample_roster.write_text(
        json.dumps(
            {
                "dataset_id": "toy-official",
                "split": "test",
                "row_indices": [0, 1],
                "sample_ids": ["test-0", "test-1"],
            }
        ),
        encoding="utf-8",
    )

    parent_dir = tmp_path / "train-repr-run"
    parent_dir.mkdir()
    representation = parent_dir / "model.pt"
    representation.write_bytes(b"verified representation checkpoint")
    representation_digest = verify_file(representation).digest
    collect_digests = {"rope": "c" * 64, "temporary": "d" * 64}
    parent_inputs = tuple(
        sorted(
            (
                InputDigest(
                    role=f"source.collect_manifest.{digest}",
                    sha256=digest,
                    size_bytes=1,
                )
                for digest in collect_digests.values()
            ),
            key=lambda item: item.role,
        )
    )
    estimator_options = {
        "n_estimators": 5,
        "norm_methods": ["none", "power"],
        "random_state": 42,
    }
    inference_contract_sha256 = official_inference_contract_sha256(
        head, estimator_options
    )
    source_lineage = {
        "schema_version": 1,
        "source_kind": "official_tabicl_bounded_activation_index",
        "reference_condition": "rope",
        "condition_checkpoints_sha256": {
            "rope": verify_file(reference_checkpoint).digest.sha256,
            "temporary": verify_file(checkpoint).digest.sha256,
        },
        "collect_parent_manifests_sha256": {
            condition: [digest]
            for condition, digest in sorted(collect_digests.items())
        },
        "alignment_sha256": {
            "training": {
                "training-dataset": {"row_interactor": "e" * 64}
            },
            "validation": {
                "validation-dataset": {"row_interactor": "f" * 64}
            },
        },
        "inference_contract_sha256": inference_contract_sha256,
        "evaluation_split": "val",
        "max_classes": 10,
    }
    parent = new_manifest(
        command="train-repr",
        model_family="tabicl-v2",
        model_revision="step-210000",
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=head,
        configuration=FileDigest("1" * 64, 1),
        checkpoint=verify_file(reference_checkpoint).digest,
        dataset_manifest=verify_file(dataset_manifest).digest,
        inputs=parent_inputs,
        condition="rope",
        sites=("row_interactor",),
        seed=42,
        artifacts=(
            ArtifactDigest(
                "model.pt",
                representation_digest.sha256,
                representation_digest.size_bytes,
            ),
        ),
        created_at_utc="2026-08-07T12:00:00Z",
    )
    (parent_dir / "manifest.json").write_text(
        json.dumps(parent.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    parent_manifest_digest = verify_file(parent_dir / "manifest.json").digest

    selection_dir = tmp_path / "validation-selection-run"
    selection_dir.mkdir()
    selection_summary = {
        "schema_version": 1,
        "analysis": "official-model-causal",
        "dataset_id": "selection-dataset",
        "fit_split": "train",
        "evaluation_split": "val",
        "roster_split": "validation",
        "evidence_scope": "exploratory-feature-selection",
        "site": "row_interactor",
        "source_evidence_level": "strict",
        "intervention": {
            "target_features": [0],
            "control_features": [1],
            "latent_baseline": 0.0,
            "random_seed": 42,
        },
        "input_bindings": {
            "model_sha": head,
            "checkpoint_sha256": verify_file(checkpoint).digest.sha256,
            "parent_manifest_sha256": parent_manifest_digest.sha256,
            "representation_model_sha256": representation_digest.sha256,
            "inference_contract_sha256": inference_contract_sha256,
        },
        "representation_source_lineage": source_lineage,
        "no_op_gates": {"passed": True},
    }
    (selection_dir / "summary.json").write_text(
        json.dumps(selection_summary, sort_keys=True), encoding="utf-8"
    )
    selection_summary_digest = verify_file(
        selection_dir / "summary.json"
    ).digest
    selection_manifest = new_manifest(
        command="model-causal",
        model_family="tabicl-v2",
        model_revision="step-210000",
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=head,
        configuration=FileDigest("2" * 64, 1),
        checkpoint=verify_file(checkpoint).digest,
        dataset_manifest=verify_file(dataset_manifest).digest,
        inputs=tuple(
            sorted(
                (
                    InputDigest(
                        "representation.model",
                        representation_digest.sha256,
                        representation_digest.size_bytes,
                    ),
                    InputDigest(
                        "representation.parent_manifest",
                        parent_manifest_digest.sha256,
                        parent_manifest_digest.size_bytes,
                    ),
                ),
                key=lambda item: item.role,
            )
        ),
        condition="temporary",
        sites=("row_interactor",),
        seed=42,
        artifacts=(
            ArtifactDigest(
                "summary.json",
                selection_summary_digest.sha256,
                selection_summary_digest.size_bytes,
            ),
        ),
        created_at_utc="2026-08-07T12:30:00Z",
    )
    (selection_dir / "manifest.json").write_text(
        json.dumps(selection_manifest.to_dict(), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    selection_manifest_digest = verify_file(
        selection_dir / "manifest.json"
    ).digest

    freeze_artifact = tmp_path / "intervention-freeze.json"
    freeze_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_scope": "validation-frozen",
                "condition": "temporary",
                "site": "row_interactor",
                "random_seed": 42,
                "evaluation_sample_roster_sha256": verify_file(
                    sample_roster
                ).digest.sha256,
                "target_features": [0],
                "control_features": [1],
                "latent_baseline": 0.0,
                "representation_model_sha256": representation_digest.sha256,
                "representation_parent_manifest_sha256": (
                    parent_manifest_digest.sha256
                ),
                "model_sha": head,
                "checkpoint_sha256": verify_file(checkpoint).digest.sha256,
                "inference_contract_sha256": inference_contract_sha256,
                "selection_parent_manifest_sha256": (
                    selection_manifest_digest.sha256
                ),
                "selection_summary_sha256": selection_summary_digest.sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    freeze_digest = verify_file(freeze_artifact).digest

    config = tmp_path / "model-causal.json"
    config.write_text(
        json.dumps(
            {
                "provenance": {
                    "model_family": "tabicl-v2",
                    "model_revision": "step-210000",
                    "condition": "temporary",
                    "sites": ["row_interactor"],
                    "checkpoint_path": str(checkpoint),
                    "dataset_manifest_path": str(dataset_manifest),
                    "training_code_root": str(repository),
                    "model_code_root": str(repository),
                    "analysis_code_root": str(repository),
                },
                "representation_run_dir": str(parent_dir),
                "dataset": {
                    "dataset_id": "toy-official",
                    "dataset_dir": str(raw_dataset),
                    "fit_split": "train",
                    "roster_split": "held_out",
                    "evaluation_split": "test",
                    "sample_roster_path": str(sample_roster),
                    "trusted_pickle": False,
                },
                "intervention": {
                    "site": "row_interactor",
                    "target_features": [0],
                    "control_features": [1],
                    "latent_baseline": 0.0,
                    "freeze_artifact_path": str(freeze_artifact),
                    "expected_freeze_artifact_sha256": freeze_digest.sha256,
                    "selection_run_dir": str(selection_dir),
                    "expected_selection_manifest_sha256": (
                        selection_manifest_digest.sha256
                    ),
                    "random_seed": 42,
                    "random_candidate_pool_size": 2,
                    "max_no_op_reconstruction_mse": 0.01,
                    "max_no_op_probability_deviation": 0.02,
                    "max_no_op_accuracy_difference": 0.005,
                },
                "official_classifier": {
                    "device": "cpu",
                    "estimator_options": estimator_options,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        repository=repository,
        head=head,
        checkpoint=checkpoint,
        reference_checkpoint=reference_checkpoint,
        dataset_manifest=dataset_manifest,
        parent_dir=parent_dir,
        representation=representation,
        source_lineage=source_lineage,
        freeze_artifact=freeze_artifact,
        selection_dir=selection_dir,
        config=config,
        output=tmp_path / "published-model-causal",
    )


def _install_workflow_fakes(
    fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    driver_temporary: bool = True,
) -> dict[str, object]:
    observed: dict[str, object] = {}

    def verify_context(
        configuration,
        *,
        command,
        seed,
        additional_input_paths,
        expected_additional_sha256,
    ):
        verified = verify_run_inputs(
            configuration_path=configuration.file.path,
            checkpoint_path=fixture.checkpoint,
            dataset_manifest_path=fixture.dataset_manifest,
            training_code_root=fixture.repository,
            model_code_root=fixture.repository,
            analysis_code_root=fixture.repository,
            command=command,
            model_family="tabicl-v2",
            model_revision="step-210000",
            condition="temporary",
            sites=("row_interactor",),
            seed=seed,
            additional_input_paths=additional_input_paths,
            expected_additional_sha256=expected_additional_sha256,
        )
        return VerifiedRunContext(
            inputs=verified,
            model_family="tabicl-v2",
            model_revision="step-210000",
            condition="temporary",
            sites=("row_interactor",),
        )

    def load_representation(verified, **_kwargs):
        observed["representation_digest"] = verified.digest.sha256
        assert verified.read_bytes() == b"verified representation checkpoint"
        metrics = {
            "explained_variance": 0.99,
            "normalized_mse": 0.01,
            "mse": 0.001,
            "dead_features": 0,
            "dead_feature_fraction": 0.0,
            "active_count": 2.0,
            "mean_active_features": 2.0,
        }
        return (
            ExactOvercompleteAutoencoder(),
            MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            {
                "qualification": {
                    "metric_split": "validation",
                    "held_out_validation": True,
                    "activation_fidelity_passed": True,
                    "explained_variance": 0.99,
                    "minimum_explained_variance": 0.95,
                    "validation_by_condition": {
                        condition: dict(metrics)
                        for condition in ("rope", "temporary")
                    },
                    "worst_condition_explained_variance": 0.99,
                    "native_score_gate": "pending",
                },
                "metadata": {"source_lineage": fixture.source_lineage},
            },
        )

    def fit_driver(
        dataset,
        checkpoint,
        *,
        context_split,
        model_sha,
        **_kwargs,
    ):
        observed["fit_split"] = context_split
        observed["fit_rows"] = len(dataset.train.y)
        evidence = verify_git_tree(fixture.repository, expected_sha=model_sha)
        return OfficialTabICLDriver(
            FakeOfficialClassifier(temporary=driver_temporary),
            adapter=TabICLAdapter(),
            model_sha=model_sha,
            checkpoint_sha=verify_file(checkpoint).digest.sha256,
            fit_context=f"talent-{context_split}",
            _source_evidence=evidence,
        )

    monkeypatch.setattr(
        official_causal, "verify_configured_run_inputs", verify_context
    )
    monkeypatch.setattr(
        official_causal,
        "load_verified_representation_checkpoint",
        load_representation,
    )
    monkeypatch.setattr(
        official_causal, "fit_official_talent_driver", fit_driver
    )
    return observed


def test_model_causal_workflow_publishes_verified_path_free_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    observed = _install_workflow_fakes(fixture, monkeypatch)

    assert (
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
        == 0
    )

    manifest = verify_run_directory(fixture.output)
    assert manifest.command == "model-causal"
    assert manifest.evidence_level == "strict"
    assert [artifact.name for artifact in manifest.artifacts] == [
        "predictions.json",
        "summary.json",
    ]
    roles = {item.role for item in manifest.inputs}
    assert {
        "representation.parent_manifest",
        "representation.model",
        "samples.roster",
        "intervention.freeze",
        "selection.parent_manifest",
        "selection.summary",
        "talent.raw.info.json",
        "talent.raw.N_test.npy",
    } <= roles
    predictions = json.loads(
        (fixture.output / "predictions.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    assert predictions["sample_ids"] == ["test-0", "test-1"]
    assert set(predictions["conditions"]) == {
        "matched_random_edit",
        "no_op_reconstruction",
        "roundtrip_restore_control",
        "target_baseline_edit",
    }
    assert summary["fit_split"] == "train"
    assert summary["evaluation_split"] == "test"
    assert summary["roster_split"] == "held_out"
    assert summary["evidence_scope"] == "confirmatory-held-out"
    assert predictions["evidence_scope"] == "confirmatory-held-out"
    assert summary["mechanistic_rescue"] == {
        "passed": False,
        "status": "paired_rescue_not_run",
        "roundtrip_restore_control_passed": True,
    }
    assert summary["no_op_gates"]["passed"] is True
    assert summary["no_op_gates"]["thresholds"] == {
        "max_no_op_accuracy_difference": 0.005,
        "max_no_op_probability_deviation": 0.02,
        "max_no_op_reconstruction_mse": 0.01,
    }
    assert summary["official_scope"]["feature_group"] == "same"
    assert summary["official_scope"]["maximum_native_classes"] == 10
    checkpoints = summary["representation_source_lineage"][
        "condition_checkpoints_sha256"
    ]
    assert checkpoints["temporary"] == verify_file(
        fixture.checkpoint
    ).digest.sha256
    assert checkpoints["rope"] == verify_file(
        fixture.reference_checkpoint
    ).digest.sha256
    assert checkpoints["temporary"] != checkpoints["rope"]
    assert observed["fit_split"] == "train"
    assert observed["fit_rows"] == 4
    assert observed["representation_digest"] == hashlib.sha256(
        b"verified representation checkpoint"
    ).hexdigest()
    published = json.dumps(
        {
            "manifest": manifest.to_dict(),
            "predictions": predictions,
            "summary": summary,
        }
    )
    assert str(tmp_path) not in published
    assert not list(tmp_path.glob(".published-model-causal.*.staging"))


def test_model_causal_workflow_rejects_tampered_parent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    fixture.representation.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash or size mismatch"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda lineage: lineage["condition_checkpoints_sha256"].__setitem__(
                "temporary", "0" * 64
            ),
            "causal checkpoint differs",
        ),
        (
            lambda lineage: lineage[
                "collect_parent_manifests_sha256"
            ].__setitem__("temporary", ["a" * 64]),
            "collect-parent lineage differs",
        ),
        (
            lambda lineage: lineage.__setitem__(
                "inference_contract_sha256", "0" * 64
            ),
            "inference contract differs",
        ),
    ],
)
def test_model_causal_rejects_tampered_multi_condition_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    mutation(fixture.source_lineage)
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match=message):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_checkpoint_identity_mode_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(
        fixture, monkeypatch, driver_temporary=False
    )

    with pytest.raises(RuntimeError, match="identity mode differs"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_tampered_freeze_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    fixture.freeze_artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_semantically_resealed_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    freeze = json.loads(fixture.freeze_artifact.read_text(encoding="utf-8"))
    freeze["latent_baseline"] = 1.0
    fixture.freeze_artifact.write_text(json.dumps(freeze), encoding="utf-8")
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"]["expected_freeze_artifact_sha256"] = verify_file(
        fixture.freeze_artifact
    ).digest.sha256
    fixture.config.write_text(json.dumps(config), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="freeze artifact differs"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_tampered_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    (fixture.selection_dir / "summary.json").write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="artifact hash or size mismatch"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_semantically_resealed_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    summary_path = fixture.selection_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence_scope"] = "confirmatory-held-out"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_digest = verify_file(summary_path).digest

    manifest_path = fixture.selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "summary.json"
    )
    summary_artifact["sha256"] = summary_digest.sha256
    summary_artifact["size_bytes"] = summary_digest.size_bytes
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = verify_file(manifest_path).digest

    freeze = json.loads(fixture.freeze_artifact.read_text(encoding="utf-8"))
    freeze["selection_parent_manifest_sha256"] = manifest_digest.sha256
    freeze["selection_summary_sha256"] = summary_digest.sha256
    fixture.freeze_artifact.write_text(json.dumps(freeze), encoding="utf-8")
    freeze_digest = verify_file(fixture.freeze_artifact).digest

    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"]["expected_selection_manifest_sha256"] = (
        manifest_digest.sha256
    )
    config["intervention"]["expected_freeze_artifact_sha256"] = (
        freeze_digest.sha256
    )
    fixture.config.write_text(json.dumps(config), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="selection summary scope differs"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_inconsistent_worst_condition_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    original_loader = official_causal.load_verified_representation_checkpoint

    def inconsistent_loader(*args, **kwargs):
        model, normalizer, metadata = original_loader(*args, **kwargs)
        metadata["qualification"]["worst_condition_explained_variance"] = 0.98
        return model, normalizer, metadata

    monkeypatch.setattr(
        official_causal,
        "load_verified_representation_checkpoint",
        inconsistent_loader,
    )

    with pytest.raises(ValueError, match="worst-condition"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_unrelated_parent_analysis_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    manifest_path = fixture.parent_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_code_sha"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="analysis ancestry"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_requires_roster_split_and_evaluation_split_alignment(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"]["roster_split"] = "validation"
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires evaluation_split='val'"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("roster_split", "scope"),
    [
        ("discovery", "exploratory-discovery"),
        ("validation", "exploratory-feature-selection"),
        ("held_out", "confirmatory-held-out"),
    ],
)
def test_model_causal_evidence_scope_is_explicit_and_validation_is_exploratory(
    roster_split: str, scope: str
) -> None:
    assert official_causal._evidence_scope_for_roster_split(roster_split) == scope
    if roster_split == "validation":
        assert "formal" not in scope and "confirmatory" not in scope


@pytest.mark.parametrize(
    ("field", "relaxed_value"),
    [
        ("max_no_op_reconstruction_mse", 0.010001),
        ("max_no_op_probability_deviation", 0.020001),
        ("max_no_op_accuracy_difference", 0.005001),
    ],
)
def test_model_causal_workflow_rejects_relaxed_protocol_thresholds(
    tmp_path: Path,
    field: str,
    relaxed_value: float,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["intervention"][field] = relaxed_value
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exceed protocol maxima"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize("random_state", [None, True, 41])
def test_model_causal_workflow_requires_aligned_explicit_random_state(
    tmp_path: Path,
    random_state: object,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["official_classifier"]["estimator_options"][
        "random_state"
    ] = random_state
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")

    expected = "explicit non-negative integer" if random_state in {None, True} else "must equal"
    with pytest.raises(ValueError, match=expected):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_workflow_requires_strict_train_repr_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    manifest_path = fixture.parent_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command"] = "collect"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="command must be train-repr"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_workflow_rejects_train_evaluation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["dataset"]["evaluation_split"] = "train"
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="never train"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_publication_failure_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    original_write = official_causal._write_json_artifact
    calls = 0

    def fail_second_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        official_causal, "_write_json_artifact", fail_second_write
    )
    with pytest.raises(RuntimeError, match="injected publication failure"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()
    assert not list(tmp_path.glob(".published-model-causal.*.staging"))
