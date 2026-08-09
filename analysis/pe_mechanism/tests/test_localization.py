from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pe_mechanism.provenance as provenance_module
import pytest
import torch

from pe_mechanism.localization import (
    _array_sha256,
    _encode_labels,
    _ensemble_schedule_sha256,
    _portable_summary,
    _validate_configuration,
    _validate_matched_checkpoint_pair_schema,
    _verify_snapshot_pair,
    evaluate_official_tabicl_rope_conditions,
    run_localization,
)
from pe_mechanism.official_tabicl import (
    OfficialInferenceMetrics,
    OfficialInferenceResult,
    OfficialTabICLSession,
)
from pe_mechanism.provenance import load_verified_json_config, verify_file


class _FakeSession:
    def __init__(self, *, break_full: bool = False) -> None:
        self.identity_state = 7
        self.policy = None
        self.break_full = break_full

    def snapshot_identity_rng(self):
        return self.identity_state

    def restore_identity_rng(self, state):
        self.identity_state = state

    @contextmanager
    def rope_policy(self, **policy):
        previous = self.policy
        self.policy = dict(policy)
        try:
            yield None
        finally:
            self.policy = previous

    def predict_proba(self, _X, *, y):
        targets = np.asarray(y, dtype=np.int64)
        shift = 0.0
        if self.policy is not None:
            if not self.policy["rotate_queries"] and not self.policy["rotate_keys"]:
                shift = 0.2
            elif self.policy["rotate_queries"] and not self.policy["rotate_keys"]:
                shift = 0.1
            elif self.policy["rotate_queries"] and self.policy["rotate_keys"]:
                shift = 1e-3 if self.break_full else 0.0
        probabilities = np.tile(np.array([[0.7 + shift, 0.3 - shift]]), (targets.size, 1))
        losses = -np.log(probabilities[np.arange(targets.size), targets])
        self.identity_state += 1
        return OfficialInferenceResult(
            probabilities=probabilities,
            baseline_probabilities=probabilities.copy(),
            classes=np.array([0, 1]),
            forward_calls=(),
            metrics=OfficialInferenceMetrics(
                accuracy=float(np.mean(probabilities.argmax(axis=1) == targets)),
                log_loss=float(losses.mean()),
                n_samples=int(targets.size),
            ),
            exact_baseline_verified=True,
            source_evidence_level="strict",
        )


class _FakeDriver:
    def __init__(self, *, break_full: bool = False) -> None:
        self.session = _FakeSession(break_full=break_full)

    @contextmanager
    def paired_session(self):
        yield self.session


class _PolicyAdapter:
    @contextmanager
    def rope_policy(self, raw, **policy):
        raw.policy = dict(policy)
        try:
            yield None
        finally:
            raw.policy = None


class _SessionDriver:
    def __init__(self) -> None:
        self.adapter = _PolicyAdapter()
        self.raw = SimpleNamespace(policy=None)

    def _validated_raw_model(self):
        return self.raw


def _schedule_driver(*, feature_shuffle=(0, 1)):
    generator = SimpleNamespace(
        n_features_in_=2,
        ensemble_configs_={"none": [(feature_shuffle, (0, 1))]},
        feature_shuffles_={"none": [feature_shuffle]},
        class_shuffles_={"none": [(0, 1)]},
    )
    estimator = SimpleNamespace(
        ensemble_generator_=generator,
        n_classes_=2,
        batch_size=1,
    )
    return SimpleNamespace(estimator=estimator)


def test_ensemble_schedule_digest_is_canonical_and_sensitive() -> None:
    first = _ensemble_schedule_sha256(_schedule_driver())
    assert first == _ensemble_schedule_sha256(_schedule_driver())
    assert first != _ensemble_schedule_sha256(
        _schedule_driver(feature_shuffle=(1, 0))
    )


def test_object_labels_are_encoded_without_pointer_dependent_bytes() -> None:
    labels = np.array(["dog", "cat", "dog"], dtype=object)
    encoded, columns = _encode_labels(labels, np.array(["cat", "dog"], dtype=object))
    np.testing.assert_array_equal(encoded, np.array([1, 0, 1], dtype=np.int64))
    assert columns == [
        {"python_type": "str", "repr": "'cat'"},
        {"python_type": "str", "repr": "'dog'"},
    ]
    with pytest.raises(TypeError, match="object arrays"):
        _array_sha256(labels)


def test_official_session_scopes_and_restores_rope_policy() -> None:
    driver = _SessionDriver()
    session = OfficialTabICLSession(driver, driver.raw)
    with session.rope_policy(blocks=[0], rotate_queries=False):
        assert driver.raw.policy == {"blocks": [0], "rotate_queries": False}
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with session.rope_policy(blocks=[1]):
                pass
    assert driver.raw.policy is None


def test_official_rope_conditions_are_rng_paired_and_restore() -> None:
    driver = _FakeDriver()
    results = evaluate_official_tabicl_rope_conditions(
        driver,
        np.zeros((2, 1)),
        np.array([0, 1]),
        (
            {"name": "full"},
            {
                "name": "off",
                "rotate_queries": False,
                "rotate_keys": False,
            },
            {
                "name": "query_only",
                "rotate_queries": True,
                "rotate_keys": False,
            },
        ),
    )

    assert tuple(results) == ("native", "full", "off", "query_only")
    assert np.array_equal(results["native"].probabilities, results["full"].probabilities)
    assert not np.array_equal(results["native"].probabilities, results["off"].probabilities)
    assert driver.session.identity_state == 7
    assert driver.session.policy is None


def test_official_rope_full_proxy_must_be_exact_noop() -> None:
    with pytest.raises(RuntimeError, match="differs from native"):
        evaluate_official_tabicl_rope_conditions(
            _FakeDriver(break_full=True),
            np.zeros((1, 1)),
            np.array([0]),
            ({"name": "full"},),
        )


def _configuration(tmp_path: Path, conditions: list[dict]) -> Path:
    private = tmp_path / "private"
    private.mkdir(parents=True)
    dataset = tmp_path / "toy"
    dataset.mkdir()
    config = {
        "schema_version": 1,
        "private_study_root": str(private),
        "datasets": [{"dataset_id": "toy", "path": str(dataset)}],
        "roster_split": "discovery",
        "evaluation_split": "val",
        "trusted_pickle": False,
        "seed": 42,
        "device": "cuda",
        "estimator_options": {
            "n_estimators": 2,
            "batch_size": 2,
            "use_amp": False,
            "use_fa3": False,
            "random_state": 42,
        },
        "checkpoint_study": "exploratory_pilot",
        "comparison_step": 250000,
        "rope_conditions": conditions,
        "secondary_checkpoint_path": str(tmp_path / "none.ckpt"),
        "expected_secondary_checkpoint_sha256": "3" * 64,
        "snapshot_manifest_path": str(tmp_path / "snapshot.json"),
        "expected_snapshot_manifest_sha256": "4" * 64,
        "provenance": {
            "expected_checkpoint_sha256": "1" * 64,
            "expected_dataset_manifest_sha256": "2" * 64,
            "expected_training_code_sha": "a" * 40,
            "expected_model_code_sha": "a" * 40,
            "expected_analysis_code_sha": "b" * 40,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _valid_conditions() -> list[dict]:
    return [
        {"name": "full"},
        {"name": "off_all", "rotate_queries": False, "rotate_keys": False},
        {
            "name": "off_block_0",
            "blocks": [0],
            "rotate_queries": False,
            "rotate_keys": False,
        },
    ]


def test_localization_config_requires_noop_global_and_layer_conditions(tmp_path: Path) -> None:
    config, specs, conditions = _validate_configuration(
        load_verified_json_config(_configuration(tmp_path, _valid_conditions()))
    )
    assert config["comparison_step"] == 250000
    assert [spec.dataset_id for spec in specs] == ["toy"]
    assert [condition.name for condition in conditions] == [
        "full",
        "off_all",
        "off_block_0",
    ]

    missing_layer = [item for item in _valid_conditions() if "block" not in item["name"]]
    with pytest.raises(ValueError, match="block-local"):
        _validate_configuration(
            load_verified_json_config(_configuration(tmp_path / "second", missing_layer))
        )


def _checkpoint(path: Path, *, mode: str, step: int) -> None:
    state_dict = {"weight": torch.ones(1)}
    if mode == "rope":
        state_dict["row_interactor.tf_row.rope.freqs"] = torch.ones(2)
    torch.save(
        {
            "config": {
                "row_identity_mode": mode,
                "row_num_blocks": 3,
                "row_nhead": 2,
                "embed_dim": 8,
            },
            "curr_step": step,
            "state_dict": state_dict,
        },
        path,
    )


def _matched_checkpoint_schema():
    common_config = {
        "row_num_blocks": 3,
        "row_nhead": 8,
        "embed_dim": 128,
    }
    common_state = {"weight": ((128, 128), "torch.float32")}
    return {
        "rope_config": {"row_identity_mode": "rope", **common_config},
        "none_config": {"row_identity_mode": "none", **common_config},
        "rope_schema": {
            **common_state,
            "row_interactor.tf_row.rope.freqs": ((8,), "torch.float32"),
        },
        "none_schema": common_state,
    }


def test_matched_checkpoint_schema_accepts_only_derived_rope_frequency_state() -> None:
    _validate_matched_checkpoint_pair_schema(**_matched_checkpoint_schema())


def test_matched_checkpoint_schema_rejects_rope_frequency_on_none_arm() -> None:
    inputs = _matched_checkpoint_schema()
    frequency = inputs["rope_schema"].pop("row_interactor.tf_row.rope.freqs")
    inputs["none_schema"]["row_interactor.tf_row.rope.freqs"] = frequency

    with pytest.raises(ValueError, match="state-schema difference"):
        _validate_matched_checkpoint_pair_schema(**inputs)


@pytest.mark.parametrize("arm", ["rope", "none"])
def test_matched_checkpoint_schema_rejects_unknown_arm_only_state(arm: str) -> None:
    inputs = _matched_checkpoint_schema()
    inputs[f"{arm}_schema"]["unexpected.state"] = ((1,), "torch.float32")

    with pytest.raises(ValueError, match="state-schema difference"):
        _validate_matched_checkpoint_pair_schema(**inputs)


@pytest.mark.parametrize(
    ("signature", "expected_error"),
    [
        (((127, 128), "torch.float32"), "tensor schemas differ"),
        (((128, 128), "torch.float64"), "tensor schemas differ"),
    ],
)
def test_matched_checkpoint_schema_rejects_common_tensor_drift(
    signature: tuple[tuple[int, ...], str], expected_error: str
) -> None:
    inputs = _matched_checkpoint_schema()
    inputs["rope_schema"]["weight"] = signature

    with pytest.raises(ValueError, match=expected_error):
        _validate_matched_checkpoint_pair_schema(**inputs)


@pytest.mark.parametrize(
    "signature",
    [((7,), "torch.float32"), ((8,), "torch.float64")],
)
def test_matched_checkpoint_schema_rejects_wrong_rope_frequency_signature(
    signature: tuple[tuple[int, ...], str],
) -> None:
    inputs = _matched_checkpoint_schema()
    inputs["rope_schema"]["row_interactor.tf_row.rope.freqs"] = signature

    with pytest.raises(ValueError, match="does not match the checkpoint config"):
        _validate_matched_checkpoint_pair_schema(**inputs)


def test_matched_checkpoint_schema_rejects_odd_head_dimension() -> None:
    inputs = _matched_checkpoint_schema()
    inputs["rope_config"]["row_nhead"] = 128
    inputs["none_config"]["row_nhead"] = 128

    with pytest.raises(ValueError, match="cannot derive the RoPE frequency shape"):
        _validate_matched_checkpoint_pair_schema(**inputs)


def test_matched_checkpoint_schema_rejects_config_drift() -> None:
    inputs = _matched_checkpoint_schema()
    inputs["rope_config"]["row_num_blocks"] = 4

    with pytest.raises(ValueError, match="outside row_identity_mode"):
        _validate_matched_checkpoint_pair_schema(**inputs)


def _snapshot_context(tmp_path: Path, *, mutate: str | None = None):
    rope_path = tmp_path / "rope.ckpt"
    none_path = tmp_path / "none.ckpt"
    _checkpoint(rope_path, mode="rope", step=250000)
    _checkpoint(none_path, mode="none", step=250000)
    rope = verify_file(rope_path)
    none = verify_file(none_path)
    arms = {
        "rope": {
            "row_identity_mode": "rope",
            "curr_step": 250000,
            "sha256": rope.digest.sha256,
            "bytes": rope.digest.size_bytes,
            "snapshot_checkpoint": rope_path.name,
            "source_checkpoint": str(rope_path),
            "source_job_id": "1",
            "state_dict_tensor_count": 2,
        },
        "none": {
            "row_identity_mode": "none",
            "curr_step": 250000,
            "sha256": none.digest.sha256,
            "bytes": none.digest.size_bytes,
            "snapshot_checkpoint": none_path.name,
            "source_checkpoint": str(none_path),
            "source_job_id": "2",
            "state_dict_tensor_count": 1,
        },
    }
    if mutate == "hash":
        arms["none"]["sha256"] = "0" * 64
    if mutate == "step":
        arms["rope"]["curr_step"] = 249999
    if mutate == "mode":
        arms["none"]["row_identity_mode"] = "rope"
    payload = {
        "schema_version": 1,
        "kind": "exploratory_same_step_pilot_checkpoint_pair",
        "formal_eligible": False,
        "comparison_step": 250000,
        "seed": 42,
        "pilot_source_commit": "a" * 40,
        "captured_at_utc": "2026-08-09T00:00:00+00:00",
        "notes": "test fixture",
        "arms": arms,
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = verify_file(snapshot_path)
    inputs = SimpleNamespace(
        checkpoint=rope,
        training_code=SimpleNamespace(head_sha="a" * 40),
        model_code=SimpleNamespace(head_sha="a" * 40),
    )
    context = SimpleNamespace(inputs=inputs)
    return context, none, snapshot


@pytest.mark.parametrize("mutate", [None, "hash", "step", "mode"])
def test_snapshot_pair_binds_both_checkpoints(tmp_path: Path, mutate: str | None) -> None:
    context, none, snapshot = _snapshot_context(tmp_path, mutate=mutate)
    if mutate is None:
        _verify_snapshot_pair(
            context,
            config={"comparison_step": 250000, "seed": 42},
            secondary=none,
            snapshot=snapshot,
        )
    else:
        with pytest.raises(ValueError):
            _verify_snapshot_pair(
                context,
                config={"comparison_step": 250000, "seed": 42},
                secondary=none,
                snapshot=snapshot,
            )


def test_portable_summary_uses_dataset_paired_effects() -> None:
    datasets = [
        {
            "conditions": {
                "rope_native": {"accuracy": 0.7, "log_loss": 0.6},
                "none_native": {"accuracy": 0.8, "log_loss": 0.5},
            }
        },
        {
            "conditions": {
                "rope_native": {"accuracy": 0.6, "log_loss": 0.7},
                "none_native": {"accuracy": 0.7, "log_loss": 0.6},
            }
        },
    ]
    summary = _portable_summary(datasets, seed=42)
    none = summary["conditions"]["none_native"]
    assert none["accuracy_effect"]["mean_effect"] == pytest.approx(0.1)
    assert none["log_loss_effect"]["mean_effect"] == pytest.approx(0.1)
    assert summary["formal_eligible"] is False


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


class _EndToEndSession:
    def __init__(self, driver) -> None:
        self.driver = driver
        self.identity_state = 11
        self.policy = None

    def snapshot_identity_rng(self):
        return self.identity_state

    def restore_identity_rng(self, state):
        self.identity_state = state

    @contextmanager
    def rope_policy(self, **policy):
        previous = self.policy
        self.policy = dict(policy)
        try:
            yield None
        finally:
            self.policy = previous

    def predict_proba(self, X, *, y):
        rows = len(X)
        class_zero = 0.70 if self.driver.mode == "rope" else 0.62
        if self.policy is not None and (
            not self.policy["rotate_queries"]
            and not self.policy["rotate_keys"]
        ):
            class_zero -= 0.08
        probabilities = np.tile([class_zero, 1.0 - class_zero], (rows, 1))
        targets = np.asarray(y, dtype=np.int64)
        losses = -np.log(probabilities[np.arange(rows), targets])
        self.identity_state += 1
        return OfficialInferenceResult(
            probabilities=probabilities,
            baseline_probabilities=probabilities.copy(),
            classes=np.array([0, 1]),
            forward_calls=(),
            metrics=OfficialInferenceMetrics(
                accuracy=float(np.mean(probabilities.argmax(axis=1) == targets)),
                log_loss=float(losses.mean()),
                n_samples=rows,
            ),
            exact_baseline_verified=True,
            source_evidence_level="strict",
        )


class _EndToEndDriver:
    def __init__(self, *, mode: str, model_sha: str, checkpoint_sha: str) -> None:
        self.mode = mode
        raw = SimpleNamespace(
            _cache=None,
            training=False,
            row_identity_mode=mode,
            row_interactor=SimpleNamespace(identity_mode=mode),
            col_embedder=SimpleNamespace(feature_group="same"),
        )
        generator = SimpleNamespace(
            n_features_in_=2,
            ensemble_configs_={"none": [((0, 1), (0, 1))]},
            feature_shuffles_={"none": [(0, 1)]},
            class_shuffles_={"none": [(0, 1)]},
        )
        self.estimator = SimpleNamespace(
            model_=raw,
            kv_cache=False,
            model_kv_cache_=None,
            support_many_classes=False,
            n_classes_=2,
            batch_size=1,
            ensemble_generator_=generator,
        )
        self.model_sha = model_sha
        self.checkpoint_sha = checkpoint_sha
        self.fit_context = "talent-train"
        self.source_evidence_level = "strict"
        self.session = _EndToEndSession(self)

    @contextmanager
    def paired_session(self):
        yield self.session


class _EndToEndFactory:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def __call__(
        self,
        dataset,
        checkpoint,
        *,
        context_split,
        device,
        model_sha,
        estimator_options,
        expected_source_root,
    ):
        del device
        assert context_split == "train"
        assert len(dataset.train.X) == 6
        assert estimator_options == {
            "n_estimators": 2,
            "batch_size": 2,
            "use_amp": False,
            "use_fa3": False,
            "random_state": 42,
        }
        assert expected_source_root.is_dir()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        mode = payload["config"]["row_identity_mode"]
        self.modes.append(mode)
        return _EndToEndDriver(
            mode=mode,
            model_sha=model_sha,
            checkpoint_sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        )


def test_localization_end_to_end_is_paired_atomic_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pe_mechanism.collect.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024**3),
    )
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.invalid")
    tracked = source / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "tracked.py")
    _git(source, "commit", "-q", "-m", "initial")
    head = _git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(provenance_module, "__file__", str(tracked))

    dataset = tmp_path / "talent" / "toy"
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text(
        json.dumps({"name": "toy", "task_type": "classification"}),
        encoding="utf-8",
    )
    for split, rows in (("train", 6), ("val", 4), ("test", 5)):
        np.save(dataset / f"N_{split}.npy", np.arange(rows * 2).reshape(rows, 2))
        np.save(dataset / f"y_{split}.npy", np.array(np.arange(rows) % 2, dtype=object))
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps({"assignments": [{"name": "toy", "split": "discovery"}]}),
        encoding="utf-8",
    )
    dataset_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.iterdir()
        if path.is_file()
    }

    rope = tmp_path / "rope.ckpt"
    none = tmp_path / "none.ckpt"
    _checkpoint(rope, mode="rope", step=250000)
    _checkpoint(none, mode="none", step=250000)
    rope_hash = hashlib.sha256(rope.read_bytes()).hexdigest()
    none_hash = hashlib.sha256(none.read_bytes()).hexdigest()
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "exploratory_same_step_pilot_checkpoint_pair",
                "formal_eligible": False,
                "comparison_step": 250000,
                "seed": 42,
                "pilot_source_commit": head,
                "captured_at_utc": "2026-08-09T00:00:00+00:00",
                "notes": "test fixture",
                "arms": {
                    "rope": {
                        "row_identity_mode": "rope",
                        "curr_step": 250000,
                        "sha256": rope_hash,
                        "bytes": rope.stat().st_size,
                        "snapshot_checkpoint": rope.name,
                        "source_checkpoint": str(rope),
                        "source_job_id": "1",
                        "state_dict_tensor_count": 2,
                    },
                    "none": {
                        "row_identity_mode": "none",
                        "curr_step": 250000,
                        "sha256": none_hash,
                        "bytes": none.stat().st_size,
                        "snapshot_checkpoint": none.name,
                        "source_checkpoint": str(none),
                        "source_job_id": "2",
                        "state_dict_tensor_count": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    private = tmp_path / "private"
    private.mkdir()
    config = tmp_path / "localize.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "private_study_root": str(private),
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "path": str(dataset),
                        "expected_input_sha256": dataset_hashes,
                    }
                ],
                "roster_split": "discovery",
                "evaluation_split": "val",
                "trusted_pickle": True,
                "seed": 42,
                "device": "cuda",
                "estimator_options": {
                    "n_estimators": 2,
                    "batch_size": 2,
                    "use_amp": False,
                    "use_fa3": False,
                    "random_state": 42,
                },
                "checkpoint_study": "exploratory_pilot",
                "comparison_step": 250000,
                "rope_conditions": _valid_conditions(),
                "secondary_checkpoint_path": str(none),
                "expected_secondary_checkpoint_sha256": none_hash,
                "snapshot_manifest_path": str(snapshot),
                "expected_snapshot_manifest_sha256": hashlib.sha256(
                    snapshot.read_bytes()
                ).hexdigest(),
                "provenance": {
                    "model_family": "tabicl-v2",
                    "model_revision": "pilot-step-250000",
                    "condition": "matched-step-rope-none",
                    "sites": [
                        "row_interactor.tf_row.blocks.0",
                        "row_interactor.tf_row.blocks.1",
                        "row_interactor.tf_row.blocks.2",
                    ],
                    "checkpoint_path": str(rope),
                    "dataset_manifest_path": str(roster),
                    "training_code_root": str(source),
                    "model_code_root": str(source),
                    "analysis_code_root": str(source),
                    "expected_checkpoint_sha256": rope_hash,
                    "expected_dataset_manifest_sha256": hashlib.sha256(
                        roster.read_bytes()
                    ).hexdigest(),
                    "expected_training_code_sha": head,
                    "expected_model_code_sha": head,
                    "expected_analysis_code_sha": head,
                },
            }
        ),
        encoding="utf-8",
    )

    factory = _EndToEndFactory()
    output = private / "run"
    summary = run_localization(config, output, driver_factory=factory)

    assert factory.modes == ["rope", "none"]
    assert summary["dataset_count"] == 1
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "predictions.npz",
        "results.json",
        "summary.json",
    }
    results_text = (output / "results.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in results_text
    results = json.loads(results_text)
    dataset_result = results["datasets"][0]
    assert results["estimator_options"]["n_estimators"] == 2
    assert results["condition_definitions"][1] == {
        "result_condition": "rope_full",
        "checkpoint_arm": "rope",
        "policy": {
            "blocks": None,
            "rotate_queries": True,
            "rotate_keys": True,
            "phase_strength": 1.0,
            "frequency_band": "all",
            "heads": None,
        },
    }
    assert len(dataset_result["ensemble_schedule_sha256"]) == 64
    assert dataset_result["label_encoding"] == (
        "zero_based_index_into_probability_columns"
    )
    assert dataset_result["probability_columns"] == [
        {"python_type": "int", "repr": "0"},
        {"python_type": "int", "repr": "1"},
    ]
    predictions = np.load(output / "predictions.npz", allow_pickle=False)
    assert set(predictions.files) == {
        "d0000_labels",
        "d0000_none_native",
        "d0000_rope_full",
        "d0000_rope_native",
        "d0000_rope_off_all",
        "d0000_rope_off_block_0",
    }
    for condition in dataset_result["conditions"].values():
        key = condition["prediction_array_key"]
        assert condition["probability_sha256"] == _array_sha256(predictions[key])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "localize"
    assert {item["role"] for item in manifest["inputs"]} >= {
        "checkpoint.none",
        "checkpoint_pair_manifest",
    }
