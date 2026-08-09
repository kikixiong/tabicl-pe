from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
import pytest
import torch
from torch import nn

import pe_mechanism.official_tabpfn as official_tabpfn
from pe_mechanism.official_tabicl import RawTalentDataset, RawTalentSplit
from pe_mechanism.official_tabpfn import fit_official_tabpfn_v26_driver


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def fake_official_runtime(tmp_path, monkeypatch):
    source_root = tmp_path / "tabpfn-source"
    source_root.mkdir()
    source_file = source_root / "official_runtime.py"
    source_file.write_text("# clean fake TabPFN source boundary\n", encoding="utf-8")
    _git("init", "-q", cwd=source_root)
    _git("config", "user.email", "tests@example.invalid", cwd=source_root)
    _git("config", "user.name", "Mechanism Tests", cwd=source_root)
    _git("add", "official_runtime.py", cwd=source_root)
    _git("commit", "-q", "-m", "fake runtime", cwd=source_root)
    model_sha = _git("rev-parse", "HEAD", cwd=source_root)

    module_name = f"_fake_tabpfn_{uuid.uuid4().hex}"
    source_module = ModuleType(module_name)
    source_module.__file__ = str(source_file)
    monkeypatch.setitem(sys.modules, module_name, source_module)

    class TabPFNV2p6(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_positional_embedding_embeddings = nn.Linear(2, 2)
            with torch.no_grad():
                self.feature_positional_embedding_embeddings.weight.copy_(
                    torch.tensor([[0.7, -0.2], [-0.4, 0.6]])
                )
                self.feature_positional_embedding_embeddings.bias.copy_(
                    torch.tensor([0.25, -0.15])
                )

    class ClassifierModelSpecs:
        def __init__(self, model, architecture_config, inference_config):
            self.model = model
            self.architecture_config = architecture_config
            self.inference_config = inference_config

    class FakeLoader:
        calls = []
        model = TabPFNV2p6().eval()
        config = SimpleNamespace(name="TabPFN-v2.6")
        inference_config = object()

        def __new__(cls, **kwargs):
            cls.calls.append(dict(kwargs))
            return (
                [cls.model],
                None,
                [cls.config],
                cls.inference_config,
            )

    class FakeTabPFNClassifier:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.fit_calls = 0
            self.predict_calls = 0
            self.fail_on_call = None
            self.drift_on_call = None
            self.predict_inputs = []
            self.__class__.instances.append(self)

        def fit(self, X, y):
            self.fit_calls += 1
            self.fit_X = np.array(X, copy=True)
            self.fit_y = np.array(y, copy=True)
            self.models_ = [self.model_path.model]
            self.models_[0].eval()
            self.classes_ = np.unique(self.fit_y)
            return self

        def predict_proba(self, X):
            self.predict_calls += 1
            if self.predict_calls == self.fail_on_call:
                raise RuntimeError("synthetic official prediction failure")
            values = np.asarray(X, dtype=np.float32)
            self.predict_inputs.append(values.copy())
            codes = torch.as_tensor(values[:, :2])
            with torch.no_grad():
                position = self.models_[
                    0
                ].feature_positional_embedding_embeddings(codes)
                score = position[:, 0] - 0.6 * position[:, 1]
                score = score + 0.2 * codes[:, 0]
                probabilities = torch.softmax(
                    torch.stack((-score, score), dim=1), dim=1
                ).cpu().numpy()
            if self.predict_calls == self.drift_on_call:
                probabilities = probabilities.copy()
                probabilities[:, 0] += 1e-4
                probabilities[:, 1] -= 1e-4
            return probabilities

    for value in (
        TabPFNV2p6,
        ClassifierModelSpecs,
        FakeLoader,
        FakeTabPFNClassifier,
    ):
        value.__module__ = module_name
        setattr(source_module, value.__name__, value)

    runtime = official_tabpfn._TabPFNRuntime(
        classifier_class=FakeTabPFNClassifier,
        model_specs_class=ClassifierModelSpecs,
        load_model_criterion_config=FakeLoader,
    )
    monkeypatch.setattr(
        official_tabpfn,
        "_load_tabpfn_runtime",
        lambda: runtime,
    )

    dataset = RawTalentDataset(
        name="strict-fixture",
        task_type="binclass",
        train=RawTalentSplit(
            X=np.array(
                [[-2.0, 0.5], [-0.3, 1.0], [0.7, -1.0], [2.0, 0.2]],
                dtype=np.float32,
            ),
            y=np.array([0, 0, 1, 1]),
        ),
        val=RawTalentSplit(
            X=np.array([[-1.1, 0.4], [1.2, -0.4]], dtype=np.float32),
            y=np.array([0, 1]),
        ),
        test=RawTalentSplit(
            X=np.array(
                [[-1.5, 0.1], [-0.1, 0.8], [0.9, -0.5]],
                dtype=np.float32,
            ),
            y=np.array([0, 0, 1]),
        ),
        n_numeric_features=2,
        n_categorical_features=0,
        info_sha256="a" * 64,
        input_sha256={},
    )
    checkpoint = tmp_path / "tabpfn-v2.6.ckpt"
    checkpoint.write_bytes(b"exact local fake TabPFN v2.6 checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return SimpleNamespace(
        source_root=source_root,
        source_file=source_file,
        model_sha=model_sha,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        dataset=dataset,
        runtime=runtime,
        loader=FakeLoader,
        classifier=FakeTabPFNClassifier,
        model=FakeLoader.model,
        module_name=module_name,
    )


def _fit(bundle):
    return fit_official_tabpfn_v26_driver(
        bundle.dataset,
        bundle.checkpoint,
        checkpoint_sha256=bundle.checkpoint_sha256,
        model_sha=bundle.model_sha,
        expected_source_root=bundle.source_root.resolve(),
        seed=17,
        device="cpu",
    )


def test_official_driver_is_offline_train_only_and_runs_paired_policies(
    fake_official_runtime,
):
    bundle = fake_official_runtime
    driver = _fit(bundle)
    estimator = bundle.classifier.instances[-1]

    assert len(bundle.loader.calls) == 1
    loader_call = bundle.loader.calls[0]
    assert loader_call == {
        "model_path": bundle.checkpoint.resolve(),
        "check_bar_distribution_criterion": False,
        "cache_trainset_representation": False,
        "which": "classifier",
        "version": "v2.6",
        "download_if_not_exists": False,
    }
    assert estimator.fit_calls == 1
    assert np.array_equal(estimator.fit_X, bundle.dataset.train.X)
    assert np.array_equal(estimator.fit_y, bundle.dataset.train.y)
    assert not np.array_equal(estimator.fit_X[:2], bundle.dataset.val.X)
    assert estimator.kwargs["n_estimators"] == 1
    assert estimator.kwargs["random_state"] == 17
    assert estimator.kwargs["fit_mode"] == "fit_preprocessors"
    assert estimator.kwargs["memory_saving_mode"] is False
    assert estimator.kwargs["categorical_features_indices"] == []
    assert estimator.models_ == [bundle.model]

    projection = bundle.model.feature_positional_embedding_embeddings
    hooks_before = tuple(projection._forward_hooks.items())
    result = driver.evaluate("test")
    hooks_after = tuple(projection._forward_hooks.items())

    assert hooks_before == hooks_after == ()
    assert estimator.fit_calls == 1
    assert estimator.predict_calls == 5
    assert tuple(result.conditions) == ("full", "weight", "bias", "none")
    assert result.classes.tolist() == [0, 1]
    assert result.model_code_sha == bundle.model_sha
    assert result.loaded_checkpoint_sha256 == bundle.checkpoint_sha256
    assert result.seed == 17
    assert result.exact_full_native_verified is True
    assert result.policies_restored_verified is True
    assert np.array_equal(
        result.conditions["full"].probabilities,
        result.native_probabilities,
    )
    assert not np.array_equal(
        result.conditions["weight"].probabilities,
        result.conditions["bias"].probabilities,
    )
    assert not np.array_equal(
        result.conditions["none"].probabilities,
        result.conditions["full"].probabilities,
    )
    for condition in result.conditions.values():
        target = bundle.dataset.test.y
        predicted = result.classes[np.argmax(condition.probabilities, axis=1)]
        expected_loss = -np.log(
            condition.probabilities[np.arange(target.shape[0]), target]
        ).mean()
        assert condition.accuracy == pytest.approx(np.mean(predicted == target))
        assert condition.log_loss == pytest.approx(expected_loss)
        assert condition.n_samples == target.shape[0]
        assert condition.probabilities.flags.writeable is False
    assert result.classes.flags.writeable is False
    assert result.native_probabilities.flags.writeable is False


def test_policy_restores_after_official_predict_failure(fake_official_runtime):
    bundle = fake_official_runtime
    driver = _fit(bundle)
    estimator = bundle.classifier.instances[-1]
    estimator.fail_on_call = 2
    projection = bundle.model.feature_positional_embedding_embeddings

    with pytest.raises(RuntimeError, match="synthetic official prediction failure"):
        driver.evaluate("val")
    assert tuple(projection._forward_hooks.items()) == ()


def test_full_policy_must_be_exact_to_native(fake_official_runtime):
    bundle = fake_official_runtime
    driver = _fit(bundle)
    estimator = bundle.classifier.instances[-1]
    estimator.drift_on_call = 5

    with pytest.raises(RuntimeError, match="byte-exact to native"):
        driver.evaluate("val")
    assert tuple(
        bundle.model.feature_positional_embedding_embeddings._forward_hooks.items()
    ) == ()


def test_rejects_hash_mismatch_dirty_source_and_outside_import(
    fake_official_runtime, monkeypatch
):
    bundle = fake_official_runtime
    with pytest.raises(ValueError, match="expected SHA-256"):
        fit_official_tabpfn_v26_driver(
            bundle.dataset,
            bundle.checkpoint,
            checkpoint_sha256="0" * 64,
            model_sha=bundle.model_sha,
            expected_source_root=bundle.source_root.resolve(),
            seed=17,
        )
    with pytest.raises(ValueError, match="expected Git SHA"):
        fit_official_tabpfn_v26_driver(
            bundle.dataset,
            bundle.checkpoint,
            checkpoint_sha256=bundle.checkpoint_sha256,
            model_sha="0" * 40,
            expected_source_root=bundle.source_root.resolve(),
            seed=17,
        )

    bundle.source_file.write_text("# dirty source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        _fit(bundle)
    _git("checkout", "--", "official_runtime.py", cwd=bundle.source_root)

    class OutsideClassifier:
        pass

    outside_runtime = official_tabpfn._TabPFNRuntime(
        classifier_class=OutsideClassifier,
        model_specs_class=bundle.runtime.model_specs_class,
        load_model_criterion_config=bundle.runtime.load_model_criterion_config,
    )
    monkeypatch.setattr(
        official_tabpfn,
        "_load_tabpfn_runtime",
        lambda: outside_runtime,
    )
    with pytest.raises(RuntimeError, match="outside expected_source_root"):
        _fit(bundle)


def test_evaluation_rechecks_checkpoint_and_source(fake_official_runtime):
    bundle = fake_official_runtime
    driver = _fit(bundle)
    bundle.checkpoint.write_bytes(b"checkpoint changed after fit")
    with pytest.raises((RuntimeError, ValueError), match="changed|SHA-256"):
        driver.evaluate("test")

    bundle.checkpoint.write_bytes(b"exact local fake TabPFN v2.6 checkpoint")
    clean_driver = _fit(bundle)
    bundle.source_file.write_text("# source changed after fit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        clean_driver.evaluate("test")


def test_rejects_wrong_loaded_architecture(fake_official_runtime, monkeypatch):
    bundle = fake_official_runtime

    class WrongLoader:
        __module__ = bundle.module_name

        def __new__(cls, **_kwargs):
            return (
                [bundle.model],
                None,
                [SimpleNamespace(name="TabPFN-v2.5")],
                object(),
            )

    setattr(sys.modules[bundle.module_name], "WrongLoader", WrongLoader)
    wrong_runtime = official_tabpfn._TabPFNRuntime(
        classifier_class=bundle.runtime.classifier_class,
        model_specs_class=bundle.runtime.model_specs_class,
        load_model_criterion_config=WrongLoader,
    )
    monkeypatch.setattr(
        official_tabpfn,
        "_load_tabpfn_runtime",
        lambda: wrong_runtime,
    )
    with pytest.raises(RuntimeError, match="exact TabPFN-v2.6"):
        _fit(bundle)
