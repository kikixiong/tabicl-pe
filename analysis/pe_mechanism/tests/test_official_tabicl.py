from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import pe_mechanism.official_tabicl as official_tabicl
from pe_mechanism.adapters.tabicl import TabICLAdapter, same_feature_group_map
from pe_mechanism.official_tabicl import (
    OfficialTabICLDriver,
    fit_official_tabicl_driver,
    fit_official_talent_driver,
    load_raw_talent_splits,
    official_inference_contract_sha256,
)
from pe_mechanism.provenance import GitEvidence


def test_official_inference_contract_is_canonical_and_split_independent() -> None:
    first = official_inference_contract_sha256(
        "a" * 40,
        {"random_state": 42, "norm_methods": ["none", "power"]},
    )
    reordered = official_inference_contract_sha256(
        "a" * 40,
        {"norm_methods": ["none", "power"], "random_state": 42},
    )
    changed = official_inference_contract_sha256(
        "a" * 40,
        {"random_state": 43, "norm_methods": ["none", "power"]},
    )

    assert first == reordered
    assert first != changed
    assert len(first) == 64
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        official_inference_contract_sha256(
            "a" * 40, {"random_state": 42, "temperature": float("nan")}
        )


class FakeColumnEmbedder(nn.Module):
    feature_group = "same"
    feature_group_size = 2

    def forward(self, X, **_kwargs):
        return X.unsqueeze(-1)


class FakeRowInteractor(nn.Module):
    def __init__(self, *, temporary: bool = False) -> None:
        super().__init__()
        self.tf_row = nn.Module()
        self.tf_row.blocks = nn.ModuleList()
        self.identity_mode = "temporary" if temporary else "none"
        self._identity_generator = torch.Generator(device="cpu")
        self._identity_generator.manual_seed(123)

    def forward(self, embeddings, **_kwargs):
        output = embeddings.mean(dim=2)
        if self.identity_mode == "temporary":
            random_offset = torch.randint(
                0, 1000, (1,), generator=self._identity_generator
            ).to(output.dtype)
            output = output + random_offset / 1000.0
        return output


class FakeICLPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tf_icl = nn.Module()
        self.tf_icl.blocks = nn.ModuleList()

    def forward(self, representations, y_train, **_kwargs):
        test = representations[:, y_train.shape[1] :, 0]
        return torch.stack((test, -0.7 * test), dim=-1)


class FakeRawTabICL(nn.Module):
    max_classes = 2

    def __init__(
        self, *, repeat_row_site: bool = False, temporary: bool = False
    ) -> None:
        super().__init__()
        self.col_embedder = FakeColumnEmbedder()
        self.row_interactor = FakeRowInteractor(temporary=temporary)
        self.icl_predictor = FakeICLPredictor()
        self.row_identity_mode = "temporary" if temporary else "none"
        self.repeat_row_site = repeat_row_site
        self._cache = None

    def forward(
        self,
        X,
        y_train,
        d=None,
        embed_with_test=False,
        feature_shuffles=None,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
        row_identity_permutation=None,
    ):
        del (
            d,
            embed_with_test,
            feature_shuffles,
            softmax_temperature,
            inference_config,
            row_identity_permutation,
        )
        embeddings = self.col_embedder(X)
        representations = self.row_interactor(embeddings)
        if self.repeat_row_site:
            self.row_interactor(embeddings)
        output = self.icl_predictor(representations, y_train)
        return output if return_logits else torch.softmax(output, dim=-1)


class FakeLabelEncoder:
    @staticmethod
    def transform(values):
        result = np.asarray(values)
        if not np.isin(result, (0, 1)).all():
            raise ValueError("unknown label")
        return result.astype(np.int64)


class FakeOfficialClassifier:
    def __init__(
        self, *, repeat_row_site: bool = False, temporary: bool = False
    ) -> None:
        self.model_ = FakeRawTabICL(
            repeat_row_site=repeat_row_site, temporary=temporary
        ).eval()
        self.classes_ = np.array([0, 1])
        self.n_classes_ = 2
        self.y_encoder_ = FakeLabelEncoder()
        self.kv_cache = False
        self.model_kv_cache_ = None
        self.support_many_classes = False
        self.batch_size = 2
        self.average_logits = True
        self.softmax_temperature = 0.9
        self.inference_config_ = object()
        self._X_train = np.array(
            [
                [0.2, 0.3, 0.7, 1.1],
                [1.0, -0.4, 0.5, 0.2],
                [-0.2, 0.8, 0.4, -0.1],
                [0.6, 0.9, -0.5, 0.3],
            ],
            dtype=np.float32,
        )
        self._y_train = np.array([0, 1, 0, 1], dtype=np.int64)
        features = OrderedDict(
            [
                (
                    "none",
                    [
                        [0, 1, 2, 3],
                        [1, 2, 3, 0],
                        [3, 2, 1, 0],
                    ],
                ),
                ("power", [[2, 0, 3, 1], [1, 3, 0, 2]]),
            ]
        )
        classes = OrderedDict(
            [
                ("none", [[0, 1], [1, 0], [0, 1]]),
                ("power", [[1, 0], [0, 1]]),
            ]
        )
        configs = OrderedDict(
            (
                norm,
                list(zip(shuffles, classes[norm], strict=True)),
            )
            for norm, shuffles in features.items()
        )
        self.ensemble_generator_ = SimpleNamespace(
            ensemble_configs_=configs,
            feature_shuffles_=features,
            class_shuffles_=classes,
            n_features_in_=4,
            y_=self._y_train,
        )

    def _batch_forward(self, Xs, ys, feature_shuffles):
        batch_size = self.batch_size or Xs.shape[0]
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
        X_chunks = np.array_split(Xs, n_batches)
        y_chunks = np.array_split(ys, n_batches)
        shuffle_chunks = np.array_split(feature_shuffles, n_batches)
        outputs = []
        for X_batch, y_batch, shuffle_batch in zip(
            X_chunks, y_chunks, shuffle_chunks, strict=True
        ):
            with torch.no_grad():
                output = self.model_(
                    X=torch.from_numpy(X_batch).float(),
                    y_train=torch.from_numpy(y_batch).float(),
                    feature_shuffles=shuffle_batch.tolist(),
                    return_logits=self.average_logits,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(output.float().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        outputs = []
        for norm, feature_shuffles in (
            self.ensemble_generator_.feature_shuffles_.items()
        ):
            scale = 1.0 if norm == "none" else 0.5
            combined = np.concatenate((self._X_train, X), axis=0) * scale
            Xs = np.stack(
                [combined[:, shuffle] for shuffle in feature_shuffles], axis=0
            )
            class_shuffles = self.ensemble_generator_.class_shuffles_[norm]
            ys = np.stack(
                [np.asarray(shuffle)[self._y_train] for shuffle in class_shuffles],
                axis=0,
            )
            outputs.append(self._batch_forward(Xs, ys, feature_shuffles))
        raw_outputs = np.concatenate(outputs, axis=0)
        all_class_shuffles = [
            shuffle
            for shuffles in self.ensemble_generator_.class_shuffles_.values()
            for shuffle in shuffles
        ]
        average = np.zeros_like(raw_outputs[0])
        for output, shuffle in zip(
            raw_outputs, all_class_shuffles, strict=True
        ):
            average += output[..., shuffle]
        average /= len(all_class_shuffles)
        shifted = average / self.softmax_temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


@pytest.fixture
def prediction_table():
    return np.array(
        [[0.1, 0.6, -0.3, 0.8], [0.9, -0.2, 0.4, 0.3]],
        dtype=np.float32,
    )


def make_driver(classifier):
    return OfficialTabICLDriver(
        classifier,
        adapter=TabICLAdapter(),
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
    )


def test_capture_preserves_exact_official_prediction_and_batch_schedule(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    direct = classifier.predict_proba(prediction_table.copy())
    driver = make_driver(classifier)

    result = driver.predict_proba(
        prediction_table,
        y=np.array([0, 1]),
        sites=("row_interactor", "icl_predictor"),
    )

    assert np.array_equal(result.probabilities, direct)
    assert np.array_equal(result.baseline_probabilities, direct)
    assert result.exact_baseline_verified
    assert [call.metadata.norm_method for call in result.forward_calls] == [
        "none",
        "none",
        "power",
    ]
    # Official np.array_split balances three views into 2+1, rather than 2+1
    # by coincidence; this also protects the less-obvious 10 -> 5+5 behavior.
    assert [call.metadata.raw_input_shape[0] for call in result.forward_calls] == [
        2,
        1,
        2,
    ]
    assert result.forward_calls[0].metadata.ensemble_indices == (0, 1)
    assert result.forward_calls[1].metadata.ensemble_indices == (2,)
    assert result.forward_calls[2].metadata.ensemble_indices == (3, 4)
    local_map = same_feature_group_map(4, 2)
    assert result.forward_calls[0].activations[
        "row_interactor"
    ].feature_group_map == local_map
    assert (
        result.forward_calls[0].metadata.post_filter_feature_group_maps[0]
        == local_map
    )
    assert (
        result.forward_calls[0].metadata.post_filter_feature_group_maps[1][0]
        == (2, 3)
    )
    assert result.metrics is not None
    assert result.metrics.n_samples == 2
    assert "forward" not in vars(classifier.model_)
    assert not any(
        module._forward_hooks for module in classifier.model_.modules()
    )


def test_intervention_changes_official_ensemble_and_capture_sees_edit(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    driver = make_driver(classifier)

    def raise_first_logit(record, metadata):
        assert metadata.raw_input_shape[0] == record.shape[0]
        edited = record.tensor.clone()
        edited[..., 0] += 2.0
        return edited

    result = driver.predict_proba(
        prediction_table,
        sites=("icl_predictor",),
        interventions={"icl_predictor": raise_first_logit},
    )

    assert not np.array_equal(
        result.probabilities, result.baseline_probabilities
    )
    assert not result.exact_baseline_verified
    for call in result.forward_calls:
        captured = call.activations["icl_predictor"].tensor
        assert captured.shape[0] == call.metadata.raw_input_shape[0]
    assert "forward" not in vars(classifier.model_)
    assert not any(
        module._forward_hooks for module in classifier.model_.modules()
    )


def test_temporary_identity_rng_is_replayed_once(prediction_table):
    classifier = FakeOfficialClassifier(temporary=True)
    initial = classifier.model_.row_interactor._identity_generator.get_state().clone()
    direct = classifier.predict_proba(prediction_table.copy())
    expected_final = (
        classifier.model_.row_interactor._identity_generator.get_state().clone()
    )
    classifier.model_.row_interactor._identity_generator.set_state(initial)

    result = make_driver(classifier).predict_proba(
        prediction_table, sites=("row_interactor",)
    )

    assert np.array_equal(result.probabilities, direct)
    assert torch.equal(
        classifier.model_.row_interactor._identity_generator.get_state(),
        expected_final,
    )


def test_direct_prediction_failure_rolls_back_temporary_identity_rng(
    prediction_table,
):
    class FailingClassifier(FakeOfficialClassifier):
        def predict_proba(self, X):
            super().predict_proba(X)
            raise RuntimeError("failure after raw inference")

    classifier = FailingClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="failure after raw inference"):
        make_driver(classifier).predict_proba(prediction_table)

    assert torch.equal(generator.get_state(), initial)


def test_ensemble_config_feature_permutation_drift_fails_closed(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    _, class_shuffle = classifier.ensemble_generator_.ensemble_configs_["none"][0]
    classifier.ensemble_generator_.ensemble_configs_["none"][0] = (
        [1, 0, 2, 3],
        class_shuffle,
    )

    with pytest.raises(RuntimeError, match="feature permutations differ"):
        make_driver(classifier).predict_proba(
            prediction_table, sites=("row_interactor",)
        )


def test_repeated_site_call_fails_and_restores_hooks_and_forward(prediction_table):
    classifier = FakeOfficialClassifier(repeat_row_site=True)
    driver = make_driver(classifier)

    with pytest.raises(RuntimeError, match="more than once"):
        driver.predict_proba(prediction_table, sites=("row_interactor",))

    assert "forward" not in vars(classifier.model_)
    assert not any(
        module._forward_hooks for module in classifier.model_.modules()
    )


def test_exclusive_guard_covers_baseline_and_releases_afterward(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    driver = make_driver(classifier)
    direct_predict = classifier.predict_proba
    entered_baseline = threading.Event()
    release_baseline = threading.Event()
    first_call_lock = threading.Lock()
    first_call = True

    def blocking_predict(X):
        nonlocal first_call
        with first_call_lock:
            should_block = first_call
            first_call = False
        if should_block:
            entered_baseline.set()
            assert release_baseline.wait(timeout=5)
        return direct_predict(X)

    classifier.predict_proba = blocking_predict
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            driver.predict_proba,
            prediction_table.copy(),
            sites=("row_interactor",),
        )
        assert entered_baseline.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="already active"):
                driver.predict_proba(
                    prediction_table.copy(), sites=("row_interactor",)
                )
        finally:
            release_baseline.set()
        assert future.result(timeout=10).exact_baseline_verified

    assert driver.predict_proba(prediction_table.copy()).probabilities.shape == (
        2,
        2,
    )


def test_reentrant_intervention_fails_and_restores_transaction(prediction_table):
    classifier = FakeOfficialClassifier()
    driver = make_driver(classifier)

    def recurse(record, metadata):
        del record, metadata
        driver.predict_proba(prediction_table.copy())

    with pytest.raises(RuntimeError, match="already active"):
        driver.predict_proba(
            prediction_table,
            interventions={"icl_predictor": recurse},
        )

    assert "forward" not in vars(classifier.model_)
    assert not any(
        module._forward_hooks for module in classifier.model_.modules()
    )
    assert driver.predict_proba(prediction_table.copy()).probabilities.shape == (
        2,
        2,
    )


def test_paired_session_rejects_reentry_and_cross_thread_use(prediction_table):
    classifier = FakeOfficialClassifier()
    driver = make_driver(classifier)
    with driver.paired_session() as session:

        def recurse(record, metadata):
            del record, metadata
            session.predict_proba(prediction_table.copy())

        with pytest.raises(RuntimeError, match="not reentrant"):
            session.predict_proba(
                prediction_table,
                interventions={"icl_predictor": recurse},
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                session.predict_proba, prediction_table.copy()
            )
            with pytest.raises(RuntimeError, match="cross threads"):
                future.result(timeout=5)
        assert session.predict_proba(
            prediction_table.copy()
        ).probabilities.shape == (2, 2)

    with pytest.raises(RuntimeError, match="no longer active"):
        session.predict_proba(prediction_table.copy())


@pytest.mark.parametrize("cache_owner", ["classifier", "raw"])
def test_cache_paths_fail_closed(cache_owner):
    classifier = FakeOfficialClassifier()
    if cache_owner == "classifier":
        classifier.model_kv_cache_ = object()
    else:
        classifier.model_._cache = object()
    with pytest.raises(ValueError, match="cache"):
        make_driver(classifier)


def test_all_missing_feature_mask_fails_before_official_prediction(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    prediction_table[:, 2] = np.nan
    with pytest.raises(ValueError, match="all-missing"):
        make_driver(classifier).predict_proba(
            prediction_table, sites=("row_interactor",)
        )


def test_class_shuffle_label_drift_fails_closed(prediction_table):
    class LabelDriftClassifier(FakeOfficialClassifier):
        def _batch_forward(self, Xs, ys, feature_shuffles):
            corrupted = ys.copy()
            corrupted[0] = 1 - corrupted[0]
            return super()._batch_forward(Xs, corrupted, feature_shuffles)

    with pytest.raises(RuntimeError, match="y_train differs"):
        make_driver(LabelDriftClassifier()).predict_proba(
            prediction_table, sites=("row_interactor",)
        )


def test_invalid_metric_labels_do_not_advance_temporary_rng(prediction_table):
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(ValueError, match="unknown label"):
        make_driver(classifier).predict_proba(
            prediction_table,
            y=np.array([0, 9]),
            sites=("row_interactor",),
        )

    assert torch.equal(generator.get_state(), initial)


def test_n_jobs_exception_restores_torch_thread_count(prediction_table):
    classifier = FakeOfficialClassifier(repeat_row_site=True)
    classifier.n_jobs = 1
    original_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="more than once"):
        make_driver(classifier).predict_proba(
            prediction_table, sites=("row_interactor",)
        )
    assert torch.get_num_threads() == original_threads


def test_ungrouped_column_optimization_is_outside_aligned_scope(
    prediction_table,
):
    classifier = FakeOfficialClassifier()
    classifier.model_.col_embedder.feature_group = False
    with pytest.raises(ValueError, match="unsupported.*feature_group"):
        make_driver(classifier).predict_proba(
            prediction_table, sites=("row_interactor",)
        )


def _write_raw_talent(root: Path) -> None:
    root.mkdir()
    (root / "info.json").write_text(
        json.dumps({"task_type": "binclass", "num_classes": 2}),
        encoding="utf-8",
    )
    numeric = {
        "train": np.array([[10.0, np.nan], [20.0, 4.0], [40.0, 8.0]]),
        "val": np.array([[30.0, 6.0]]),
        "test": np.array([[50.0, 10.0]]),
    }
    categorical = {
        "train": np.array([["red"], ["blue"], [None]], dtype=object),
        "val": np.array([["red"]], dtype=object),
        "test": np.array([["unseen"]], dtype=object),
    }
    labels = {
        "train": np.array([0, 1, 0]),
        "val": np.array([1]),
        "test": np.array([0]),
    }
    for split in ("train", "val", "test"):
        np.save(root / f"N_{split}.npy", numeric[split])
        np.save(root / f"C_{split}.npy", categorical[split])
        np.save(root / f"y_{split}.npy", labels[split])


def test_raw_talent_loader_preserves_values_types_and_missingness(tmp_path):
    root = tmp_path / "raw_talent"
    _write_raw_talent(root)

    with pytest.raises(ValueError, match="trusted_pickle"):
        load_raw_talent_splits(root)
    dataset = load_raw_talent_splits(root, trusted_pickle=True)

    assert dataset.n_numeric_features == 2
    assert dataset.n_categorical_features == 1
    assert list(dataset.train.X.columns) == [
        "numerical_0",
        "numerical_1",
        "categorical_0",
    ]
    assert dataset.train.X["numerical_0"].tolist() == [10.0, 20.0, 40.0]
    assert np.isnan(dataset.train.X.loc[0, "numerical_1"])
    assert dataset.train.X.loc[0, "categorical_0"] == "red"
    assert dataset.train.X.loc[2, "categorical_0"] is None
    assert dataset.test.X.loc[0, "categorical_0"] == "unseen"
    assert dataset.info_sha256 == hashlib.sha256(
        (root / "info.json").read_bytes()
    ).hexdigest()
    assert set(dataset.input_sha256) == {
        "C_test.npy",
        "C_train.npy",
        "C_val.npy",
        "N_test.npy",
        "N_train.npy",
        "N_val.npy",
        "info.json",
        "y_test.npy",
        "y_train.npy",
        "y_val.npy",
    }


def test_talent_context_choice_is_explicit(tmp_path, monkeypatch):
    root = tmp_path / "raw_talent"
    _write_raw_talent(root)
    dataset = load_raw_talent_splits(root, trusted_pickle=True)
    received = []

    def fake_fit(checkpoint, X_train, y_train, **kwargs):
        received.append((checkpoint, X_train.copy(), y_train.copy(), kwargs))
        return kwargs["fit_context"]

    monkeypatch.setattr(
        official_tabicl, "fit_official_tabicl_driver", fake_fit
    )
    train_result = fit_official_talent_driver(
        dataset,
        "unused.ckpt",
        context_split="train",
        model_sha="c" * 40,
    )
    combined_result = fit_official_talent_driver(
        dataset,
        "unused.ckpt",
        context_split="train+val",
        model_sha="c" * 40,
    )

    assert train_result == "talent-train"
    assert combined_result == "talent-train+val"
    assert len(received[0][1]) == 3
    assert len(received[1][1]) == 4
    assert received[0][1].loc[0, "numerical_0"] == 10.0
    with pytest.raises(ValueError, match="context_split"):
        fit_official_talent_driver(
            dataset,
            "unused.ckpt",
            context_split="automatic",
            model_sha="c" * 40,
        )


def test_factory_locks_official_options_and_passes_raw_dataframe(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"local checkpoint bytes")
    pandas = pytest.importorskip("pandas")
    X_train = pandas.DataFrame(
        {"number": [1.0, 2.0], "category": ["a", "b"]}
    )
    y_train = np.array([0, 1])

    class RecordingClassifier(FakeOfficialClassifier):
        instance = None

        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.model_path_ = kwargs["model_path"]
            self.fit_X = None
            self.fit_y = None
            RecordingClassifier.instance = self

        def fit(self, X, y):
            self.fit_X = X
            self.fit_y = y
            return self

    import tabicl

    monkeypatch.setattr(tabicl, "TabICLClassifier", RecordingClassifier)
    driver = fit_official_tabicl_driver(
        checkpoint,
        X_train,
        y_train,
        model_sha="d" * 40,
        estimator_options={"n_estimators": 1, "norm_methods": "none"},
    )

    recorded = RecordingClassifier.instance
    assert recorded is not None
    assert recorded.fit_X is X_train
    assert recorded.fit_y is y_train
    assert recorded.kwargs["allow_auto_download"] is False
    assert recorded.kwargs["kv_cache"] is False
    assert recorded.kwargs["support_many_classes"] is False
    assert recorded.kwargs["device"] == "cpu"
    assert recorded.kwargs["n_estimators"] == 1
    assert driver.checkpoint_sha == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="cannot override"):
        fit_official_tabicl_driver(
            checkpoint,
            X_train,
            y_train,
            model_sha="d" * 40,
            estimator_options={"kv_cache": True},
        )


def test_factory_verifies_source_git_and_checkpoint_stability(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"stable checkpoint")
    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([0, 1])

    class RecordingClassifier(FakeOfficialClassifier):
        def __init__(self, **kwargs):
            super().__init__()
            self.model_path_ = kwargs["model_path"]

        def fit(self, X, y):
            del X, y
            return self

    import tabicl

    monkeypatch.setattr(tabicl, "TabICLClassifier", RecordingClassifier)
    expected_root = Path(__file__).parents[3].resolve()
    evidence_checks = []

    def fake_verify_git_tree(root, *, expected_sha):
        assert Path(root).resolve() == expected_root
        assert expected_sha == "e" * 40
        evidence_checks.append("measured")
        return GitEvidence(
            head_sha="e" * 40,
            evidence_level="strict",
            legacy_reasons=(),
            root=expected_root,
            status_sha256="0" * 64,
        )

    monkeypatch.setattr(
        official_tabicl, "verify_git_tree", fake_verify_git_tree
    )
    monkeypatch.setattr(
        GitEvidence,
        "assert_unchanged",
        lambda self: evidence_checks.append("unchanged"),
    )
    driver = fit_official_tabicl_driver(
        checkpoint,
        X_train,
        y_train,
        model_sha="e" * 40,
        expected_source_root=expected_root,
    )

    assert evidence_checks == ["measured", "unchanged"]
    assert driver.source_evidence_level == "strict"
    assert driver.checkpoint_sha == hashlib.sha256(
        b"stable checkpoint"
    ).hexdigest()


def test_factory_rejects_checkpoint_mutation_and_symlink(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"initial checkpoint")
    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([0, 1])

    class MutatingClassifier(FakeOfficialClassifier):
        def __init__(self, **kwargs):
            super().__init__()
            self.model_path_ = kwargs["model_path"]

        def fit(self, X, y):
            del X, y
            Path(self.model_path_).write_bytes(b"replacement checkpoint")
            return self

    import tabicl

    monkeypatch.setattr(tabicl, "TabICLClassifier", MutatingClassifier)
    with pytest.raises(ValueError, match="does not match"):
        fit_official_tabicl_driver(
            checkpoint,
            X_train,
            y_train,
            model_sha="f" * 40,
        )

    stable = tmp_path / "stable.ckpt"
    stable.write_bytes(b"stable")
    alias = tmp_path / "alias.ckpt"
    alias.symlink_to(stable)
    with pytest.raises(ValueError, match="symlink"):
        fit_official_tabicl_driver(
            alias,
            X_train,
            y_train,
            model_sha="f" * 40,
        )
