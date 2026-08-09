from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from pe_mechanism.adapters.base import ActivationRecord
from pe_mechanism.engine import (
    RopeCondition,
    default_rope_conditions,
    evaluate_tabicl_rope_conditions,
    prepare_tabicl_raw_batch,
)


class FakeModel(nn.Module):
    def __init__(self, max_classes: int = 2) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.child = nn.Linear(1, 1)


class FakeAdapter:
    def __init__(self, *, break_full: bool = False) -> None:
        self.policy = None
        self.capture_buffer = None
        self.break_full = break_full
        self.seen_policies = []

    @contextmanager
    def rope_policy(self, _model, **policy):
        previous = self.policy
        self.policy = dict(policy)
        try:
            yield None
        finally:
            self.policy = previous

    @contextmanager
    def capture(
        self,
        _model,
        *,
        sites,
        model_sha,
        checkpoint_sha,
        preprocessing_view_id,
        feature_group_map=None,
    ):
        previous = self.capture_buffer
        buffer = {}
        self.capture_buffer = (
            buffer,
            tuple(sites),
            model_sha,
            checkpoint_sha,
            preprocessing_view_id,
            feature_group_map,
        )
        try:
            yield buffer
        finally:
            self.capture_buffer = previous

    def predict(self, model, batch):
        model.eval()
        self.seen_policies.append(None if self.policy is None else dict(self.policy))
        test = batch["X"][:, batch.y_train.shape[1] :, 0]
        shift = torch.zeros_like(test)
        policy = self.policy
        if policy is not None:
            if not policy["rotate_queries"] and not policy["rotate_keys"]:
                shift = shift + 0.8
            elif policy["rotate_queries"] and not policy["rotate_keys"]:
                shift = shift + 0.4
            elif not policy["rotate_queries"] and policy["rotate_keys"]:
                shift = shift - 0.4
            elif policy["phase_strength"] != 1.0:
                shift = shift + policy["phase_strength"]
            elif policy["frequency_band"] == "high":
                shift = shift + 0.2
            elif policy["frequency_band"] == "low":
                shift = shift - 0.2
            elif policy["heads"] is not None:
                shift = shift + 0.1 * len(policy["heads"])
            elif self.break_full:
                shift = shift + 0.01
        logits = torch.stack((test + shift, -test - shift), dim=-1)
        probabilities = logits.softmax(dim=-1)
        if self.capture_buffer is not None:
            buffer, sites, model_sha, checkpoint_sha, view, group_map = (
                self.capture_buffer
            )
            for site in sites:
                tensor = test.detach().cpu().clone()
                buffer[site] = ActivationRecord(
                    tensor=tensor,
                    site=site,
                    axis_names=("table", "test_row"),
                    shape=tuple(tensor.shape),
                    model_sha=model_sha,
                    checkpoint_sha=checkpoint_sha,
                    preprocessing_view_id=view,
                    feature_group_map=group_map,
                )
        return probabilities


@pytest.fixture
def split():
    return (
        np.array([[0.0, 1.0], [2.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        np.array([9, 4, 9]),
        np.array([[1.5, 0.0], [-0.5, 3.0]], dtype=np.float64),
        np.array([4, 9]),
    )


def test_prepare_raw_batch_maps_labels_and_exposes_only_model_fields(split):
    batch = prepare_tabicl_raw_batch(*split, max_classes=2)

    assert batch.X.shape == (1, 5, 2)
    assert batch.X.dtype == torch.float32
    assert batch.y_train.shape == (1, 3)
    assert batch.y_train.tolist() == [[1, 0, 1]]
    assert batch.y_test.tolist() == [0, 1]
    assert batch.classes == (4, 9)
    assert tuple(batch) == ("X", "y_train", "return_logits")
    assert batch["return_logits"] is False
    assert batch.decode(np.array([1, 0])).tolist() == [9, 4]


def test_prepare_raw_batch_rejects_unseen_test_class_and_bad_shapes(split):
    X_train, y_train, X_test, _ = split
    with pytest.raises(ValueError, match="absent from the training"):
        prepare_tabicl_raw_batch(X_train, y_train, X_test, np.array([4, 12]))
    with pytest.raises(ValueError, match="same number of features"):
        prepare_tabicl_raw_batch(X_train, y_train, X_test[:, :1], np.array([4, 9]))
    with pytest.raises(ValueError, match="supports 1"):
        prepare_tabicl_raw_batch(*split, max_classes=1)


def test_default_roster_covers_all_requested_policy_families():
    conditions = default_rope_conditions(blocks=[1], head_selection=[0, 2])
    by_name = {condition.name: condition for condition in conditions}

    assert set(by_name) == {
        "full",
        "rope_off",
        "query_only",
        "key_only",
        "phase_strength_0.5",
        "frequency_high",
        "frequency_low",
        "heads_0_2",
    }
    assert by_name["full"].is_full_policy
    assert by_name["rope_off"].blocks == (1,)
    assert not by_name["query_only"].rotate_keys
    assert by_name["frequency_high"].frequency_band == "high"
    assert by_name["heads_0_2"].heads == (0, 2)


def test_evaluate_returns_probabilities_metrics_captures_and_restores_state(split):
    batch = prepare_tabicl_raw_batch(*split)
    adapter, model = FakeAdapter(), FakeModel()
    model.train()
    model.child.eval()
    original_states = tuple(module.training for module in model.modules())

    results = evaluate_tabicl_rope_conditions(
        adapter,
        model,
        batch,
        capture_sites=["probe"],
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
        preprocessing_view_id="raw-0",
    )

    assert set(results) == {
        "baseline",
        "full",
        "rope_off",
        "query_only",
        "key_only",
        "phase_strength_0.5",
        "frequency_high",
        "frequency_low",
        "heads_0",
    }
    baseline, full = results["baseline"], results["full"]
    assert baseline.probabilities.shape == (2, 2)
    assert np.array_equal(full.probabilities, baseline.probabilities)
    assert baseline.accuracy == np.mean(
        baseline.probabilities.argmax(axis=1) == batch.y_test
    )
    expected_loss = -np.log(
        baseline.probabilities[np.arange(batch.test_size), batch.y_test]
    ).mean()
    assert baseline.log_loss == pytest.approx(expected_loss)
    assert baseline.activations is not None
    assert isinstance(baseline.activations["probe"], ActivationRecord)
    assert baseline.activations["probe"].preprocessing_view_id == "raw-0"
    assert not np.array_equal(
        results["query_only"].probabilities, results["key_only"].probabilities
    )
    assert adapter.policy is None
    assert adapter.capture_buffer is None
    assert tuple(module.training for module in model.modules()) == original_states


def test_mapping_conditions_are_cli_friendly_and_class_limit_is_enforced(split):
    batch = prepare_tabicl_raw_batch(*split)
    adapter = FakeAdapter()
    result = evaluate_tabicl_rope_conditions(
        adapter,
        FakeModel(),
        batch,
        conditions=[
            {
                "name": "selected_band",
                "blocks": [0],
                "frequency_band": [0, 1],
                "heads": [1],
            }
        ],
    )
    assert tuple(result) == ("baseline", "selected_band")
    assert adapter.seen_policies[1]["frequency_band"] == (0, 1)

    too_many = FakeModel(max_classes=1)
    with pytest.raises(ValueError, match="model supports 1"):
        evaluate_tabicl_rope_conditions(FakeAdapter(), too_many, batch)


def test_misaligned_full_policy_fails_closed_and_restores_every_context(split):
    batch = prepare_tabicl_raw_batch(*split)
    adapter, model = FakeAdapter(break_full=True), FakeModel()
    model.train()
    model.child.eval()
    original_states = tuple(module.training for module in model.modules())

    with pytest.raises(RuntimeError, match="not exactly equal to baseline"):
        evaluate_tabicl_rope_conditions(
            adapter,
            model,
            batch,
            conditions=[RopeCondition("full-check")],
            capture_sites=["probe"],
        )

    assert adapter.policy is None
    assert adapter.capture_buffer is None
    assert tuple(module.training for module in model.modules()) == original_states


def test_condition_names_must_be_unique_and_baseline_is_reserved(split):
    batch = prepare_tabicl_raw_batch(*split)
    with pytest.raises(ValueError, match="unique"):
        evaluate_tabicl_rope_conditions(
            FakeAdapter(),
            FakeModel(),
            batch,
            conditions=[RopeCondition("same"), RopeCondition("same")],
        )
    with pytest.raises(ValueError, match="reserved"):
        evaluate_tabicl_rope_conditions(
            FakeAdapter(),
            FakeModel(),
            batch,
            conditions=[RopeCondition("baseline")],
        )
