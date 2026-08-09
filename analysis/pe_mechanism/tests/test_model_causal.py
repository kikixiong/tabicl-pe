from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from pe_mechanism.adapters.base import ActivationRecord, ActivationSite
from pe_mechanism.model_causal import run_model_causal_edits
from pe_mechanism.representation import DenseAutoencoder, MeanRMSNormalizer


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.site = nn.Identity()
        self.child = nn.Identity()

    def forward(self, batch):
        # Two site calls exercise replacement slicing for adapters whose capture
        # buffers merge repeated hooks along the leading semantic axis.
        chunks = torch.chunk(batch["x"], 2, dim=0)
        activation = torch.cat([self.site(chunk) for chunk in chunks], dim=0)
        score = 2.0 * activation[:, 0] + activation[:, 1]
        return torch.stack((score, -score), dim=-1).softmax(dim=-1)


class FakeAdapter:
    model_family = "fake"

    def __init__(self, *, fail_during_intervention: bool = False) -> None:
        self.fail_during_intervention = fail_during_intervention
        self.intervention_active = False

    def list_sites(self, _model):
        return {"site": ActivationSite("site", ("sample", "embedding"))}

    @contextmanager
    def capture(
        self,
        model,
        *,
        sites,
        model_sha,
        checkpoint_sha,
        preprocessing_view_id,
        feature_group_map=None,
    ):
        assert tuple(sites) == ("site",)
        buffer = {}

        def hook(_module, _args, output):
            tensor = output.detach().cpu().clone()
            previous = buffer.get("site")
            if previous is not None:
                tensor = torch.cat((previous.tensor, tensor), dim=0)
            buffer["site"] = ActivationRecord(
                tensor=tensor,
                site="site",
                axis_names=("sample", "embedding"),
                shape=tuple(tensor.shape),
                model_sha=model_sha,
                checkpoint_sha=checkpoint_sha,
                preprocessing_view_id=preprocessing_view_id,
                feature_group_map=feature_group_map,
            )

        handle = model.site.register_forward_hook(hook)
        try:
            yield buffer
        finally:
            handle.remove()

    @contextmanager
    def intervene(self, model, interventions):
        assert tuple(interventions) == ("site",)

        def hook(_module, _args, output):
            record = ActivationRecord(
                tensor=output,
                site="site",
                axis_names=("sample", "embedding"),
                shape=tuple(output.shape),
                model_sha="",
                checkpoint_sha="",
                preprocessing_view_id="",
            )
            return interventions["site"](record)

        handle = model.site.register_forward_hook(hook)
        self.intervention_active = True
        try:
            yield None
        finally:
            self.intervention_active = False
            handle.remove()

    def predict(self, model, batch):
        model.eval()
        output = model(batch)
        if self.intervention_active and self.fail_during_intervention:
            raise RuntimeError("injected prediction failure")
        return output


def identity_overcomplete_autoencoder():
    model = DenseAutoencoder(2, 4, activation="linear")
    with torch.no_grad():
        model.encoder.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        )
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(
            torch.tensor([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]])
        )
        model.decoder.bias.zero_()
    return model


QUALIFIED = {
    "metric_split": "validation",
    "activation_fidelity_passed": True,
    "explained_variance": 1.0,
}


@pytest.fixture
def batches():
    primary_x = torch.tensor([[2.0, 0.2], [-2.0, -0.2], [1.0, 0.1], [-1.0, -0.1]])
    paired_x = torch.tensor([[-1.5, -0.4], [1.5, 0.4], [-0.5, -0.3], [0.5, 0.3]])
    return (
        {"x": primary_x},
        np.array([0, 1, 0, 1]),
        {"x": paired_x},
        np.array([1, 0, 1, 0]),
    )


def test_activation_edits_change_predictions_and_record_controls_and_rescues(batches):
    batch, targets, paired_batch, paired_targets = batches
    model, adapter = FakeModel(), FakeAdapter()
    autoencoder = identity_overcomplete_autoencoder()
    normalizer = MeanRMSNormalizer(torch.zeros(2), torch.ones(2))
    model.train()
    model.child.eval()
    autoencoder.train()
    model_states = tuple(module.training for module in model.modules())
    representation_states = tuple(module.training for module in autoencoder.modules())

    result = run_model_causal_edits(
        adapter,
        model,
        batch,
        site="site",
        autoencoder=autoencoder,
        normalizer=normalizer,
        target_features=[0],
        dataset_id="toy-dataset",
        sample_ids=list(range(4)),
        representation_qualification=QUALIFIED,
        targets=targets,
        latent_baseline="zero",
        paired_batch=paired_batch,
        paired_dataset_id="toy-dataset",
        paired_sample_ids=list(range(4)),
        paired_targets=paired_targets,
        random_seed=5,
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
    )

    assert result.baseline_activation.shape == (4, 2)
    assert result.no_op_reconstruction_mse == pytest.approx(0.0)
    assert set(result.conditions) == {
        "no_op_reconstruction",
        "target_baseline_edit",
        "matched_random_edit",
        "rescue",
        "paired_no_op_reconstruction",
        "paired_activation_edit",
        "paired_reverse",
        "paired_rescue",
    }
    target = result.conditions["target_baseline_edit"]
    assert not np.array_equal(
        target.prediction.probabilities,
        result.conditions["no_op_reconstruction"].prediction.probabilities,
    )
    assert target.prediction.true_class_log_loss.shape == (4,)
    assert target.delta_log_loss_vs_model_baseline.shape == (4,)
    assert target.delta_log_loss_vs_reconstruction.shape == (4,)
    random_control = result.conditions["matched_random_edit"]
    assert random_control.control_features == result.matched_control_features
    assert len(random_control.control_features) == 1
    assert random_control.control_features != (0,)
    assert np.array_equal(
        result.conditions["rescue"].prediction.probabilities,
        result.conditions["no_op_reconstruction"].prediction.probabilities,
    )
    assert result.conditions["paired_reverse"].reference_scope == "paired"
    assert not np.array_equal(
        result.conditions["paired_reverse"].prediction.probabilities,
        result.conditions["paired_no_op_reconstruction"].prediction.probabilities,
    )
    assert np.array_equal(
        result.conditions["paired_rescue"].prediction.probabilities,
        result.conditions["paired_no_op_reconstruction"].prediction.probabilities,
    )
    assert len(model.site._forward_hooks) == 0
    assert not adapter.intervention_active
    assert tuple(module.training for module in model.modules()) == model_states
    assert (
        tuple(module.training for module in autoencoder.modules())
        == representation_states
    )


def test_noop_reconstruction_threshold_is_enforced_and_hooks_restore(batches):
    batch, targets, _, _ = batches
    autoencoder = identity_overcomplete_autoencoder()
    with torch.no_grad():
        autoencoder.decoder.bias.fill_(0.25)
    model, adapter = FakeModel(), FakeAdapter()

    with pytest.raises(RuntimeError, match="reconstruction MSE"):
        run_model_causal_edits(
            adapter,
            model,
            batch,
            site="site",
            autoencoder=autoencoder,
            normalizer=MeanRMSNormalizer(torch.zeros(2), torch.ones(2)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification=QUALIFIED,
            targets=targets,
            max_no_op_reconstruction_mse=0.001,
        )

    assert len(model.site._forward_hooks) == 0


def test_strict_shape_and_finite_checks_fail_closed(batches):
    batch, targets, _, _ = batches
    model, adapter = FakeModel(), FakeAdapter()
    wrong_width = DenseAutoencoder(3, 4, activation="linear")
    with pytest.raises(ValueError, match="input_dim"):
        run_model_causal_edits(
            adapter,
            model,
            batch,
            site="site",
            autoencoder=wrong_width,
            normalizer=MeanRMSNormalizer(torch.zeros(3), torch.ones(3)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification=QUALIFIED,
            targets=targets,
        )

    nonfinite = identity_overcomplete_autoencoder()
    with torch.no_grad():
        nonfinite.decoder.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        run_model_causal_edits(
            adapter,
            model,
            batch,
            site="site",
            autoencoder=nonfinite,
            normalizer=MeanRMSNormalizer(torch.zeros(2), torch.ones(2)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification=QUALIFIED,
            targets=targets,
        )
    assert len(model.site._forward_hooks) == 0


def test_paired_activation_shape_must_match_primary(batches):
    batch, targets, _, _ = batches
    short_pair = {
        "x": torch.tensor(
            [[1.0, 0.0, 0.2], [-1.0, 0.0, 0.2], [0.5, 0.0, 0.2], [-0.5, 0.0, 0.2]]
        )
    }
    model = FakeModel()
    with pytest.raises(ValueError, match="exactly match"):
        run_model_causal_edits(
            FakeAdapter(),
            model,
            batch,
            site="site",
            autoencoder=identity_overcomplete_autoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(2), torch.ones(2)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification=QUALIFIED,
            targets=targets,
            paired_batch=short_pair,
            paired_dataset_id="toy-dataset",
            paired_sample_ids=list(range(4)),
            paired_targets=targets,
        )
    assert len(model.site._forward_hooks) == 0


def test_prediction_failure_removes_intervention_hook_and_restores_modes(batches):
    batch, targets, _, _ = batches
    model = FakeModel().train()
    original_states = tuple(module.training for module in model.modules())
    adapter = FakeAdapter(fail_during_intervention=True)

    with pytest.raises(RuntimeError, match="injected prediction failure"):
        run_model_causal_edits(
            adapter,
            model,
            batch,
            site="site",
            autoencoder=identity_overcomplete_autoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(2), torch.ones(2)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification=QUALIFIED,
            targets=targets,
        )

    assert len(model.site._forward_hooks) == 0
    assert not adapter.intervention_active
    assert tuple(module.training for module in model.modules()) == original_states


def test_unqualified_representation_is_rejected_before_model_execution(batches):
    batch, targets, _, _ = batches
    model = FakeModel()
    with pytest.raises(RuntimeError, match="qualification gate"):
        run_model_causal_edits(
            FakeAdapter(),
            model,
            batch,
            site="site",
            autoencoder=identity_overcomplete_autoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(2), torch.ones(2)),
            target_features=[0],
            dataset_id="toy-dataset",
            sample_ids=list(range(4)),
            representation_qualification={
                "metric_split": "training",
                "activation_fidelity_passed": True,
            },
            targets=targets,
        )
    assert len(model.site._forward_hooks) == 0
