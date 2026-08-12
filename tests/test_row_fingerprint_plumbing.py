from __future__ import annotations

import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tabicl._model.kv_cache import TabICLCache
from tabicl._model.tabicl import TabICL
from tabicl._model.inference_config import InferenceConfig
from tabicl.train._train_config import build_parser
from tabicl.train._run import Trainer


def _fingerprint_helper(num_cls: int = 2) -> TabICL:
    # Exercise the pure aggregation helper without constructing the full model;
    # RowInteraction has its own focused tests for learned Q/K injection.
    model = TabICL.__new__(TabICL)
    nn.Module.__init__(model)
    model.row_fingerprint = True
    model.row_num_cls = num_cls
    return model


def test_row_fingerprint_uses_only_training_rows_and_zeros_cls_slots():
    model = _fingerprint_helper()
    embeddings = torch.randn(2, 7, 5, 4, requires_grad=True)

    actual = model._compute_row_fingerprint(embeddings, train_size=3)
    expected = embeddings[:, :3].mean(dim=1)

    torch.testing.assert_close(actual[:, :2], torch.zeros_like(actual[:, :2]))
    torch.testing.assert_close(actual[:, 2:], expected[:, 2:])

    actual.sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, :3, 2:]) > 0
    assert torch.count_nonzero(embeddings.grad[:, 3:]) == 0


def test_row_fingerprint_is_invariant_to_test_embedding_changes():
    model = _fingerprint_helper()
    embeddings = torch.randn(1, 6, 5, 4)
    changed_test = embeddings.clone()
    changed_test[:, 3:] = torch.randn_like(changed_test[:, 3:]) * 1000

    expected = model._compute_row_fingerprint(embeddings, train_size=3)
    actual = model._compute_row_fingerprint(changed_test, train_size=3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_row_fingerprint_rejects_empty_or_out_of_range_training_prefix():
    model = _fingerprint_helper()
    embeddings = torch.randn(1, 3, 5, 4)

    with pytest.raises(ValueError, match="non-empty training prefix"):
        model._compute_row_fingerprint(embeddings, train_size=0)
    with pytest.raises(ValueError, match="non-empty training prefix"):
        model._compute_row_fingerprint(embeddings, train_size=4)


def test_row_fingerprint_fails_closed_when_column_embedding_can_see_test_rows():
    model = _fingerprint_helper()
    X = torch.randn(1, 4, 3)
    y_train = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="forbids embed_with_test=True"):
        model._train_forward(X, y_train, embed_with_test=True)
    with pytest.raises(ValueError, match="forbids embed_with_test=True"):
        model._inference_forward(X, y_train, embed_with_test=True)


def test_cache_preserves_fingerprint_through_serialization_slice_move_and_concat():
    fingerprint = torch.randn(2, 5, 4)
    cache = TabICLCache(train_shape=(2, 3, 3), row_fingerprint=fingerprint)

    buffer = io.BytesIO()
    torch.save(cache, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    torch.testing.assert_close(restored.row_fingerprint, fingerprint)

    first = cache.slice_batch(0, 1)
    second = cache.slice_batch(1, 2)
    torch.testing.assert_close(first.row_fingerprint, fingerprint[:1])
    torch.testing.assert_close(cache.to("cpu").row_fingerprint, fingerprint)
    combined = TabICLCache.concat([first, second])
    torch.testing.assert_close(combined.row_fingerprint, fingerprint)


def test_cache_concat_rejects_partial_fingerprint_state():
    with_fingerprint = TabICLCache(
        train_shape=(1, 3, 3), row_fingerprint=torch.randn(1, 5, 4)
    )
    without_fingerprint = TabICLCache(train_shape=(1, 3, 3))

    with pytest.raises(ValueError, match="row_fingerprint"):
        TabICLCache.concat([with_fingerprint, without_fingerprint])


@pytest.mark.parametrize(
    "other",
    [torch.randn(1, 6, 4), torch.randn(1, 5, 4, dtype=torch.float64)],
)
def test_cache_concat_rejects_incompatible_fingerprint_schema(other):
    first = TabICLCache(
        train_shape=(1, 3, 3), row_fingerprint=torch.randn(1, 5, 4)
    )
    second = TabICLCache(train_shape=(1, 3, 3), row_fingerprint=other)

    with pytest.raises(ValueError, match="row_fingerprint tensors"):
        TabICLCache.concat([first, second])


def test_cache_use_rejects_missing_or_extraneous_fingerprint_before_compute():
    model = TabICL.__new__(TabICL)
    nn.Module.__init__(model)
    model._cache = None
    X_test = torch.randn(1, 2, 3)

    model.row_fingerprint = True
    missing = TabICLCache(
        train_shape=(1, 3, 3), row_repr=torch.randn(1, 3, 8)
    )
    with pytest.raises(ValueError, match="missing its training-only summary"):
        model.forward_with_cache(X_test=X_test, cache=missing)

    model.row_fingerprint = False
    extraneous = TabICLCache(
        train_shape=(1, 3, 3), row_fingerprint=torch.randn(1, 5, 4)
    )
    with pytest.raises(ValueError, match="model has it disabled"):
        model.forward_with_cache(X_test=X_test, cache=extraneous)


def test_training_flags_default_off_and_are_opt_in():
    parser = build_parser()
    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        [
            "--row_identity_mode",
            "none",
            "--row_fingerprint",
            "true",
            "--row_fingerprint_dim",
            "12",
            "--fail_on_oom",
            "true",
            "--fail_on_nonfinite",
            "true",
        ]
    )

    assert defaults.row_fingerprint is False
    assert defaults.row_fingerprint_dim == 16
    assert defaults.fail_on_oom is False
    assert defaults.fail_on_nonfinite is False
    assert enabled.row_identity_mode == "none"
    assert enabled.row_fingerprint is True
    assert enabled.row_fingerprint_dim == 12
    assert enabled.fail_on_oom is True
    assert enabled.fail_on_nonfinite is True


def _tiny_fingerprint_model() -> TabICL:
    return TabICL(
        max_classes=2,
        embed_dim=12,
        col_num_blocks=1,
        col_nhead=3,
        col_num_inds=4,
        col_feature_group=False,
        row_num_blocks=2,
        row_nhead=3,
        row_num_cls=2,
        row_identity_mode="none",
        row_fingerprint=True,
        row_fingerprint_dim=4,
        icl_num_blocks=1,
        icl_nhead=3,
        ff_factor=2,
        zero_init=False,
    )


def _cpu_inference_config() -> InferenceConfig:
    component = {"device": "cpu", "use_amp": False, "use_fa3": False}
    return InferenceConfig(
        COL_CONFIG=dict(component),
        ROW_CONFIG=dict(component),
        ICL_CONFIG=dict(component),
    )


@pytest.mark.parametrize("cache_mode", ["repr", "kv"])
def test_fingerprint_cached_inference_matches_uncached(cache_mode):
    torch.manual_seed(9)
    model = _tiny_fingerprint_model().eval()
    X_train = torch.randn(1, 4, 3)
    y_train = torch.tensor([[0, 1, 0, 1]])
    X_test = torch.randn(1, 3, 3)
    inference_config = _cpu_inference_config()

    expected = model(
        torch.cat((X_train, X_test), dim=1),
        y_train,
        inference_config=inference_config,
    )

    model.clear_cache()
    assert model.forward_with_cache(
        X_train=X_train,
        y_train=y_train,
        store_cache=True,
        use_cache=False,
        cache_mode=cache_mode,
        inference_config=inference_config,
    ) is None
    assert model._cache.row_fingerprint is not None
    actual = model.forward_with_cache(
        X_test=X_test,
        store_cache=False,
        use_cache=True,
        cache_mode=cache_mode,
        inference_config=inference_config,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fingerprint_full_train_forward_is_finite_and_differentiable():
    torch.manual_seed(10)
    model = _tiny_fingerprint_model().train()
    X = torch.randn(1, 7, 3)
    y_train = torch.tensor([[0, 1, 0, 1]])

    output = model(X, y_train)
    assert output.shape == (1, 3, 2)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.row_interactor.fingerprint_k_gates.grad is not None


def test_fail_on_nonfinite_stops_before_backward():
    class NanModel(nn.Module):
        def forward(self, X, y_train, d, row_identity_permutation=None):
            test_size = X.shape[1] - y_train.shape[1]
            return torch.full((X.shape[0], test_size, 2), float("nan"))

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        device="cpu", ignore_d=True, fail_on_nonfinite=True
    )
    trainer.ddp = False
    trainer.regression = False
    trainer.model = NanModel()
    trainer.raw_model = SimpleNamespace(_num_row_identity_tokens=lambda _: 3)
    trainer.identity_rng = SimpleNamespace(sample_for_micro_batch=lambda **_: None)
    trainer.amp_ctx = nullcontext()
    trainer.curr_step = 7
    micro_batch = (
        torch.randn(1, 4, 3),
        torch.tensor([[0, 1, 0, 1]]),
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([2]),
    )

    with pytest.raises(FloatingPointError, match="non-finite loss.*step 7"):
        trainer.run_micro_batch(micro_batch, 0, 1)


def test_fail_on_oom_reraises_first_microbatch_error():
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        batch_size=1,
        micro_batch_size=1,
        fail_on_oom=True,
        gradient_clipping=0,
    )
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.regression = False
    trainer.curr_step = 9

    def raise_oom(*_args):
        raise torch.cuda.OutOfMemoryError("pilot OOM")

    trainer.run_micro_batch = raise_oom
    batch = (
        torch.randn(1, 4, 3),
        torch.tensor([[0, 1, 0, 1]]),
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([2]),
    )

    with pytest.raises(torch.cuda.OutOfMemoryError, match="pilot OOM"):
        trainer.run_batch(batch)
