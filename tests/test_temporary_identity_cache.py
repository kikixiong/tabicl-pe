import torch
import pytest

from tabicl._model.interaction import RowInteraction
from tabicl._model.kv_cache import TabICLCache
from tabicl._model.tabicl import TabICL


def _interactor() -> RowInteraction:
    return RowInteraction(
        embed_dim=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        num_cls=2,
        identity_mode="temporary",
    )


def test_explicit_identity_permutation_is_shared_by_every_row():
    interactor = _interactor()
    embeddings = torch.zeros(2, 3, 6, 8)
    embeddings[:, :, 2:, 0] = torch.arange(4.0)
    permutations = torch.tensor([[3, 1, 0, 2], [2, 0, 3, 1]])

    actual, _ = interactor._apply_temporary_feature_identity(
        embeddings, row_identity_permutation=permutations
    )

    for table in range(2):
        expected = permutations[table].to(torch.float32)
        for row in range(3):
            torch.testing.assert_close(actual[table, row, 2:, 0], expected)


def test_explicit_identity_validation_supports_fullgraph_compile():
    interactor = _interactor()
    embeddings = torch.zeros(2, 3, 6, 8)
    permutations = torch.tensor([[3, 1, 0, 2], [2, 0, 3, 1]])

    def apply_identity(values, identity):
        return interactor._apply_temporary_feature_identity(
            values, row_identity_permutation=identity
        )[0]

    compiled_apply = torch.compile(apply_identity, backend="eager", fullgraph=True)
    actual = compiled_apply(embeddings, permutations)
    expected = apply_identity(embeddings, permutations)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="feature permutation"):
        compiled_apply(embeddings, torch.tensor([[0, 0, 2, 3], [0, 1, 2, 3]]))


def test_cache_serialization_slice_move_and_concat_preserve_identity():
    identity = torch.tensor([[2, 0, 1], [1, 2, 0]])
    cache = TabICLCache(train_shape=(2, 4, 3), row_identity_permutation=identity)

    torch.testing.assert_close(cache.slice_batch(1, 2).row_identity_permutation, identity[1:2])
    torch.testing.assert_close(cache.to("cpu").row_identity_permutation, identity)
    combined = TabICLCache.concat([cache.slice_batch(0, 1), cache.slice_batch(1, 2)])
    torch.testing.assert_close(combined.row_identity_permutation, identity)


def test_cache_concat_rejects_partial_identity_state():
    with_identity = TabICLCache(
        train_shape=(1, 4, 3),
        row_identity_permutation=torch.tensor([[2, 0, 1]]),
    )
    without_identity = TabICLCache(train_shape=(1, 4, 3))

    with pytest.raises(ValueError, match="row_identity_permutation"):
        TabICLCache.concat([with_identity, without_identity])


def test_cached_train_and_test_paths_reuse_the_same_temporary_identity(monkeypatch):
    model = TabICL(
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        row_identity_mode="temporary",
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        zero_init=False,
    ).eval()
    seen = []
    original = model.row_interactor._inference_forward

    def record_identity(embeddings, mgr_config=None, row_identity_permutation=None):
        seen.append(row_identity_permutation.detach().cpu().clone())
        return original(embeddings, mgr_config, row_identity_permutation)

    monkeypatch.setattr(model.row_interactor, "_inference_forward", record_identity)
    X_train = torch.randn(1, 4, 3)
    y_train = torch.tensor([[0, 1, 0, 1]])
    X_test = torch.randn(1, 2, 3)

    assert model.forward_with_cache(
        X_train=X_train,
        y_train=y_train,
        store_cache=True,
        use_cache=False,
        cache_mode="repr",
    ) is None
    model.forward_with_cache(
        X_test=X_test,
        store_cache=False,
        use_cache=True,
        cache_mode="repr",
    )

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    torch.testing.assert_close(
        seen[0], model._cache.row_identity_permutation.cpu(), rtol=0, atol=0
    )
