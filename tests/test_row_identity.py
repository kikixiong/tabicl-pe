import pytest
import torch

from tabicl._model.interaction import RowInteraction


def make_interactor(identity_mode: str) -> RowInteraction:
    return RowInteraction(
        embed_dim=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        num_cls=2,
        identity_mode=identity_mode,
    )


def test_identity_mode_controls_rope():
    assert make_interactor("rope").tf_row.rope is not None
    assert make_interactor("temporary").tf_row.rope is not None
    assert make_interactor("none").tf_row.rope is None


def test_invalid_identity_mode_is_rejected():
    with pytest.raises(ValueError, match="identity_mode"):
        make_interactor("stable_semantics")


def test_temporary_identity_is_table_wise_and_mask_aligned():
    interactor = make_interactor("temporary")
    embeddings = torch.zeros(2, 3, 6, 8)
    embeddings[:, :, :2, 0] = torch.tensor([[-2.0, -1.0]])
    embeddings[:, :, 2:, 0] = torch.arange(4.0)

    key_mask = torch.zeros(2, 3, 6, dtype=torch.bool)
    key_mask[:, :, 2:] = torch.tensor([False, True, False, True])

    torch.manual_seed(0)
    permuted, permuted_mask = interactor._apply_temporary_feature_identity(embeddings, key_mask)

    torch.testing.assert_close(permuted[:, :, :2], embeddings[:, :, :2])
    for table in range(2):
        for row in range(1, 3):
            torch.testing.assert_close(permuted[table, row], permuted[table, 0])
        assert sorted(permuted[table, 0, 2:, 0].tolist()) == [0.0, 1.0, 2.0, 3.0]
        expected_mask = permuted[table, 0, 2:, 0].remainder(2).bool()
        torch.testing.assert_close(permuted_mask[table, 0, 2:], expected_mask)


@pytest.mark.parametrize("identity_mode", ["rope", "temporary", "none"])
def test_identity_modes_support_forward_and_backward(identity_mode):
    interactor = make_interactor(identity_mode).train()
    input_embeddings = torch.randn(2, 3, 6, 8, requires_grad=True)
    # RowInteraction receives the non-leaf output of ColEmbedding in the full model.
    embeddings = input_embeddings + 0
    output = interactor(embeddings, d=torch.tensor([4, 3]))
    assert output.shape == (2, 3, 16)
    output.square().mean().backward()
    assert input_embeddings.grad is not None
