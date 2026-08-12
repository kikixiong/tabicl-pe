from __future__ import annotations

import torch

from tabicl._model.interaction import RowInteraction
from tabicl._model.layers import MultiheadAttentionBlock


def _row(*, fingerprint_dim):
    return RowInteraction(
        embed_dim=12,
        num_blocks=3,
        nhead=3,
        dim_feedforward=24,
        num_cls=2,
        identity_mode="none",
        fingerprint_dim=fingerprint_dim,
        zero_init=False,
    )


def test_qk_identity_changes_q_and_k_but_not_v_or_residual_input(monkeypatch):
    block = MultiheadAttentionBlock(12, 3, 24, zero_init=False)
    q = torch.randn(2, 5, 12)
    identity = torch.randn_like(q)
    captured = {}

    original = block.attn.forward

    def capture(query, key=None, value=None, **kwargs):
        captured["query"] = query.detach().clone()
        captured["key"] = key.detach().clone()
        captured["value"] = value.detach().clone()
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(block.attn, "forward", capture)
    block(q, q_identity=identity, k_identity=identity)

    normed = block.norm1(q)
    torch.testing.assert_close(captured["query"], normed + identity)
    torch.testing.assert_close(captured["key"], normed + identity)
    torch.testing.assert_close(captured["value"], normed)


def test_zero_fingerprint_gates_match_nope_exactly_and_shared_init_is_equal():
    torch.manual_seed(123)
    baseline = _row(fingerprint_dim=None)
    after_baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    fingerprint = _row(fingerprint_dim=4)
    after_fingerprint_rng = torch.random.get_rng_state()

    assert torch.equal(after_baseline_rng, after_fingerprint_rng)
    fingerprint_state = fingerprint.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, fingerprint_state[name], rtol=0, atol=0)

    embeddings = torch.randn(2, 4, 7, 12)
    fp = torch.randn(2, 7, 12)
    fp[:, :2] = 0
    zeros = torch.zeros(3, 2)
    baseline.train()
    fingerprint.train()
    expected = baseline(embeddings.clone())
    actual = fingerprint(
        embeddings.clone(), row_fingerprint=fp, fingerprint_layer_gates=zeros
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fingerprint_gradients_reach_source_and_qk_projections():
    model = _row(fingerprint_dim=4).train()
    # RowInteraction replaces reserved CLS slots in place, matching the
    # non-leaf ColEmbedding output used by the real model.
    embeddings = torch.randn(2, 3, 6, 12, requires_grad=True) * 1.0
    fingerprint = torch.randn(2, 6, 12, requires_grad=True)
    with torch.no_grad():
        fingerprint[:, :2] = 0

    out = model(embeddings, row_fingerprint=fingerprint)
    out.square().mean().backward()

    assert fingerprint.grad is not None
    assert torch.count_nonzero(fingerprint.grad[:, 2:]) > 0
    assert model.fingerprint_q_projections[0][1].weight.grad is not None
    assert model.fingerprint_k_projections[-1][1].weight.grad is not None


def test_permuted_intervention_requires_valid_tablewise_permutation():
    model = _row(fingerprint_dim=4).train()
    embeddings = torch.randn(1, 2, 6, 12)
    fingerprint = torch.randn(1, 6, 12)
    fingerprint[:, :2] = 0

    try:
        model(
            embeddings,
            row_fingerprint=fingerprint,
            fingerprint_intervention="permuted",
        )
    except ValueError as error:
        assert "requires a permutation" in str(error)
    else:
        raise AssertionError("missing permutation was accepted")

    output = model(
        embeddings,
        row_fingerprint=fingerprint,
        fingerprint_intervention="permuted",
        fingerprint_permutation=torch.tensor([[3, 2, 1, 0]]),
    )
    assert output.shape == (1, 2, 24)
