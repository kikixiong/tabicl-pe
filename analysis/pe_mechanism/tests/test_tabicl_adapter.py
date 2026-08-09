from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pe_mechanism.adapters.tabicl import TabICLAdapter, same_feature_group_map


class FakeRoPE:
    interleaved = False
    default_seq_dim = -2
    use_xpos = False

    def __init__(self) -> None:
        self.freqs = torch.tensor([1.0, 0.2])

    @staticmethod
    def get_seq_pos(seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def rotate_queries_or_keys(self, tensor, seq_dim=None, offset=0, scale=None):
        assert seq_dim in (None, -2)
        positions = self.get_seq_pos(
            tensor.shape[-2], tensor.device, tensor.dtype, offset
        )
        angles = positions[:, None] * self.freqs.to(tensor.device)[None]
        cosine, sine = angles.cos(), angles.sin()
        first, second = tensor[..., :2], tensor[..., 2:]
        out = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), dim=-1
        )
        return out if scale is None else out * scale


class FakeAttention(nn.Module):
    def __init__(self, embedding: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.num_heads = 2

    def _split(self, tensor):
        return tensor.unflatten(
            -1, (self.num_heads, self.embedding // self.num_heads)
        ).transpose(-3, -2)

    def _merge(self, tensor):
        return tensor.transpose(-3, -2).flatten(-2)

    def forward(self, query, key=None, value=None, rope=None, **_kwargs):
        del value
        key = query if key is None else key
        q_heads, k_heads = self._split(query), self._split(key)
        if rope is not None:
            q_heads = rope.rotate_queries_or_keys(q_heads)
            k_heads = rope.rotate_queries_or_keys(k_heads)
        combined = q_heads + 0.3 * k_heads.mean(dim=-2, keepdim=True)
        return self._merge(combined)


class FakeBlock(nn.Module):
    def __init__(self, embedding: int) -> None:
        super().__init__()
        self.attn = FakeAttention(embedding)
        self.linear2 = nn.Linear(embedding, embedding, bias=False)
        nn.init.eye_(self.linear2.weight)
        self.linear2.weight.data.mul_(0.1)

    def forward(self, q, k=None, v=None, rope=None, **kwargs):
        attention = self.attn(q, key=k, value=v, rope=rope, **kwargs)
        return q + attention + self.linear2(q)


class FakeColumnEmbedder(nn.Module):
    feature_group = "same"
    feature_group_size = 3
    reserve_cls_tokens = 2

    def forward(self, X, **_kwargs):
        feature_count = X.shape[-1]
        grouped = []
        for group in same_feature_group_map(feature_count, self.feature_group_size):
            value = X[..., list(group)].mean(dim=-1)
            grouped.append(
                torch.stack(
                    (
                        value,
                        value.square(),
                        value.sin(),
                        value.cos(),
                        value,
                        -value,
                        value / 2,
                        value * 2,
                    ),
                    dim=-1,
                )
            )
        features = torch.stack(grouped, dim=-2)
        cls = torch.zeros(*X.shape[:2], self.reserve_cls_tokens, 8, dtype=X.dtype)
        return torch.cat((cls, features), dim=-2)


class FakeRowInteraction(nn.Module):
    num_cls = 2

    def __init__(self) -> None:
        super().__init__()
        self.tf_row = nn.Module()
        self.tf_row.blocks = nn.ModuleList([FakeBlock(8) for _ in range(3)])
        self.tf_row.rope = FakeRoPE()

    def forward(self, embeddings, **_kwargs):
        for block in self.tf_row.blocks[:-1]:
            embeddings = block(embeddings, rope=self.tf_row.rope)
        output = self.tf_row.blocks[-1](
            q=embeddings[..., : self.num_cls, :],
            k=embeddings,
            v=embeddings,
            rope=self.tf_row.rope,
        )
        return output.flatten(-2)


class FakeICLPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tf_icl = nn.Module()
        self.tf_icl.blocks = nn.ModuleList([FakeBlock(16) for _ in range(2)])
        self.tf_icl.rope = None
        self.decoder = nn.Linear(16, 3, bias=False)
        torch.manual_seed(4)
        nn.init.normal_(self.decoder.weight, std=0.2)

    def forward(self, representations, y_train, **_kwargs):
        output = representations
        for block in self.tf_icl.blocks:
            output = block(output)
        return self.decoder(output[:, y_train.shape[1] :])


class FakeTabICL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.col_embedder = FakeColumnEmbedder()
        self.row_interactor = FakeRowInteraction()
        self.icl_predictor = FakeICLPredictor()
        self._cache = None

    def forward(self, X, y_train, **kwargs):
        del kwargs
        return self.icl_predictor(self.row_interactor(self.col_embedder(X)), y_train)


@pytest.fixture
def prepared():
    torch.manual_seed(10)
    return {"X": torch.randn(1, 7, 5), "y_train": torch.tensor([[0, 1, 0, 1]])}


def test_same_group_mapping_is_the_real_circular_power_of_two_mapping():
    assert same_feature_group_map(7, 3) == (
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 0),
        (5, 6, 1),
        (6, 0, 2),
        (0, 1, 3),
    )


def test_capture_labels_axes_infers_groups_and_rejects_repeated_calls(prepared):
    adapter, model = TabICLAdapter(), FakeTabICL().eval()
    sites = adapter.list_sites(model)
    assert sites["row_interactor.tf_row.blocks.2"].axis_names == (
        "table",
        "row",
        "cls",
        "embedding",
    )
    requested = [
        "col_embedder",
        "row_interactor.tf_row.blocks.0.attn",
        "row_interactor.tf_row.blocks.2",
        "row_interactor",
        "icl_predictor.tf_icl.blocks.1.linear2",
    ]
    with adapter.capture(
        model,
        sites=requested,
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
        preprocessing_view_id="toy-view",
    ) as captured:
        adapter.predict(model, prepared)
        with pytest.raises(RuntimeError, match="more than once"):
            adapter.predict(model, prepared)

    assert captured["col_embedder"].shape == (1, 7, 7, 8)
    assert captured["row_interactor.tf_row.blocks.2"].shape == (1, 7, 2, 8)
    assert captured["row_interactor"].shape == (1, 7, 16)
    assert captured["col_embedder"].feature_group_map == same_feature_group_map(5, 3)
    assert captured["row_interactor"].feature_group_map == same_feature_group_map(5, 3)
    assert captured["row_interactor"].preprocessing_view_id == "toy-view"


def test_activation_intervention_and_context_exit_restore_predictions(prepared):
    adapter, model = TabICLAdapter(), FakeTabICL().eval()
    baseline = adapter.predict(model, prepared)
    with adapter.intervene(
        model, {"row_interactor": lambda record: torch.zeros_like(record.tensor)}
    ):
        changed = adapter.predict(model, prepared)
    restored = adapter.predict(model, prepared)
    assert not torch.allclose(changed, baseline)
    assert torch.equal(restored, baseline)


def test_rope_policies_cover_q_k_phase_band_and_heads_and_restore(prepared):
    adapter, model = TabICLAdapter(), FakeTabICL().eval()
    baseline = adapter.predict(model, prepared)

    outputs = {}
    policies = {
        "off": dict(rotate_queries=False, rotate_keys=False),
        "query_only": dict(rotate_queries=True, rotate_keys=False),
        "key_only": dict(rotate_queries=False, rotate_keys=True),
        "half_phase": dict(phase_strength=0.5),
        "high_head0": dict(frequency_band="high", heads=[0]),
        "low_head1": dict(frequency_band="low", heads=[1]),
    }
    for name, policy in policies.items():
        with adapter.rope_policy(model, blocks=[0], **policy):
            outputs[name] = adapter.predict(model, prepared)
        assert torch.equal(adapter.predict(model, prepared), baseline)

    assert not torch.allclose(outputs["off"], baseline)
    assert not torch.allclose(outputs["query_only"], outputs["key_only"])
    assert not torch.allclose(outputs["half_phase"], baseline)
    assert not torch.allclose(outputs["high_head0"], outputs["low_head1"])

    with pytest.raises(RuntimeError):
        with adapter.rope_policy(model, blocks=[1], phase_strength=0.25):
            raise RuntimeError("probe")
    assert torch.equal(adapter.predict(model, prepared), baseline)


def test_rope_policy_rejects_cached_keys(prepared):
    del prepared
    model = FakeTabICL()
    model._cache = SimpleNamespace(existing=True)
    with pytest.raises(ValueError, match="cache"):
        with TabICLAdapter().rope_policy(model):
            pass


def test_missing_checkpoint_fails_without_trying_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TabICLAdapter().load_model(tmp_path / "missing.ckpt", device="cpu")
