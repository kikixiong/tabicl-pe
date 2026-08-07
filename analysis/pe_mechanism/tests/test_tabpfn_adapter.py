from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from pe_mechanism.adapters.tabpfn_v26 import (
    TabPFNV26Adapter,
    position_components,
)


class AddThinkingRows(nn.Module):
    def forward(self, values, single_eval_pos):
        thinking = torch.full(
            (values.shape[0], 1, values.shape[2], values.shape[3]),
            0.05,
            dtype=values.dtype,
            device=values.device,
        )
        return torch.cat((thinking, values), dim=1), single_eval_pos + 1


class Branch(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, values, **_kwargs):
        # Mix the sequence axis so the target token depends on feature tokens and
        # the fake model can detect position/component interventions.
        mixed = torch.tanh(values.mean(dim=-2, keepdim=True)) * self.scale
        return mixed.expand_as(values)


class FakeTabPFNBlock(nn.Module):
    def __init__(self, embedding: int) -> None:
        super().__init__()
        self.per_sample_attention_between_features = Branch(0.2)
        self.per_column_attention_between_cells = Branch(0.3)
        self.mlp = nn.Sequential(
            nn.Linear(embedding, embedding * 2, bias=False),
            nn.GELU(),
            nn.Linear(embedding * 2, embedding, bias=False),
        )
        torch.manual_seed(20)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.08)

    def forward(self, values, single_eval_pos, save_peak_memory_factor=None):
        del single_eval_pos, save_peak_memory_factor
        table_count, row_count, token_count, embedding = values.shape
        row_branch = self.per_sample_attention_between_features(
            values.reshape(table_count * row_count, token_count, embedding)
        ).reshape_as(values)
        values = values + row_branch
        column_values = values.transpose(1, 2).contiguous()
        column_branch = self.per_column_attention_between_cells(
            column_values.reshape(table_count * token_count, row_count, embedding)
        ).reshape_as(column_values)
        values = values + column_branch.transpose(1, 2)
        flat = values.reshape(table_count * row_count * token_count, embedding)
        return values + self.mlp(flat).reshape_as(values)


class FakeTabPFNV26(nn.Module):
    features_per_group = 3

    def __init__(self) -> None:
        super().__init__()
        self.feature_group_embedder = nn.Linear(6, 8, bias=False)
        self.feature_positional_embedding_embeddings = nn.Linear(2, 8, bias=True)
        self.add_thinking_rows = AddThinkingRows()
        self.blocks = nn.ModuleList([FakeTabPFNBlock(8), FakeTabPFNBlock(8)])
        self.output_projection = nn.Sequential(nn.Linear(8, 3, bias=False))
        torch.manual_seed(5)
        nn.init.normal_(self.feature_group_embedder.weight, std=0.1)
        nn.init.normal_(self.feature_positional_embedding_embeddings.weight, std=0.2)
        nn.init.constant_(self.feature_positional_embedding_embeddings.bias, 0.25)
        nn.init.normal_(self.output_projection[0].weight, std=0.1)

    def forward(
        self,
        x,
        y,
        *,
        only_return_standard_out=True,
        categorical_inds=None,
        performance_options=None,
        task_type=None,
    ):
        del categorical_inds, task_type
        self.last_performance_options = performance_options
        rows, tables, columns = x.shape
        groups = (columns + self.features_per_group - 1) // self.features_per_group
        padded = torch.nn.functional.pad(
            x, (0, groups * self.features_per_group - columns)
        )
        grouped = padded.reshape(rows, tables, groups, self.features_per_group)
        indicators = torch.zeros_like(grouped)
        encoded = torch.cat((grouped, indicators), dim=-1)
        embedded = self.feature_group_embedder(
            encoded.reshape(rows, tables * groups, 6)
        )
        embedded = embedded.reshape(rows, tables, groups, 8).permute(1, 0, 2, 3)
        codes = torch.stack(
            (
                torch.arange(groups, dtype=x.dtype, device=x.device),
                torch.ones(groups, device=x.device),
            ),
            dim=-1,
        )
        embedded = (
            embedded + self.feature_positional_embedding_embeddings(codes)[None, None]
        )
        target = torch.zeros(tables, rows, 1, 8, dtype=x.dtype, device=x.device)
        values = torch.cat((embedded, target), dim=2)
        values, eval_position = self.add_thinking_rows(values, int(y.shape[0]))
        for block in self.blocks:
            values = block(values, eval_position, None)
        test_embeddings = values[:, eval_position:, -1].transpose(0, 1)
        standard = self.output_projection(test_embeddings)
        if only_return_standard_out:
            return standard
        return {"standard": standard, "test_embeddings": test_embeddings}


@pytest.fixture
def prepared():
    torch.manual_seed(30)
    return {"x": torch.randn(7, 1, 5), "y": torch.tensor([[0.0], [1.0], [0.0], [1.0]])}


def test_position_components_are_exact_linear_decomposition():
    projection = nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.3]]))
        projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    codes = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    pieces = position_components(projection, codes)
    assert torch.equal(pieces["full"], projection(codes))
    assert torch.equal(pieces["full"], pieces["weight"] + pieces["bias"])
    assert torch.count_nonzero(pieces["none"]) == 0
    assert torch.equal(pieces["bias"][0], pieces["bias"][1])


def test_sites_and_branch_capture_include_axis_metadata_and_reject_repeated_calls(
    prepared,
):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    sites = adapter.list_sites(model)
    assert sites["blocks.0"].axis_names == ("table", "row", "token", "embedding")
    assert (
        sites["blocks.0.per_sample_attention_between_features"].axis_names[0]
        == "table_x_row"
    )
    requested = [
        "feature_group_embedder",
        "feature_positional_embedding.input",
        "feature_positional_embedding.full",
        "feature_positional_embedding.weight",
        "feature_positional_embedding.bias",
        "feature_positional_embedding.none",
        "add_thinking_rows",
        "blocks.0.per_sample_attention_between_features",
        "blocks.0.per_column_attention_between_cells",
        "blocks.0.mlp",
        "blocks.1",
    ]
    group_map = ((0, 1, 2), (3, 4))
    with adapter.capture(
        model,
        sites=requested,
        model_sha="c" * 40,
        checkpoint_sha="d" * 64,
        preprocessing_view_id="tabpfn-view",
        feature_group_map=group_map,
    ) as captured:
        adapter.predict(model, prepared)
        with pytest.raises(RuntimeError, match="more than once"):
            adapter.predict(model, prepared)

    assert captured["feature_positional_embedding.input"].shape == (2, 2)
    assert captured["feature_positional_embedding.full"].shape == (2, 8)
    assert torch.allclose(
        captured["feature_positional_embedding.full"].tensor,
        captured["feature_positional_embedding.weight"].tensor
        + captured["feature_positional_embedding.bias"].tensor,
    )
    assert (
        torch.count_nonzero(captured["feature_positional_embedding.none"].tensor) == 0
    )
    assert captured["blocks.1"].shape[0] == 1
    assert captured["blocks.1"].feature_group_map == group_map
    assert captured["blocks.0.mlp"].axis_names == ("token_instance", "embedding")


def test_position_policy_changes_only_projection_and_restores(prepared):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    baseline = adapter.predict(model, prepared)
    outputs = {}
    for component in ("weight", "bias", "none"):
        with adapter.position_policy(model, component):
            outputs[component] = adapter.predict(model, prepared)
        assert torch.equal(adapter.predict(model, prepared), baseline)
    assert not torch.allclose(outputs["weight"], baseline)
    assert not torch.allclose(outputs["bias"], baseline)
    assert not torch.allclose(outputs["none"], baseline)
    assert not torch.allclose(outputs["weight"], outputs["bias"])


def test_capture_keeps_mathematical_full_while_active_policy_is_weight(prepared):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    with adapter.capture(
        model,
        sites=[
            "feature_positional_embedding.active",
            "feature_positional_embedding.full",
            "feature_positional_embedding.weight",
            "feature_positional_embedding.bias",
        ],
        model_sha="e" * 40,
        checkpoint_sha="f" * 64,
        preprocessing_view_id="weight-policy",
    ) as captured:
        with adapter.position_policy(model, "weight"):
            adapter.predict(model, prepared)
    assert torch.equal(
        captured["feature_positional_embedding.active"].tensor,
        captured["feature_positional_embedding.weight"].tensor,
    )
    assert torch.equal(
        captured["feature_positional_embedding.full"].tensor,
        captured["feature_positional_embedding.weight"].tensor
        + captured["feature_positional_embedding.bias"].tensor,
    )


def test_activation_intervention_restores_after_context(prepared):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    baseline = adapter.predict(model, prepared)
    with adapter.intervene(
        model, {"blocks.0": lambda record: torch.zeros_like(record.tensor)}
    ):
        changed = adapter.predict(model, prepared)
    assert not torch.allclose(changed, baseline)
    assert torch.equal(adapter.predict(model, prepared), baseline)


def test_position_component_interventions_preserve_the_complement(prepared):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    with adapter.position_policy(model, "bias"):
        expected_bias_only = adapter.predict(model, prepared)
    with adapter.position_policy(model, "weight"):
        expected_weight_only = adapter.predict(model, prepared)

    def zero(record):
        return torch.zeros_like(record.tensor)
    with adapter.intervene(model, {"feature_positional_embedding.weight": zero}):
        without_weight = adapter.predict(model, prepared)
    with adapter.intervene(model, {"feature_positional_embedding.bias": zero}):
        without_bias = adapter.predict(model, prepared)

    assert torch.equal(without_weight, expected_bias_only)
    assert torch.equal(without_bias, expected_weight_only)


def test_multiple_position_component_interventions_are_rejected(prepared):
    del prepared
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    def zero(record):
        return torch.zeros_like(record.tensor)
    with pytest.raises(ValueError, match="at most one feature-position component"):
        with adapter.intervene(
            model,
            {
                "feature_positional_embedding.weight": zero,
                "feature_positional_embedding.bias": zero,
            },
        ):
            pass


def test_missing_gated_weight_has_actionable_offline_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="gated.*never downloads"):
        TabPFNV26Adapter().load_model(tmp_path / "tabpfn-v2.6.ckpt", device="cpu")


def test_load_model_requires_both_exact_v26_class_and_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / "tabpfn-v2.6.ckpt"
    checkpoint.write_bytes(b"offline-test-checkpoint")

    class TabPFNV2p6(FakeTabPFNV26):
        pass

    def install_loader(model, config_name):
        package = ModuleType("tabpfn")
        loading = ModuleType("tabpfn.model_loading")
        loading.load_model_criterion_config = lambda *_args, **_kwargs: (
            [model],
            None,
            [SimpleNamespace(name=config_name)],
            None,
        )
        package.model_loading = loading
        monkeypatch.setitem(sys.modules, "tabpfn", package)
        monkeypatch.setitem(sys.modules, "tabpfn.model_loading", loading)

    install_loader(FakeTabPFNV26(), "TabPFN-v2.6")
    with pytest.raises(ValueError, match="not TabPFN v2.6"):
        TabPFNV26Adapter().load_model(checkpoint, device="cpu")

    install_loader(TabPFNV2p6(), "wrong-config")
    with pytest.raises(ValueError, match="not TabPFN v2.6"):
        TabPFNV26Adapter().load_model(checkpoint, device="cpu")

    exact = TabPFNV2p6()
    install_loader(exact, "TabPFN-v2.6")
    assert TabPFNV26Adapter().load_model(checkpoint, device="cpu") is exact


def test_predict_disables_recompute_and_chunked_hook_paths(prepared):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    adapter.predict(model, prepared)
    options = model.last_performance_options
    assert options.force_recompute_layer is False
    assert options.save_peak_memory_factor is None
    assert options.use_chunkwise_inference is False


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("save_peak_memory_factor", 2),
        ("force_recompute_layer", True),
        ("use_chunkwise_inference", True),
    ],
)
def test_predict_rejects_instrumentation_unsafe_performance_options(
    prepared, field, unsafe_value
):
    adapter, model = TabPFNV26Adapter(), FakeTabPFNV26().eval()
    values = {
        "save_peak_memory_factor": None,
        "force_recompute_layer": False,
        "use_chunkwise_inference": False,
    }
    values[field] = unsafe_value
    batch = {
        **prepared,
        "performance_options": SimpleNamespace(**values),
    }
    with pytest.raises(ValueError, match=field):
        adapter.predict(model, batch)
