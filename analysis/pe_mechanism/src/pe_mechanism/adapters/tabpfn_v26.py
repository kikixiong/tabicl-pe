"""Optional TabPFN v2.6 adapter with exact feature-position decomposition."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from .base import ActivationRecord, ActivationSite, Intervention, ModelAdapter


PositionComponent = Literal["full", "weight", "bias", "none"]
_POSITION_PREFIX = "feature_positional_embedding"
_POSITION_COMPONENTS = ("input", "active", "full", "weight", "bias", "none")


def _raw_model(model: Any) -> Any:
    models = getattr(model, "models_", None)
    if models is not None:
        if len(models) != 1:
            raise ValueError(
                "mechanism tracing requires exactly one TabPFN ensemble model"
            )
        return models[0]
    return model


def _module_map(model: Any) -> dict[str, Any]:
    raw = _raw_model(model)
    if not hasattr(raw, "named_modules"):
        raise TypeError("TabPFN adapter requires a torch.nn.Module-like model")
    return dict(raw.named_modules())


def _first_tensor(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _first_tensor(output[0])
    if isinstance(output, Mapping):
        for key in ("standard", "logits", "output"):
            if key in output:
                return _first_tensor(output[key])
    raise TypeError(f"hook output {type(output).__name__} does not contain a tensor")


def _replace_first_tensor(output: Any, tensor: Any) -> Any:
    if hasattr(output, "shape"):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    if isinstance(output, list) and output:
        return [tensor, *output[1:]]
    if isinstance(output, Mapping):
        updated = dict(output)
        for key in ("standard", "logits", "output"):
            if key in updated:
                updated[key] = tensor
                return updated
    raise TypeError(f"cannot replace tensor inside {type(output).__name__}")


def _snapshot(tensor: Any) -> Any:
    if not hasattr(tensor, "detach"):
        raise TypeError("captured activation must be a torch tensor")
    return tensor.detach().to("cpu").clone()


def _append_record(
    buffer: dict[str, ActivationRecord], record: ActivationRecord
) -> None:
    """Record exactly one hook invocation per site and capture context.

    A repeated call is not another element of the site's leading semantic axis.
    Merging it there would erase invocation/view boundaries and could attach the
    first invocation's feature-group map to later, incompatible tensors.
    """

    if record.site in buffer:
        raise RuntimeError(
            f"site {record.site!r} was called more than once in one capture context; "
            "capture one model invocation per context so tensor axes and "
            "feature_group_map stay aligned"
        )
    buffer[record.site] = record


def position_components(projection: Any, codes: Any) -> dict[str, Any]:
    """Compute the exact ``Wp+b``, ``Wp``, ``b`` and zero components."""

    import torch
    import torch.nn.functional as functional

    if not hasattr(projection, "weight"):
        raise TypeError("position projection must expose a linear weight")
    weight = functional.linear(codes, projection.weight, bias=None)
    bias_parameter = getattr(projection, "bias", None)
    if bias_parameter is None:
        bias = torch.zeros_like(weight)
    else:
        bias = bias_parameter.to(device=weight.device, dtype=weight.dtype)
        bias = bias.expand_as(weight)
    return {
        "full": weight + bias,
        "weight": weight,
        "bias": bias,
        "none": torch.zeros_like(weight),
    }


def _instrumentation_safe_performance_options(model: Any) -> Any:
    """Disable execution paths that repeat or chunk hooks during tracing.

    TabPFN's memory-saving modes may recompute a block or invoke its internal
    branches once per chunk.  Those paths are valid for prediction but change
    capture/intervention call semantics.  Mechanism runs therefore use the
    architecture's own options object with every such mode explicitly off.
    """

    factory = getattr(model, "get_default_performance_options", None)
    if callable(factory):
        defaults = factory()
        overrides = {
            name: value
            for name, value in (
                ("save_peak_memory_factor", None),
                ("force_recompute_layer", False),
                ("use_chunkwise_inference", False),
            )
            if hasattr(defaults, name)
        }
        if is_dataclass(defaults):
            return replace(defaults, **overrides)
        # A non-dataclass options carrier is unusual; a simple immutable-by-
        # convention namespace still exposes the exact attributes v2.6 reads.
        return SimpleNamespace(**overrides)
    try:
        from tabpfn.architectures.interface import PerformanceOptions

        return PerformanceOptions(
            save_peak_memory_factor=None,
            force_recompute_layer=False,
            use_chunkwise_inference=False,
        )
    except (ImportError, TypeError):
        # Fake/minimal models used in adapter contract tests need no TabPFN
        # dependency, but still receive explicit instrumentation-safe flags.
        return SimpleNamespace(
            save_peak_memory_factor=None,
            force_recompute_layer=False,
            use_chunkwise_inference=False,
        )


def _validate_instrumentation_safe_performance_options(options: Any) -> None:
    """Reject caller options that can repeat or chunk instrumented modules."""

    missing = object()

    def value(name: str) -> Any:
        if isinstance(options, Mapping):
            return options.get(name, missing)
        return getattr(options, name, missing)

    expected = {
        "save_peak_memory_factor": None,
        "force_recompute_layer": False,
        "use_chunkwise_inference": False,
    }
    unsafe = [
        name for name, required in expected.items() if value(name) is not required
    ]
    if unsafe:
        raise ValueError(
            "performance_options must explicitly disable instrumentation-unsafe "
            f"chunk/recompute fields: {unsafe}"
        )


class TabPFNV26Adapter(ModelAdapter):
    """Adapter for the single-file TabPFN v2.6 architecture."""

    @property
    def model_family(self) -> str:
        return "tabpfn-v2.6"

    def load_model(
        self,
        checkpoint: Path,
        *,
        device: str,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "TabPFN v2.6 released checkpoint is unavailable at "
                f"{checkpoint}. The v2.6 weights are gated: accept the Prior-Labs/"
                "tabpfn_2_6 terms and place the exact checkpoint locally. This adapter "
                "never downloads weights or falls back to another TabPFN version."
            )
        supplied = dict(options or {})
        allowed = {"cache_trainset_representation", "model_index"}
        unknown = supplied.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported TabPFN v2.6 load options: {sorted(unknown)}")
        try:
            from tabpfn.model_loading import load_model_criterion_config
        except ImportError as error:
            raise ImportError(
                "TabPFN v2.6 analysis requires a compatible `tabpfn` installation; "
                "install it explicitly alongside this optional adapter"
            ) from error

        models, _criterion, configs, _inference_config = load_model_criterion_config(
            checkpoint,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=bool(
                supplied.get("cache_trainset_representation", False)
            ),
            which="classifier",
            version="v2.6",
            download_if_not_exists=False,
        )
        index = int(supplied.get("model_index", 0))
        if index < 0 or index >= len(models):
            raise IndexError(f"TabPFN model_index {index} outside [0, {len(models)})")
        model = models[index]
        config = configs[index]
        config_name = getattr(config, "name", "")
        if type(model).__name__ != "TabPFNV2p6" or config_name != "TabPFN-v2.6":
            raise ValueError(
                f"checkpoint resolved to {type(model).__name__}/{config_name!r}, not TabPFN v2.6"
            )
        return model.to(device).eval()

    def list_sites(self, model: Any) -> Mapping[str, ActivationSite]:
        raw = _raw_model(model)
        modules = _module_map(raw)
        required = {
            "feature_group_embedder",
            "feature_positional_embedding_embeddings",
            "add_thinking_rows",
            "output_projection",
        }
        missing = sorted(required - modules.keys())
        if missing:
            raise ValueError(f"model is missing TabPFN v2.6 modules: {missing}")

        sites: dict[str, ActivationSite] = {
            "feature_group_embedder": ActivationSite(
                "feature_group_embedder",
                ("row", "table_x_feature_group", "embedding"),
                "Embedded, grouped feature values before column identity is added.",
            ),
            f"{_POSITION_PREFIX}.input": ActivationSite(
                f"{_POSITION_PREFIX}.input",
                ("feature_group", "position_subspace"),
                "Deterministic 48-dimensional column code p.",
            ),
            f"{_POSITION_PREFIX}.active": ActivationSite(
                f"{_POSITION_PREFIX}.active",
                ("feature_group", "embedding"),
                "Position projection actually returned under the active policy.",
            ),
            f"{_POSITION_PREFIX}.full": ActivationSite(
                f"{_POSITION_PREFIX}.full",
                ("feature_group", "embedding"),
                "Exact full projection Wp+b.",
            ),
            f"{_POSITION_PREFIX}.weight": ActivationSite(
                f"{_POSITION_PREFIX}.weight",
                ("feature_group", "embedding"),
                "Exact weight contribution Wp.",
            ),
            f"{_POSITION_PREFIX}.bias": ActivationSite(
                f"{_POSITION_PREFIX}.bias",
                ("feature_group", "embedding"),
                "Broadcast linear bias b.",
            ),
            f"{_POSITION_PREFIX}.none": ActivationSite(
                f"{_POSITION_PREFIX}.none",
                ("feature_group", "embedding"),
                "Zero position contribution with unchanged shape.",
            ),
            "add_thinking_rows": ActivationSite(
                "add_thinking_rows",
                ("table", "row", "token", "embedding"),
                "Feature groups plus target token after thinking rows are prepended.",
            ),
            "output_projection": ActivationSite(
                "output_projection",
                ("test_row", "table", "class_or_bucket"),
                "Final raw output projection.",
            ),
        }
        blocks = tuple(getattr(raw, "blocks", ()))
        for index, block in enumerate(blocks):
            prefix = f"blocks.{index}"
            sites[prefix] = ActivationSite(
                prefix,
                ("table", "row", "token", "embedding"),
                "Complete TabPFN v2.6 block output.",
            )
            if hasattr(block, "per_sample_attention_between_features"):
                branch = f"{prefix}.per_sample_attention_between_features"
                sites[branch] = ActivationSite(
                    branch,
                    ("table_x_row", "token", "embedding"),
                    "Attention between feature-group tokens within each row.",
                )
            if hasattr(block, "per_column_attention_between_cells"):
                branch = f"{prefix}.per_column_attention_between_cells"
                sites[branch] = ActivationSite(
                    branch,
                    ("table_x_token", "row", "embedding"),
                    "Attention between cells down each feature-group column.",
                )
            if hasattr(block, "mlp"):
                branch = f"{prefix}.mlp"
                sites[branch] = ActivationSite(
                    branch,
                    ("token_instance", "embedding"),
                    "Feed-forward branch before residual addition.",
                )
        return sites

    @staticmethod
    def _position_projection(model: Any) -> Any:
        raw = _raw_model(model)
        projection = getattr(raw, "feature_positional_embedding_embeddings", None)
        if projection is None:
            raise ValueError("model has no TabPFN v2.6 feature position projection")
        return projection

    @contextmanager
    def capture(
        self,
        model: Any,
        *,
        sites: Sequence[str],
        model_sha: str,
        checkpoint_sha: str,
        preprocessing_view_id: str,
        feature_group_map: tuple[tuple[int, ...], ...] | None = None,
    ) -> Iterator[Mapping[str, ActivationRecord]]:
        raw = _raw_model(model)
        available = self.list_sites(raw)
        requested = tuple(sites)
        if len(set(requested)) != len(requested):
            raise ValueError("capture sites must not contain duplicates")
        unknown = sorted(set(requested) - available.keys())
        if unknown:
            raise KeyError(f"unknown TabPFN v2.6 capture sites: {unknown}")

        buffer: dict[str, ActivationRecord] = {}
        handles: list[Any] = []

        def record(name: str, value: Any) -> None:
            tensor = _snapshot(value)
            _append_record(
                buffer,
                ActivationRecord(
                    tensor=tensor,
                    site=name,
                    axis_names=available[name].axis_names,
                    shape=tuple(tensor.shape),
                    model_sha=model_sha,
                    checkpoint_sha=checkpoint_sha,
                    preprocessing_view_id=preprocessing_view_id,
                    feature_group_map=feature_group_map,
                ),
            )

        positional_requested = {
            name for name in requested if name.startswith(f"{_POSITION_PREFIX}.")
        }
        if positional_requested:
            projection = self._position_projection(raw)
            pending_inputs: list[Any] = []

            def position_pre_hook(
                _module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]
            ) -> None:
                if not args:
                    raise RuntimeError(
                        "position projection was called without column codes"
                    )
                pending_inputs.append(args[0])
                if f"{_POSITION_PREFIX}.input" in positional_requested:
                    record(f"{_POSITION_PREFIX}.input", args[0])

            def position_hook(
                _module: Any, _args: tuple[Any, ...], output: Any
            ) -> None:
                if not pending_inputs:
                    raise RuntimeError("position projection hook lost its paired input")
                codes = pending_inputs.pop(0)
                components = position_components(projection, codes)
                active_name = f"{_POSITION_PREFIX}.active"
                if active_name in positional_requested:
                    record(active_name, _first_tensor(output))
                for component in ("full", "weight", "bias", "none"):
                    name = f"{_POSITION_PREFIX}.{component}"
                    if name in positional_requested:
                        record(name, components[component])

            handles.append(
                projection.register_forward_pre_hook(
                    position_pre_hook, with_kwargs=True
                )
            )
            handles.append(projection.register_forward_hook(position_hook))

        modules = _module_map(raw)
        for site_name in requested:
            if site_name.startswith(f"{_POSITION_PREFIX}."):
                continue

            def hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = site_name,
            ) -> None:
                record(name, _first_tensor(output))

            handles.append(modules[site_name].register_forward_hook(hook))
        try:
            yield buffer
        finally:
            for handle in reversed(handles):
                handle.remove()

    @contextmanager
    def intervene(
        self, model: Any, interventions: Mapping[str, Intervention]
    ) -> Iterator[None]:
        raw = _raw_model(model)
        available = self.list_sites(raw)
        unknown = sorted(set(interventions) - available.keys())
        if unknown:
            raise KeyError(f"unknown TabPFN v2.6 intervention sites: {unknown}")
        positional_sites = [
            name for name in interventions if name.startswith(f"{_POSITION_PREFIX}.")
        ]
        if len(positional_sites) > 1:
            raise ValueError(
                "at most one feature-position component may be intervened on per "
                "context; combine component edits explicitly"
            )
        handles: list[Any] = []
        modules = _module_map(raw)
        for site_name, intervention in interventions.items():
            if not callable(intervention):
                raise TypeError(f"intervention for {site_name!r} must be callable")
            if site_name.startswith(f"{_POSITION_PREFIX}."):
                component = site_name.rsplit(".", 1)[-1]
                if component == "input":
                    raise ValueError(
                        "intervene on a projected position component, not its input code"
                    )
                projection = self._position_projection(raw)

                def position_hook(
                    module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    *,
                    name: str = site_name,
                    selected: str = component,
                ) -> Any:
                    codes = args[0]
                    components = position_components(module, codes)
                    source = (
                        _first_tensor(output)
                        if selected == "active"
                        else components[selected]
                    )
                    record = ActivationRecord(
                        tensor=source,
                        site=name,
                        axis_names=available[name].axis_names,
                        shape=tuple(source.shape),
                        model_sha="",
                        checkpoint_sha="",
                        preprocessing_view_id="",
                    )
                    replacement = interventions[name](record)
                    if isinstance(replacement, ActivationRecord):
                        replacement = replacement.tensor
                    if not hasattr(replacement, "shape") or tuple(
                        replacement.shape
                    ) != tuple(source.shape):
                        raise ValueError(
                            f"intervention for {name!r} must preserve shape {tuple(source.shape)}"
                        )
                    if selected == "weight":
                        return replacement + components["bias"]
                    if selected == "bias":
                        return components["weight"] + replacement
                    return replacement

                handles.append(
                    projection.register_forward_hook(position_hook, prepend=True)
                )
                continue

            def hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = site_name,
            ) -> Any:
                original = _first_tensor(output)
                record = ActivationRecord(
                    tensor=original,
                    site=name,
                    axis_names=available[name].axis_names,
                    shape=tuple(original.shape),
                    model_sha="",
                    checkpoint_sha="",
                    preprocessing_view_id="",
                )
                replacement = interventions[name](record)
                if isinstance(replacement, ActivationRecord):
                    replacement = replacement.tensor
                if not hasattr(replacement, "shape") or tuple(
                    replacement.shape
                ) != tuple(original.shape):
                    raise ValueError(
                        f"intervention for {name!r} must preserve shape {tuple(original.shape)}"
                    )
                return _replace_first_tensor(output, replacement)

            handles.append(modules[site_name].register_forward_hook(hook))
        try:
            yield None
        finally:
            for handle in reversed(handles):
                handle.remove()

    @contextmanager
    def position_policy(
        self, model: Any, component: PositionComponent
    ) -> Iterator[None]:
        """Return exactly ``Wp+b``, ``Wp``, ``b`` or zero from the PE projection."""

        if component not in ("full", "weight", "bias", "none"):
            raise ValueError("position component must be full, weight, bias, or none")
        if component == "full":
            yield None
            return
        projection = self._position_projection(model)

        def hook(module: Any, args: tuple[Any, ...], _output: Any) -> Any:
            if not args:
                raise RuntimeError(
                    "position projection was called without column codes"
                )
            return position_components(module, args[0])[component]

        handle = projection.register_forward_hook(hook, prepend=True)
        try:
            yield None
        finally:
            handle.remove()

    def predict(self, model: Any, batch: Any) -> Any:
        raw = _raw_model(model)
        if isinstance(batch, Mapping):
            if "x" not in batch or "y" not in batch:
                raise KeyError("TabPFN v2.6 prepared batch requires x and y")
            allowed = {
                "only_return_standard_out",
                "categorical_inds",
                "performance_options",
                "task_type",
            }
            unknown = batch.keys() - {"x", "y"} - allowed
            if unknown:
                raise ValueError(
                    f"unsupported TabPFN v2.6 batch fields: {sorted(unknown)}"
                )
            args = (batch["x"], batch["y"])
            kwargs = {key: batch[key] for key in allowed if key in batch}
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            args = (batch[0], batch[1])
            kwargs = {}
        else:
            raise TypeError("TabPFN v2.6 batch must be a mapping or (x, y) pair")
        import torch

        if "performance_options" in kwargs:
            _validate_instrumentation_safe_performance_options(
                kwargs["performance_options"]
            )
        else:
            kwargs["performance_options"] = _instrumentation_safe_performance_options(
                raw
            )
        raw.eval()
        with torch.inference_mode():
            return raw(*args, **kwargs)


__all__ = [
    "PositionComponent",
    "TabPFNV26Adapter",
    "position_components",
]
