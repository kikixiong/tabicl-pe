"""TabICLv2 activation tracing and scoped Row-RoPE interventions.

The adapter intentionally consumes the raw, already-numerical tensors accepted by
``TabICL.forward``.  Dataset preprocessing and ensemble-view construction stay outside
this module so every captured activation can be tied to an explicit view identifier.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from .base import ActivationRecord, ActivationSite, Intervention, ModelAdapter


FrequencyBand = Literal["all", "high", "low"] | tuple[int, int]


def same_feature_group_map(
    num_features: int, group_size: int = 3
) -> tuple[tuple[int, ...], ...]:
    """Return TabICLv2's exact circular ``feature_group='same'`` mapping.

    The implementation groups token ``j`` with original columns
    ``(j + 2**i) % H``.  In particular, size three means offsets ``(1, 2, 4)``;
    it is neither a consecutive window nor a group containing column ``j``.
    """

    if isinstance(num_features, bool) or num_features <= 0:
        raise ValueError("num_features must be a positive integer")
    if isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    return tuple(
        tuple((group_index + 2**offset) % num_features for offset in range(group_size))
        for group_index in range(num_features)
    )


def _raw_model(model: Any) -> Any:
    return getattr(model, "model_", model)


def _module_map(model: Any) -> dict[str, Any]:
    raw = _raw_model(model)
    if not hasattr(raw, "named_modules"):
        raise TypeError("TabICL adapter requires a torch.nn.Module-like model")
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

    Concatenating repeated calls onto the first tensor axis is unsafe: that axis
    already has a model-specific meaning (usually ``table``), and doing so loses
    invocation/view boundaries while retaining only the first feature-group map.
    Callers must open a fresh capture context for every model invocation.
    """

    if record.site in buffer:
        raise RuntimeError(
            f"site {record.site!r} was called more than once in one capture context; "
            "capture one model invocation per context so tensor axes and "
            "feature_group_map stay aligned"
        )
    buffer[record.site] = record


class _SelectiveRoPE:
    """Per-forward RoPE proxy that distinguishes the q call from the k call."""

    def __init__(
        self,
        base: Any,
        *,
        rotate_queries: bool,
        rotate_keys: bool,
        phase_strength: float,
        frequency_band: FrequencyBand,
        heads: tuple[int, ...] | None,
    ) -> None:
        self.base = base
        self.rotate_queries = rotate_queries
        self.rotate_keys = rotate_keys
        self.phase_strength = phase_strength
        self.frequency_band = frequency_band
        self.heads = heads
        self._call_index = 0

    def rotate_queries_or_keys(
        self,
        tensor: Any,
        seq_dim: int | None = None,
        offset: int = 0,
        scale: Any = None,
    ) -> Any:
        role_is_query = self._call_index == 0
        self._call_index += 1
        enabled = self.rotate_queries if role_is_query else self.rotate_keys
        if not enabled or self.phase_strength == 0.0:
            return tensor

        # Delegate the unmodified case so baseline numerical behaviour is byte-for-byte
        # owned by the installed TabICL version.
        if (
            self.phase_strength == 1.0
            and self.frequency_band == "all"
            and self.heads is None
        ):
            return self.base.rotate_queries_or_keys(
                tensor, seq_dim=seq_dim, offset=offset, scale=scale
            )

        if getattr(self.base, "use_xpos", False):
            raise ValueError("selective RoPE policies do not support XPOS")
        seq_dim = (
            getattr(self.base, "default_seq_dim", -2) if seq_dim is None else seq_dim
        )
        if seq_dim not in (-2, tensor.ndim - 2):
            raise ValueError(
                "selective RoPE currently requires the sequence axis at -2"
            )

        import torch

        pair_count = tensor.shape[-1] // 2
        if pair_count == 0 or tensor.shape[-1] % 2:
            raise ValueError("RoPE head dimension must be a positive even number")
        base_freqs = self.base.freqs[:pair_count].to(device=tensor.device)
        positions = self.base.get_seq_pos(
            tensor.shape[-2], device=tensor.device, dtype=tensor.dtype, offset=offset
        )
        angles = positions.to(base_freqs.dtype)[:, None] * base_freqs[None, :]
        angles = angles * self.phase_strength

        band_mask = torch.zeros(pair_count, dtype=torch.bool, device=tensor.device)
        if self.frequency_band == "all":
            band_mask[:] = True
        elif self.frequency_band == "high":
            band_mask[: (pair_count + 1) // 2] = True
        elif self.frequency_band == "low":
            band_mask[pair_count // 2 :] = True
        else:
            start, stop = self.frequency_band
            if start < 0 or stop <= start or stop > pair_count:
                raise ValueError(
                    f"frequency band {self.frequency_band!r} is outside [0, {pair_count})"
                )
            band_mask[start:stop] = True
        angles = torch.where(band_mask[None, :], angles, torch.zeros_like(angles))
        cosine = angles.cos().to(tensor.dtype)
        sine = angles.sin().to(tensor.dtype)

        if getattr(self.base, "interleaved", True):
            paired = tensor.unflatten(-1, (pair_count, 2))
            first, second = paired.unbind(dim=-1)
            rotated = torch.stack(
                (first * cosine - second * sine, second * cosine + first * sine), dim=-1
            ).flatten(-2)
        else:
            first, second = tensor[..., :pair_count], tensor[..., pair_count:]
            rotated = torch.cat(
                (first * cosine - second * sine, second * cosine + first * sine), dim=-1
            )

        if scale is not None:
            rotated = rotated * scale
        if self.heads is None:
            return rotated.to(tensor.dtype)
        num_heads = tensor.shape[-3]
        if any(head < 0 or head >= num_heads for head in self.heads):
            raise ValueError(
                f"head selection {self.heads!r} is outside [0, {num_heads})"
            )
        head_mask = torch.zeros(num_heads, dtype=torch.bool, device=tensor.device)
        head_mask[list(self.heads)] = True
        shape = [1] * tensor.ndim
        shape[-3] = num_heads
        return torch.where(head_mask.view(shape), rotated, tensor).to(tensor.dtype)


class TabICLAdapter(ModelAdapter):
    """Adapter for raw TabICLv2 classifier modules."""

    @property
    def model_family(self) -> str:
        return "tabiclv2"

    def load_model(
        self,
        checkpoint: Path,
        *,
        device: str,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TabICLv2 checkpoint does not exist: {checkpoint}")
        try:
            import torch
            from tabicl._model.tabicl import TabICL
        except ImportError as error:
            raise ImportError(
                "TabICLv2 analysis requires a compatible `tabicl` installation; "
                "the adapter never downloads code or weights"
            ) from error

        allowed = {"strict"}
        supplied = dict(options or {})
        unknown = supplied.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported TabICL load options: {sorted(unknown)}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or not {"config", "state_dict"} <= payload.keys()
        ):
            raise ValueError("TabICLv2 checkpoint must contain config and state_dict")
        model = TabICL(**payload["config"])
        model.load_state_dict(
            payload["state_dict"], strict=bool(supplied.get("strict", True))
        )
        model._pe_mechanism_device = str(torch.device(device))
        return model.to(device).eval()

    def list_sites(self, model: Any) -> Mapping[str, ActivationSite]:
        raw = _raw_model(model)
        modules = _module_map(raw)
        required = {"col_embedder", "row_interactor", "icl_predictor"}
        missing = sorted(required - modules.keys())
        if missing:
            raise ValueError(f"model is missing TabICLv2 modules: {missing}")

        sites: dict[str, ActivationSite] = {
            "col_embedder": ActivationSite(
                "col_embedder",
                ("table", "row", "feature_group_or_cls", "embedding"),
                "Column-group embeddings; CLS-reserved slots precede feature groups.",
            ),
            "row_interactor": ActivationSite(
                "row_interactor",
                ("table", "row", "row_representation"),
                "Flattened outputs of all RowInteraction CLS slots.",
            ),
            "icl_predictor": ActivationSite(
                "icl_predictor",
                ("table", "test_row", "class_or_quantile"),
                "Final raw prediction tensor.",
            ),
        }
        row_blocks = getattr(getattr(raw.row_interactor, "tf_row", None), "blocks", ())
        for index, block in enumerate(row_blocks):
            token_axis = (
                "cls" if index == len(row_blocks) - 1 else "feature_group_or_cls"
            )
            prefix = f"row_interactor.tf_row.blocks.{index}"
            axes = ("table", "row", token_axis, "embedding")
            sites[prefix] = ActivationSite(prefix, axes, "RowInteraction block output.")
            if hasattr(block, "attn"):
                sites[f"{prefix}.attn"] = ActivationSite(
                    f"{prefix}.attn", axes, "Attention branch before residual addition."
                )
            if hasattr(block, "linear2"):
                sites[f"{prefix}.linear2"] = ActivationSite(
                    f"{prefix}.linear2",
                    axes,
                    "Feed-forward output projection before residual addition.",
                )
        icl_blocks = getattr(getattr(raw.icl_predictor, "tf_icl", None), "blocks", ())
        for index, block in enumerate(icl_blocks):
            prefix = f"icl_predictor.tf_icl.blocks.{index}"
            axes = ("table", "row", "row_representation")
            sites[prefix] = ActivationSite(
                prefix, axes, "In-context-learning block output."
            )
            if hasattr(block, "attn"):
                sites[f"{prefix}.attn"] = ActivationSite(
                    f"{prefix}.attn",
                    axes,
                    "In-context attention branch before residual addition.",
                )
            if hasattr(block, "linear2"):
                sites[f"{prefix}.linear2"] = ActivationSite(
                    f"{prefix}.linear2",
                    axes,
                    "In-context feed-forward output projection.",
                )
        return sites

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
            raise KeyError(f"unknown TabICLv2 capture sites: {unknown}")

        buffer: dict[str, ActivationRecord] = {}
        handles: list[Any] = []
        runtime_group_map = {"value": feature_group_map}
        col = raw.col_embedder
        group_mode = getattr(col, "feature_group", None)
        group_size = int(getattr(col, "feature_group_size", 1))

        def infer_groups(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            if runtime_group_map["value"] is not None or group_mode not in (
                True,
                "same",
            ):
                return
            inputs = args[0] if args else kwargs.get("X")
            if (
                inputs is not None
                and hasattr(inputs, "shape")
                and len(inputs.shape) >= 3
            ):
                runtime_group_map["value"] = same_feature_group_map(
                    int(inputs.shape[-1]), group_size
                )

        handles.append(raw.register_forward_pre_hook(infer_groups, with_kwargs=True))

        modules = _module_map(raw)
        for site_name in requested:
            site = available[site_name]

            def hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = site_name,
            ) -> None:
                tensor = _snapshot(_first_tensor(output))
                record = ActivationRecord(
                    tensor=tensor,
                    site=name,
                    axis_names=available[name].axis_names,
                    shape=tuple(tensor.shape),
                    model_sha=model_sha,
                    checkpoint_sha=checkpoint_sha,
                    preprocessing_view_id=preprocessing_view_id,
                    feature_group_map=runtime_group_map["value"],
                )
                _append_record(buffer, record)

            handles.append(modules[site.name].register_forward_hook(hook))
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
            raise KeyError(f"unknown TabICLv2 intervention sites: {unknown}")
        modules = _module_map(raw)
        handles: list[Any] = []
        for site_name, intervention in interventions.items():
            if not callable(intervention):
                raise TypeError(f"intervention for {site_name!r} must be callable")

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
    def rope_policy(
        self,
        model: Any,
        *,
        blocks: Sequence[int] | None = None,
        rotate_queries: bool = True,
        rotate_keys: bool = True,
        phase_strength: float = 1.0,
        frequency_band: FrequencyBand = "all",
        heads: Sequence[int] | None = None,
    ) -> Iterator[None]:
        """Apply a reversible Row-RoPE policy to selected RowInteraction blocks.

        ``high`` selects the first (largest angular-frequency) half of RoPE pairs;
        ``low`` selects the remaining half.  KV/repr caching must be disabled because
        cached keys were rotated before this scoped policy was installed.
        """

        raw = _raw_model(model)
        if getattr(raw, "_cache", None) is not None:
            raise ValueError(
                "Row-RoPE intervention requires TabICL inference cache to be clear"
            )
        if not isinstance(phase_strength, (int, float)) or phase_strength < 0:
            raise ValueError("phase_strength must be a non-negative number")
        if not isinstance(rotate_queries, bool) or not isinstance(rotate_keys, bool):
            raise TypeError("rotate_queries and rotate_keys must be booleans")
        if frequency_band not in ("all", "high", "low") and not (
            isinstance(frequency_band, tuple)
            and len(frequency_band) == 2
            and all(isinstance(value, int) for value in frequency_band)
        ):
            raise ValueError(
                "frequency_band must be all/high/low or a (start, stop) pair"
            )

        row_encoder = getattr(raw.row_interactor, "tf_row", None)
        row_blocks = tuple(getattr(row_encoder, "blocks", ()))
        if not row_blocks:
            raise ValueError("model has no RowInteraction blocks")
        selected = (
            tuple(range(len(row_blocks)))
            if blocks is None
            else tuple(int(i) for i in blocks)
        )
        if len(set(selected)) != len(selected) or any(
            i < 0 or i >= len(row_blocks) for i in selected
        ):
            raise ValueError(f"blocks must be unique indices in [0, {len(row_blocks)})")
        selected_heads = None if heads is None else tuple(int(head) for head in heads)
        if selected_heads is not None and len(set(selected_heads)) != len(
            selected_heads
        ):
            raise ValueError("heads must not contain duplicates")

        handles: list[Any] = []
        for index in selected:

            def pre_hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                base_rope = kwargs.get("rope")
                if base_rope is None:
                    if rotate_queries or rotate_keys:
                        raise ValueError("selected TabICL block has no RoPE to modify")
                    return args, kwargs
                updated = dict(kwargs)
                updated["rope"] = _SelectiveRoPE(
                    base_rope,
                    rotate_queries=rotate_queries,
                    rotate_keys=rotate_keys,
                    phase_strength=float(phase_strength),
                    frequency_band=frequency_band,
                    heads=selected_heads,
                )
                return args, updated

            handles.append(
                row_blocks[index].register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        try:
            yield None
        finally:
            for handle in reversed(handles):
                handle.remove()

    def predict(self, model: Any, batch: Any) -> Any:
        raw = _raw_model(model)
        if isinstance(batch, Mapping):
            if "X" not in batch or "y_train" not in batch:
                raise KeyError("TabICLv2 prepared batch requires X and y_train")
            allowed = {
                "d",
                "embed_with_test",
                "feature_shuffles",
                "return_logits",
                "softmax_temperature",
                "inference_config",
            }
            unknown = batch.keys() - {"X", "y_train"} - allowed
            if unknown:
                raise ValueError(f"unsupported TabICL batch fields: {sorted(unknown)}")
            args = (batch["X"], batch["y_train"])
            kwargs = {key: batch[key] for key in allowed if key in batch}
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            args = (batch[0], batch[1])
            kwargs = {}
        else:
            raise TypeError("TabICLv2 batch must be a mapping or (X, y_train) pair")
        import torch

        if "inference_config" not in kwargs and hasattr(raw, "_pe_mechanism_device"):
            from tabicl._model.inference_config import InferenceConfig

            resolved_device = str(raw._pe_mechanism_device)
            use_amp = torch.device(resolved_device).type == "cuda"
            inference_config = InferenceConfig()
            for manager_config in (
                inference_config.COL_CONFIG,
                inference_config.ROW_CONFIG,
                inference_config.ICL_CONFIG,
            ):
                manager_config.update(
                    {
                        "device": resolved_device,
                        "use_amp": use_amp,
                        "use_fa3": False,
                        "offload": False,
                        "use_async": False,
                    }
                )
            kwargs["inference_config"] = inference_config

        raw.eval()
        with torch.inference_mode():
            return raw(*args, **kwargs)


__all__ = ["FrequencyBand", "TabICLAdapter", "same_feature_group_map"]
