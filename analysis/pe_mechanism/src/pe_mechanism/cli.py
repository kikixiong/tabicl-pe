"""Command-line entry point for PE mechanism analysis workflows."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

from .manifest import validate_output_dir


Runner = Callable[[argparse.Namespace], int | None]

_COMMAND_MODULES = {
    "collect": "collect",
    "official-collect": "official_collect",
    "ablate": "ablate",
    "localize": "localization",
    "tabpfn-localize": "tabpfn_localization",
    "train-repr": "representation",
    "reconstruction-sensitivity": "causal",
    "model-causal": "official_causal",
    "select-features": "feature_selection",
    "confirm-features": "confirmation",
}


def _external_output_dir(value: str) -> Path:
    try:
        return validate_output_dir(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _config_path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pe-mechanism",
        description="Run reproducible positional-encoding mechanism analyses.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    descriptions = {
        "collect": "Collect documented internal activations.",
        "official-collect": (
            "Collect activations through official TabICL sklearn inference."
        ),
        "ablate": "Run component and positional-encoding ablations.",
        "localize": "Run matched-step fixed-weight PE localization.",
        "tabpfn-localize": (
            "Run official TabPFN v2.6 fixed-weight position localization."
        ),
        "train-repr": "Train dense or sparse representation models.",
        "reconstruction-sensitivity": (
            "Run reconstruction-space sensitivity diagnostics; this is not a model causal test."
        ),
        "model-causal": (
            "Run official TabICL model-in-the-loop representation interventions."
        ),
        "select-features": (
            "Select validation-replicated features and freeze held-out choices."
        ),
        "confirm-features": (
            "Confirm frozen features across the complete held-out roster."
        ),
    }
    for command in _COMMAND_MODULES:
        command_parser = subparsers.add_parser(command, help=descriptions[command])
        command_parser.add_argument(
            "--config",
            type=_config_path,
            required=True,
            help="JSON configuration for this workflow.",
        )
        command_parser.add_argument(
            "--output-dir",
            type=_external_output_dir,
            required=True,
            help="Absolute output directory outside the Git source tree.",
        )

    return parser


def _load_runner(command: str) -> Runner:
    module_name = _COMMAND_MODULES[command]
    module: ModuleType = importlib.import_module(f".{module_name}", package=__package__)
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise RuntimeError(f"pe_mechanism.{module_name} must define callable run(args)")
    return runner


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _load_runner(args.command)(args)
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    raise TypeError(f"workflow returned {type(result).__name__}; expected int or None")
