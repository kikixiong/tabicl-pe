from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from pe_mechanism import cli


@pytest.mark.parametrize(
    ("command", "module_name"),
    [
        ("collect", ".collect"),
        ("official-collect", ".official_collect"),
        ("ablate", ".ablate"),
        ("train-repr", ".representation"),
        ("reconstruction-sensitivity", ".causal"),
        ("model-causal", ".official_causal"),
        ("select-features", ".feature_selection"),
        ("confirm-features", ".confirmation"),
    ],
)
def test_cli_dispatches_to_lazy_workflow_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    module_name: str,
) -> None:
    observed: dict[str, object] = {}

    def run(args: argparse.Namespace) -> int:
        observed["args"] = args
        return 7

    def import_module(name: str, package: str | None = None) -> SimpleNamespace:
        observed["import"] = (name, package)
        return SimpleNamespace(run=run)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)
    output = tmp_path / "external-output"
    result = cli.main([command, "--config", "config.json", "--output-dir", str(output)])

    assert result == 7
    assert observed["import"] == (module_name, "pe_mechanism")
    args = observed["args"]
    assert isinstance(args, argparse.Namespace)
    assert args.command == command
    assert args.config == Path("config.json")
    assert args.output_dir == output.resolve()


def test_cli_rejects_relative_output_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(*args: object, **kwargs: object) -> None:
        raise AssertionError("workflow module must not be imported")

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)
    with pytest.raises(SystemExit) as error:
        cli.main(["collect", "--config", "config.json", "--output-dir", "relative"])
    assert error.value.code == 2


def test_cli_rejects_absolute_output_inside_source_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(*args: object, **kwargs: object) -> None:
        raise AssertionError("workflow module must not be imported")

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)
    project_output = Path(__file__).resolve().parents[1] / "generated-output"
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["collect", "--config", "config.json", "--output-dir", str(project_output)]
        )
    assert error.value.code == 2


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2


def test_cli_requires_run_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli.importlib, "import_module", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(RuntimeError, match=r"must define callable run\(args\)"):
        cli.main(
            ["collect", "--config", "config.json", "--output-dir", str(tmp_path / "output")]
        )
