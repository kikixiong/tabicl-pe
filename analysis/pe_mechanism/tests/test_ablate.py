from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pe_mechanism.provenance as provenance_module
import pytest
from pe_mechanism.ablate import run, summarize_conditions


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "initial")
    monkeypatch.setattr(provenance_module, "__file__", str(repository / "tracked.txt"))
    checkpoint = tmp_path / "raw.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"datasets":["a","b"]}\n', encoding="utf-8")
    return {
        "model_family": "tabicl-v2",
        "model_revision": "step-180000",
        "condition": "rope-off",
        "sites": ["blocks.0"],
        "checkpoint_path": str(checkpoint),
        "dataset_manifest_path": str(dataset),
        "training_code_root": str(repository),
        "model_code_root": str(repository),
        "analysis_code_root": str(repository),
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metric(value: float, *, roster: str = "a" * 64) -> dict[str, object]:
    return {"accuracy": value, "prediction_roster_sha256": roster}


def test_ablation_summary_is_paired_and_path_free(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    condition = tmp_path / "condition.json"
    _write(baseline, {"a": _metric(0.7), "b": _metric(0.8, roster="b" * 64)})
    _write(condition, {"a": _metric(0.6), "b": _metric(0.7, roster="b" * 64)})
    summary = summarize_conditions(
        {
            "schema_version": 1,
            "metric": "accuracy",
            "higher_is_better": True,
            "baseline_metrics": str(baseline),
            "conditions": [{"name": "rope-off", "metrics": str(condition)}],
            "bootstrap_resamples": 100,
            "seed": 3,
        }
    )
    result = summary["conditions"][0]
    assert result["name"] == "rope-off"
    assert result["mean_effect"] < 0.0
    assert result["count"] == 2
    assert len(summary["prediction_pairing_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(summary)


def test_missing_provenance_leaves_no_ablation_outputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    condition = tmp_path / "condition.json"
    _write(baseline, {"a": _metric(0.7)})
    _write(condition, {"a": _metric(0.6)})
    config = tmp_path / "config.json"
    _write(
        config,
        {
            "schema_version": 1,
            "baseline_metrics": str(baseline),
            "conditions": [{"name": "rope-off", "metrics": str(condition)}],
        },
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="provenance"):
        run(argparse.Namespace(config=config, output_dir=output))
    assert not output.exists()


def test_ablation_run_hashes_metric_inputs_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    condition = tmp_path / "condition.json"
    _write(baseline, {"a": _metric(0.7), "b": _metric(0.8, roster="b" * 64)})
    _write(condition, {"a": _metric(0.6), "b": _metric(0.7, roster="b" * 64)})
    config = tmp_path / "config.json"
    _write(
        config,
        {
            "schema_version": 1,
            "metric": "accuracy",
            "baseline_metrics": {
                "path": str(baseline),
                "expected_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            },
            "conditions": [
                {
                    "name": "rope-off",
                    "metrics": {
                        "path": str(condition),
                        "expected_sha256": hashlib.sha256(condition.read_bytes()).hexdigest(),
                    },
                }
            ],
            "bootstrap_resamples": 100,
            "provenance": _provenance(tmp_path, monkeypatch),
        },
    )
    output = tmp_path / "published"
    assert run(argparse.Namespace(config=config, output_dir=output)) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [item["role"] for item in manifest["inputs"]] == [
        "metrics.baseline",
        "metrics.condition.000000",
    ]
    assert {path.name for path in output.iterdir()} == {
        "ablation-summary.json",
        "manifest.json",
    }
    assert not list(tmp_path.glob(".published.*.staging"))


def test_ablation_rejects_string_boolean_nonfinite_and_path_like_public_names(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    condition = tmp_path / "condition.json"
    _write(baseline, {"a": _metric(0.7)})
    _write(condition, {"a": _metric(0.6)})
    config = {
        "schema_version": 1,
        "baseline_metrics": str(baseline),
        "conditions": [{"name": "rope-off", "metrics": str(condition)}],
    }
    with pytest.raises(ValueError, match="JSON boolean"):
        summarize_conditions({**config, "higher_is_better": "false"})

    _write(condition, {"a": _metric(float("inf"))})
    with pytest.raises(ValueError, match="finite"):
        summarize_conditions(config)

    _write(condition, {"a": _metric(0.6)})
    with pytest.raises(ValueError, match="portable"):
        summarize_conditions(
            {**config, "conditions": [{"name": "/private/run", "metrics": str(condition)}]}
        )


def test_ablation_rejects_unmatched_prediction_rosters(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    condition = tmp_path / "condition.json"
    _write(baseline, {"a": _metric(0.7, roster="a" * 64)})
    _write(condition, {"a": _metric(0.6, roster="b" * 64)})

    with pytest.raises(ValueError, match="prediction roster differs"):
        summarize_conditions(
            {
                "schema_version": 1,
                "baseline_metrics": str(baseline),
                "conditions": [{"name": "rope-off", "metrics": str(condition)}],
            }
        )
