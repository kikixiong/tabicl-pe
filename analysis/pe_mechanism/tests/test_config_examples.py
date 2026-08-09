from __future__ import annotations

import json
from pathlib import Path

import pe_mechanism.confirmation as confirmation
import pe_mechanism.feature_selection as selection


EXAMPLES = Path(__file__).parents[1] / "examples"


def _load(name: str) -> dict:
    value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_select_features_example_matches_the_strict_config_shape() -> None:
    config = selection._exact_object(
        _load("select-features.example.json"),
        label="example",
        fields=selection._TOP_LEVEL_FIELDS,
    )
    parameters = selection._selection_parameters(config["selection"])
    assert parameters.minimum_validation_datasets == 8
    assert parameters.maximum_selections == 1

    candidates = config["candidates"]
    assert isinstance(candidates, list) and len(candidates) == 1
    candidate = selection._exact_object(
        candidates[0],
        label="candidate example",
        fields=selection._CANDIDATE_FIELDS,
    )
    assert len(candidate["validation_runs"]) == 8
    for run in candidate["validation_runs"]:
        values = selection._exact_object(
            run, label="validation example", fields=selection._RUN_FIELDS
        )
        selection._required_sha256(
            values["expected_manifest_sha256"],
            name="expected_manifest_sha256",
        )

    heldout = selection._exact_object(
        config["heldout"], label="heldout example", fields=selection._HELDOUT_FIELDS
    )
    assert len(heldout["sample_rosters"]) == 8
    selection._exact_object(
        heldout["confirmation"],
        label="confirmation example",
        fields=selection._CONFIRMATION_FIELDS,
    )


def test_confirm_features_example_matches_the_strict_config_shape() -> None:
    config = selection._exact_object(
        _load("confirm-features.example.json"),
        label="example",
        fields=confirmation._TOP_LEVEL_FIELDS,
    )
    selection._required_sha256(
        config["expected_selection_manifest_sha256"],
        name="expected_selection_manifest_sha256",
    )
    assert len(config["heldout_runs"]) == 8
    for run in config["heldout_runs"]:
        values = selection._exact_object(
            run,
            label="heldout run example",
            fields=confirmation._HELDOUT_RUN_FIELDS,
        )
        selection._required_sha256(
            values["expected_manifest_sha256"],
            name="expected_manifest_sha256",
        )
