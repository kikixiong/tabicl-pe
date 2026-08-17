from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_fingerprint_beyondarena_exploratory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_fingerprint_beyondarena_exploratory", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FULLSIZE = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
    "row_fingerprint_dim": 16,
}


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "iid-hash",
                "dataset_name": "iid",
                "tid": 1,
                "problem_type": "binary",
                "eval_metric": "roc_auc",
                "task_type": "random",
                "num_instances": 100,
                "n_features": 4,
                "num_text_cols": 0,
            },
            {
                "dataset": "grouped-hash",
                "dataset_name": "grouped",
                "tid": 2,
                "problem_type": "multiclass",
                "eval_metric": "log_loss",
                "task_type": "grouped",
                "num_instances": 200,
                "n_features": 8,
                "num_text_cols": 0,
            },
        ]
    )


def _checkpoint(path: Path, *, arm: str, step: int = 50_000) -> None:
    config = dict(FULLSIZE)
    config.update(
        row_identity_mode="rope" if arm == "rope" else "none",
        row_fingerprint=arm == "fingerprint",
    )
    torch.save(
        {
            "curr_step": step,
            "config": config,
            "state_dict": {"weight": torch.zeros(2, 3)},
            "prior_stream": {
                "cursor": step,
                "experiment_seed": 42,
                "ddp_rank": 0,
                "world_size": 1,
                "schema_sha256": "a" * 64,
            },
        },
        path,
    )


def test_default_smoke_covers_all_three_split_regimes() -> None:
    assert MODULE.DEFAULT_DATASETS == (
        "blood_transfusion",
        "parkinsons_biomedical_voice_measurements",
        "ghanas_indigenous_intel",
    )


def test_dataset_metadata_preserves_request_order_and_maps_random_to_iid() -> None:
    roster, expected = MODULE._select_dataset_metadata(
        _metadata(), ("grouped", "iid")
    )
    assert roster == ("grouped-hash", "iid-hash")
    assert expected["iid-hash"]["split_regime"] == "iid"
    assert expected["grouped-hash"]["split_regime"] == "grouped"
    assert expected["grouped-hash"]["metric"] == "log_loss"


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (("iid", "iid"), "unique"),
        (("missing",), "lacks requested"),
    ],
)
def test_dataset_metadata_fails_closed(requested: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._select_dataset_metadata(_metadata(), requested)


def test_dataset_metadata_rejects_text_tasks() -> None:
    metadata = _metadata()
    metadata.loc[metadata.dataset_name == "iid", "num_text_cols"] = 1
    with pytest.raises(ValueError, match="text"):
        MODULE._select_dataset_metadata(metadata, ("iid",))


def test_checkpoint_contract_accepts_matched_fullsize_pair(tmp_path: Path) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope")
    _checkpoint(fingerprint, arm="fingerprint")
    contract = MODULE._checkpoint_contract(
        rope, fingerprint, comparison_step=50_000
    )
    assert contract["rope"]["curr_step"] == 50_000
    assert contract["fingerprint"]["model_state_elements"] == 6


def test_checkpoint_contract_rejects_prior_mismatch(tmp_path: Path) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope")
    _checkpoint(fingerprint, arm="fingerprint")
    payload = torch.load(fingerprint, map_location="cpu", weights_only=True)
    payload["prior_stream"]["schema_sha256"] = "b" * 64
    torch.save(payload, fingerprint)
    with pytest.raises(ValueError, match="prior streams are not exactly matched"):
        MODULE._checkpoint_contract(rope, fingerprint, comparison_step=50_000)
