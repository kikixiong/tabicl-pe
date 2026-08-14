from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_fingerprint_tabarena_exploratory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_fingerprint_tabarena_exploratory", SCRIPT
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


def _checkpoint(path: Path, *, arm: str, step: int = 5000) -> None:
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


def test_fullsize_checkpoint_contract_accepts_matched_pair(tmp_path: Path) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope")
    _checkpoint(fingerprint, arm="fingerprint")

    contract = MODULE._checkpoint_contract(
        rope,
        fingerprint,
        comparison_step=5000,
        model_scale="fullsize",
    )

    assert contract["rope"]["curr_step"] == 5000
    assert contract["fingerprint"]["model_state_elements"] == 6
    assert contract["rope"]["model_state_tensors"] == 1


def test_checkpoint_contract_rejects_wrong_step(tmp_path: Path) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope", step=4999)
    _checkpoint(fingerprint, arm="fingerprint")

    with pytest.raises(ValueError, match="rope checkpoint is not at step 5000"):
        MODULE._checkpoint_contract(
            rope,
            fingerprint,
            comparison_step=5000,
            model_scale="fullsize",
        )


def test_checkpoint_contract_rejects_wrong_scale(tmp_path: Path) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope")
    _checkpoint(fingerprint, arm="fingerprint")

    with pytest.raises(ValueError, match="expected compact architecture"):
        MODULE._checkpoint_contract(
            rope,
            fingerprint,
            comparison_step=5000,
            model_scale="compact",
        )


def test_checkpoint_contract_rejects_mismatched_prior_stream(
    tmp_path: Path,
) -> None:
    rope = tmp_path / "rope.ckpt"
    fingerprint = tmp_path / "fingerprint.ckpt"
    _checkpoint(rope, arm="rope")
    _checkpoint(fingerprint, arm="fingerprint")
    payload = torch.load(fingerprint, map_location="cpu", weights_only=True)
    payload["prior_stream"]["schema_sha256"] = "b" * 64
    torch.save(payload, fingerprint)

    with pytest.raises(ValueError, match="prior streams are not exactly matched"):
        MODULE._checkpoint_contract(
            rope,
            fingerprint,
            comparison_step=5000,
            model_scale="fullsize",
        )
