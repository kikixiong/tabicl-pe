import os

import pytest
import torch

from tabicl.train._checkpoint_io import atomic_torch_save


def test_atomic_save_replaces_checkpoint_without_visible_partial(tmp_path):
    checkpoint = tmp_path / "step-10.ckpt"
    atomic_torch_save({"step": 1}, checkpoint)
    atomic_torch_save({"step": 2}, checkpoint)

    assert torch.load(checkpoint, weights_only=True) == {"step": 2}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step-10.ckpt"]


def test_failed_serialization_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-10.ckpt"
    atomic_torch_save({"step": 1}, checkpoint)

    def fail_save(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_torch_save({"step": 2}, checkpoint)

    assert torch.load(checkpoint, weights_only=True) == {"step": 1}
    assert not any(path.suffix == ".ckpt" and path != checkpoint for path in tmp_path.iterdir())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step-10.ckpt"]


def test_failed_replace_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-10.ckpt"
    atomic_torch_save({"step": 1}, checkpoint)

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_torch_save({"step": 2}, checkpoint)

    assert torch.load(checkpoint, weights_only=True) == {"step": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step-10.ckpt"]
