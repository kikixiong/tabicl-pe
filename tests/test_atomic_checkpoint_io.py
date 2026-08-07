import os

import pytest
import torch

import tabicl.train._checkpoint_io as checkpoint_io
from tabicl.train._checkpoint_io import (
    CheckpointSizeLimitError,
    atomic_torch_save,
)


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


def test_real_torch_save_accepts_exact_size_and_rejects_one_byte_less(tmp_path):
    checkpoint = tmp_path / "step-10.ckpt"
    value = {"step": 1, "tensor": torch.arange(32)}
    atomic_torch_save(value, checkpoint)
    exact_size = checkpoint.stat().st_size

    atomic_torch_save(value, checkpoint, max_bytes=exact_size)
    trusted_bytes = checkpoint.read_bytes()
    with pytest.raises(CheckpointSizeLimitError, match="max_checkpoint_bytes"):
        atomic_torch_save(value, checkpoint, max_bytes=exact_size - 1)

    assert checkpoint.read_bytes() == trusted_bytes
    assert torch.load(checkpoint, weights_only=True)["step"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [checkpoint.name]


def test_bounded_writer_rejects_c_plus_one_before_writing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-10.ckpt"
    checkpoint.write_bytes(b"trusted")
    original_named_temporary_file = checkpoint_io.tempfile.NamedTemporaryFile
    tracked = []

    class SizeTrackingTemporary:
        def __init__(self, *args, **kwargs):
            self._temporary = original_named_temporary_file(*args, **kwargs)
            self.name = self._temporary.name
            self.max_size = 0
            tracked.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._temporary.close()

        def write(self, data):
            written = self._temporary.write(data)
            self._temporary.flush()
            self.max_size = max(
                self.max_size, os.fstat(self._temporary.fileno()).st_size
            )
            return written

        def seek(self, offset, whence=os.SEEK_SET):
            return self._temporary.seek(offset, whence)

        def tell(self):
            return self._temporary.tell()

        def truncate(self, size=None):
            result = self._temporary.truncate(size)
            self._temporary.flush()
            self.max_size = max(
                self.max_size, os.fstat(self._temporary.fileno()).st_size
            )
            return result

        def flush(self):
            self._temporary.flush()

        def fileno(self):
            return self._temporary.fileno()

    monkeypatch.setattr(
        checkpoint_io.tempfile,
        "NamedTemporaryFile",
        SizeTrackingTemporary,
    )

    def write_past_ceiling(_value, handle):
        assert not hasattr(handle, "fileno")
        assert not hasattr(handle, "raw")
        assert handle.write(b"1234") == 4
        handle.write(b"5")

    monkeypatch.setattr(torch, "save", write_past_ceiling)
    with pytest.raises(CheckpointSizeLimitError, match=r"5 > 4"):
        atomic_torch_save({}, checkpoint, max_bytes=4)

    assert checkpoint.read_bytes() == b"trusted"
    assert len(tracked) == 1
    assert tracked[0].max_size == 4
    assert sorted(path.name for path in tmp_path.iterdir()) == [checkpoint.name]


def test_bounded_writer_rejects_seek_past_ceiling_before_moving(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-10.ckpt"
    checkpoint.write_bytes(b"trusted")

    def seek_past_ceiling(_value, handle):
        assert handle.tell() == 0
        handle.seek(5)

    monkeypatch.setattr(torch, "save", seek_past_ceiling)
    with pytest.raises(CheckpointSizeLimitError, match=r"5 > 4"):
        atomic_torch_save({}, checkpoint, max_bytes=4)

    assert checkpoint.read_bytes() == b"trusted"
    assert sorted(path.name for path in tmp_path.iterdir()) == [checkpoint.name]


@pytest.mark.parametrize(
    "offset,whence,initial_bytes",
    [
        (-1, os.SEEK_SET, b""),
        (-1, os.SEEK_CUR, b""),
        (-5, os.SEEK_END, b"1234"),
    ],
)
def test_bounded_writer_rejects_negative_seek_without_moving(
    tmp_path, monkeypatch, offset, whence, initial_bytes
):
    checkpoint = tmp_path / "step-10.ckpt"
    checkpoint.write_bytes(b"trusted")

    def seek_negative(_value, handle):
        handle.write(initial_bytes)
        handle.seek(0)
        position = handle.tell()
        try:
            handle.seek(offset, whence)
        except ValueError:
            assert handle.tell() == position
            raise

    monkeypatch.setattr(torch, "save", seek_negative)
    with pytest.raises(ValueError, match="non-negative"):
        atomic_torch_save({}, checkpoint, max_bytes=4)

    assert checkpoint.read_bytes() == b"trusted"
    assert sorted(path.name for path in tmp_path.iterdir()) == [checkpoint.name]


def test_bounded_writer_rejects_seek_end_past_high_water(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-10.ckpt"
    checkpoint.write_bytes(b"trusted")

    def seek_past_end(_value, handle):
        handle.write(b"1234")
        handle.seek(1, os.SEEK_END)

    monkeypatch.setattr(torch, "save", seek_past_end)
    with pytest.raises(CheckpointSizeLimitError, match=r"5 > 4"):
        atomic_torch_save({}, checkpoint, max_bytes=4)

    assert checkpoint.read_bytes() == b"trusted"
    assert sorted(path.name for path in tmp_path.iterdir()) == [checkpoint.name]


@pytest.mark.parametrize("max_bytes", [True, False, 0, -1, 1.5, "10"])
def test_atomic_save_rejects_invalid_ceiling_without_creating_parent(
    tmp_path, max_bytes
):
    checkpoint = tmp_path / "missing" / "step-10.ckpt"
    with pytest.raises(ValueError, match="positive integer or None"):
        atomic_torch_save({}, checkpoint, max_bytes=max_bytes)
    assert not checkpoint.parent.exists()
