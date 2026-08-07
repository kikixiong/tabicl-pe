"""Durable same-filesystem checkpoint replacement."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import torch


class CheckpointSizeLimitError(ValueError):
    """Raised before checkpoint serialization can cross its byte ceiling."""


class _BoundedCheckpointWriter:
    """Restrict every file-like write, seek and truncate to ``max_bytes``."""

    def __init__(self, file_object: Any, max_bytes: int) -> None:
        self._file_object = file_object
        self._max_bytes = max_bytes

    def _require_within_ceiling(self, position: int) -> None:
        if position < 0:
            raise ValueError(
                f"checkpoint serialization position must be non-negative ({position})"
            )
        if position > self._max_bytes:
            raise CheckpointSizeLimitError(
                "checkpoint serialization exceeds max_checkpoint_bytes "
                f"({position} > {self._max_bytes})"
            )

    def write(self, data: Any) -> int:
        position = self.tell()
        attempted_position = position + len(data)
        self._require_within_ceiling(attempted_position)
        written = self._file_object.write(data)
        self._require_within_ceiling(self.tell())
        return written

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        current = self.tell()
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = current + offset
        elif whence == os.SEEK_END:
            try:
                self._file_object.seek(0, os.SEEK_END)
                end = self._file_object.tell()
            finally:
                self._file_object.seek(current, os.SEEK_SET)
            self._require_within_ceiling(end)
            target = end + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        self._require_within_ceiling(target)
        return self._file_object.seek(offset, whence)

    def truncate(self, size: int | None = None) -> int:
        target = self.tell() if size is None else size
        self._require_within_ceiling(target)
        return self._file_object.truncate(size)

    def tell(self) -> int:
        return self._file_object.tell()

    def flush(self) -> None:
        self._file_object.flush()


def _validate_max_bytes(max_bytes: int | None) -> int | None:
    if max_bytes is None:
        return None
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer or None")
    return max_bytes


def atomic_torch_save(
    value: Any,
    path: str | os.PathLike[str],
    *,
    max_bytes: int | None = None,
) -> None:
    """Save ``value`` durably and replace ``path`` without exposing partial ckpts."""
    max_bytes = _validate_max_bytes(max_bytes)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = (
                temporary
                if max_bytes is None
                else _BoundedCheckpointWriter(temporary, max_bytes)
            )
            torch.save(value, writer)
            temporary.flush()
            if max_bytes is not None and os.fstat(temporary.fileno()).st_size > max_bytes:
                raise CheckpointSizeLimitError(
                    "checkpoint serialization exceeded max_checkpoint_bytes"
                )
            os.fsync(temporary.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
