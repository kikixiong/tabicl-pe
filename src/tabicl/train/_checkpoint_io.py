"""Durable same-filesystem checkpoint replacement."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def atomic_torch_save(value: Any, path: str | os.PathLike[str]) -> None:
    """Save ``value`` durably and replace ``path`` without exposing partial ckpts."""
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
            torch.save(value, temporary)
            temporary.flush()
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
