"""Validation for strings that may cross the private-to-public artifact boundary."""

from __future__ import annotations

import math
import re
from typing import Any


_PORTABLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\[\]-]{0,191}$")


def require_portable_identifier(value: Any, *, name: str = "identifier") -> str:
    resolved = value if isinstance(value, str) else str(value)
    if not _PORTABLE_IDENTIFIER.fullmatch(resolved):
        raise ValueError(
            f"{name} must be a portable identifier without paths, whitespace, or control characters"
        )
    return resolved


def require_public_label(value: Any, *, name: str = "label") -> str:
    """Allow human-readable labels while rejecting paths and control text."""

    resolved = value if isinstance(value, str) else str(value)
    if (
        not resolved
        or len(resolved) > 191
        or resolved != resolved.strip()
        or resolved in {".", ".."}
        or "/" in resolved
        or "\\" in resolved
        or any(ord(character) < 32 or ord(character) == 127 for character in resolved)
    ):
        raise ValueError(f"{name} must be a public-safe label without path separators or controls")
    return resolved


def validate_public_value(value: Any, *, name: str = "metadata") -> None:
    """Reject path-like or arbitrary strings in recursively public metadata."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} numbers must be finite")
        return
    if isinstance(value, str):
        require_portable_identifier(value, name=name)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_public_value(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            portable_key = require_portable_identifier(key, name=f"{name} key")
            validate_public_value(item, name=f"{name}.{portable_key}")
        return
    raise ValueError(f"{name} contains unsupported public value type {type(value).__name__}")


__all__ = ["require_portable_identifier", "require_public_label", "validate_public_value"]
