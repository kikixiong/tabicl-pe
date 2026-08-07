from __future__ import annotations

import pytest

from pe_mechanism.identifiers import (
    require_portable_identifier,
    require_public_label,
    validate_public_value,
)


def test_portable_identifier_accepts_roster_tokens_and_rejects_paths() -> None:
    assert require_portable_identifier("row.blocks[0]") == "row.blocks[0]"
    for value in ("/private/data", "two words", "..\\private", "line\nbreak"):
        with pytest.raises(ValueError, match="portable"):
            require_portable_identifier(value)


def test_public_metadata_is_recursive_and_finite() -> None:
    validate_public_value({"split": "held_out", "seed": 42, "features": [1, 2]})
    with pytest.raises(ValueError, match="portable"):
        validate_public_value({"source": "/private/data"})
    with pytest.raises(ValueError, match="finite"):
        validate_public_value({"effect": float("nan")})


def test_public_label_allows_display_spaces_but_not_paths() -> None:
    assert require_public_label("Dataset 003") == "Dataset 003"
    with pytest.raises(ValueError, match="public-safe"):
        require_public_label("private/data")
