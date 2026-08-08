#!/usr/bin/env python3
"""Fail before formal work when the exact runtime environment has drifted."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _import_exact_provenance(root: Path):
    preloaded = sorted(
        name for name in sys.modules if name == "tabicl" or name.startswith("tabicl.")
    )
    if preloaded:
        raise ValueError("tabicl was imported before exact-T isolation")
    sys.path.insert(0, os.fspath(root / "src"))
    from tabicl.train import _provenance as provenance

    expected = root / "src" / "tabicl" / "train" / "_provenance.py"
    raw_file = getattr(provenance, "__file__", None)
    if not isinstance(raw_file, str) or Path(raw_file) != expected:
        raise ValueError("formal environment preflight imported tabicl outside exact T")
    actual = Path(raw_file).resolve(strict=True)
    if actual != expected.resolve(strict=True) or not actual.is_file():
        raise ValueError("formal environment preflight imported tabicl outside exact T")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-gpus", required=True, type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal environment preflight requires Python -I -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("formal environment preflight requires PYTHONNOUSERSITE=1")
    root = Path(__file__).resolve(strict=True).parent.parent
    if not args.exact_root.is_absolute() or args.exact_root.resolve(strict=True) != root:
        raise ValueError("formal environment preflight is not running from exact T")
    if _HEX64.fullmatch(args.expected_sha256) is None:
        raise ValueError("expected environment digest is malformed")
    provenance = _import_exact_provenance(root)
    environment = provenance.runtime_environment_manifest(require_formal_runtime=True)
    if environment["payload"]["visible_cuda_device_count"] != args.expected_gpus:
        raise ValueError("formal environment visible CUDA count mismatch")
    if environment["sha256"] != args.expected_sha256:
        raise ValueError("formal environment differs from the precommitted digest")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"formal environment rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
